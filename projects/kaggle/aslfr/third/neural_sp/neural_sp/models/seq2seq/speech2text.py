# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Speech to text sequence-to-sequence model."""

import copy
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn

from neural_sp.bin.train_utils import load_checkpoint
from neural_sp.models.base import ModelBase
from neural_sp.models.lm.rnnlm import RNNLM
from neural_sp.models.seq2seq.decoders.beam_search import BeamSearch
from neural_sp.models.seq2seq.decoders.build import build_decoder
from neural_sp.models.seq2seq.decoders.fwd_bwd_attention import fwd_bwd_attention
from neural_sp.models.seq2seq.decoders.las import RNNDecoder
from neural_sp.models.seq2seq.decoders.rnn_transducer import RNNTransducer as RNNT
from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder
from neural_sp.models.seq2seq.encoders.build import build_encoder
from neural_sp.models.seq2seq.frontends.frame_stacking import stack_frame
from neural_sp.models.seq2seq.frontends.input_noise import add_input_noise
from neural_sp.models.seq2seq.frontends.sequence_summary import SequenceSummaryNetwork
from neural_sp.models.seq2seq.frontends.spec_augment import SpecAugment
from neural_sp.models.seq2seq.frontends.splicing import splice
from neural_sp.models.seq2seq.frontends.streaming import Streaming
from neural_sp.models.torch_utils import (
    np2tensor,
    tensor2np,
    pad_list
)
from neural_sp.utils import mkdir_join

random.seed(1)

logger = logging.getLogger(__name__)


class Speech2Text(ModelBase):
    """Speech to text sequence-to-sequence model."""

    def __init__(self, args, save_path=None, idx2token=None):

        super(ModelBase, self).__init__()

        self.save_path = save_path

        # for encoder, decoder
        self.input_type = args.input_type
        self.input_dim = args.input_dim
        self.enc_type = args.enc_type
        self.dec_type = args.dec_type

        # for OOV resolution
        self.enc_n_layers = args.enc_n_layers
        self.enc_n_layers_sub1 = args.enc_n_layers_sub1
        self.subsample = [int(s) for s in args.subsample.split('_')]

        # for decoder
        self.vocab = args.vocab
        self.vocab_sub1 = args.vocab_sub1
        self.vocab_sub2 = args.vocab_sub2
        self.blank = 0
        self.unk = 1
        self.eos = 2
        self.pad = 3
        # NOTE: reserved in advance

        # for the sub tasks
        self.main_weight = args.total_weight - args.sub1_weight - args.sub2_weight
        self.sub1_weight = args.sub1_weight
        self.sub2_weight = args.sub2_weight
        self.mtl_per_batch = args.mtl_per_batch
        self.task_specific_layer = args.task_specific_layer

        # for CTC
        self.ctc_weight = min(args.ctc_weight, self.main_weight)
        self.ctc_weight_sub1 = min(args.ctc_weight_sub1, self.sub1_weight)
        self.ctc_weight_sub2 = min(args.ctc_weight_sub2, self.sub2_weight)

        # for backward decoder
        self.bwd_weight = min(args.bwd_weight, self.main_weight)
        self.fwd_weight = self.main_weight - self.bwd_weight - self.ctc_weight
        self.fwd_weight_sub1 = self.sub1_weight - self.ctc_weight_sub1
        self.fwd_weight_sub2 = self.sub2_weight - self.ctc_weight_sub2

        # for MBR
        self.mbr_training = args.mbr_training
        self.recog_params = vars(args)
        self.idx2token = idx2token

        # for discourse-aware model
        self.utt_id_prev = None

        # Feature extraction
        self.input_noise_std = args.input_noise_std
        self.n_stacks = args.n_stacks
        self.n_skips = args.n_skips
        self.n_splices = args.n_splices
        self.weight_noise_std = args.weight_noise_std
        self.specaug = None
        if args.n_freq_masks > 0 or args.n_time_masks > 0:
            assert args.n_stacks == 1 and args.n_skips == 1
            assert args.n_splices == 1
            self.specaug = SpecAugment(F=args.freq_width,
                                       T=args.time_width,
                                       n_freq_masks=args.n_freq_masks,
                                       n_time_masks=args.n_time_masks,
                                       p=args.time_width_upper,
                                       adaptive_number_ratio=args.adaptive_number_ratio,
                                       adaptive_size_ratio=args.adaptive_size_ratio,
                                       max_n_time_masks=args.max_n_time_masks)

        # Frontend
        self.ssn = None
        if args.sequence_summary_network:
            assert args.input_type == 'speech'
            self.ssn = SequenceSummaryNetwork(args.input_dim,
                                              n_units=512,
                                              n_layers=3,
                                              bottleneck_dim=100,
                                              dropout=0,
                                              param_init=args.param_init)

        # Encoder
        self.enc = build_encoder(args)
        if args.freeze_encoder:
            for n, p in self.enc.named_parameters():
                if 'bridge' in n or 'sub1' in n:
                    continue
                p.requires_grad = False
                logger.info('freeze %s' % n)

        special_symbols = {
            'blank': self.blank,
            'unk': self.unk,
            'eos': self.eos,
            'pad': self.pad,
        }

        # main task
        external_lm = None
        directions = []
        if self.fwd_weight > 0 or (self.bwd_weight == 0 and self.ctc_weight > 0):
            directions.append('fwd')
        if self.bwd_weight > 0:
            directions.append('bwd')

        for dir in directions:
            # Load the LM for LM fusion and decoder initialization
            if args.external_lm and dir == 'fwd':
                external_lm = RNNLM(args.lm_conf)
                load_checkpoint(args.external_lm, external_lm)
                # freeze LM parameters
                for n, p in external_lm.named_parameters():
                    p.requires_grad = False

            # Decoder
            dec = build_decoder(args, special_symbols,
                                self.enc.output_dim,
                                args.vocab,
                                self.ctc_weight,
                                self.main_weight - self.bwd_weight if dir == 'fwd' else self.bwd_weight,
                                external_lm)
            setattr(self, 'dec_' + dir, dec)

        # sub task
        for sub in ['sub1', 'sub2']:
            if getattr(self, sub + '_weight') > 0:
                args_sub = copy.deepcopy(args)
                if hasattr(args, 'dec_config_' + sub):
                    for k, v in getattr(args, 'dec_config_' + sub).items():
                        setattr(args_sub, k, v)
                # NOTE: Other parameters are the same as the main decoder
                dec_sub = build_decoder(args_sub, special_symbols,
                                        getattr(self.enc, 'output_dim_' + sub),
                                        getattr(self, 'vocab_' + sub),
                                        getattr(self, 'ctc_weight_' + sub),
                                        getattr(self, sub + '_weight'),
                                        external_lm)
                setattr(self, 'dec_fwd_' + sub, dec_sub)

        if args.input_type == 'text':
            if args.vocab == args.vocab_sub1:
                # Share the embedding layer between input and output
                self.embed = dec.embed
            else:
                self.embed = nn.Embedding(args.vocab_sub1, args.emb_dim,
                                          padding_idx=self.pad)
                self.dropout_emb = nn.Dropout(p=args.dropout_emb)

        # Initialize bias in forget gate with 1
        # self.init_forget_gate_bias_with_one()

        # Fix all parameters except for the gating parts in deep fusion
        if args.lm_fusion == 'deep' and external_lm is not None:
            for n, p in self.named_parameters():
                if 'output' in n or 'output_bn' in n or 'linear' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    def trigger_scheduled_sampling(self):
        # main task
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).trigger_scheduled_sampling()

        # sub task
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).trigger_scheduled_sampling()

    def trigger_quantity_loss(self):
        # main task only now
        if hasattr(self, 'dec_fwd'):
            getattr(self, 'dec_fwd').trigger_quantity_loss()
            getattr(self, 'dec_fwd').trigger_latency_loss()

    def trigger_stableemit(self):
        # main task only now
        if hasattr(self, 'dec_fwd'):
            getattr(self, 'dec_fwd').trigger_stableemit()

    def reset_session(self):
        # main task
        for dir in ['fwd', 'bwd']:
            if hasattr(self, 'dec_' + dir):
                getattr(self, 'dec_' + dir).reset_session()

        # sub task
        for sub in ['sub1', 'sub2']:
            if hasattr(self, 'dec_fwd_' + sub):
                getattr(self, 'dec_fwd_' + sub).reset_session()

    def forward(self, batch, task, is_eval=False, teacher=None, teacher_lm=None):
        """Forward pass.

        Args:
            batch (dict):
                xs (List): input data of size `[T, input_dim]`
                xlens (List): lengths of each element in xs
                ys (List): reference labels in the main task of size `[L]`
                ys_sub1 (List): reference labels in the 1st auxiliary task of size `[L_sub1]`
                ys_sub2 (List): reference labels in the 2nd auxiliary task of size `[L_sub2]`
                utt_ids (List): name of utterances
                speakers (List): name of speakers
            task (str): all/ys*/ys_sub*
            is_eval (bool): evaluation mode
                This should be used in inference model for memory efficiency.
            teacher (Speech2Text): used for knowledge distillation from ASR
            teacher_lm (RNNLM): used for knowledge distillation from LM
        Returns:
            loss (FloatTensor): `[1]`
            observation (dict):

        """
        if is_eval:
            self.eval()
            with torch.no_grad():
                loss, observation = self._forward(batch, task)
        else:
            self.train()
            loss, observation = self._forward(batch, task, teacher, teacher_lm)

        return loss, observation

    def _forward(self, batch, task, teacher=None, teacher_lm=None):
        # Encode input features
        if self.input_type == 'speech':
            if self.mtl_per_batch:
                eout_dict = self.encode(batch['xs'], task)
            else:
                eout_dict = self.encode(batch['xs'], 'all')
        else:
            eout_dict = self.encode(batch['ys_sub1'])

        observation = {}
        loss = torch.zeros((1,), dtype=torch.float32, device=self.device)

        # for the forward decoder in the main task
        if (self.fwd_weight > 0 or (self.bwd_weight == 0 and self.ctc_weight > 0) or self.mbr_training) and task in ['all', 'ys', 'ys.ctc', 'ys.mbr']:
            teacher_logits = None
            if teacher is not None:
                teacher.eval()
                teacher_logits = teacher.generate_logits(batch)
                # TODO(hirofumi): label smoothing, scheduled sampling, dropout?
            elif teacher_lm is not None:
                teacher_lm.eval()
                teacher_logits = self.generate_lm_logits(batch['ys'], lm=teacher_lm)

            loss_fwd, obs_fwd = self.dec_fwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'],
                                             batch['ys'], task,
                                             teacher_logits, self.recog_params, self.idx2token,
                                             batch['trigger_points'])
            loss += loss_fwd
            if isinstance(self.dec_fwd, RNNT):
                observation['loss.transducer'] = obs_fwd['loss_transducer']
            else:
                observation['acc.att'] = obs_fwd['acc_att']
                observation['ppl.att'] = obs_fwd['ppl_att']
                observation['loss.att'] = obs_fwd['loss_att']
                observation['loss.mbr'] = obs_fwd['loss_mbr']
                if 'loss_quantity' not in obs_fwd.keys():
                    obs_fwd['loss_quantity'] = None
                observation['loss.quantity'] = obs_fwd['loss_quantity']

                if 'loss_latency' not in obs_fwd.keys():
                    obs_fwd['loss_latency'] = None
                observation['loss.latency'] = obs_fwd['loss_latency']

            observation['loss.ctc'] = obs_fwd['loss_ctc']

        # for the backward decoder in the main task
        if self.bwd_weight > 0 and task in ['all', 'ys.bwd']:
            loss_bwd, obs_bwd = self.dec_bwd(eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'], task)
            loss += loss_bwd
            observation['loss.att-bwd'] = obs_bwd['loss_att']
            observation['acc.att-bwd'] = obs_bwd['acc_att']
            observation['ppl.att-bwd'] = obs_bwd['ppl_att']
            observation['loss.ctc-bwd'] = obs_bwd['loss_ctc']

        # only fwd for sub tasks
        for sub in ['sub1', 'sub2']:
            # for the forward decoder in the sub tasks
            if (getattr(self, 'fwd_weight_' + sub) > 0 or getattr(self, 'ctc_weight_' + sub) > 0) and task in ['all', 'ys_' + sub, 'ys_' + sub + '.ctc']:
                if len(batch['ys_' + sub]) == 0:
                    continue  # NOTE: for evaluation at the end of every epoch

                loss_sub, obs_fwd_sub = getattr(self, 'dec_fwd_' + sub)(
                    eout_dict['ys_' + sub]['xs'], eout_dict['ys_' + sub]['xlens'],
                    batch['ys_' + sub], task)
                loss += loss_sub
                if isinstance(getattr(self, 'dec_fwd_' + sub), RNNT):
                    observation['loss.transducer-' + sub] = obs_fwd_sub['loss_transducer']
                else:
                    observation['loss.att-' + sub] = obs_fwd_sub['loss_att']
                    observation['acc.att-' + sub] = obs_fwd_sub['acc_att']
                    observation['ppl.att-' + sub] = obs_fwd_sub['ppl_att']
                observation['loss.ctc-' + sub] = obs_fwd_sub['loss_ctc']

        return loss, observation

    def generate_logits(self, batch, temperature=1.0):
        # Encode input features
        if self.input_type == 'speech':
            eout_dict = self.encode(batch['xs'], task='ys')
        else:
            eout_dict = self.encode(batch['ys_sub1'], task='ys')

        # for the forward decoder in the main task
        logits = self.dec_fwd.forward_att(
            eout_dict['ys']['xs'], eout_dict['ys']['xlens'], batch['ys'],
            return_logits=True)
        return logits

    def generate_lm_logits(self, ys, lm, temperature=5.0):
        # Append <sos> and <eos>
        eos = next(lm.parameters()).new_zeros(1).fill_(self.eos).long()
        ys = [np2tensor(np.fromiter(y, dtype=np.int64), self.device)for y in ys]
        ys_in = pad_list([torch.cat([eos, y], dim=0) for y in ys], self.pad)
        lmout, _ = lm.decode(ys_in, None)
        logits = lm.output(lmout)
        return logits

    def encode(self, xs, task='all', streaming=False,
               cnn_lookback=False, cnn_lookahead=False, xlen_block=-1):
        """Encode acoustic or text features.

        Args:
            xs (List): length `[B]`, which contains Tensor of size `[T, input_dim]`
            task (str): all/ys*/ys_sub1*/ys_sub2*
            streaming (bool): streaming encoding
            cnn_lookback (bool): truncate leftmost frames for lookback in CNN context
            cnn_lookahead (bool): truncate rightmost frames for lookahead in CNN context
            xlen_block (int): input length in a block in the streaming mode
        Returns:
            eout_dict (dict):

        """
        if self.input_type == 'speech':
            # Frame stacking
            if self.n_stacks > 1:
                xs = [stack_frame(x, self.n_stacks, self.n_skips) for x in xs]

            # Splicing
            if self.n_splices > 1:
                xs = [splice(x, self.n_splices, self.n_stacks) for x in xs]

            if streaming:
                xlens = torch.IntTensor([xlen_block])
            else:
                xlens = torch.IntTensor([len(x) for x in xs])
            xs = pad_list([np2tensor(x, self.device).float() for x in xs], 0.)

            # SpecAugment
            if self.specaug is not None and self.training:
                xs = self.specaug(xs)

            # Weight noise injection
            if self.weight_noise_std > 0 and self.training:
                self.add_weight_noise(std=self.weight_noise_std)

            # Input Gaussian noise injection
            if self.input_noise_std > 0 and self.training:
                xs = add_input_noise(xs, std=self.input_noise_std)

            # Sequence summary network
            if self.ssn is not None:
                xs = self.ssn(xs, xlens)

        elif self.input_type == 'text':
            xlens = torch.IntTensor([len(x) for x in xs])
            xs = [np2tensor(np.fromiter(x, dtype=np.int64), self.device) for x in xs]
            xs = pad_list(xs, self.pad)
            xs = self.dropout_emb(self.embed(xs))
            # TODO(hirofumi): fix for Transformer

        # encoder
        eout_dict = self.enc(xs, xlens, task.split('.')[0], streaming,
                             cnn_lookback, cnn_lookahead)

        if self.main_weight < 1 and self.enc_type in ['conv', 'tds', 'gated_conv']:
            for sub in ['sub1', 'sub2']:
                eout_dict['ys_' + sub]['xs'] = eout_dict['ys']['xs'].clone()
                eout_dict['ys_' + sub]['xlens'] = eout_dict['ys']['xlens'][:]

        return eout_dict

    def get_ctc_probs(self, xs, task='ys', temperature=1, topk=None):
        """Get CTC top-K probabilities.

        Args:
            xs (FloatTensor): `[B, T, idim]`
            task (str): task to evaluate
            temperature (float): softmax temperature
            topk (int): top-K classes to sample
        Returns:
            probs (np.ndarray): `[B, T, vocab]`
            topk_ids (np.ndarray): `[B, T, topk]`
            elens (IntTensor): `[B]`

        """
        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, task)
            dir = 'fwd' if self.fwd_weight >= self.bwd_weight else 'bwd'
            if task == 'ys_sub1':
                dir += '_sub1'
            elif task == 'ys_sub2':
                dir += '_sub2'

            if task == 'ys':
                assert self.ctc_weight > 0
            elif task == 'ys_sub1':
                assert self.ctc_weight_sub1 > 0
            elif task == 'ys_sub2':
                assert self.ctc_weight_sub2 > 0

            probs = getattr(self, 'dec_' + dir).ctc.probs(eout_dict[task]['xs'])
            if topk is None:
                topk = probs.size(-1)  # return all classes
            _, topk_ids = torch.topk(probs, k=topk, dim=-1, largest=True, sorted=True)

            return tensor2np(probs), tensor2np(topk_ids), eout_dict[task]['xlens']

    def ctc_forced_align(self, xs, ys, task='ys'):
        """CTC-based forced alignment.

        Args:
            xs (FloatTensor): `[B, T, idim]`
            ys (List): length `B`, each of which contains a list of size `[L]`
        Returns:
            trigger_points (np.ndarray): `[B, L]`

        """
        from neural_sp.models.seq2seq.decoders.ctc import CTCForcedAligner
        forced_aligner = CTCForcedAligner()

        self.eval()
        with torch.no_grad():
            eout_dict = self.encode(xs, 'ys')
            # NOTE: support the main task only
            ctc = getattr(self, 'dec_fwd').ctc
            logits = ctc.output(eout_dict[task]['xs'])
            ylens = np2tensor(np.fromiter([len(y) for y in ys], dtype=np.int32))
            trigger_points = forced_aligner(logits, eout_dict[task]['xlens'], ys, ylens)

        return tensor2np(trigger_points)

    def plot_attention(self):
        """Plot attention weights during training."""
        # encoder
        self.enc._plot_attention(mkdir_join(self.save_path, 'enc_att_weights'))
        # decoder
        self.dec_fwd._plot_attention(mkdir_join(self.save_path, 'dec_att_weights'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_attention(mkdir_join(self.save_path, 'dec_att_weights_sub2'))

    def plot_ctc(self):
        """Plot CTC posteriors during training."""
        self.dec_fwd._plot_ctc(mkdir_join(self.save_path, 'ctc'))
        if getattr(self, 'dec_fwd_sub1', None) is not None:
            self.dec_fwd_sub1._plot_ctc(mkdir_join(self.save_path, 'ctc_sub1'))
        if getattr(self, 'dec_fwd_sub2', None) is not None:
            self.dec_fwd_sub2._plot_ctc(mkdir_join(self.save_path, 'ctc_sub2'))

    def encode_streaming(self, xs, params, task='ys'):
        """Simulate streaming encoding. Decoding is performed in the offline mode.
        Args:
            xs (FloatTensor): `[B, T, idim]`
            params (dict): hyper-parameters for decoding
            task (str): task to evaluate
        Returns:
            eout (FloatTensor): `[B, T, idim]`
            elens (IntTensor): `[B]`

        """
        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.fwd_weight > 0
        assert len(xs) == 1  # batch size
        streaming = Streaming(xs[0], params, self.enc)

        self.enc.reset_cache()
        while True:
            # Encode input features block by block
            x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block = streaming.extract_feat()
            eout_block_dict = self.encode([x_block], 'all',
                                          streaming=True,
                                          cnn_lookback=cnn_lookback,
                                          cnn_lookahead=cnn_lookahead,
                                          xlen_block=xlen_block)
            eout_block = eout_block_dict[task]['xs']
            streaming.cache_eout(eout_block)
            streaming.next_block()
            if is_last_block:
                break

        eout = streaming.pop_eouts()
        elens = torch.IntTensor([eout.size(1)])

        return eout, elens

    @torch.no_grad()
    def decode_streaming(self, xs, params, idx2token, exclude_eos=False,
                         speaker=None, task='ys'):
        """Simulate streaming encoding+decoding. Both encoding and decoding are performed in the online mode."""
        self.eval()
        block_size = params.get('recog_block_sync_size')  # before subsampling
        cache_emb = params.get('recog_cache_embedding')
        ctc_weight = params.get('recog_ctc_weight')
        backoff = True

        assert task == 'ys'
        assert self.input_type == 'speech'
        assert self.ctc_weight > 0
        assert self.fwd_weight > 0 or self.ctc_weight == 1.0
        assert len(xs) == 1  # batch size
        assert params.get('recog_block_sync')
        # assert params.get('recog_length_norm')

        streaming = Streaming(xs[0], params, self.enc, idx2token)
        factor = self.enc.subsampling_factor
        block_size //= factor
        assert block_size >= 1, "block_size is too small."
        is_transformer_enc = 'former' in self.enc.enc_type

        hyps = None
        hyps_nobd = []
        best_hyp_id_session = []
        is_reset = False

        helper = BeamSearch(params.get('recog_beam_width'),
                            self.eos,
                            params.get('recog_ctc_weight'),
                            params.get('recog_lm_weight'),
                            self.device)

        lm = getattr(self, 'lm_fwd', None)
        lm_second = getattr(self, 'lm_second', None)
        lm = helper.verify_lm_eval_mode(lm, params.get('recog_lm_weight'), cache_emb)
        if lm is not None:
            assert isinstance(lm, RNNLM)
        lm_second = helper.verify_lm_eval_mode(lm_second, params.get('recog_lm_second_weight'), cache_emb)

        # cache token embeddings
        if cache_emb and self.fwd_weight > 0:
            self.dec_fwd.cache_embedding(self.device)

        self.enc.reset_cache()
        eout_block_tail = None
        x_block_prev, xlen_block_prev = None, None
        while True:
            # Encode input features block by block
            x_block, is_last_block, cnn_lookback, cnn_lookahead, xlen_block = streaming.extract_feat()
            if not is_transformer_enc and is_reset:
                self.enc.reset_cache()
                if backoff:
                    self.encode([x_block_prev], 'all',
                                streaming=True,
                                cnn_lookback=cnn_lookback,
                                cnn_lookahead=cnn_lookahead,
                                xlen_block=xlen_block_prev)
            x_block_prev = x_block
            xlen_block_prev = xlen_block
            eout_block_dict = self.encode([x_block], 'all',
                                          streaming=True,
                                          cnn_lookback=cnn_lookback,
                                          cnn_lookahead=cnn_lookahead,
                                          xlen_block=xlen_block)
            eout_block = eout_block_dict[task]['xs']
            if eout_block_tail is not None:
                eout_block = torch.cat([eout_block_tail, eout_block], dim=1)
                eout_block_tail = None

            if eout_block.size(1) > 0:
                streaming.cache_eout(eout_block)

                # Block-synchronous decoding
                if ctc_weight == 1 or self.ctc_weight == 1:
                    end_hyps, hyps = self.dec_fwd.ctc.beam_search_block_sync(
                        eout_block, params, helper, idx2token, hyps, lm)
                elif isinstance(self.dec_fwd, RNNT):
                    end_hyps, hyps = self.dec_fwd.beam_search_block_sync(
                        eout_block, params, helper, idx2token, hyps, lm)
                elif isinstance(self.dec_fwd, RNNDecoder):
                    n_frames = getattr(self.dec_fwd, 'n_frames', 0)
                    for i in range(math.ceil(eout_block.size(1) / block_size)):
                        eout_block_i = eout_block[:, i * block_size:(i + 1) * block_size]
                        end_hyps, hyps, hyps_nobd = self.dec_fwd.beam_search_block_sync(
                            eout_block_i, params, helper, idx2token, hyps, hyps_nobd, lm,
                            speaker=speaker)
                elif isinstance(self.dec_fwd, TransformerDecoder):
                    raise NotImplementedError
                else:
                    raise NotImplementedError(self.dec_fwd)

                # CTC-based reset point detection
                is_reset = False
                if streaming.enable_ctc_reset_point_detection:
                    if self.ctc_weight_sub1 > 0:
                        ctc_probs_block = self.dec_fwd_sub1.ctc.probs(eout_block_dict['ys_sub1']['xs'])
                        # TODO: consider subsampling
                    else:
                        ctc_probs_block = self.dec_fwd.ctc.probs(eout_block)
                    is_reset = streaming.ctc_reset_point_detection(ctc_probs_block)

                merged_hyps = sorted(end_hyps + hyps + hyps_nobd, key=lambda x: x['score'], reverse=True)
                if len(merged_hyps) > 0:
                    best_hyp_id_prefix = np.array(merged_hyps[0]['hyp'][1:])
                    best_hyp_id_prefix_viz = np.array(merged_hyps[0]['hyp'][1:])

                    if (len(best_hyp_id_prefix) > 0 and best_hyp_id_prefix[-1] == self.eos):
                        # reset beam if <eos> is generated from the best hypothesis
                        best_hyp_id_prefix = best_hyp_id_prefix[:-1]  # exclude <eos>
                        # Condition 2:
                        # If <eos> is emitted from the decoder (not CTC),
                        # the current block is segmented.
                        if (not is_reset) and (not streaming.safeguard_reset):
                            is_reset = True

                    if len(best_hyp_id_prefix_viz) > 0:
                        n_frames = self.dec_fwd.ctc.n_frames if ctc_weight == 1 or self.ctc_weight == 1 else self.dec_fwd.n_frames
                        print('\rStreaming (T:%.3f [sec], offset:%d [frame], blank:%d [frame]): %s' %
                              ((streaming.offset + eout_block.size(1) * factor) / 100,
                               n_frames * factor,
                               streaming.n_blanks * factor,
                               idx2token(best_hyp_id_prefix_viz)))

            if is_reset:

                # pick up the best hyp from ended and active hypotheses
                if len(best_hyp_id_prefix) > 0:
                    best_hyp_id_session.extend(best_hyp_id_prefix)

                # reset
                streaming.reset()
                hyps = None
                hyps_nobd = []

            streaming.next_block()
            if is_last_block:
                break

        # pick up the best hyp
        if not is_reset and len(best_hyp_id_prefix) > 0:
            best_hyp_id_session.extend(best_hyp_id_prefix)

        if len(best_hyp_id_session) > 0:
            return [[np.stack(best_hyp_id_session, axis=0)]], [None]
        else:
            return [[[]]], [None]

    def streamable(self):
        return getattr(self.dec_fwd, 'streamable', False)

    def quantity_rate(self):
        return getattr(self.dec_fwd, 'quantity_rate', 1.0)

    def last_success_frame_ratio(self):
        return getattr(self.dec_fwd, 'last_success_frame_ratio', 0)

    @torch.no_grad()
    def decode(self, xs, params, idx2token, exclude_eos=False,
               refs_id=None, refs=None, utt_ids=None, speakers=None,
               task='ys', ensemble_models=[], trigger_points=None, teacher_force=False):
        """Decode in the inference stage.

        Args:
            xs (List): length `[B]`, which contains arrays of size `[T, input_dim]`
            params (dict): hyper-parameters for decoding
            idx2token (): converter from index to token
            exclude_eos (bool): exclude <eos> from best_hyps_id
            refs_id (List): gold token IDs to compute log likelihood
            refs (List): gold transcriptions
            utt_ids (List): utterance id list
            speakers (List): speaker list
            task (str): ys* or ys_sub1* or ys_sub2*
            ensemble_models (List): Speech2Text classes
            trigger_points (np.ndarray): `[B, L]`
            teacher_force (bool): conduct teacher-forcing
        Returns:
            nbest_hyps_id (List[List[np.ndarray]]): length `[B]`, which contains a list of length `[n_best]` which contains arrays of size `[L]`
            aws (List[np.ndarray]): length `[B]`, which contains arrays of size `[L, T, n_heads]`

        """
        self.eval()
        if task.split('.')[0] == 'ys':
            dir = 'bwd' if self.bwd_weight > 0 and params['recog_bwd_attention'] else 'fwd'
        elif task.split('.')[0] == 'ys_sub1':
            dir = 'fwd_sub1'
        elif task.split('.')[0] == 'ys_sub2':
            dir = 'fwd_sub2'
        else:
            raise ValueError(task)

        if utt_ids is not None:
            if self.utt_id_prev != utt_ids[0]:
                self.reset_session()
            self.utt_id_prev = utt_ids[0]

        # Encode input features
        if params['recog_streaming_encoding']:
            eouts, elens = self.encode_streaming(xs, params, task)
        else:
            eout_dict = self.encode(xs, task)
            eouts = eout_dict[task]['xs']
            elens = eout_dict[task]['xlens']

        # CTC
        if (self.fwd_weight == 0 and self.bwd_weight == 0) or (self.ctc_weight > 0 and params['recog_ctc_weight'] == 1):
            lm = getattr(self, 'lm_' + dir, None)
            lm_second = getattr(self, 'lm_second', None)
            lm_second_bwd = None  # TODO

            if params.get('recog_beam_width') == 1:
                nbest_hyps_id = getattr(self, 'dec_' + dir).ctc.greedy(
                    eouts, elens)
            else:
                nbest_hyps_id = getattr(self, 'dec_' + dir).ctc.beam_search(
                    eouts, elens, params, idx2token,
                    lm, lm_second, lm_second_bwd,
                    1, refs_id, utt_ids, speakers)
            return nbest_hyps_id, None

        # Attention/RNN-T
        elif params['recog_beam_width'] == 1 and not params['recog_fwd_bwd_attention']:
            best_hyps_id, aws = getattr(self, 'dec_' + dir).greedy(
                eouts, elens, params['recog_max_len_ratio'], idx2token,
                exclude_eos, refs_id, utt_ids, speakers)
            nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
        else:
            assert params['recog_batch_size'] == 1

            scores_ctc = None
            if params['recog_ctc_weight'] > 0:
                scores_ctc = self.dec_fwd.ctc.scores(eouts)

            # forward-backward decoding
            if params['recog_fwd_bwd_attention']:
                lm = getattr(self, 'lm_fwd', None)
                lm_bwd = getattr(self, 'lm_bwd', None)

                # forward decoder
                nbest_hyps_id_fwd, aws_fwd, scores_fwd = self.dec_fwd.beam_search(
                    eouts, elens, params, idx2token,
                    lm, None, lm_bwd, scores_ctc,
                    params['recog_beam_width'], False, refs_id, utt_ids, speakers)

                # backward decoder
                nbest_hyps_id_bwd, aws_bwd, scores_bwd, _ = self.dec_bwd.beam_search(
                    eouts, elens, params, idx2token,
                    lm_bwd, None, lm, scores_ctc,
                    params['recog_beam_width'], False, refs_id, utt_ids, speakers)

                # forward-backward attention
                best_hyps_id = fwd_bwd_attention(
                    nbest_hyps_id_fwd, aws_fwd, scores_fwd,
                    nbest_hyps_id_bwd, aws_bwd, scores_bwd,
                    self.eos, params['recog_gnmt_decoding'], params['recog_length_penalty'],
                    idx2token, refs_id)
                nbest_hyps_id = [[hyp] for hyp in best_hyps_id]
                aws = None
            else:
                # ensemble
                ensmbl_eouts, ensmbl_elens, ensmbl_decs = [], [], []
                if len(ensemble_models) > 0:
                    for i_e, model in enumerate(ensemble_models):
                        enc_outs_e = model.encode(xs, task)
                        ensmbl_eouts += [enc_outs_e[task]['xs']]
                        ensmbl_elens += [enc_outs_e[task]['xlens']]
                        ensmbl_decs += [getattr(model, 'dec_' + dir)]
                        # NOTE: only support for the main task now

                lm = getattr(self, 'lm_' + dir, None)
                lm_second = getattr(self, 'lm_second', None)
                lm_bwd = getattr(self, 'lm_bwd' if dir == 'fwd' else 'lm_bwd', None)

                nbest_hyps_id, aws, scores = getattr(self, 'dec_' + dir).beam_search(
                    eouts, elens, params, idx2token,
                    lm, lm_second, lm_bwd, scores_ctc,
                    params['recog_beam_width'], exclude_eos, refs_id, utt_ids, speakers,
                    ensmbl_eouts, ensmbl_elens, ensmbl_decs)

        return nbest_hyps_id, aws
