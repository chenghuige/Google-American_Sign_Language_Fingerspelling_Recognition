#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate ASR model."""

import copy
import logging
import os
import sys
import time

from neural_sp.bin.args_asr import parse_args_eval
from neural_sp.bin.eval_utils import (
    average_checkpoints,
    load_lm
)
from neural_sp.bin.train_utils import (
    load_config,
    set_logger
)
from neural_sp.datasets.asr.build import build_dataloader
from neural_sp.evaluators.accuracy import eval_accuracy
from neural_sp.evaluators.character import eval_char
from neural_sp.evaluators.phone import eval_phone
from neural_sp.evaluators.ppl import eval_ppl
from neural_sp.evaluators.word import eval_word
from neural_sp.evaluators.wordpiece import eval_wordpiece
from neural_sp.evaluators.wordpiece_bleu import eval_wordpiece_bleu
from neural_sp.models.seq2seq.speech2text import Speech2Text

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'decode.log')):
        os.remove(os.path.join(args.recog_dir, 'decode.log'))
    set_logger(os.path.join(args.recog_dir, 'decode.log'), stdout=args.recog_stdout)

    # Load ASR model
    model = Speech2Text(args, dir_name)
    average_checkpoints(model, args.recog_model[0], n_average=args.recog_n_average)

    # Ensemble
    ensemble_models = [model]
    if len(args.recog_model) > 1:
        for recog_model_e in args.recog_model[1:]:
            conf_e = load_config(os.path.join(os.path.dirname(recog_model_e), 'conf.yml'))
            args_e = copy.deepcopy(args)
            for k, v in conf_e.items():
                if 'recog' not in k:
                    setattr(args_e, k, v)
            model_e = Speech2Text(args_e)
            average_checkpoints(model_e, recog_model_e, n_average=args.recog_n_average)
            ensemble_models += [model_e]

    # Load LM for shallow fusion
    if not args.lm_fusion:
        if args.recog_lm is not None and args.recog_lm_weight > 0:
            lm = load_lm(args.recog_lm, args.recog_mem_len)
            if lm.backward:
                model.lm_bwd = lm
            else:
                model.lm_fwd = lm

        # second pass (forward)
        if args.recog_lm_second is not None and args.recog_lm_second_weight > 0:
            model.lm_second = load_lm(args.recog_lm_second, args.recog_mem_len)

        # second pass (backward)
        if args.recog_lm_bwd is not None and args.recog_lm_bwd_weight > 0:
            model.lm_bwd = load_lm(args.recog_lm_bwd, args.recog_mem_len)

    if not args.recog_unit:
        args.recog_unit = args.unit

    logger.info('recog unit: %s' % args.recog_unit)
    logger.info('recog metric: %s' % args.recog_metric)
    logger.info('recog oracle: %s' % args.recog_oracle)
    logger.info('batch size: %d' % args.recog_batch_size)
    logger.info('beam width: %d' % args.recog_beam_width)
    logger.info('min length ratio: %.3f' % args.recog_min_len_ratio)
    logger.info('max length ratio: %.3f' % args.recog_max_len_ratio)
    logger.info('length penalty: %.3f' % args.recog_length_penalty)
    logger.info('length norm: %s' % args.recog_length_norm)
    logger.info('coverage penalty: %.3f' % args.recog_coverage_penalty)
    logger.info('coverage threshold: %.3f' % args.recog_coverage_threshold)
    logger.info('CTC weight: %.3f' % args.recog_ctc_weight)
    logger.info('fist LM path: %s' % args.recog_lm)
    logger.info('second LM path: %s' % args.recog_lm_second)
    logger.info('backward LM path: %s' % args.recog_lm_bwd)
    logger.info('LM weight (first-pass): %.3f' % args.recog_lm_weight)
    logger.info('LM weight (second-pass): %.3f' % args.recog_lm_second_weight)
    logger.info('LM weight (backward): %.3f' % args.recog_lm_bwd_weight)
    logger.info('ensemble: %d' % (len(ensemble_models)))
    logger.info('ASR decoder state carry over: %s' % (args.recog_asr_state_carry_over))
    logger.info('LM state carry over: %s' % (args.recog_lm_state_carry_over))
    logger.info('model average: %d' % (args.recog_n_average))

    # GPU setting
    if args.recog_n_gpus >= 1:
        model.cudnn_setting(deterministic=True, benchmark=False)
        for m in ensemble_models:
            m.cuda()

    wer_avg, cer_avg, per_avg = 0, 0, 0
    ppl_avg, loss_avg = 0, 0
    acc_avg = 0
    bleu_avg = 0
    for i, s in enumerate(args.recog_sets):
        # Load dataloader
        dataloader = build_dataloader(args=args,
                                      tsv_path=s,
                                      batch_size=1,
                                      is_test=True,
                                      first_n_utterances=args.recog_first_n_utt,
                                      longform_max_n_frames=args.recog_longform_max_n_frames)

        start_time = time.time()

        if args.recog_metric == 'edit_distance':
            if args.recog_unit in ['word', 'word_char']:
                wer, cer, _ = eval_word(ensemble_models, dataloader, args,
                                        save_dir=args.recog_dir,
                                        progressbar=True,
                                        fine_grained=True,
                                        oracle=True)
                wer_avg += wer
                cer_avg += cer
            elif args.recog_unit == 'wp':
                wer, cer = eval_wordpiece(ensemble_models, dataloader, args,
                                          save_dir=args.recog_dir,
                                          streaming=args.recog_streaming,
                                          progressbar=True,
                                          edit_distance=args.recog_longform_max_n_frames == 0,
                                          fine_grained=True,
                                          oracle=True)
                wer_avg += wer
                cer_avg += cer
            elif 'char' in args.recog_unit:
                wer, cer = eval_char(ensemble_models, dataloader, args,
                                     save_dir=args.recog_dir,
                                     progressbar=True,
                                     task_idx=0,
                                     fine_grained=True,
                                     oracle=True)
                #  task_idx=1 if args.recog_unit and 'char' in args.recog_unit else 0)
                wer_avg += wer
                cer_avg += cer
            elif 'phone' in args.recog_unit:
                per = eval_phone(ensemble_models, dataloader, args,
                                 save_dir=args.recog_dir,
                                 progressbar=True,
                                 fine_grained=True,
                                 oracle=True)
                per_avg += per
            else:
                raise ValueError(args.recog_unit)
        elif args.recog_metric in ['ppl', 'loss']:
            ppl, loss = eval_ppl(ensemble_models, dataloader, progressbar=True)
            ppl_avg += ppl
            loss_avg += loss
        elif args.recog_metric == 'accuracy':
            acc_avg += eval_accuracy(ensemble_models, dataloader, progressbar=True)
        elif args.recog_metric == 'bleu':
            bleu = eval_wordpiece_bleu(ensemble_models, dataloader, args,
                                       save_dir=args.recog_dir,
                                       streaming=args.recog_streaming,
                                       progressbar=True,
                                       fine_grained=True,
                                       oracle=True)
            bleu_avg += bleu
        else:
            raise NotImplementedError(args.recog_metric)
        elapsed_time = time.time() - start_time
        logger.info('Elapsed time: %.3f [sec]' % elapsed_time)
        logger.info('RTF: %.3f' % (elapsed_time / (dataloader.n_frames * 0.01)))

    if args.recog_metric == 'edit_distance':
        if 'phone' in args.recog_unit:
            logger.info('PER (avg.): %.2f %%\n' % (per_avg / len(args.recog_sets)))
        else:
            logger.info('WER / CER (avg.): %.2f / %.2f %%\n' %
                        (wer_avg / len(args.recog_sets), cer_avg / len(args.recog_sets)))
    elif args.recog_metric in ['ppl', 'loss']:
        logger.info('PPL (avg.): %.2f\n' % (ppl_avg / len(args.recog_sets)))
        print('PPL (avg.): %.3f' % (ppl_avg / len(args.recog_sets)))
        logger.info('Loss (avg.): %.2f\n' % (loss_avg / len(args.recog_sets)))
        print('Loss (avg.): %.3f' % (loss_avg / len(args.recog_sets)))
    elif args.recog_metric == 'accuracy':
        logger.info('Accuracy (avg.): %.2f\n' % (acc_avg / len(args.recog_sets)))
        print('Accuracy (avg.): %.3f' % (acc_avg / len(args.recog_sets)))
    elif args.recog_metric == 'bleu':
        logger.info('BLEU (avg.): %.2f\n' % (bleu / len(args.recog_sets)))
        print('BLEU (avg.): %.3f' % (bleu / len(args.recog_sets)))


if __name__ == '__main__':
    main()
