#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Plot CTC posteriors."""

import logging
import os
import shutil
import sys

from neural_sp.bin.args_asr import parse_args_eval
from neural_sp.bin.eval_utils import average_checkpoints
from neural_sp.bin.plot_utils import plot_ctc_probs
from neural_sp.bin.train_utils import set_logger
from neural_sp.datasets.asr.build import build_dataloader
from neural_sp.models.seq2seq.speech2text import Speech2Text
from neural_sp.utils import mkdir_join

logger = logging.getLogger(__name__)


def main():

    # Load configuration
    args, dir_name = parse_args_eval(sys.argv[1:])

    # Setting for logging
    if os.path.isfile(os.path.join(args.recog_dir, 'plot.log')):
        os.remove(os.path.join(args.recog_dir, 'plot.log'))
    set_logger(os.path.join(args.recog_dir, 'plot.log'), stdout=args.recog_stdout)

    # Load ASR model
    model = Speech2Text(args, dir_name)
    average_checkpoints(model, args.recog_model[0], n_average=args.recog_n_average)

    if not args.recog_unit:
        args.recog_unit = args.unit

    logger.info('recog unit: %s' % args.recog_unit)
    logger.info('batch size: %d' % args.recog_batch_size)

    # GPU setting
    if args.recog_n_gpus >= 1:
        model.cudnn_setting(deterministic=True, benchmark=False)
        model.cuda()

    for i, s in enumerate(args.recog_sets):
        # Load dataloader
        dataloader = build_dataloader(args=args,
                                      tsv_path=s,
                                      batch_size=1,
                                      is_test=True,
                                      first_n_utterances=args.recog_first_n_utt,
                                      longform_max_n_frames=args.recog_longform_max_n_frames)

        save_path = mkdir_join(args.recog_dir, 'ctc_probs')

        # Clean directory
        if save_path is not None and os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

        for batch in dataloader:
            nbest_hyps_id, _ = model.decode(batch['xs'], args, dataloader.idx2token[0])
            best_hyps_id = [h[0] for h in nbest_hyps_id]

            # Get CTC probs
            ctc_probs, topk_ids, xlens = model.get_ctc_probs(
                batch['xs'], temperature=1, topk=min(100, model.vocab))
            # NOTE: ctc_probs: '[B, T, topk]'

            for b in range(len(batch['xs'])):
                tokens = dataloader.idx2token[0](best_hyps_id[b], return_list=True)
                spk = batch['speakers'][b]

                plot_ctc_probs(
                    ctc_probs[b, :xlens[b]], topk_ids[b],
                    factor=args.subsample_factor,
                    spectrogram=batch['xs'][b][:, :dataloader.input_dim],
                    save_path=mkdir_join(save_path, spk, batch['utt_ids'][b] + '.png'),
                    figsize=(20, 8))

                hyp = ' '.join(tokens)
                logger.info('utt-id: %s' % batch['utt_ids'][b])
                logger.info('Ref: %s' % batch['text'][b].lower())
                logger.info('Hyp: %s' % hyp)
                logger.info('-' * 50)


if __name__ == '__main__':
    main()
