#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for RNN encoder."""

import importlib
import math
import numpy as np
import pytest
import torch

from neural_sp.models.torch_utils import (
    np2tensor,
    pad_list
)


def make_args(**kwargs):
    args = dict(
        input_dim=80,
        enc_type='blstm',
        n_units=16,
        n_projs=0,
        last_proj_dim=0,
        n_layers=4,
        n_layers_sub1=0,
        n_layers_sub2=0,
        dropout_in=0.1,
        dropout=0.1,
        subsample="1_1_1_1",
        subsample_type='drop',
        n_stacks=1,
        n_splices=1,
        frontend_conv=None,
        bidir_sum_fwd_bwd=False,
        task_specific_layer=False,
        param_init=0.1,
        chunk_size_current="0",
        chunk_size_right="0",
        cnn_lookahead=True,
        rsp_prob=0,
    )
    args.update(kwargs)
    return args


def make_args_conv(**kwargs):
    args = dict(
        input_dim=80,
        in_channel=1,
        channels="32_32",
        kernel_sizes="(3,3)_(3,3)",
        strides="(1,1)_(1,1)",
        poolings="(2,2)_(2,2)",
        dropout=0.1,
        normalization='',
        residual=False,
        bottleneck_dim=0,
        param_init=0.1,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args, args_conv",
    [
        # RNN type
        ({'enc_type': 'blstm'}, {}),
        ({'enc_type': 'lstm'}, {}),
        ({'enc_type': 'lstm', 'rsp_prob': 0.5}, {}),
        # 2dCNN-RNN
        ({'enc_type': 'conv_blstm'}, {}),
        ({'enc_type': 'conv_blstm', 'input_dim': 240}, {'input_dim': 240, 'in_channel': 3}),
        # 1dCNN-RNN
        ({'enc_type': 'conv_blstm'}, {'kernel_sizes': "3_3", 'strides': "1_1", 'poolings': "2_2"}),
        ({'enc_type': 'conv_blstm', 'input_dim': 240},
         {'input_dim': 240, 'in_channel': 3, 'kernel_sizes': "3_3", 'strides': "1_1", 'poolings': "2_2"}),
        # normalization
        ({'enc_type': 'conv_blstm'}, {'normalization': 'batch_norm'}),
        ({'enc_type': 'conv_blstm'}, {'normalization': 'layer_norm'}),
        # projection
        ({'enc_type': 'blstm', 'n_projs': 8}, {}),
        ({'enc_type': 'lstm', 'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'last_proj_dim': 5}, {}),
        ({'enc_type': 'blstm', 'last_proj_dim': 5, 'n_projs': 8}, {}),
        ({'enc_type': 'lstm', 'last_proj_dim': 5, 'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 5}, {}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True, 'last_proj_dim': 5, 'n_projs': 8}, {}),
        # subsampling
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop'}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat'}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool'}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'conv1d'}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add'}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop',
          'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat',
          'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool',
          'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'conv1d',
          'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add',
          'bidir_sum_fwd_bwd': True}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'drop',
          'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'concat',
          'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'max_pool',
          'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'conv1d',
          'n_projs': 8}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_2_1", 'subsample_type': 'add',
          'n_projs': 8}, {}),
        # LC-BLSTM
        ({'enc_type': 'blstm', 'chunk_size_current': "0", 'chunk_size_right': "40"}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "0", 'chunk_size_right': "40"}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),
        ({'enc_type': 'conv_blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),
        ({'enc_type': 'conv_blstm', 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40", 'rsp_prob': 0.5}, {}),
        # LC-BLSTM + subsampling
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1",
          'chunk_size_right': "40"}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1",
          'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_right': "40"}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),
        ({'enc_type': 'blstm', 'subsample': "1_2_1_1", 'bidir_sum_fwd_bwd': True,
          'chunk_size_current': "40", 'chunk_size_right': "40", 'rsp_prob': 0.5}, {}),
        # Multi-task
        ({'enc_type': 'blstm', 'n_layers_sub1': 2}, {}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True}, {}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "0", 'chunk_size_right': "40"}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "40", 'chunk_size_right': "40"}, {}),  # LC-BLSTM
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "0", 'chunk_size_right': "40",
          'rsp_prob': 0.5}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'task_specific_layer': True,
          'chunk_size_current': "40", 'chunk_size_right': "40",
          'rsp_prob': 0.5}, {}),  # LC-BLSTM
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'n_layers_sub2': 1}, {}),
        ({'enc_type': 'blstm', 'n_layers_sub1': 2, 'n_layers_sub2': 1,
          'task_specific_layer': True}, {}),
        # Multi-task + subsampling
        ({'enc_type': 'blstm', 'subsample': "2_1_1_1", 'n_layers_sub1': 2,
          'chunk_size_current': "0", 'chunk_size_right': "40",
          'task_specific_layer': True}, {}),  # BLSTM for PT
        ({'enc_type': 'blstm', 'subsample': "2_1_1_1", 'n_layers_sub1': 2,
          'chunk_size_current': "40", 'chunk_size_right': "40",
          'task_specific_layer': True}, {}),  # LC-BLSTM
    ]
)
def test_forward(args, args_conv):
    device = "cpu"

    args = make_args(**args)
    if 'conv' in args['enc_type']:
        conv_module = importlib.import_module('neural_sp.models.seq2seq.encoders.conv')
        args_conv = make_args_conv(**args_conv)
        args['frontend_conv'] = conv_module.ConvEncoder(**args_conv).to(device)

    bs = 4
    xmaxs = [40, 45] if int(args['chunk_size_current'].split('_')[0]) == -1 else [400, 455]

    module = importlib.import_module('neural_sp.models.seq2seq.encoders.rnn')
    enc = module.RNNEncoder(**args).to(device)

    for xmax in xmaxs:
        xs = np.random.randn(bs, xmax, args['input_dim']).astype(np.float32)
        xlens = torch.IntTensor([len(x) - i * enc.subsampling_factor for i, x in enumerate(xs)])

        # shuffle
        perm_ids = torch.randperm(bs)
        xs = xs[perm_ids]
        xlens = xlens[perm_ids]

        xs = pad_list([np2tensor(x, device).float() for x in xs], 0.)
        eout_dict = enc(xs, xlens, task='all')

        assert eout_dict['ys']['xs'].size(0) == bs
        assert eout_dict['ys']['xs'].size(1) == eout_dict['ys']['xlens'].max()
        for b in range(bs):
            if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', 'conv1d', 'drop', 'add']:
                assert eout_dict['ys']['xlens'][b].item() == math.ceil(xlens[b].item() / enc.subsampling_factor)
            else:
                assert eout_dict['ys']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor

        if args['n_layers_sub1'] > 0:
            # all outputs
            assert eout_dict['ys_sub1']['xs'].size(0) == bs
            assert eout_dict['ys_sub1']['xs'].size(1) == eout_dict['ys_sub1']['xlens'].max()
            for b in range(bs):
                if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', 'conv1d', 'drop', 'add']:
                    assert eout_dict['ys_sub1']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor_sub1)
                else:
                    assert eout_dict['ys_sub1']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor_sub1
            # single output
            eout_dict_sub1 = enc(xs, xlens, task='ys_sub1')
            assert eout_dict_sub1['ys_sub1']['xs'].size(0) == bs
            assert eout_dict_sub1['ys_sub1']['xs'].size(1) == eout_dict['ys_sub1']['xlens'].max()

        if args['n_layers_sub2'] > 0:
            # all outputs
            assert eout_dict['ys_sub2']['xs'].size(0) == bs
            assert eout_dict['ys_sub2']['xs'].size(1) == eout_dict['ys_sub2']['xlens'].max()
            for b in range(bs):
                if 'conv' in args['enc_type'] or args['subsample_type'] in ['max_pool', 'conv1d', 'drop', 'add']:
                    assert eout_dict['ys_sub2']['xlens'][b].item() == math.ceil(
                        xlens[b].item() / enc.subsampling_factor_sub2)
                else:
                    assert eout_dict['ys_sub2']['xlens'][b].item() == xlens[b].item() // enc.subsampling_factor_sub2
            # single output
            eout_dict_sub2 = enc(xs, xlens, task='ys_sub2')
            assert eout_dict_sub2['ys_sub2']['xs'].size(0) == bs
            assert eout_dict_sub2['ys_sub2']['xs'].size(1) == eout_dict_sub2['ys_sub2']['xlens'].max()
