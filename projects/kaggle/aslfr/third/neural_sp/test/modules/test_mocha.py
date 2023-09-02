#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test for monotonic (multihead) chunkwise atteniton (HMA/MoChA/MMA)."""

import importlib
import pytest
import torch


def make_args(**kwargs):
    args = dict(
        kdim=32,
        qdim=32,
        adim=16,
        odim=32,
        atype='add',
        chunk_size=1,
        n_heads_mono=1,
        n_heads_chunk=1,
        conv1d=False,
        init_r=-4,
        eps=1e-6,
        noise_std=1.0,
        no_denominator=False,
        sharpening_factor=1.0,
        dropout=0.1,
        dropout_head=0.,
        bias=True,
        param_init='',
        decot=False,
        decot_delta=0,
        share_chunkwise_attention=True,
        stableemit_weight=0.0,
    )
    args.update(kwargs)
    return args


@pytest.mark.parametrize(
    "args",
    [
        # hard monotonic attention
        ({'n_heads_mono': 1, 'chunk_size': 1}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'conv1d': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'no_denominator': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'bias': False}),
        # MoChA
        ({'n_heads_mono': 1, 'chunk_size': 4}),
        # Milk
        ({'n_heads_mono': 1, 'chunk_size': -1}),
        # MMA
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot',
          'share_chunkwise_attention': False}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'share_chunkwise_attention': False}),
        # HeadDrop
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'dropout_head': 0.5}),
        # StableEmit
        ({'stableemit_weight': 0.1}),
        # initialization
        ({'param_init': 'xavier_uniform'}),
    ]
)
def test_forward_soft(args):
    args = make_args(**args)

    bs = 4
    klen = 40
    qlen = 5
    device = "cpu"

    key = torch.randn(bs, klen, args['kdim'], device=device)
    value = torch.randn(bs, klen, args['kdim'], device=device)
    query = torch.randn(bs, qlen, args['qdim'], device=device)
    src_mask = key.new_ones(bs, 1, klen).byte()

    module = importlib.import_module('neural_sp.models.modules.mocha.mocha')
    mocha = module.MoChA(**args)
    mocha = mocha.to(device)

    mocha.train()
    for linear_decoding in [True, False]:
        alpha = None
        mocha.reset()
        for i in range(qlen):
            out = mocha(key, value, query[:, i:i + 1], mask=src_mask, aw_prev=alpha,
                        mode='parallel', cache=True, linear_decoding=linear_decoding)
            assert len(out) == 3
            cv, alpha, attn_state = out
            assert cv.size() == (bs, 1, value.size(2))
            assert alpha.size() == (bs, args['n_heads_mono'], 1, klen)
            assert isinstance(attn_state, dict)
            beta = attn_state['beta']
            p_choose = attn_state['p_choose']
            assert p_choose.size() == (bs, args['n_heads_mono'], 1, klen)
            if args['chunk_size'] > 1:
                assert beta is not None
                assert beta.size() == (bs, args['n_heads_mono'] * args['n_heads_chunk'], 1, klen)


@pytest.mark.parametrize(
    "args", [
        # hard monotonic attention
        ({'n_heads_mono': 1, 'chunk_size': 1}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'conv1d': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'no_denominator': True}),
        ({'n_heads_mono': 1, 'chunk_size': 1, 'bias': False}),
        # MoChA
        ({'n_heads_mono': 1, 'chunk_size': 4}),
        ({'n_heads_mono': 1, 'chunk_size': 4, 'decot': True, 'decot_delta': 2}),
        # Milk
        ({'n_heads_mono': 1, 'chunk_size': -1}),
        # MMA
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 1, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 1, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot'}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 1, 'chunk_size': 4, 'atype': 'scaled_dot',
          'share_chunkwise_attention': False}),
        ({'n_heads_mono': 4, 'n_heads_chunk': 4, 'chunk_size': 4, 'atype': 'scaled_dot',
          'share_chunkwise_attention': False}),
    ]
)
def test_forward_hard(args):
    args = make_args(**args)

    bs = 4
    klen = 40
    qlen = 5
    device = "cpu"

    key = torch.randn(bs, klen, args['kdim'], device=device)
    value = torch.randn(bs, klen, args['kdim'], device=device)
    query = torch.randn(bs, qlen, args['qdim'], device=device)

    module = importlib.import_module('neural_sp.models.modules.mocha.mocha')
    mocha = module.MoChA(**args)
    mocha = mocha.to(device)

    mocha.eval()
    alpha = None
    trigger_points = None
    if args['decot']:
        trigger_points = torch.arange(qlen).unsqueeze(0).repeat(bs, 1)
    for linear_decoding in [True, False]:
        for i in range(qlen):
            out = mocha(key, value, query[:, i:i + 1], mask=None, aw_prev=alpha,
                        mode='hard', cache=False, trigger_points=trigger_points,
                        eps_wait=-1, linear_decoding=linear_decoding)
            assert len(out) == 3
            cv, alpha, attn_state = out
            assert cv.size() == (bs, 1, value.size(2))
            assert alpha.size() == (bs, args['n_heads_mono'], 1, klen)
            assert isinstance(attn_state, dict)
            beta = attn_state['beta']
            p_choose = attn_state['p_choose']
            assert p_choose.size() == (bs, args['n_heads_mono'], 1, klen)
            if args['chunk_size'] > 1:
                assert beta is not None
                assert beta.size() == (bs, args['n_heads_mono'] * args['n_heads_chunk'], 1, klen)
