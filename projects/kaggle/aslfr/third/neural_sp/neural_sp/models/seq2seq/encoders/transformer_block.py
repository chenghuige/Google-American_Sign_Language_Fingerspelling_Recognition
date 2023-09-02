# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder block."""

import logging
import random
import torch
import torch.nn as nn

from neural_sp.models.modules.multihead_attention import MultiheadAttentionMechanism as MHA
from neural_sp.models.modules.positionwise_feed_forward import PositionwiseFeedForward as FFN
from neural_sp.models.modules.relative_multihead_attention import RelativeMultiheadAttentionMechanism as RelMHA

random.seed(1)

logger = logging.getLogger(__name__)


class TransformerEncoderBlock(nn.Module):
    """A single layer of the Transformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        dropout_layer (float): LayerDrop probability
        layer_norm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        param_init (str): parameter initialization method
        pe_type (str): type of positional encoding
        clamp_len (int): maximum relative distance from each position
        ffn_bottleneck_dim (int): bottleneck dimension for the light-weight FFN layer

    """

    def __init__(self, d_model, d_ff, n_heads,
                 dropout, dropout_att, dropout_layer,
                 layer_norm_eps, ffn_activation, param_init,
                 pe_type, clamp_len, ffn_bottleneck_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.n_heads = n_heads
        self.rel_attn = pe_type in ['relaive', 'relative_xl']

        # self-attention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        mha = RelMHA if self.rel_attn else MHA
        self.self_attn = mha(kdim=d_model,
                             qdim=d_model,
                             adim=d_model,
                             odim=d_model,
                             n_heads=n_heads,
                             dropout=dropout_att,
                             param_init=param_init,
                             xl_like=pe_type == 'relative_xl',
                             clamp_len=clamp_len)

        # position-wise feed-forward
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = FFN(d_model, d_ff, dropout, ffn_activation, param_init,
                                ffn_bottleneck_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_layer = dropout_layer  # probability to skip
        logger.info('Stochastic depth prob: %.3f' % dropout_layer)

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, xs, xx_mask=None, cache=None,
                pos_embs=None, rel_bias=(None, None)):
        """Transformer encoder layer definition.

        Args:
            xs (FloatTensor): `[B, T (query), d_model]`
            xx_mask (ByteTensor): `[B, T (query), T (key)]`
            cache (dict):
                input_san: `[B, n_cache, d_model]`
            pos_embs (LongTensor): `[T (query), 1, d_model]`
            rel_bias (tuple):
                u_bias (FloatTensor): global parameter for relative positional encoding
                v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            xs (FloatTensor): `[B, T (query), d_model]`
            new_cache (dict):
                input_san: `[B, n_cache+T, d_model]`

        """
        self.reset_visualization()
        new_cache = {}
        qlen = xs.size(1)
        u_bias, v_bias = rel_bias

        # LayerDrop
        if self.dropout_layer > 0:
            if self.training and random.random() < self.dropout_layer:
                return xs, new_cache
            else:
                xs = xs / (1 - self.dropout_layer)

        ##################################################
        # self-attention
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm1(xs)  # pre-norm

        # cache
        if cache is not None:
            xs = torch.cat([cache['input_san'], xs], dim=1)
        new_cache['input_san'] = xs

        xs_kv = xs
        if cache is not None:
            xs = xs[:, -qlen:]
            residual = residual[:, -qlen:]  # `[B, qlen, d_model]`
            xx_mask = xx_mask[:, -qlen:]

        if self.rel_attn:
            xs, self._xx_aws = self.self_attn(xs_kv, xs, pos_embs, xx_mask, u_bias, v_bias)  # k/q/m
        else:
            xs, self._xx_aws = self.self_attn(xs_kv, xs_kv, xs, mask=xx_mask)[:2]  # k/v/q
        xs = self.dropout(xs) + residual

        ##################################################
        # position-wise feed-forward
        ##################################################
        residual = xs  # `[B, qlen, d_model]`
        xs = self.norm2(xs)  # pre-norm
        xs = self.feed_forward(xs)
        xs = self.dropout(xs) + residual

        return xs, new_cache
