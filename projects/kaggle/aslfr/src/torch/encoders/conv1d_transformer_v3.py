#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model2.py
#        \author   chenghuige
#          \date   2023-07-13 00:12:33.561837
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import Conv1DBlocks, InstanceDropout

def relpos_att(x, y):
  b, h, n, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  r = y.shape[1]
  
  x = x.permute(2, 0, 1, 3).view(n, -1, d)
  x = torch.matmul(x, y.permute(0,2,1)).view(n, b, h, r).permute(1,2,0,3)
  return x
#https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
# 
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

# class MultiHeadSelfAttention(nn.Module):
#   def __init__(self, input_dim, num_heads=4, dim_head=64, dropout=0, max_pos_emb=512, **kwargs):
#     super().__init__(**kwargs)
#     self.dim = dim_head * num_heads
#     # TODO or self.scale = dim_head**-0.5
#     self.scale = self.dim**-0.5
#     self.num_heads = num_heads
#     self.qkv = nn.Linear(input_dim, 3 * self.dim, bias=False)
#     self.drop = nn.Dropout(dropout)
#     self.softmax = nn.Softmax(dim=-1)
#     self.proj = nn.Linear(self.dim, input_dim, bias=False)
    
#     self.max_pos_emb = max_pos_emb
#     if FLAGS.relpos_att:
#       self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
    
#   def forward(self, inputs):
#     qkv = self.qkv(inputs)
#     qkv = qkv.view(inputs.shape[0], inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads)
#     qkv = qkv.permute(0, 2, 1, 3)
#     q, k, v = torch.split(qkv, [self.dim // self.num_heads] * 3, dim=-1)

#     attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale

#     # ## 这里相对位置编码影响很大 但是很耗时...
#     ## 另外 FIMXE keras to tflite 转换einsum似乎有问题。。。 带来极大的不一致性 但是下面的转换速度过慢..
#     # # shaw's relative positional embedding 
#     if FLAGS.relpos_att:
#       device = q.device
#       n = inputs.shape[-2]
#       max_pos_emb = self.max_pos_emb
#       seq = torch.arange(n, device=device)
#       dist = seq.unsqueeze(-1) - seq.unsqueeze(0)
#       # dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
#       dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
#       rel_pos_emb = self.rel_pos_emb(dist).to(q)
#       if FLAGS.allow_einsum:
#         pos_attn = torch.einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
#       else:
#         pos_attn = relpos_att(q, rel_pos_emb) * self.scale        
    
#       attn = attn + pos_attn

#     attn = self.softmax(attn)
#     attn = self.drop(attn)

#     x = attn @ v
#     x = x.permute(0, 2, 1, 3)
#     x = x.reshape(x.shape[0], x.shape[1], self.dim)
#     x = self.proj(x)
#     return x
  
class TransformerBlock(nn.Module):
  def __init__(self, 
                input_dim,
                dim_head=64,
                num_heads=4,
                expand=4,
                attn_dropout=0.2,
                drop_rate=0.2):
    super().__init__()
    
    self.bn = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.mhsa = MultiHeadAttention(input_dim, dim_head=dim_head, num_heads=num_heads, dropout=attn_dropout)
    self.bn2 = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.fc = nn.Sequential(
                  nn.Linear(input_dim, input_dim * expand, bias=False),
                  nn.SiLU(),
                  nn.Linear(input_dim * expand, input_dim, bias=False),
                  )
    if not FLAGS.inst_drop:
      self.drop = nn.Dropout(drop_rate)
    else:
      self.drop = InstanceDropout(drop_rate)
  
  def forward(self, inputs):
    x = inputs
    x = self.bn(x)
    x = self.mhsa(x)
    x = self.drop(x)
    x = x * FLAGS.skip_factor + inputs
    attn_out = x
    x = self.bn2(x)
    x = self.fc(x)
    x = self.drop(x)
    x = x * FLAGS.skip_factor + attn_out
    return x

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.encoder = nn.Sequential(*[
      nn.Sequential(
        Conv1DBlocks(FLAGS.encoder_units, ksize_vals=FLAGS.conv1d_ksize_vals),
        TransformerBlock(FLAGS.encoder_units, 
                         dim_head=FLAGS.mhatt_dimhead,
                         num_heads=FLAGS.mhatt_heads, 
                         expand=FLAGS.conv1d_expansion_factor),
      ) for _ in range(FLAGS.encoder_layers)
    ])
    gezi.set('torch2tf', True)
    
  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
  
