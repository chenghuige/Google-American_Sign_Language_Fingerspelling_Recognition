#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   optimizers.py
#        \author   chenghuige
#          \date   2018-10-29 07:06:55.090940
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

import tensorflow as tf
import sys
import os

import torch
from torch.optim.optimizer import Optimizer

# https://github.com/google/automl/blob/master/lion/lion_pytorch.py
class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss
# https://github.com/mgrankin/over9000/blob/master/lookahead.py
from collections import defaultdict
class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def LookaheadAdam(params, alpha=0.5, k=6, eps=1e-8, weight_decay=0, *args, **kwargs):
  from torch.optim import Adam
  adam = Adam(params, eps=eps, weight_decay=weight_decay, *args, **kwargs)
  return Lookahead(adam, alpha, k)

def LookaheadRAdam(params, alpha=0.5, k=6, eps=1e-8, weight_decay=0, *args, **kwargs):
  from torch.optim import RAdam
  adam = RAdam(params, eps=eps, weight_decay=weight_decay, *args, **kwargs)
  return Lookahead(adam, alpha, k)
   

# http://nlp.seas.harvard.edu/2018/04/03/attention.html
class OptWrapper:

  def __init__(self, optimizer, lr=0.):
    self._step = 0
    self._rate = 0.
    self.start_lr = lr
    self.optimizer = optimizer
    self.param_groups = self.optimizer.param_groups
    if self.start_lr:
      for p in self.optimizer.param_groups:
        p['ratio'] = p['lr'] / self.start_lr

  def set_step(self, step):
    self._step = step

  def step(self):
    "Update parameters and rate"
    self._step += 1

    rate = self.rate()

    for p in self.optimizer.param_groups:
      #p['lr'] = rate
      if 'ratio' in p:
        p['lr'] = rate * p['ratio']

    self._rate = rate
    self.optimizer.step()

  def zero_grad(self):
    self.optimizer.zero_grad()

  def state_dict(self):
    return self.optimizer.state_dict()

  def load_state_dict(self, x):
    return self.optimizer.load_state_dict(x)


class NoamOpt(OptWrapper):
  "Optim wrapper that implements rate."

  def __init__(self, model_size, factor, warmup, optimizer):
    super(NoamOpt, self).__init__(optimizer)
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size

  def rate(self, step=None):
    "Implement `lrate` above"
    if step is None:
      step = self._step
    return self.factor * \
        (self.model_size ** (-0.5) *
        min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
  return NoamOpt(
      model.src_embed[0].d_model, 2, 4000,
      torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def lr_poly(base_lr, iter, max_iter, end_learning_rate, power):
  return (base_lr - end_learning_rate) * (
      (1 - float(iter) / max_iter)**(power)) + end_learning_rate


class BertOpt(OptWrapper):
  "Optim wrapper that implements learning rate."

  def __init__(self, lr, min_lr, num_train_steps, warmup, optimizer, power=1.):
    super(BertOpt, self).__init__(optimizer, lr)
    self.warmup = warmup
    self.lr = lr
    self.ori_min_lr = min_lr
    self.min_lr = min_lr
    self.num_train_steps = num_train_steps
    self.power = power
    #print('---------param_groups', self.optimizer.param_groups)

  def rate(self, step=None):
    #print('-------------here')
    "Implement `lrate` above"
    if step is None:
      step = self._step

    warmup_percent_done = step / self.warmup
    warmup_learning_rate = self.lr * warmup_percent_done

    is_warmup = step < self.warmup
    learning_rate = lr_poly(self.lr, step, self.num_train_steps, self.min_lr,
                            self.power)
    learning_rate = ((1.0 - is_warmup) * learning_rate +
                     is_warmup * warmup_learning_rate)
    #print('-----------------', is_warmup, warmup_percent_done, warmup_learning_rate, warmup_learning_rate)
    return learning_rate

  def update(self, num_train_steps, num_warmup_steps):
    self.num_train_steps = num_train_steps
    self.num_warmup_steps = num_warmup_steps


class MultipleOpt(object):

  def __init__(self, *optimizers):
    self.optimizers = optimizers

  def set_step(self, step):
    for op in self.optimizers:
      op._step = step

  def rate(self, step=None):
    return self.optimizers[0].rate(step)

  def rates(self, step=None):
    return [op.rate(step) for op in self.optimizers]

  @property
  def param_groups(self):
    param_groups = []
    for optimizer in self.optimizers:
      param_groups.extend(optimizer.param_groups)
    return param_groups

  def zero_grad(self):
    """ ? """
    for op in self.optimizers:
      op.zero_grad()

  def step(self):
    """ ? """
    for op in self.optimizers:
      op.step()

  @property
  def state(self):
    """ ? """
    return {k: v for op in self.optimizers for k, v in op.state.items()}

  def state_dict(self):
    """ ? """
    return [op.state_dict() for op in self.optimizers]

  def load_state_dict(self, state_dicts):
    assert len(state_dicts) == len(self.optimizers)
    for i in range(len(self.optimizers)):
      self.optimizers[i].load_state_dict(state_dicts[i])


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import numpy as np

  steps = 2326
  for i in range(steps):
    lr = lr_poly(0.1, i, steps, 1e-6, 1.)
    print(i, lr)

  opts = [
      NoamOpt(512, 1, 4000, None),
      NoamOpt(512, 1, 8000, None),
      NoamOpt(256, 1, 4000, None),
      NoamOpt(200, 2, 4000, None),
      NoamOpt(256, 2, 4000, None),
      NoamOpt(300, 2, 4000, None),
      NoamOpt(128, 2, 4000, None)
  ]
  plt.plot(np.arange(1, 20000),
           [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
  plt.legend([
      "512:4000", "512:8000", "256:4000", "200:2:4000", "256:2:4000",
      "300:2:4000", "128:2:4000"
  ])

  for i in range(1, 40000, 1000):
    print(i, NoamOpt(200, 2, 2000, None).rate(i))

  plt.savefig('/home/gezi/tmp/lr.png')
