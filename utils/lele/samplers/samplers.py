#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   sampler.py
#        \author   chenghuige  
#          \date   2023-01-31 20:34:07.554702
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
if not os.path.exists('/kaggle'):
  from gezi.common import * 
from typing import Callable, Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
# from torch.utils.data import DistributedSampler
# from torchnlp.samplers.distributed_sampler import DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler

# https://github.com/ufoym/imbalanced-dataset-sampler/blob/c2ef9d9529f2eb25306aab5e199a99eff455b2cd/torchsampler/imbalanced.py#L40
# from torchsampler import ImbalancedDatasetSampler

# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     sampler=ImbalancedDatasetSampler(train_dataset),
#     batch_size=args.batch_size,
#     **kwargs
# )
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/samplers/distributed_sampler.html
class DistributedSampler(Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)
      

class WeightsSampler(torch.utils.data.sampler.Sampler):
  def __init__(self, weights, shuffle=True, seed=None):
    self.num_samples = weights.sum()
    ic(self.num_samples, len(weights))
    self.weights = weights
    self.shuffle = shuffle
    self.rng = np.random.default_rng(seed)
    
  def __iter__(self):
    indices = []
    for i, weight in enumerate(self.weights):
      indices.extend([i] * weight)
    if self.shuffle:
      self.rng.shuffle(indices)
    return iter(indices)

  def __len__(self):
    return self.num_samples

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
  """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

  def __init__(self,
               dataset,
               indices: list = None,
               num_samples: int = None,
               weights: list = None,
               callback_get_label: Callable = None):
    # if indices is not provided, all elements in the dataset will be considered
    self.indices = list(range(len(dataset))) if indices is None else indices

    # define custom callback
    self.callback_get_label = callback_get_label

    # if num_samples is not provided, draw `len(indices)` samples in each iteration
    self.num_samples = len(self.indices) if num_samples is None else num_samples

    if weights is None:
      # distribution of classes in the dataset
      df = pd.DataFrame()
      df["label"] = self._get_labels(dataset)
      df.index = self.indices
      df = df.sort_index()

      label_to_count = df["label"].value_counts()

      weights = 1.0 / label_to_count[df["label"]]
    else:
      weights = np.asarray(weights)
      weights = weights / weights.sum()

    self.weights = torch.DoubleTensor(weights)

  def _get_labels(self, dataset):
    if self.callback_get_label:
      return self.callback_get_label(dataset)
    # elif isinstance(dataset, torchvision.datasets.MNIST):
    #     return dataset.train_labels.tolist()
    # elif isinstance(dataset, torchvision.datasets.ImageFolder):
    #     return [x[1] for x in dataset.imgs]
    # elif isinstance(dataset, torchvision.datasets.DatasetFolder):
    #     return dataset.samples[:][1]
    # elif isinstance(dataset, torch.utils.data.Subset):
    #     return dataset.dataset.imgs[:][1]
    elif isinstance(dataset, torch.utils.data.Dataset):
      return dataset.get_labels()
    else:
      raise NotImplementedError

  def __iter__(self):
    return (self.indices[i] for i in torch.multinomial(
        self.weights, self.num_samples, replacement=True))

  def __len__(self):
    return self.num_samples
  
# https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py
class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_last=True):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        groups = np.unique(self.group_ids).tolist()

        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in groups}
        self.drop_last = drop_last

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

      
class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)

