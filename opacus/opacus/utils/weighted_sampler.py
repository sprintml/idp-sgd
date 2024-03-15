from typing import List

import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import Sampler


class WeightedWithReplacementSampler(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, *, pp_sample_rates: ndarray, generator=None):
        r"""
        Args:
            pp_sample_rates: per-point probabilities used in sampling.
            generator: Generator used in sampling.
        """
        self.pp_sample_rates = pp_sample_rates
        self.generator = generator
        self.unique_sample_rates = np.unique(self.pp_sample_rates)
        self.num_samples = len(self.pp_sample_rates)
        assert all(self.pp_sample_rates >= 0) and all(
            self.pp_sample_rates < 1), "pp_sample_rates must be >=0 and <1!"

    def __len__(self):
        ratios = np.array([sum(self.pp_sample_rates == rate) / self.num_samples
                           for rate in self.unique_sample_rates])
        return round(1 / np.dot(ratios, self.unique_sample_rates))

    def __iter__(self):
        num_batches = len(self)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < torch.Tensor(self.pp_sample_rates)
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices
            num_batches -= 1
