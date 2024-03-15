# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import numpy as np

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis


class RDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    DEFAULT_MIN_ALPHA, DEFAULT_MAX_ALPHA = 1.001, 10_000

    def __init__(self):
        super().__init__()
        self.alpha = None

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_privacy_spent(
        self, *, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0

        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        rdp = sum(
            [
                privacy_analysis.compute_rdp(
                    q=sample_rate,
                    noise_multiplier=noise_multiplier,
                    steps=num_steps,
                    orders=alphas,
                )
                for (noise_multiplier, sample_rate, num_steps) in self.history
            ]
        )
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta,
        )
        return float(eps), float(best_alpha)

    def get_privacy_spent_optimal(
            self, delta: float, min_alpha: Optional[float] = None,
            max_alpha: Optional[float] = None,
            from_prev_alpha: bool = False) -> Tuple[float, float]:
        """
        Args:
            delta: target delta
            min_alpha: lower limit of RDP order to be searched for
            max_alpha: upper limit of RDP order to be searched for
            from_prev_alpha: if True, limits of RDP order to be searched for are
                calculated from previous alpha
        """
        if not self.history:
            return 0, 0
        min_alpha = self.DEFAULT_MIN_ALPHA if min_alpha is None else min_alpha
        max_alpha = self.DEFAULT_MAX_ALPHA if max_alpha is None else max_alpha
        assert max_alpha > min_alpha > 1, f"min_alpha={min_alpha}, " \
                                          f"max_alpha={max_alpha}"
        if from_prev_alpha and self.alpha is not None:
            max_alpha = self.alpha  # optimal RDP order drops monotonically
            min_alpha = max(1 + 0.5 * (max_alpha - 1), min_alpha)
        eps, best_alpha = 0, max_alpha
        while min_alpha / max_alpha < 0.99:
            d_thirds = (max_alpha - min_alpha) / 3
            alphas = [min_alpha + i * d_thirds for i in range(4)]
            rdp = sum(
                [
                    privacy_analysis.compute_rdp(
                        q=sample_rate,
                        noise_multiplier=noise_multiplier,
                        steps=num_steps,
                        orders=alphas,
                    )
                    for (noise_multiplier, sample_rate, num_steps)
                    in self.history
                ]
            )
            rdp = [r if r >= 0 else np.nan for r in rdp]
            if np.isnan(rdp).all():     # for numerical errors use alternative
                return self.get_privacy_spent(delta=delta)
            eps, best_alpha = privacy_analysis.get_privacy_spent(
                orders=alphas, rdp=rdp, delta=delta, suppress_warning=True,
            )
            if best_alpha == alphas[0]:
                max_alpha = alphas[1]
            elif best_alpha == alphas[1]:
                max_alpha = alphas[2]
            elif best_alpha == alphas[2]:
                min_alpha = alphas[1]
            else:
                min_alpha = alphas[2]
        return float(eps), float(best_alpha)

    def get_epsilon(
        self,
        delta: float,
        alphas: Optional[List[Union[float, int]]] = None,
        optimal: Optional[bool] = None,
        **kwargs,
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal
                conversion between RDP and (eps, delta)-DP
            optimal: if True, the best RDP order is searched via binary search
        """
        if optimal:
            eps, alpha = self.get_privacy_spent_optimal(delta=delta, **kwargs)
        else:
            eps, alpha = self.get_privacy_spent(delta=delta, alphas=alphas)
        self.alpha = alpha
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "rdp"