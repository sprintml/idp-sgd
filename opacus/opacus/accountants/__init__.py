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
from typing import List

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .idp import IndividualAccountant
from .prv import PRVAccountant
from .rdp import RDPAccountant


__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
    "PRVAccountant",
    "IndividualAccountant",
]


def create_accountant(mechanism: str) -> IAccountant:
    if mechanism == "rdp":
        return RDPAccountant()
    elif mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "prv":
        return PRVAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")


def create_individual_accountant(mechanism: str, n_groups: int) -> IAccountant:
    accountants = [create_accountant(mechanism=mechanism)
                   for _ in range(n_groups)]
    return IndividualAccountant(accountants=accountants, n_groups=n_groups)
