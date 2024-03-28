# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional
from functools import partial

import jax
from flax import linen as nn
from jax import numpy as jnp

from netket import config
from netket.hilbert import DiscreteHilbert
from netket.nn import to_array
from netket.utils.types import PyTree, SeedT, DType

from .base import Sampler, SamplerState

class RandomSampler(Sampler):
    """
    This sampler generates samples using hilbert.random_state()
    """

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        machine_pow: int = 2,
        dtype: DType = float,
    ):
        """
        Construct an exact sampler.

        Args:
            hilbert: The Hilbert space to sample.
            machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """
        super().__init__(hilbert, machine_pow=machine_pow, dtype=dtype)

    def _init_state(
        sampler,
        machine: nn.Module,
        parameters: PyTree,
        seed: Optional[SeedT] = None,
    ):
        return seed

    def _reset(sampler, machine, parameters, state):
        return state

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(
        sampler,
        machine: nn.Module,
        parameters: PyTree,
        state: SamplerState,
        chain_length: int,
    ) -> tuple[jnp.ndarray, SeedT]:
        new_rng, rng = jax.random.split(state)
        samples = sampler.hilbert.random_state(rng, size=(sampler.n_batches, chain_length), dtype=sampler.dtype)
        return samples, new_rng
