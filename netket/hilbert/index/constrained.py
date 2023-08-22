# Copyright 2023 The NetKet Authors - All rights reserved.
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

from functools import lru_cache

from .unconstrained import UnconstrainedHilbertIndex

import jax
import jax.numpy as jnp

import numpy as np
from jax.tree_util import Partial

from typing import Tuple, Callable
from netket.utils.types import Array, DType

import itertools

from flax import struct
# for __pre__init__
from netket.utils.struct import dataclass as nk_struct_dataclass

def compute_constrained_to_bare_conversion_table(hilbert_index, constraint_fun, *, chunk_size: int = 65536):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    This function operates on blocks of `chunk_size` states at a time in order
    to lower the memory cost. The default chunk size has been chosen by instinct
    and is likely wrong.
    """

    with jax.ensure_compile_time_eval():
        n_chunks = int(np.ceil(hilbert_index.n_states / chunk_size))
        bare_number_chunks = []
        for i in range(n_chunks):
            id_start = chunk_size * i
            id_end = np.minimum(chunk_size * (i + 1), hilbert_index.n_states)
            ids = jnp.arange(id_start, id_end, dtype=hilbert_index.dtype)
            states = hilbert_index.numbers_to_states(ids)
            # TODO jit the constraint_fun
            is_constrained = constraint_fun(states)
            (chunk_bare_number,) = jnp.nonzero(is_constrained)
            bare_number_chunks.append(chunk_bare_number + id_start)
        bare_numbers = jnp.concatenate(bare_number_chunks)
    return bare_numbers


@nk_struct_dataclass
class ConstrainedHilbertIndex:
    _unconstrained_index : UnconstrainedHilbertIndex
    _constraint_fun : Callable = struct.field(pytree_node=False)
    _bare_numbers : Array

    def __pre_init__(self, local_states, size, constraint_fun, dtype=None, **kwargs):
        hilbert_index = UnconstrainedHilbertIndex(local_states, size, dtype)
        # TODO make it optional
        bare_numbers = compute_constrained_to_bare_conversion_table(hilbert_index, constraint_fun, **kwargs)
        return (hilbert_index, constraint_fun, bare_numbers), {}

    @property
    def dtype(self):
        return self._unconstrained_index.dtype

    @property
    def n_states(self):
        return self._bare_numbers.shape[0]

    @property
    def size(self) -> int:
        return self._unconstrained_index.size

    @property
    def local_states(self) -> Array:
        return self._unconstrained_index.local_states

    @property
    def local_size(self) -> int:
        return self._unconstrained_index.local_size

    def states_to_numbers(self, states):
        out = self._unconstrained_index.states_to_numbers(states)
        return jnp.searchsorted(self._bare_numbers, out)

    def numbers_to_states(self, numbers):
        # convert to original space
        numbers = self._bare_numbers[numbers]
        return self._unconstrained_index.numbers_to_states(numbers)

    def all_states(self):
        return self.numbers_to_states(jnp.arange(self.n_states))

    def to_lookup_table(self):
        return LookupTableHilbertIndex(self.all_states())


@nk_struct_dataclass
class SumConstrainedHilbertIndex:
    shape : Tuple[int] = struct.field(pytree_node=False)
    n_particles : int = struct.field(pytree_node=False)
    dtype : DType = struct.field(pytree_node=False)

    @property
    def n_states(self):
        if self.n_max == 1:
            return math.comb(self.size, self.n_particles)
        else:
            raise NotImplementedError  # use to_lookup_table

    @property
    def size(self):
        return len(self.shape)

    def states_to_numbers(self, states):
        raise NotImplementedError  # use to_lookup_table

    def numbers_to_states(self, numbers):
        raise NotImplementedError  # use to_lookup_table

    def all_states(self):
        raise NotImplementedError  # use to_lookup_table

    def _all_states(self):
        c = jnp.repeat(jnp.eye(self.size, dtype=self.dtype), np.array(self.shape) - 1, axis=0)
        combs = jnp.array(list(itertools.combinations(np.arange(len(c)), self.n_particles)))
        _all_states = c[combs].sum(axis=1, dtype=self.dtype)
        if (np.array(self.shape) > 1).any():
            with jax.ensure_compile_time_eval():
                _all_states = jnp.unique(_all_states, axis=0)
        return jnp.asarray(_all_states)

    def to_lookup_table(self):
        return LookupTableHilbertIndex(self._all_states())
