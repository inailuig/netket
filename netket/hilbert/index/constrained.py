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

import numpy as np
from jax.tree_util import Partial

import itertools

# for __pre__init__
from netket.utils.struct import dataclass as nk_struct_dataclass

def compute_constrained_to_bare_conversion_table(
    hilbert_index, constraint_fn, *, chunk_size: int = 100000
):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    This function operates on blocks of `chunk_size` states at a time in order
    to lower the memory cost. The default chunk size has been chosen by instinct
    and is likely wrong.
    """

    n_chunks = int(np.ceil(hilbert_index.n_states / chunk_size))
    bare_number_chunks = []
    for i in range(n_chunks):
        id_start = chunk_size * i
        id_end = np.minimum(chunk_size * (i + 1), hilbert_index.n_states)
        ids = np.arange(id_start, id_end)

        states = hilbert_index.numbers_to_states(ids)
        is_constrained = constraint_fn(states)
        (chunk_bare_number,) = np.nonzero(is_constrained)
        bare_number_chunks.append(chunk_bare_number + id_start)

    return np.concatenate(bare_number_chunks)



# the generic one

@nk_struct_dataclass
class ConstrainedHilbertIndex:
    _unconstrained_index : UnconstrainedHilbertIndex
    _constraint_fn : Partial

    def __pre_init__(self, local_states, size, constraint_fun):
        if not isinstance(constraint_fun, Partial):
            constraint_fun = Partial(constraint_fun)
        return (UnconstrainedHilbertIndex(local_states, size), constraint_fun), {}

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
        return self._unconstrained_index.number_to_state(numbers[i])

    def all_states(self):
        return self.numbers_to_states(jnp.arange(self.n_states))

@nk_struct_dataclass
class SumConstrainedHilbertIndex:
    _unconstrained_index : UnconstrainedHilbertIndexBoson
    n_particles : int = struct.field(pytree_node=False)

    @property
    def n_states(self):
        if self.n_max == 1:
            return math.comb(self.size, self.n_particles)
        else:
            raise NotImplementedError  # use LookupTableHilbertIndex instead
            # return self.all_states().shape[0]

    @property
    def n_max(self):
        return self.local_size

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
        raise NotImplementedError  # use LookupTableHilbertIndex instead

    def numbers_to_states(self, numbers):
        raise NotImplementedError  # use LookupTableHilbertIndex instead

    @property
    def _dtype(self):
        return self._unconstrained_index._dtype

    def all_states(self):
         c = np.repeat(np.eye(N, dtype=self._dtype), np.array(self._shape) - 1, axis=0)  # TODO dtype=np.int32
        _all_states = np.array(list(itertools.combinations(list(c), self.n_particles))).sum(axis=1, dtype=self._dtype)
        if self.n_max > 1:
            _all_states = np.unique(_all_states, axis=0)
        return jnp.asarray(_all_states)
