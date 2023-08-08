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

import jax
import jax.numpy as jnp
import numpy as np

from functools import wraps

from netket.utils.types import DType
from jax import Array

# for __pre__init__
from netket.utils.struct import dataclass as nk_struct_dataclass
from flax import struct

from .base import HilbertIndex


def _sort(x):
    if x.ndim == 1:
        return jnp.sort(self._all_states)
    else:
        return sort_lexicographic(x)

def _searchsorted(a, x):
    if a.ndim == 1:
        return jnp.searchsorted(a, x)
    else:
        return searchsorted_lexicographic(a, x)

@struct.dataclass
class LookupTableHilbertIndex(HilbertIndex):
    # TODO eventually add support for pytree states
    _all_states : Array

    def __post_init__(self):
        # ensure the local states are sorted
        object.__setattr__(self, "_all_states", _sort(self._all_states)

    @property
    def n_states(self) -> int:
        return self._all_states.shape[0]

    def numbers_to_states(self, numbers: Array) -> Array:
        return self._all_states[numbers]

    def states_to_numbers(self, states: Array) -> Array:
        return _searchsorted(self._all_states, states)

    def all_states(self) -> Array:
        return self._all_states


@struct.dataclass
class UnsignedIntegerHilbertIndex(HilbertIndex):
    # state and index are identical

    n_states : int = struct.field(pytree_node=False)

    def numbers_to_states(self, numbers: Array) -> Array:
        return numbers

    def states_to_numbers(self, states: Array) -> Array:
        return states

    def all_states(self) -> Array:
        return jnp.arange(n_states)



@sturct.dataclass
class UniformTensorProductHilbertIndex(HilbertIndex):
    # tensor product with uniform local space

    _local_index : HilbertIndex
    _size : int

    @property
    def size(self) -> int:
        return self._size

    @property
    def local_size(self) -> int:
        return self._local_index.n_states

    @property
    def n_states(self):
        return self.local_size**self._size

    @property
    def local_states(self):
        return self._local_index.all_states()

    @property
    def _basis(self):
        return self.local_size**jax.lax.iota(int, self.size)[::-1]

    def states_to_numbers(self, states):
        local_numbers = self._local_index.states_to_numbers(states)
        return local_numbers@self._basis

    def numbers_to_states(self, numbers):
        local_numbers = (numbers[..., None] // self._basis) % self.local_size
        return self._local_index.numbers_to_states(local_numbers)

    def all_states(self, out=None):
        return self.numbers_to_states(jnp.arange(self.n_states))


@nk_struct_dataclass
class UnconstrainedHilbertIndex(UniformTensorProductHilbertIndex):

    def __pre_init__(self, local_states: Array, size: int):
        return (LookupTableHilbertIndex(local_states), size), {}


@nk_struct_dataclass
class UnconstrainedHilbertIndexBoson(UniformTensorProductHilbertIndex):
    # _size is inherited
    n_max : int = struct.field(pytree_node=False)

    # override _local_index
    @property
    def _local_index(self):
        return UnsignedIntegerHilbertIndex(self.n_max)

    @property
    def _dtype(self):
        if n_max <= 256:
            return jnp.uint8
        # TODO 16 bit?
        elif n_max <= 2**32:
            return jnp.uint32
        else:
            return jnp.uint64
