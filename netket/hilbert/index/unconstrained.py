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

from typing import Tuple, Callable
from netket.utils.types import Array, DType

from flax import struct
# for __pre__init__
from netket.utils.struct import dataclass as nk_struct_dataclass

from .base import HilbertIndex


def _sort(x):
    if x.ndim == 1:
        return jnp.sort(x)
    else:
        return sort_lexicographic(x)

def _searchsorted(a, x, dtype=None):
    if a.ndim == 1:
        res = jnp.searchsorted(a, x)
    else:
        res = searchsorted_lexicographic(a, x)
    if dtype is not None:
        res = res.astype(dtype)
    return res

@struct.dataclass
class LookupTableHilbertIndex(HilbertIndex):
    # TODO eventually add support for pytree states
    _all_states : Array
    _dtype : DType = struct.field(pytree_node=False)

    def __post_init__(self):
        # ensure the local states are sorted
        object.__setattr__(self, "_all_states", _sort(self._all_states))

    @property
    def n_states(self) -> int:
        return self._all_states.shape[0]

    def numbers_to_states(self, numbers: Array) -> Array:
        return self._all_states[numbers]

    def states_to_numbers(self, states: Array) -> Array:
        return _searchsorted(self._all_states, states, self._dtype)

    def all_states(self) -> Array:
        return self._all_states

    @property
    def dtype(self):
        if self._dtype is None:
            if self.n_states-1 <= np.iinfo(np.uint8).max:
                return np.uint8
            # TODO use 16 bit too?
            elif self.n_states-1 <= np.iinfo(np.uint32).max:
                return np.uint32
            else:
                # TODO check it's representable
                return jax.dtypes.canonicalize_dtype(np.uint64)
        else:
            return self._dtype

@struct.dataclass
class UnsignedIntegerHilbertIndex(HilbertIndex):
    # state and index are identical

    n_states : int = struct.field(pytree_node=False)
    dtype : DType = struct.field(pytree_node=False)

    def numbers_to_states(self, numbers: Array) -> Array:
        return numbers

    def states_to_numbers(self, states: Array) -> Array:
        return states

    def all_states(self) -> Array:
        if self.dtype is None:
            if self.n_states -1 <= np.iinfo(np.uint8).max:
                dtype = np.uint8
            # TODO use 16 bit too?
            elif self.n_states -1 <= np.iinfo(np.uint32).max:
                dtype = np.uint32
            else:
                # TODO check it's representable
                dtype = jax.dtypes.canonicalize_dtype(np.uint64)
        return jnp.arange(n_states, dtype=dtype)



@struct.dataclass
class UniformTensorProductHilbertIndex(HilbertIndex):
    # tensor product with uniform local space

    _local_index : HilbertIndex
    _size : int = struct.field(pytree_node=False)
    _dtype : DType = struct.field(pytree_node=False)

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> DType:
        if self._dtype is None:
            if self._size*self.local_size < np.iinfo(np.uint8).max:
                return np.uint8
            # TODO use 16 bit too?
            elif self._size*self.local_size < np.iinfo(np.uint32).max:
                return np.uint32
            else:
                # TODO check its representable
                return jax.dtypes.canonicalize_dtype(np.uint64)
        else:
            return self._dtype

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
        return self._local_index.numbers_to_states(local_numbers.astype(self._local_index.dtype))

    def all_states(self, out=None):
        return self.numbers_to_states(jnp.arange(self.n_states, dtype=self.dtype))


@nk_struct_dataclass
class UnconstrainedHilbertIndex(UniformTensorProductHilbertIndex):

    def __pre_init__(self, local_states: Array, size: int, dtype: DType = None, local_dtype: DType = None):
        return (LookupTableHilbertIndex(local_states, local_dtype), size, dtype), {}


@nk_struct_dataclass
class UnconstrainedHilbertIndexBoson(UniformTensorProductHilbertIndex):
    # _size is inherited
    n_max : int = struct.field(pytree_node=False)

    # override _local_index
    @property
    def _local_index(self):
        return UnsignedIntegerHilbertIndex(self.n_max, self._dtype)
