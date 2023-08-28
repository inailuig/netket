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
from jax import numpy as jnp

from netket.hilbert import Qubit
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: Qubit, key, batches: int):
    shape = (batches,)
    # we special case for float here (use int/bool and cast)
    if jnp.issubdtype(hilb.dtype, jnp.integer):
        return jax.random.randint(key, shape=shape+(hilb.size,), minval=0, maxval=2, dtype=hilb.dtype)
    else:
        rs = jax.random.randint(key, shape=shape+(hilb.size,), minval=0, maxval=2)
        return rs.astype(hilb.dtype)


@dispatch
def flip_state_scalar(hilb: Qubit, key, x, i):
    return x.at[i].set(-x[i] + 1), x[i]
