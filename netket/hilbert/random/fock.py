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
import numpy as np
from jax import numpy as jnp

from netket.hilbert import Fock
from netket.utils.dispatch import dispatch

from functools import partial


@dispatch
def random_state(hilb: Fock, key, batches: int):
    shape = (batches,)

    # If unconstrained space, use fast sampling
    if hilb.n_particles is None:
        return _random_states(hilb, key, shape)
    else:
        return _random_states_with_constraint(hilb, key, shape)

@partial(jax.jit, static_argnames=('hilb','shape'))
def _random_states(hilb, key, shape):
    assert hilb.n_particles is None
    if jnp.issubdtype(hilb.dtype, jnp.integer):
        return jax.random.randint(key, shape=shape+(hilb.size,), minval=0, maxval=hilb.n_max+1, dtype=hilb.dtype)
    else:
        return jax.random.randint(key, shape=shape+(hilb.size,), minval=0, maxval=hilb.n_max+1).astype(hilb.dtype)

def _choice(key, p):
    # p needs to be in [0, 1], of type integer or bool
    # in the following all the sites are indexed starting from 1
    # to distinguish between site 0 (now 1) and not selecting it
    # e.g  take p = [[1 0 0 1 0 1 1 0]]
    cs = jnp.cumsum(p, axis=-1) # now  cs = [[1 1 1 2 2 3 4 4]]
    n_candidates = cs[..., -1] # == p.sum(axis=-1, keepdims=True)
    # 1 is exlusive in random.uniform
    # +1 because we index starting from 1
    r = jax.random.uniform(key, p.shape[:-1]) * n_candidates + 1
    # now cs*p = [[1 0 0 2 0 3 4 0]] and floor(r) in [1,2,3,4]
    return (cs*p) == jax.lax.floor(r).astype(cs.dtype)[..., None]


@partial(jax.jit, static_argnames=('hilb', 'shape'))
def _random_states_with_constraint(hilb, key, shape):
    assert hilb.n_particles is not None
    # distribute uniformly, excluding fully occupied sites

    # sites = jnp.arange(hilb.size)

    # use shape (per site n_max)
    n_max = jnp.array(hilb.shape)-1
    keys = jax.random.split(key, hilb.n_particles)

    def body_fun(x, key):
        p = x < n_max
        carry = x + _choice(key, p)
        return carry, None

    init = jnp.zeros(shape+(hilb.size,), dtype=hilb.dtype)
    return jax.lax.scan(body_fun, init, keys)[0]


@dispatch
def flip_state_scalar(hilb: Fock, key, σ, idx):
    if hilb._n_max == 0:
        return σ, σ[idx]

    n_states = hilb._n_max + 1

    σi_old = σ[idx]
    r = jax.random.uniform(key)
    σi_new = jax.numpy.floor(r * (n_states - 1))
    σi_new = σi_new + (σi_new >= σi_old)
    σi_new = σi_new.astype(σ.dtype)

    return σ.at[idx].set(σi_new), σi_old
