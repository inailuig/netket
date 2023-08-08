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


@dispatch
def random_state(hilb: Fock, key, shape):


    # If unconstrained space, use fast sampling
    if hilb.n_particles is None:
        return _random_states(hilb, rngkey, shape)

    else:
        state = jax.pure_callback(
            lambda rng: _random_states_with_constraint(hilb, rng, batches, dtype),
            jax.ShapeDtypeStruct(shape, dtype),
            key,
        )

        return state

@partial(jax.jit, static_argnames='shape')
def _random_states(hilb, key, shape):
    assert hilb.n_particles is None
    return jax.random.randint(key, shape=shape+(hilb.size,), minval=0, maxval=hilb.n_max+1, dtype=hilb.dtype)

def _choice(key, p):
    # p needs to be in [0, 1]
    # p needs to be integer or bool
    #
    # TODO the following can probably be implemented more efficiently
    n_candidates = p.sum(axis=-1)
    p_padded = jnp.pad(p, [(0,0)]*(p.ndim-1)+[(1,0)])
    cs = jnp.cumsum(p_padded, axis=-1)
    n_candidates = cs[..., -1]
    cs = cs[..., :-1]*p
    r = jax.random.uniform(key, p.shape[:-1]) * n_candidates # 1 is exlusive
    return cs == r.astype(cs.dtype)


@partial(jax.jit, static_argnames='shape')
def _random_states_with_constraint(hilb, key, shape):
    assert hilb.n_particles is not None
    # distribute uniformly, excluding fully occupied sites

    sites = jnp.arange(hilb.size)

    # use shape (per site n_max)
    n_max = jnp.array(hilb.shape)-1
    keys = jax.random.split(key, hilb.n_particles)

    def body_fun(x, key):
        p = x < n_max
        carry = x + _choice(key, p)
        return carry, None

    init = key, jnp.zeros(shape+(hilb.size,), dtype=hilb.dtype)
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
