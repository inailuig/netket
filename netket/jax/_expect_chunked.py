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

# The score function (REINFORCE) gradient estimator of an expectation

from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics
from netket.utils import mpi

from ._vjp_chunked import vjp_chunked
from ._vmap_chunked import apply_chunked
from ._utils_tree import eval_shape


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _expect_chunked(
    n_chains, chunk_size, in_axes, log_pdf, expected_fun, pars, σ, *expected_fun_args
):
    if in_axes is None:
        # only chunk samples axis
        in_axes = (None, 0) + tuple(None for _ in expected_fun_args)
    else:
        n_args = 2 + len(expected_fun_args)
        assert n_args == len(in_axes)
    expected_fun = apply_chunked(
        expected_fun,
        chunk_size=chunk_size,
        in_axes=in_axes,
    )
    #from IPython import embed; embed()
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    L̄_σ = mpi_statistics(L_σ.reshape((n_chains, -1)) if n_chains else L_σ)

    return L̄_σ.mean, L̄_σ


def _expect_chunked_fwd(
    n_chains, chunk_size, in_axes, log_pdf, expected_fun, pars, σ, *expected_fun_args
):
    if in_axes is None:  # only chunk samples axis by default
        in_axes = (None, 0) + tuple(None for _ in expected_fun_args)
        chunk_argnums = tuple(i for i, v in enumerate(in_axes) if v is not None)
    else:
        assert 2 + len(expected_fun_args) == len(in_axes)

    print(
        "vjp chunked with",
        jax.tree.map(lambda x: x.shape, (pars, σ, *expected_fun_args)),
    )
    pb_L = vjp_chunked(
        expected_fun,
        pars,
        σ,
        *expected_fun_args,
        chunk_size=chunk_size,
        chunk_argnums=chunk_argnums,
        # _cotangent_is_scalar=True,
        return_forward=True,
    )

    out_shape = eval_shape(expected_fun, pars, σ, *expected_fun_args)
    vec = jnp.ones_like(out_shape) / len(out_shape)
    print("pb_L(vec) with", jax.tree.map(lambda x: x.shape, (vec,)))
    L_σ, gradL_σ = pb_L(vec)
    print("got shape L_σ", L_σ.shape)

    L̄_stat = mpi_statistics(L_σ.reshape((n_chains, -1)) if n_chains else L_σ)
    ΔL_σ = L_σ - L̄_stat.mean

    return (L̄_stat.mean, L̄_stat), (pars, σ, expected_fun_args, gradL_σ, ΔL_σ)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def _expect_chunked_bwd(
    n_chains, chunk_size, in_axes, log_pdf, expected_fun, residuals, dout
):
    pars, σ, cost_args, gradL_σ, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    def term1_fun(pars, σ, ΔL_σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        return term1

    # capture ΔL_σ to not differentiate through it
    print("evaluating...σ", σ.shape,)
    print("evaluating...ΔL_σ", ΔL_σ.shape,)
    pb_term1 = vjp_chunked(
        term1_fun,
        pars,
        σ,
        ΔL_σ,
        chunk_size=chunk_size,
        chunk_argnums=(1,2,),
        nondiff_argnums=(2,),
        # _cotangent_is_scalar=True,
        return_forward=False,
    )

    vec = jnp.ones(σ.shape[0]) / σ.shape[0]
    grad_term1 = pb_term1(vec)

    grad_f = jax.tree_util.tree_map(
        lambda x, y: mpi.mpi_mean_jax(dL̄ * (x + y))[0], grad_term1, gradL_σ
    )
    print("the pars.    is:", jax.tree.map(lambda g: (g.shape, g.dtype), pars))
    print("the gradient is:", jax.tree.map(lambda g: (g.shape, g.dtype), grad_f))

    return grad_f


_expect_chunked.defvjp(_expect_chunked_fwd, _expect_chunked_bwd)
