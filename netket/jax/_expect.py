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

from typing import Callable, Optional, Union
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
from netket.utils.types import PyTree

from ._vjp import vjp as nkvjp
from ._expect_chunked import _expect_chunked


def expect(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: Optional[int] = None,
    chunk_size: Optional[int] = None,
    in_axes: Optional[tuple[Union[int, None], ...]] = None,
) -> tuple[jnp.ndarray, Stats]:
    r"""
    Computes the expectation value over a log-pdf, equivalent to

    .. math::

        \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[f(\sigma)] = \sum_{\mathbf{x}} p(\mathbf{x}) f(\mathbf{x})

    where the evaluation of the expectation value is approximated using the sample average, with
    samples :math:`\sigma` that are assumed to be drawn from the probability distribution :math:`p(x)`.

    .. math::

        \langle f \rangle \approx \frac{1}{N} \sum_{i=1}^{N} f(\sigma_i)

    This function ensures that the backward pass is computed correctly, by first differentiating the first equation
    above, and then by approximating the expectation values again using the sample average. The resulting
    backward gradient is

    .. math::

            \nabla \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[(\nabla \log p(\sigma)) f(\sigma) + \nabla f(\sigma)]

    where again, the expectation values are comptued using the sample average.

    Args:
        log_pdf: The log-pdf function from which the samples are drawn. This should output real values, and have a signature
            :code:`log_pdf(pars, σ) -> jnp.ndarray`.
        expected_fun: The function to compute the expectation value of. This should have a signature
            :code:`expected_fun(pars, σ, *expected_fun_args) -> jnp.ndarray`.
        pars: The parameters of the model.
        σ: The samples to compute the expectation value over.
        expected_fun_args: Additional arguments to pass to the expected_fun function (will be differentiated; to avoid
            differentiation, capture them as constants inside of the expected_fun).
        n_chains: The number of chains to use in the computation. If None, the number of chains is inferred from the shape of the input.
        chunk_size: The size of the chunks to use in the computation. If None, no chunking is used.
        in_axes: The axes along which to perform the chunking. If none, only the samples are chunked, otherwise this must be
            the sharding declaration of the samples and the additional arguments to the expected_fun function (must have length
            equal to the number of expected_fun_args + 2).

    Returns:
        A tuple where the first element is the scalar value containing the expectation value, and the second element is
        a :class:`netket.stats.Stats` object containing the statistics (including the mean) of the expectation value.
    """
    if chunk_size is not None:
        if σ.shape[0] <= chunk_size:
            chunk_size = None
    chunk_size = 2

    if chunk_size is None:
        return _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args)
    else:
        return _expect_chunked(
            n_chains,
            chunk_size,
            in_axes,
            log_pdf,
            expected_fun,
            pars,
            σ,
            *expected_fun_args,
        )


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ)
    # L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ


def _expect_fwd(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    # The forward pass is in principle as easy as calling and averaging over
    # expected_fun. However, to avoid calling this function twice, in here and
    # in the backward pass, we directly build the vjp, with an auxiliary output.
    #
    # L_σ = expected_fun(pars, σ, *expected_fun_args)

    def f(pars, σ, *cost_args):
        # This is the real forward pass.
        L_σ = expected_fun(pars, σ, *cost_args)
        L̄_stat = mpi_statistics(L_σ.reshape((n_chains, -1)) if n_chains else L_σ)

        # We have two terms in the gradient: ∇(p⋅L) = p⋅(∇logp⋅L + ∇L)
        # Below we write a function whose gradient will lead to the gradient above
        # (Excluding the p(x) which is implicit because of sampling estimation).
        # We do not really care about the output of this function. Only of its gradient!

        # We will use the baseline trick to reduce the variance
        ΔL_σ = L_σ - L̄_stat.mean

        # We first compute something whose gradient evaluates to the first term
        log_p = log_pdf(pars, σ)
        log_p_L_σ = jax.vmap(jnp.multiply)(jax.lax.stop_gradient(ΔL_σ), log_p)

        # And we add to it the term whose gradient evaluates to ∇L
        out = mpi_mean(log_p_L_σ + L_σ, axis=0)
        out = out.sum()
        return out, L̄_stat

    L_σ, pb, L̄_stat = nkvjp(f, pars, σ, *expected_fun_args, has_aux=True)

    return (L̄_stat.mean, L̄_stat), (pb,)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def _expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    (pb,) = residuals
    dL̄, dL̄_stats = dout
    grad_f = pb(dL̄)
    return grad_f


_expect.defvjp(_expect_fwd, _expect_bwd)
