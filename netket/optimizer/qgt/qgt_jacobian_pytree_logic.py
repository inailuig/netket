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

from typing import Any, Optional, Tuple
from functools import partial, wraps
from netket.jax import compose

import jax
import jax.flatten_util
import jax.numpy as jnp

import flax

import numpy as np

from netket.stats import subtract_mean
from netket.utils import mpi
import netket.jax as nkjax

from netket.utils.types import Array, Callable, PyTree, Scalar

import netket.jax as nkjax
from netket.jax import tree_cast, tree_conj, tree_axpy, tree_to_real

from .pytreearray import *

# TODO better name and move it somewhere sensible
def single_sample(forward_fn):
    """
    A decorator to make the forward_fn accept a single sample
    """

    def f(W, σ):
        return forward_fn(W, σ[jnp.newaxis, :])[0]

    return f


# TODO move it somewhere reasonable
def tree_subtract_mean(oks: PyTree) -> PyTree:
    """
    subtract the mean with MPI along axis 0 of every leaf
    """
    return jax.tree_map(partial(subtract_mean, axis=0), oks)  # MPI


@partial(jax.vmap, in_axes=(None, None, 0))
def jacobian_real_holo(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→R or holomorphic C→C, so single grad is enough

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """
    y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
    res, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    return res


@partial(jax.vmap, in_axes=(None, None, 0, None))
def _jacobian_cplx(
    forward_fn: Callable, params: PyTree, samples: Array, _build_fn: Callable
) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→C, backpropagates 1 and -1j

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """
    y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
    gr, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    gi, _ = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return _build_fn(gr, gi)


@partial(wraps(_jacobian_cplx))
def jacobian_cplx(
    forward_fn, params, samples, _build_fn=partial(jax.tree_multimap, jax.lax.complex)
):
    return _jacobian_cplx(forward_fn, params, samples, _build_fn)


centered_jacobian_real_holo = compose(tree_subtract_mean, jacobian_real_holo)
centered_jacobian_cplx = compose(tree_subtract_mean, jacobian_cplx)


def _sqrt_n_samp(samples):
    n_samp = samples.shape[0] * mpi.n_nodes  # MPI
    return np.sqrt(n_samp)


def stack_jacobian(centered_oks: PyTree) -> PyTree:
    """
    Return the real and imaginary parts of ΔOⱼₖ stacked along the sample axis
    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]
    """
    return jax.tree_map(
        lambda x: jnp.concatenate([x.real, x.imag], axis=0), centered_oks
    )


def stack_jacobian_tuple(centered_oks_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along the sample axis

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        centered_oks_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    return jax.tree_multimap(
        lambda re, im: jnp.concatenate([re, im], axis=0), *centered_oks_re_im
    )


def _scale(centered_oks: PyTreeArrayT):
    """
    compute √Sₖₖ
    to do scale-invariant regularization (Becca & Sorella 2017, pp. 143)
    Sₖₗ/(√Sₖₖ√Sₗₗ) = ΔOₖᴴΔOₗ/(√Sₖₖ√Sₗₗ) = (ΔOₖ/√Sₖₖ)ᴴ(ΔOₗ/√Sₗₗ)
    """
    O = centered_oks
    scale = (O * O.conj()).real.sum(axis=0) ** 0.5
    return scale


# ==============================================================================
# the logic above only works for R→R, R→C and holomorphic C→C
# here the other modes are converted


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def prepare_centered_oks(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    model_state: Optional[PyTree],
    mode: str,
    rescale_shift: bool,
    flatten: bool,
) -> PyTree:
    """
    compute ΔOⱼₖ = Oⱼₖ - ⟨Oₖ⟩ = ∂/∂pₖ ln Ψ(σⱼ) - ⟨∂/∂pₖ ln Ψ⟩
    divided by √n

    In a somewhat intransparent way this also internally splits all parameters to real
    in the 'real' and 'complex' modes (for C→R, R&C→R, R&C→C and general C→C) resulting in the respective ΔOⱼₖ
    which is only compatible with split-to-real pytree vectors

    Args:
        apply_fun: The forward pass of the Ansatz
        params : a pytree of parameters p
        samples : an array of (n in total) batched samples σ
        model_state: untrained state parameters of the model
        mode: differentiation mode, must be one of 'real', 'complex', 'holomorphic'
        rescale_shift: whether scale-invariant regularisation should be used (default: True)

    Returns:
        if not rescale_shift:
            a pytree representing the centered jacobian of ln Ψ evaluated at the samples σ, divided by √n;
            None
        else:
            the same pytree, but the entries for each parameter normalised to unit norm;
            pytree containing the norms that were divided out (same shape as params)

    """
    # un-batch the samples
    samples = samples.reshape((-1, samples.shape[-1]))

    # pre-apply the model state
    def forward_fn(W, σ):
        return apply_fun(flax.core.freeze({"params": W, **model_state}), σ)

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        centered_jacobian_fun = centered_jacobian_real_holo
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C
        # centered_jacobian_fun = compose(stack_jacobian, centered_jacobian_cplx)

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        centered_jacobian_fun = compose(
            stack_jacobian_tuple,
            partial(centered_jacobian_cplx, _build_fn=lambda *x: x),
        )
    elif mode == "holomorphic":
        split_complex_params = False
        centered_jacobian_fun = centered_jacobian_real_holo
    else:
        raise NotImplementedError(
            'Differentiation mode should be one of "real", "complex", or "holomorphic", got {}'.format(
                mode
            )
        )

    if split_complex_params:
        # doesn't do anything if the params are already real
        params, reassemble = tree_to_real(params)

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    else:
        f = forward_fn

    if flatten:
        params_flat, unravel = jax.flatten_util.ravel_pytree(params)

        def f_flat(params, samples):
            return f(unravel(params), samples)

    centered_oks = centered_jacobian_fun(
        f_flat if flatten else f,
        params_flat if flatten else params,
        samples,
    )
    # starting from here we use PyTreeArray
    centered_oks = PyTreeArray2(centered_oks) / _sqrt_n_samp(samples)

    if rescale_shift:
        scale = _scale(centered_oks)
        return centered_oks / scale, scale
    else:
        return centered_oks, None
