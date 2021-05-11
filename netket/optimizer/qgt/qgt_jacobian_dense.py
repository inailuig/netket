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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree
from netket.utils import mpi
import netket.jax as nkjax

from ..linear_operator import LinearOperator, Uninitialized


def QGTJacobianDense(
    vstate, *, mode, rescale_shift=False, **kwargs
) -> "QGTJacobianDenseT":
    O, scale = gradients(
        vstate._apply_fun,
        vstate.parameters,
        vstate.samples,
        vstate.model_state,
        mode,
        rescale_shift,
    )

    return QGTJacobianDenseT(O=O, scale=scale, **kwargs)


@struct.dataclass
class QGTJacobianDenseT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    O: jnp.ndarray = Uninitialized
    """Gradients O_ij = ∂log ψ(σ_i)/∂p_j of the neural network 
    for all samples σ_i at given values of the parameters p_j
    Average <O_j> subtracted for each parameter
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, columns normalised to unit norm
    """

    scale: Optional[jnp.ndarray] = None
    """If not None, contains 2-norm of each column of the gradient matrix,
    i.e., the sqrt of the diagonal elements of the S matrix
    """

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        if not hasattr(vec, "ndim"):
            vec, unravel = nkjax.tree_ravel(vec)
        else:
            unravel = None

        if self.scale is not None:
            vec = vec * self.scale

        result = (
            mpi.mpi_sum_jax(((self.O @ vec).T.conj() @ self.O).T.conj())[0]
            + self.diag_shift * vec
        )

        if self.scale is not None:
            result = result * self.scale

        if unravel is None:
            return result
        else:
            return unravel(result)

    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterataive solver.

        Args:
            y: the vector y in the system above.
            x0: optional initial guess for the solution.

        Returns:
            x: the PyTree solving the system.
            info: optional additional informations provided by the solver. Might be
                None if there are no additional informations provided.
        """

        # Ravel input PyTrees, record unravelling function too
        grad, unravel = nkjax.tree_ravel(y)

        if x0 is not None:
            x0, _ = nkjax.tree_ravel(x0)
            if self.scale is not None:
                x0 = x0 * self.scale

        if self.scale is not None:
            grad = grad / self.scale

        # to pass the object LinearOperator itself down
        # but avoid rescaling, we pass down an object with
        # scale = None
        unscaled_self = self.replace(scale=None)

        out, info = solve_fun(unscaled_self, grad, x0=x0)

        if self.scale is not None:
            out = out / self.scale

        return unravel(out), info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        if self.scale is None:
            O = self.O
            diag = jnp.eye(self.O.shape[1])
        else:
            O = self.O * self.scale[jnp.newaxis, :]
            diag = jnp.diag(self.scale ** 2)

        return mpi.mpi_sum_jax(O.T.conj() @ O)[0] + self.diag_shift * diag


@partial(jax.jit, static_argnums=(0, 4, 5))
def gradients(
    apply_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    params: PyTree,
    samples: jnp.ndarray,
    model_state: Optional[PyTree],
    mode: str,
    rescale_shift: bool,
):
    """Calculates the gradients O_ij by backpropagating every sample separately,
    vectorising the loop using vmap
    If rescale_shift is True, columns of O are rescaled to unit norm, and
    scale factor sqrt(S_kk) returned as a separate vector for
    scale-invariant regularisation as per Becca & Sorella p. 143.
    """
    # Ravel the parameter PyTree and obtain the unravelling function
    params, unravel = nkjax.tree_ravel(params)

    if jnp.ndim(samples) != 2:
        samples = jnp.reshape(samples, (-1, samples.shape[-1]))
    n_samples = samples.shape[0] * mpi.n_nodes

    if mode == "holomorphic":
        # Preapply the model state so that when computing gradient
        # we only get gradient of parameters
        # Also divide through sqrt(n_samples) to normalise S matrix in the end
        def fun(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[0]
                / n_samples ** 0.5
            )

        grads = _grad_vmap_minus_mean(fun, params, samples, True)
    elif mode == "R2R":

        def fun(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[
                    0
                ].real
                / n_samples ** 0.5
            )

        grads = _grad_vmap_minus_mean(fun, params, samples, False)
    elif mode == "R2C":

        def fun1(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[
                    0
                ].real
                / n_samples ** 0.5
            )

        def fun2(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[
                    0
                ].imag
                / n_samples ** 0.5
            )

        # Stack real and imaginary parts as real matrixes along the "sample"
        # axis to get Re(O†O) directly
        grads = jnp.concatenate(
            (
                _grad_vmap_minus_mean(fun1, params, samples, False),
                _grad_vmap_minus_mean(fun2, params, samples, False),
            ),
            axis=0,
        )
    else:
        raise NotImplementedError(
            'Differentation mode must be one of "R2R", "R2C", "holomorphic", got "{}"'.format(
                mode
            )
        )

    if rescale_shift:
        sqrt_Skk = (
            mpi.mpi_sum_jax(
                jnp.sum((grads * grads.conj()).real, axis=0, keepdims=True)
            )[0]
            ** 0.5
        )
        return grads / sqrt_Skk, sqrt_Skk.flatten()
    else:
        return grads, None


def _grad_vmap_minus_mean(
    fun: Callable, params: jnp.ndarray, samples: jnp.ndarray, holomorphic: bool
):
    """Calculates the gradient of a neural network for a number of samples
    efficiently using vmap(grad),
    and subtracts their mean for each parameter, i.e., each column
    """
    grads = jax.vmap(
        jax.grad(fun, holomorphic=holomorphic), in_axes=(None, 0), out_axes=0
    )(params, samples)
    return grads - mpi.mpi_sum_jax(grads.sum(axis=0, keepdims=True))[0] / (
        grads.shape[0] * mpi.n_nodes
    )
