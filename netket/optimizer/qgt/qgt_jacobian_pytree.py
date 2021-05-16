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

from netket.utils.types import PyTree, Array
from netket.utils import mpi
from netket.stats import sum_inplace
import netket.jax as nkjax

from ..linear_operator import LinearOperator, Uninitialized

from .qgt_jacobian_pytree_logic import prepare_centered_oks

from .pytreearray import *


def QGTJacobianPyTree(
    vstate, *, mode="auto", rescale_shift=False, flatten=False, **kwargs
) -> "QGTJacobianPyTreeT":
    # Choose sensible default mode
    if mode == "auto":
        complex_output = nkjax.is_complex(
            jax.eval_shape(
                vstate._apply_fun,
                {"params": vstate.parameters, **vstate.model_state},
                vstate.samples.reshape(-1, vstate.samples.shape[-1]),
            )
        )

        mode = "complex" if complex_output else "real"

    O, scale = prepare_centered_oks(
        vstate._apply_fun,
        vstate.parameters,
        vstate.samples.reshape(-1, vstate.samples.shape[-1]),
        vstate.model_state,
        mode,
        rescale_shift,
        flatten,
    )

    return QGTJacobianPyTreeT(
        O=O, scale=scale, params=vstate.parameters, mode=mode, flatten=flatten, **kwargs
    )


@struct.dataclass
class QGTJacobianPyTreeT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    O: PyTree = Uninitialized
    """Centred gradients ΔO_ij = O_ij - <O_j> of the neural network, where
    O_ij = ∂log ψ(σ_i)/∂p_j, for all samples σ_i at given values of the parameters p_j
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, O_ij for is normalised to unit norm for each parameter j
    """

    scale: Optional[PyTree] = None
    """If not None, contains 2-norm of each column of the gradient matrix,
    i.e., the sqrt of the diagonal elements of the S matrix
    """

    params: PyTree = Uninitialized
    """Parameters of the network. Its only purpose is to represent its own shape when scale is None"""

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode:
        - "real": for real-valued R->R and C->R ansatze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C ansatze, splits the complex
                  inputs and outputs into real and imaginary part
        - "holomorphic": for any ansatze. Does not split complex values.
        - "auto": autoselect real or complex.
    """

    flatten: bool = struct.field(pytree_node=False, default=False)

    _in_solve: bool = struct.field(pytree_node=False, default=False)
    """Internal flag used to signal that we are inside the _solve method and matmul should
    not take apart into real and complex parts the other vector"""

    def __matmul__(self, vec: Union[PyTree, Array]) -> Union[PyTree, Array]:
        return _matmul(self, vec)

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
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
            In R→R and R→C modes, real and imaginary parts of parameters get own rows/columns
        """
        # TODO take real for real params
        return _to_dense(self)

    def _to_matrix(self):
        # TODO take real for real params
        return _to_mat(self)

    def to_matrix(self):
        return QGTJacobianMatrixT(self.diag_shift, self._to_matrix())


# TODO name?
@struct.dataclass
class QGTJacobianMatrixT(LinearOperator):
    S: PyTree = Uninitialized

    # TODO scale-invariant-regularization

    def __matmul__(self, v):
        return _matmul2(self.S, v)

    def __add__(self, eps):
        return self.replace(S=S.add_diag_scalar(eps), diag_shift=self.diag_shift + eps)

    def to_dense(self):
        return _to_dense2(self.S)

    # TODO custom replace for diag_shift


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


@jax.jit
def _matmul(
    self: QGTJacobianPyTreeT, vec: Union[PyTree, Array]
) -> Union[PyTree, Array]:

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = nkjax.tree_to_real(vec)

    ravel_or_unravel = None
    if self.flatten:
        if not hasattr(vec, "ndim"):
            vec, ravel_or_unravel = nkjax.tree_ravel(vec)
    else:
        if hasattr(vec, "ndim"):
            p = self.params
            if reassemble is not None:
                p, _ = nkjax.tree_to_real(p)
            _, unravel = nkjax.tree_ravel(p)
            vec = unravel(vec)
            ravel_or_unravel = lambda x: nkjax.tree_ravel(x)[0]

    # +++++++++++++++++++++++++++++
    vec = PyTreeArray(vec)
    # begin PyTreeArray

    if self.scale is not None:
        vec = vec * self.scale

    # TODO MPI
    result = ((self.O @ vec).T.conj() @ self.O).T.conj()
    result = result.astype(vec) + self.diag_shift * vec

    if self.scale is not None:
        result = result * self.scale

    # end PyTreeArray
    result = result.tree
    # +++++++++++++++++++++++++++++

    if ravel_or_unravel is not None:
        result = ravel_or_unravel(result)

    if reassemble is not None:
        result = reassemble(result)

    return result


@jax.jit
def _solve(
    self: QGTJacobianPyTreeT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:
    # Real-imaginary split RHS in R→R and R→C modes

    if self.mode != "holomorphic":
        y, reassemble = nkjax.tree_to_real(y)

    if self.flatten:
        if not hasattr(y, "ndim"):
            y, unravel = nkjax.tree_ravel(y)
    else:
        unravel = None

    # +++++++++++++++++++++++++++++
    y = PyTreeArray(y)
    if x0 is not None:
        x0 = PyTreeArray(x0)
    # begin PyTreeArray

    if self.scale is not None:
        y = y / self.scale
        if x0 is not None:
            x0 = x0 * self.scale

    # to pass the object LinearOperator itself down
    # but avoid rescaling, we pass down an object with
    # scale = None
    # mode=holomoprhic to disable splitting the complex part
    unscaled_self = self.replace(scale=None, _in_solve=True)

    # end PyTreeArray
    # TODO make it work with the solvers, like FrozenDict does; why doesnt pytree_node=False work???
    y = y.tree
    if x0 is not None:
        x0 = x0.tree
    out, info = solve_fun(unscaled_self, y, x0=x0)
    out = PyTreeArray(out)
    # begin PyTreeArray

    if self.scale is not None:
        out = out / self.scale

    # end PyTreeArray
    out = out.tree
    # +++++++++++++++++++++++++++++

    if unravel is not None:
        out = unravel(out)

    # Reassemble real-imaginary split as needed
    if self.mode != "holomorphic":
        out = reassemble(out)

    return out, info


@jax.jit
def _to_mat(self):
    # TODO MPI
    S = self.O.T.conj() @ self.O
    return S.add_diag_scalar(self.diag_shift)


@jax.jit
def _to_dense(self):
    # TODO MPI
    res = self._to_matrix().to_dense()
    return res


@jax.jit
def _matmul2(S, v):
    # TODO MPI
    return (S @ v).tree


@jax.jit
def _to_dense2(S):
    # TODO MPI
    return S.to_dense()
