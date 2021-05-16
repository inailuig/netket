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

    if flatten:
        params_flat, unravel = jax.flatten_util.ravel_pytree(vstate.parameters)

        def apply_fun_flat(variables, *args, **kwargs):
            variables = variables.copy({"params": unravel(variables["params"])})
            return vstate._apply_fun(variables, *args, **kwargs)

    O, scale = prepare_centered_oks(
        apply_fun_flat if flatten else vstate._apply_fun,
        params_flat if flatten else vstate.parameters,
        vstate.samples.reshape(-1, vstate.samples.shape[-1]),
        vstate.model_state,
        mode,
        rescale_shift,
    )

    O = PyTreeArray2(O)

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
        return _to_dense(self)

    def to_matrix(self) -> PyTree:
        # TODO diag_shift
        return _to_matrix(self)


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


@jax.jit
def _matmul(
    self: QGTJacobianPyTreeT, vec: Union[PyTree, Array]
) -> Union[PyTree, Array]:
    # Turn vector RHS into PyTree

    if self.flatten:
        do_ravel = False
        if not hasattr(vec, "ndim"):
            vec, unravel = nkjax.tree_ravel(self.params)
            do_unravel = True
        else:
            do_unravel = False
    else:
        do_unravel = False
        if hasattr(vec, "ndim"):
            _, unravel = nkjax.tree_ravel(self.params)
            vec = unravel(vec)
            do_ravel = True
        else:
            do_ravel = False

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = nkjax.tree_to_real(vec)

    if self.scale is not None:
        vec = jax.tree_multimap(jnp.multiply, vec, self.scale)

    if not isinstance(vec, PyTreeArrayT):
        vec = PyTreeArray(vec)

    result = nkjax.tree_cast(((self.O @ vec).H @ self.O).H, vec) + self.diag_shift * vec
    result = result.tree  # remove PyTreeArrayT wrapper for now

    if self.scale is not None:
        result = jax.tree_multimap(jnp.multiply, result, self.scale)

    # Reassemble real-imaginary split as needed
    if reassemble is not None:
        result = reassemble(result)

    # Ravel PyTree back into vector as needed
    if do_ravel:
        result, _ = nkjax.tree_ravel(result)
    if do_unravel:
        result = unravel(result)
    return result


@jax.jit
def _solve(
    self: QGTJacobianPyTreeT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:
    # Real-imaginary split RHS in R→R and R→C modes
    if self.mode != "holomorphic":
        y, reassemble = nkjax.tree_to_real(y)

    if self.scale is not None:
        y = jax.tree_multimap(jnp.divide, y, self.scale)
        if x0 is not None:
            x0 = jax.tree_multimap(jnp.multiply, x0, self.scale)

    # to pass the object LinearOperator itself down
    # but avoid rescaling, we pass down an object with
    # scale = None
    # mode=holomoprhic to disable splitting the complex part
    unscaled_self = self.replace(scale=None, _in_solve=True)

    out, info = solve_fun(unscaled_self, y, x0=x0)

    if self.scale is not None:
        out = jax.tree_multimap(jnp.divide, out, self.scale)

    # Reassemble real-imaginary split as needed
    if self.mode != "holomorphic":
        out = reassemble(out)

    return out, info


@jax.jit
def _to_matrix(self: QGTJacobianPyTreeT) -> PyTree:
    # TODO MPI
    return self.O.T.conj() @ self.O


@jax.jit
def _to_dense(self: QGTJacobianPyTreeT) -> jnp.ndarray:
    # TODO MPI
    return self.to_matrix().to_dense()
