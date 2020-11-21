from functools import partial, singledispatch
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes

import jax
import jax.numpy as jnp

from jax import jit
from jax.scipy.sparse.linalg import cg
from jax._src.scipy.sparse.linalg import _cg_solve
from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
from netket.vmc_common import jax_shape_for_update
from netket.utils import n_nodes, mpi4jax_available

from ._sr_onthefly import delta_odagov

# TODO optionally pass vjp_fun
def _S_grad_mul_onthefly(forward_fn, params, samples, v, n_samp):
    # forward_fn is vectorised in samples
    # TODO where to do the n_samp division
    return delta_odagov(samples, params, v, forward_fn, factor=n_samp)

@singledispatch
def _S_grad_mul(oks, v, n_samp):
    r"""
    Computes y = 1/N * ( O^\dagger * O * v ) where v is a vector of
    length n_parameters, and O is a matrix (n_samples, n_parameters)
    """
    v_tilde = jnp.matmul(oks, v) / n_samp
    y = jnp.matmul(oks.conjugate().transpose(), v_tilde)
    return y

# TODO avoid workaround
# TODO where to jit
# TODO optionally pass vjp_fun
@_S_grad_mul.register
def _S_grad_mul_onthefly_test(args: tuple, v, n_samp):
    forward_fn, params, samples = args
    return _S_grad_mul_onthefly(forward_fn, params, samples, v, n_samp)


def _compose_result_cmplx(v, y, diag_shift):
    return v * diag_shift + y



def _compose_result_real(v, y, diag_shift):
    return (v * diag_shift + y).real


# Note: n_samp must be the total number of samples across all MPI processes!
# Note: _sum_inplace can only be jitted through if we are in single process.

def _matvec_cmplx(v, oks, n_samp, diag_shift):
    y = _S_grad_mul(oks, v, n_samp)
    return _compose_result_cmplx(v, _sum_inplace(y), diag_shift)


def _matvec_real(v, oks, n_samp, diag_shift):
    y = _S_grad_mul(oks, v, n_samp)
    return _compose_result_real(v, _sum_inplace(y), diag_shift)


@partial(jit, static_argnums=1)
def _jax_cg_solve(
    x0, mat_vec, oks, grad, diag_shift, n_samp, sparse_tol, sparse_maxiter
):
    r"""
    Solves the SR flow equation using the conjugate gradient method
    """

    _mat_vec = partial(mat_vec, oks=oks, diag_shift=diag_shift, n_samp=n_samp)

    out, _ = cg(_mat_vec, grad, x0=x0, tol=sparse_tol, maxiter=sparse_maxiter)

    return out
# cant use the other one cause wee need static_argnums for forward_fn
@partial(jit, static_argnums=(1,2))
def _jax_cg_solve_onthefly(
    x0, mat_vec, forward_fn, params, samples, grad, diag_shift, n_samp, sparse_tol, sparse_maxiter
):
    r"""
    Solves the SR flow equation using the conjugate gradient method
    """

    _mat_vec = partial(mat_vec, oks=(forward_fn, params, samples), diag_shift=diag_shift, n_samp=n_samp)

    # jax AD gives error with our _mat_vec
    # (when it tries to transpose it w/ jax.linear_transpose)
    # TODO debug it
    # out, _ = cg(_mat_vec, grad, x0=x0, tol=sparse_tol, maxiter=sparse_maxiter)
    # so just use the non differentiable _cg_solve for now:

    # adapted from jax.scipy.sparse.linalg.cg
    if sparse_maxiter is None:
        sparse_maxiter = 10 * grad.size  # copied from scipy

    out = _cg_solve(_mat_vec, grad, x0=x0, tol=sparse_tol, maxiter=sparse_maxiter)
    return out


@jit
def _shape_for_sr(grads, jac=None):
    r"""Reshapes grads and jax from tree like structures to arrays if jax_available

    Args:
        grads,jac: pytrees of jax arrays or numpy array

    Returns:
        A 1D array of gradients and a 2D array of the jacobian
    """

    grads = jnp.concatenate(tuple(fd.reshape(-1) for fd in tree_flatten(grads)[0]))
    if jac is None:
        return grads
    else:
        jac = jnp.concatenate(
            tuple(fd.reshape(len(fd), -1) for fd in tree_flatten(jac)[0]), -1
        )
        return grads, jac


@jit
def _flatten_grad_and_oks(grad, oks):
    grad, oks = _shape_for_sr(grad, oks) # size n_parameters
    oks -= jnp.mean(oks, axis=0) # size n_samples x n_parameters
    return grad, oks


class SR:
    r"""
    Performs stochastic reconfiguration (SR) updates.
    """

    def __init__(
        self,
        machine,
        lsq_solver=None,
        diag_shift=0.01,
        use_iterative=True,
        svd_threshold=None,
        sparse_tol=None,
        sparse_maxiter=None,
    ):

        if n_nodes > 1 and not mpi4jax_available:
            raise RuntimeError(
                """
                Cannot use Jax-Stochastic Reconfiguration with multiple MPI processes unless mpi4jax package is installed.
                Please run `pip install mpi4jax` and restart python.
                """
            )

        self._lsq_solver = lsq_solver
        self._diag_shift = diag_shift
        self._use_iterative = use_iterative
        self._has_complex_parameters = None
        self._machine = None

        # Quantities for sparse solver
        self.sparse_tol = 1.0e-5 if sparse_tol is None else sparse_tol
        self.sparse_maxiter = sparse_maxiter
        self._lsq_solver = lsq_solver
        self._x0 = None
        self._mat_vec = None
        self._init_solver()

        if not self._use_iterative:
            raise NotImplementedError("JaxSR supports only iterative solvers")

        if machine is not None:
            self.setup(machine)

    def _init_solver(self):
        lsq_solver = self._lsq_solver

        if lsq_solver in ["gmres", "cg", "minres", "jaxcg"]:
            self._use_iterative = True
        if lsq_solver in ["ColPivHouseholder", "QR", "SVD", "Cholesky"]:
            self._use_iterative = False

        if self._use_iterative:
            if lsq_solver is None or lsq_solver == "cg":
                self._lsq_solver = "cg"
            elif lsq_solver in ["gmres", "minres"]:
                raise Warning(
                    "Conjugate gradient is the only sparse solver currently implemented in Jax. Defaulting to cg"
                )
                self._lsq_solver = "cg"
            else:
                raise RuntimeError("Unknown sparse lsq_solver " + lsq_solver + ".")

    def setup(self, machine):
        r"""
        Sets up this Sr object to work with the selected machine.
        This mainly sets internal flags `has_complex_parameters` and the
        method used to flatten/unflatten the gradients.

        Args:
            machine: the machine
        """
        self._machine = machine
        self._has_complex_parameters = machine.has_complex_parameters

        if self._has_complex_parameters:
            self._mat_vec = _matvec_cmplx
        else:
            self._mat_vec = _matvec_real

        # captured here
        # TODO move it to jax machine??
        _, unravel_pytree  = ravel_pytree(self._machine.parameters)
        def forward_fn_flat(params_flat, inputs, **kwargs):
            par = unravel_pytree(params_flat)
            return self._machine.jax_forward_nj(par, inputs, **kwargs)
        self._forward_fn_flat = forward_fn_flat


    def compute_update(self, oks, grad, out=None, debug=True):
        r"""
        Solves the SR flow equation for the parameter update ẋ.

        The SR update is computed by solving the linear equation
           Sẋ = f
        where S is the covariance matrix of the partial derivatives
        O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
        gradient).

        Args:
            oks: A pytree of the jacobians ∂/∂x_i log Ψ(v_j)
            grad: A pytree of the forces f
            out: A pytree of the parameter updates that will be ignored
        """

        if self.has_complex_parameters is None or self._machine is None:
            raise ValueError("This SR object is not properly initialized.")

        grad, oks = _flatten_grad_and_oks(grad, oks)  # also subtracts the mean from ok

        n_samp = oks.shape[0] * n_nodes

        n_par = grad.shape[0]

        if self._x0 is None:
            if self.has_complex_parameters:
                self._x0 = jnp.zeros(n_par, dtype=jnp.complex128)
            else:
                self._x0 = jnp.zeros(n_par, dtype=jnp.float64)

        if debug:
            return (                self._x0,
                                    self._mat_vec,
                                    oks,
                                    grad,
                                    self._diag_shift,
                                    n_samp,
                                    self.sparse_tol,
                                    self.sparse_maxiter,
                                    )

        if self.has_complex_parameters:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = _jax_cg_solve(
                        self._x0,
                        self._mat_vec,
                        oks,
                        grad,
                        self._diag_shift,
                        n_samp,
                        self.sparse_tol,
                        self.sparse_maxiter,
                    )
                self._x0 = out
        else:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = _jax_cg_solve(
                        self._x0,
                        self._mat_vec,
                        oks,
                        grad.real,
                        self._diag_shift,
                        n_samp,
                        self.sparse_tol,
                        self.sparse_maxiter,
                    )
                self._x0 = out

        out = jax_shape_for_update(out, self._machine.parameters)

        return out

    def compute_update_onthefly(self, samples, grad, out=None, debug=False):
        r"""
        Solves the SR flow equation for the parameter update ẋ.

        The SR update is computed by solving the linear equation
           Sẋ = f
        where S is the covariance matrix of the partial derivatives
        O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
        gradient).

        Args:
            samples: An array of samples
            grad: A pytree of the forces f
            out: A pytree of the parameter updates that will be ignored
        """

        # TODO pass vjp_fun from gradient calculation
        # which can be reused for delta_odagov

        if self.has_complex_parameters is None or self._machine is None:
            raise ValueError("This SR object is not properly initialized.")

        grad_flat, unravel_pytree  = ravel_pytree(grad)
        params_flat, _  = ravel_pytree(self._machine.parameters)
        n_samp = samples.shape[0]
        n_par = params_flat.shape[0]

        if self._x0 is None:
            if self.has_complex_parameters:
                self._x0 = jnp.zeros(n_par, dtype=jnp.complex128)
            else:
                self._x0 = jnp.zeros(n_par, dtype=jnp.float64)

        if debug:
            return(
                    self._x0,
                    self._mat_vec,
                    self._forward_fn_flat,
                    params_flat,
                    samples,
                    grad_flat,
                    self._diag_shift,
                    n_samp,
                    self.sparse_tol,
                    self.sparse_maxiter)

        if self.has_complex_parameters:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = _jax_cg_solve_onthefly(
                        self._x0,
                        self._mat_vec,
                        self._forward_fn_flat,
                        params_flat,
                        samples,
                        grad_flat,
                        self._diag_shift,
                        n_samp,
                        self.sparse_tol,
                        self.sparse_maxiter,
                    )
                self._x0 = out
        else:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = _jax_cg_solve_onthefly(
                        self._x0,
                        self._mat_vec,
                        self._forward_fn_flat,
                        params_flat,
                        samples,
                        grad_flat.real,
                        self._diag_shift,
                        n_samp,
                        self.sparse_tol,
                        self.sparse_maxiter,
                    )
                self._x0 = out

        out = unravel_pytree(out)

        return out



    def __repr__(self):
        rep = "SR(solver="

        if self._use_iterative:
            rep += "iterative"

        rep += ", has_complex_parameters=" + str(self._has_complex_parameters) + ")"
        return rep

    def info(self, depth=0):
        indent = " " * 4 * depth
        rep = indent
        rep += "Stochastic reconfiguration method for "
        rep += (
            "complex-parameters" if self._has_complex_parameters else "real-parameters"
        )
        rep += " wavefunctions\n"

        rep += indent + "Solver: "

        if self._use_iterative:
            rep += "iterative (Conjugate Gradient)"

        rep += "\n"

        return rep

    @property
    def has_complex_parameters(self):
        return self._has_complex_parameters
