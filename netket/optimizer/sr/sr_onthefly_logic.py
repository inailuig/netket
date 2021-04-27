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
import jax.numpy as jnp
from functools import partial
from netket.stats import sum_inplace, subtract_mean
from netket.utils import n_nodes
import netket.jax as nkjax
from netket.jax import tree_conj, tree_dot, tree_cast, tree_axpy

# Stochastic Reconfiguration with jvp and vjp

# Here O (Oks) is the jacobian (derivatives w.r.t. params) of the vectorised (in x) log wavefunction (forward_fn) evaluated at all samples.
# instead of computing (and storing) the full jacobian matrix jvp and vjp are used to implement the matrix vector multiplications with it.
# Expectation values are then just the mean over the leading dimension.


def O_jvp(x, params, v, forward_fn):
    # TODO apply the transpose of sum_inplace (allreduce) to v here
    # in order to get correct transposition with MPI
    _, res = jax.jvp(lambda p: forward_fn(p, x), (params,), (v,))
    return res


def O_vjp(x, params, v, forward_fn):
    _, vjp_fun = jax.vjp(forward_fn, params, x)
    res, _ = vjp_fun(v)
    return jax.tree_map(sum_inplace, res)  # allreduce w/ MPI.SUM


def O_mean(samples, params, forward_fn):
    r"""
    compute \langle O \rangle
    i.e. the mean of the rows of the jacobian of forward_fn
    """

    # determine the output type of the forward pass
    dtype = jax.eval_shape(forward_fn, params, samples).dtype
    w = jnp.ones(samples.shape[0], dtype=dtype) * (1.0 / (samples.shape[0] * n_nodes))

    if not nkjax.tree_leaf_iscomplex(params) and nkjax.is_complex_dtype(dtype):
        # R->C
        return nkjax.vjp(forward_fn, params, samples)[1](w)[0]
    else:
        # R->R and C->C
        return O_vjp(samples, params, w, forward_fn)
    # TODO inhomogeneous


def OH_w(samples, params, w, forward_fn):
    r"""
    compute  O^H w
    (where ^H is the hermitian transpose)
    """

    # O^H w = (w^H O)^H
    # The transposition of the 1D arrays is omitted in the implementation:
    # (w^H O)^H -> (w* O)*

    # TODO The allreduce in O_vjp could be deferred until after the tree_cast
    # where the amount of data to be transferred would potentially be smaller
    res = tree_conj(O_vjp(samples, params, w.conjugate(), forward_fn))

    return tree_cast(res, params)


def Odagger_O_v(samples, params, v, forward_fn, *, center=False):
    r"""
    if center=False (default):
        compute \langle O^\dagger O \rangle v

    else (center=True):
        compute \langle O^\dagger \Delta O \rangle v
        where \Delta O = O - \langle O \rangle
    """

    # w is an array of size n_samples; each MPI rank has its own slice
    w = O_jvp(samples, params, v, forward_fn)
    # w /= n_samples (elementwise):
    w = w * (1.0 / (samples.shape[0] * n_nodes))

    if center:
        w = subtract_mean(w)  # w/ MPI

    return OH_w(samples, params, w, forward_fn)


Odagger_DeltaO_v = partial(Odagger_O_v, center=True)


def DeltaOdagger_DeltaO_v(samples, params, v, forward_fn):

    r"""
    compute \langle \Delta O^\dagger \Delta O \rangle v

    where \Delta O = O - \langle O \rangle
    """

    omean = O_mean(samples, params, forward_fn)

    def forward_fn_centered(params, x):
        return forward_fn(params, x) - tree_dot(params, omean)

    return Odagger_O_v(samples, params, v, forward_fn_centered)


# TODO block the computations (in the same way as done with MPI) if memory consumtion becomes an issue
def mat_vec(v, forward_fn, params, samples, diag_shift, centered=True):
    r"""
    compute (S + diag_shift) v

    where the elements of S are given by one of the following equivalent formulations:

    if centered=True (default): S_kl = \langle \Delta O_k^\dagger \Delta O_l \rangle
    if centered=False : S_kl = \langle O_k^\dagger \Delta O_l \rangle

    where \Delta O_k = O_k - \langle O_k \rangle
    and O_k (operator) is derivative of the log wavefunction w.r.t parameter k
    The expectation values are calculated as mean over the samples

    v: a pytree with the same structure as params
    forward_fn(params, x): a vectorised function returning the logarithm of the wavefunction for each configuration in x
    params: pytree of parameters with arrays as leaves
    samples: an array of samples (when using MPI each rank has its own slice of samples)
    diag_shift: a scalar diagonal shift
    """

    if centered:
        f = DeltaOdagger_DeltaO_v
    else:
        f = Odagger_DeltaO_v
    res = f(samples, params, v, forward_fn)
    # add diagonal shift:
    res = tree_axpy(diag_shift, v, res)  # res += diag_shift * v
    return res
