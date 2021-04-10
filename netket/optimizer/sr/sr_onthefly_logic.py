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
from functools import partial, wraps
from compose import compose

from netket.stats import sum_inplace, subtract_mean
from netket.utils import n_nodes

# Stochastic Reconfiguration with jvp and vjp

# Here O (Oks) is the jacobian (derivatives w.r.t. params) of the vectorised (in x) log wavefunction (forward_fn) evaluated at all samples.
# instead of computing (and storing) the full jacobian matrix jvp and vjp are used to implement the matrix vector multiplications with it.
# Expectation values are then just the mean over the leading dimension.


def tree_conj(t):
    r"""
    conjugate all complex leaves
    The real leaves are left untouched.

    t: pytree
    """
    return jax.tree_map(lambda x: jax.lax.conj(x) if jnp.iscomplexobj(x) else x, t)


def tree_dot(a, b):
    r"""
    compute the dot product of of the flattened arrays of a and b (without actually flattening)

    a, b: pytrees with the same treedef
    """
    res = jax.tree_util.tree_reduce(
        jax.numpy.add, jax.tree_map(jax.numpy.sum, jax.tree_multimap(jax.lax.mul, a, b))
    )
    # convert shape from () to (1,)
    # this is needed for automatic broadcasting to work also when transposed with linear_transpose
    return jnp.expand_dims(res, 0)


def tree_cast(x, target):
    r"""
    Cast each leaf of x to the dtype of the corresponding leaf in target.
    The imaginary part of complex leaves which are cast to real is discarded

    x: a pytree with arrays as leaves
    target: a pytree with the same treedef as x where only the dtypes of the leaves are accessed
    """
    # astype alone would also work, however that raises ComplexWarning when casting complex to real
    # therefore the real is taken first where needed
    return jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        x,
        target,
    )


def tree_axpy(a, x, y):
    r"""
    compute a * x + y

    a: scalar
    x, y: pytrees with the same treedef
    """
    return jax.tree_multimap(lambda x_, y_: a * x_ + y_, x, y)


# TODO check that the 1 is actually optimized away by jit
tree_xpy = partial(tree_axpy, 1)


def unbatch(x):
    return x.reshape((-1,) + x.shape[2:])


def batch(x, batchsize=None):
    # batchsize=None -> add just a dummy batch dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if batchsize is None:
        batchsize = n
    else:
        assert (n % batchsize) == 0
    return x.reshape((n // batchsize, batchsize) + x.shape[1:])


def scan_accum(f, x, op):
    res0 = f(jax.tree_map(lambda x: x[0], x))

    def f_(carry, x):
        return op(carry, f(x)), None

    res, _ = jax.lax.scan(
        f_, res0, jax.tree_map(lambda x: x[1:], x), unroll=1
    )  # make sure it uses the loop impl
    return res


def scan_append(f, x):
    def f_(carry, x):
        return None, f(x)

    _, res = jax.lax.scan(f_, None, x, unroll=1)  # make sure it uses the loop impl
    return res


# TODO add in_axes a la vmap to avoid the w workaround below
def scanmap(f, scan_fun, **kwargs):
    @wraps(f)
    def f_(x, *fargs, **fkwargs):
        return scan_fun(lambda x_: f(x_, *fargs, **fkwargs), x, **kwargs)

    return f_


# TODO apply the transpose of sum_inplace (allreduce) to the argument v
# in order to get correct transposition with MPI
# @partial(compose, unbatch)
@partial(
    lambda f: wraps(f)(compose(unbatch, f))
)  # w workaround part 1: flatten the batched w
@partial(scanmap, scan_fun=scan_append)
def O_jvp(x, params, v, forward_fn):
    _, res = jax.jvp(lambda p: forward_fn(p, x), (params,), (v,))
    return res


# a decorator which allreduces the result tree
def allreduce(f):
    # TODO move the tree_map to sum_inplace ?
    return wraps(f)(compose(partial(jax.tree_map, sum_inplace), f))


def w_workaround_3(f):
    def _f(x, params, w, forward_fn):
        w = batch(w, x.shape[1])
        xw = (x, w)
        return f(xw, params, forward_fn)

    return _f


def w_workaround_2(f):
    def _f(xw, params, forward_fn):
        x, w = xw
        return f(x, params, w, forward_fn)

    return _f


def _O_vjp(x, params, w, forward_fn):
    _, vjp_fun = jax.vjp(forward_fn, params, x)
    res, _ = vjp_fun(w)
    return res


@allreduce
@w_workaround_3  # put x and w in a single arg and batch w
@partial(scanmap, scan_fun=scan_accum, op=tree_xpy)
@w_workaround_2  # split xw arg into x and w
def O_vjp(*args, **kwargs):
    return _O_vjp(*args, **kwargs)


@allreduce
@partial(scanmap, scan_fun=scan_accum, op=tree_xpy)
def O_vjp2(*args, **kwargs):
    return _O_vjp(*args, **kwargs)


def O_mean(samples, params, forward_fn):
    r"""
    compute \langle O \rangle
    i.e. the mean of the rows of the jacobian of forward_fn
    """

    # determine the output type of the forward pass
    dtype = jax.eval_shape(forward_fn, params, samples[0]).dtype
    # same v for every batch
    v = jnp.ones(samples.shape[1], dtype=dtype) * (
        1.0 / (samples.shape[0] * samples.shape[1] * n_nodes)
    )
    return O_vjp2(samples, params, v, forward_fn)


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
    w = w * (1.0 / (samples.shape[0] * samples.shape[1] * n_nodes))

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


def mat_vec_batched(v, forward_fn, params, samples, diag_shift, centered=True):
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
    samples: an array of batched samples (when using MPI each rank has its own slice of batched samples)
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


# TODO remove once its no longer needed used anywhere
def mat_vec(v, forward_fn, params, samples, diag_shift, centered=True):
    return mat_vec_batched(
        v, forward_fn, params, jnp.expand_dims(samples, 0), diag_shift, centered=True
    )
