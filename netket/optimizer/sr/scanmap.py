import jax
import jax.numpy as jnp
from functools import partial, wraps
from compose import compose

from netket.jax import tree_conj, tree_dot, tree_cast, tree_axpy

from .batch_utils import *


# TODO check that the 1 is actually optimized away by jit
tree_xpy = partial(tree_axpy, 1)


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


# TODO add multiple in_axes a la vmap to avoid the w workaround below


def scanmap(f, scan_fun, argnum=0):
    @wraps(f)
    def f_(*fargs, **fkwargs):
        x = fargs[argnum]
        return scan_fun(
            lambda x_: f(*fargs[:argnum], x_, *fargs[argnum + 1 :], **fkwargs), x
        )

    return f_


# w workaround part 1: flatten the batched w
def w_workaround_1(f):
    return wraps(f)(compose(unbatch, f))


# split xw arg into x and w
def w_workaround_2(f):
    def _f(forward_fn, params, xw):
        x, w = xw
        return f(forward_fn, params, x, w)

    return _f


# put x and w in a single arg
def w_workaround_3(f):
    def _f(forward_fn, params, x, w):
        xw = (x, w)
        return f(forward_fn, params, xw)

    return _f


# batch w
def w_workaround_4(f):
    def _f(forward_fn, params, x, w):
        w = batch(w, x.shape[1])
        return f(forward_fn, params, x, w)

    return _f
