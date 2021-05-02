import jax
import jax.numpy as jnp

from jax import linear_util as lu
from jax.api_util import argnums_partial

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


# TODO in_axes a la vmap?
def scanmap(fun, scan_fun, argnums=0):
    # @wraps(f)
    def f_(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args)
        return scan_fun(lambda x: f_partial.call_wrapped(*x), dyn_args)

    return f_
