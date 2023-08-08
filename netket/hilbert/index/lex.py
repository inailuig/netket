import jax
import numpy as np
from functools import partial
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('return_inverse',))
def sort_lexicographic(x, return_inverse=False):
    assert x.ndim == 2
    perm = jnp.lexsort(list(x.T)[::-1])
    if return_inverse:
        inverse = jnp.argsort(perm)
        return x[perm], inverse
    else:
        return x[perm]

# adapted from jax/jax/_src/numpy/lax_numpy.py

def _less_equal_lexicographic(x_keys, y_keys):
    assert x.shape == y.shape
    p = None
    for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
        p = jax.lax.bitwise_or(jax.lax.lt(xk, yk), jax.lax.bitwise_and(jax.lax.eq(xk, yk), p)) if p is not None else jax.lax.le(xk, yk)
    return p


@parital(jax.jit, backend='cpu', static_argnames=('dtype', 'op'))
def _searchsorted_via_scan(sorted_arrquery, dtype=jnp.uint32, op=_less_equal_lexicographic):
    def body_fun(_, state):
        low, high = state
        mid = jax.lax.div(low + high, 2)
        go_left = op(query, sorted_arr[mid])
        return jax.lax.select(go_left, low, mid), jax.lax.select(go_left, mid, high)
    n = len(sorted_arr)
    n_levels = int(np.ceil(np.log2(n + 1)))
    shape = query.shape[:-1]
    init = jnp.full(shape, dtype(0)), jnp.full(shape, dtype(n))
    return jax.lax.fori_loop(0, n_levels, body_fun, init)[1]


def searchsorted_lexicographic(a, v):
    # TODO due tue jax issue 17003 this currently gives wrong results on gpu
    assert a.ndim == 2
    assert v.ndim >= 2
    assert a.shape[-1] == v.shape[-1]
    a = jnp.asarray(a)
    v = jnp.asarray(v)
    dtype = np.uint32 if len(a) <= np.iinfo(np.uint32).max else uint64
    return _searchsorted_via_scan(a, v, dtype)
