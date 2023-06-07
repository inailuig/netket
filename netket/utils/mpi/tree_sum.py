# transposable version of mpi_tree_map(mpi.mpi_sum_jax, ...)
# some parts are inspired by https://github.com/google/jax/issues/13298#issue-1453697688

import jax
from jax import custom_transpose
from . import mpi_tree_map, mpi_sum_jax

class _custom_transpose:
    def __init__(self, out_types, fun):
        from jax.custom_transpose import custom_transpose

        self.out_types = out_types
        self.fun = custom_transpose(fun)

    def __getattr__(self, name):
        return getattr(self.fun, name)

    def __call__(self, *args):
        return self.fun(self.out_types, *args)

def _mpi_tree_sum(_, x):
    token = jax.lax.create_token()
    res, token = mpi_tree_map(mpi_sum_jax, x, token=token)
    return res

def _mpi_tree_sum_transposed(_, x):
    return x

def mpi_tree_sum(x):
    m = _mpi_tree_sum
    m_T = _mpi_tree_sum_transposed

    out_types = jax.tree_map(lambda x: jax.core.get_aval(x).at_least_vspace(), m((), x))
    inp_types = jax.tree_map(lambda x: jax.core.get_aval(x).at_least_vspace(), x)

    m = _custom_transpose(out_types, m)
    m_T = _custom_transpose(inp_types, m_T)
    m.def_transpose(m_T)
    m_T.def_transpose(m)
    return m((), a)
