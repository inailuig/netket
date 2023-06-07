# transposable version of mpi_tree_map(mpi.mpi_sum_jax, ...)
import jax
from jax import custom_derivatives
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
    return custom_derivatives.linear_call(_mpi_tree_allreduce_sum, _mpi_tree_allreduce_sum_transposed, (), x)
