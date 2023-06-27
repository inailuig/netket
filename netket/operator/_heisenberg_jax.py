import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ._jax import JaxOperator
from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert
from netket.utils.numbers import dtype as _dtype
from netket.utils.types import DType
from typing import Optional

class HeisenbergJax(JaxOperator): # TODO inherit from sth else
    pass # TODO


@partial(jax.jit, inline=True)
def _heisenberg_kernel_jax(x, edges, Jz=1, Jxy=1, weights=1):
    n_sites = x.shape[-1]
    n_conn = edges.shape[0] + 1
    i = edges[:, 0]
    j = edges[:, 1]
    x_i = x[..., i]
    x_j = x[..., j]
    mels = jnp.zeros((x.shape[0], n_conn))  # TODO dtype?
    mels = mels.at[..., 0].set(Jz * (weights * x_i * x_j).sum(axis=-1))
    mels = mels.at[..., 1:].set(2 * Jxy * weights * (x_i != x_j))
    x_prime = jax.lax.broadcast_in_dim(x, (x.shape[0], n_conn, n_sites), (0, 2))
    x_prime = x_prime.at[..., jnp.arange(1, x_prime.shape[1]), i].set(x_j)
    x_prime = x_prime.at[..., jnp.arange(1, x_prime.shape[1]), j].set(x_i)
    return x_prime, mels


# def get_heisenberg_kernel_jax(edges, Jz=1, sign_rule=True):
#     Jxy = -1 if sign_rule else 1
#     return Partial(_heisenberg_kernel_jax, edges=edges, Jz=Jz, Jxy=Jxy)
