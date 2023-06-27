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
from jax.tree_util import Partial

class IsingJax(JaxOperator): # TODO inherit from sth else

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        h: float,
        J: float = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs the Ising Operator from an hilbert space and a
        graph specifying the connectivity.

        Args:
            hilbert: Hilbert space the operator acts on.
            h: The strength of the transverse field.
            J: The strength of the coupling. Default is 1.0.
            dtype: The dtype of the matrix elements.

        Examples:
            Constructs an ``Ising`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
            >>> op = nk.operator.IsingJax(h=1.321, hilbert=hi, J=0.5, graph=g)
            >>> print(op)
            IsingJax(J=0.5, h=1.321; dim=20)
        """
        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space"

        super().__init__(hilbert)

        if dtype is None:
            dtype = jnp.promote_types(_dtype(h), _dtype(J))
        self._dtype = dtype

        self._h = np.array(h, dtype=dtype)
        self._J = np.array(J, dtype=dtype)
        self._edges = np.asarray(
            [[u, v] for u, v in graph.edges()],
            dtype=np.int32,
        )

        if self.h == 0:
            self._flip = None
        else:
            self._flip = jnp.eye(
                self.max_conn_size, self.hilbert.size, k=-1, dtype=bool
            )

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "IsingJax only supports Hamiltonians with two local states"
            )
        self._hi_local_states = tuple(self.hilbert.local_states)

    def n_conn(self, x):
        return _ising_n_conn_jax(x, self._edges, self.h, self.J)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self.hilbert.size + 1

    @property
    def h(self) -> float:
        """The magnitude of the transverse field"""
        if self._h is None:
            return 0
        return self._h

    @property
    def J(self) -> float:
        """The magnitude of the hopping"""
        return self._J

    @property
    def edges(self) -> jnp.ndarray:
        return self._edges

    @property
    def is_hermitian(self) -> bool:
        return True

    @property
    def dtype(self) -> DType:
        return self._dtype

    def get_get_conn_padded_closure(self):
        return Partial(_ising_kernel_jax, self._edges, self._flip, self._h, self._J, self._hi_local_states)


@partial(jax.vmap, in_axes=(0, None, None, None, None))
def _ising_mels_jax(x, edges, h, J, h_is_0):
    if h_is_0:
        max_conn_size = 1
    else:
        max_conn_size = x.size + 1

    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    mels = jnp.empty((max_conn_size,), dtype=J.dtype)
    mels = mels.at[0].set(J * (2 * same_spins - 1).sum())
    if not h_is_0:
        mels = mels.at[1:].set(-h)
    return mels


def _flip_if(cond, x, local_states):
    # TODO here we could special-case for qubit / ising
    # by taking -x / 1 - x
    # i.e
    # if local_states[0] + local_states[1] == 0:
    #     return jnp.where(cond, -x, x)
    # elif local_states[0] == 0:
    #     return jnp.where(cond, local_states[1] - x, x)
    # elif local_states[1] == 0:
    #     return jnp.where(cond, local_states[0] - x, x)
    # else:
    #     ...
    was_state_0 = x == local_states[0]
    state_0 = jnp.asarray(local_states[0], dtype=x.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, state_0, state_1)


@partial(jax.vmap, in_axes=(0, None, None))
def _ising_conn_states_jax(x, flip, local_states):
    return _flip_if(flip, x, local_states)


@partial(jax.jit, inline=True)
def _ising_kernel_jax(edges, flip, h, J, local_states, x):
    h_is_0 = flip is None
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, x.shape[-1]))

    mels = _ising_mels_jax(x, edges, h, J, h_is_0)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    if h_is_0:
        x_prime = jnp.expand_dims(x, axis=1)
    else:
        x_prime = _ising_conn_states_jax(x, flip, local_states)
    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    return x_prime, mels


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def _ising_n_conn_jax(x, edges, h, J):
    n_sites = x.size
    n_conn_X = jnp.asarray(h != 0, dtype=jnp.int32) * n_sites
    same_spins = x[edges[:, 0]] == x[edges[:, 1]]
    # TODO duplicated with _ising_mels_jax
    mels_ZZ = J * (2 * same_spins - 1).sum()
    n_conn_ZZ = mels_ZZ != 0
    return n_conn_X + n_conn_ZZ
