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

from typing import Optional

import numpy as np
import jax.numpy as jnp
import math

from netket.graph import AbstractGraph, Graph
from netket.hilbert import Fock
from netket.utils.types import DType
from netket.utils.numbers import dtype as _dtype

from . import boson
from ._local_operator import LocalOperator
from ._hamiltonian import SpecialHamiltonian
from ._discrete_operator_jax import DiscreteJaxOperator


@partial(jax.jit, static_argnums=0)
def _bose_hubbard_kernel_jax(max_conn_size, edges, U, V, J, mu, n_max, x):
    i = edges[:, 0]
    j = edges[:, 1]
    n_i = x[..., i]
    n_j = x[..., j]

    mask0 = jnp.full(x.shape[:-1]+(1,), True)
    mels0 = 0
    xmels_0_is_nonzero = True
    if not isinstance(mu, StaticZero):
        xmels_0_is_nonzero=False
        mels0 -= mu * x.sum(axis=-1, keepdims=True)
    if not isinstance(U, StaticZero):
        xmels_0_is_nonzero=False
        mels0 += U * (x * (x - 1)).sum(axis=-1, keepdims=True)
    if not isinstance(V, StaticZero):
        xmels_0_is_nonzero=False
        mels0 += V * (n_i * n_j).sum(axis=-1)
    x_prime0 = x[..., None]
    if not xmels_0_is_nonzero:
        x_prime0 = x_prime0[..., :0, :]
        mask0 = mask0[..., :0]
        mels0 = jnp.zeros(x.shape[:-1]+(0,), dtype = J.dtype)
        
    # destroy on i create on j
    mask1 = (n_i > 0) * (n_j < n_max)
    mels1 = mask1*(-J * jnp.sqrt(n_i) * jnp.sqrt(n_j + 1))
    x_prime1 = x[..., None]*mask1[..., None]
    x_prime1 = x_prime1.at[:,:, i].add(-1)
    x_prime1 = x_prime1.at[:,:, j].add(+1)

    # destroy on j create on i
    mask2 = (n_j > 0) * (n_i < n_max)
    mels2 = mask2*(-J * jnp.sqrt(n_j) * jnp.sqrt(n_i + 1))
    x_prime2 = x[..., None]*mask2[..., None]
    x_prime2 = x_prime2.at[:,:, j].add(-1)
    x_prime2 = x_prime2.at[:,:, i].add(+1)

    if max_conn_size is None:
        xp = jnp.concatenate([mask0, mask1, mask2], axis=-2)
        mels = jnp.concatenate([mels0, mels1, mels2], axis=-1)
        return xp, mels, None
    else:
        # move the nonzeros to the beginning
        # we pad with 0 times vacuum, for fill value -1 below
        mask_all = jnp.concatenate([mask0, mask1, mask2, jnp.zeros_like(mask0)], axis=-1)
        mels_all = jnp.concatenate([mels0, mels1, mels2, jnp.zeros_like(mels0)], axis=-1)
        xp_all = jnp.concatenate([x_prime0, x_prime1, x_prime2, jnp.zeros_like(x_prime0)], axis=-2)

        index_nonzero = jnp.where(mask_all, size=max_conn_size, fill_value=-1)
        n_conn = mask_all.sum(axis=-1)
        return xp_all[index_nonzero], mels_all[index_nonzero], n_conn


class BoseHubbardJax(BoseHubbardBase, DiscreteJaxOperator):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    def __init__(
        self,
        hilbert: Fock,
        graph: AbstractGraph,
        U: float,
        V: float = 0.,
        J: float = 1.0,
        mu: float = 0.,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs a new BoseHubbard operator given a hilbert space, a graph
        specifying the connectivity and the interaction strength.
        The chemical potential and the density-density interaction strength
        can be specified as well.

        Args:
           hilbert: Hilbert space the operator acts on.
           U: The on-site interaction term.
           V: The strength of density-density interaction term.
           J: The hopping amplitude.
           mu: The chemical potential.
           dtype: The dtype of the matrix elements.

        Examples:
           Constructs a BoseHubbard operator for a 2D system.

           >>> import netket as nk
           >>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
           >>> hi = nk.hilbert.Fock(n_max=3, n_particles=6, N=g.n_nodes)
           >>> op = nk.operator.BoseHubbard(hi, U=4.0, graph=g)
           >>> print(op.hilbert.size)
           9
        """
        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space."

        assert isinstance(hilbert, Fock)
        super().__init__(hilbert)

        self._dtype = dtype

        if not isinstance(j, jax.Array) and (j == 0 or j is None):
            j = StaticZero()
        if not isinstance(mu, jax.Array) and (mu == 0 or mu is None):
            mu = StaticZero()


        V = jnp.array(V, dtype=dtype)
        J = jnp.array(J, dtype=dtype)
        if not isinstance(j, StaticZero):
            j = jnp.array(j, dtype=dtype)
        if not isinstance(mu, StaticZero):
            mu = jnp.array(mu, dtype=dtype)

        self._n_max = hilbert.n_max
        self._n_sites = hilbert.size
        self._edges = jnp.asarray(self.edges, dtype=jnp.int32)
        self._max_conn = 1 + self._edges.shape[0] * 2

        @property
        def max_conn_size(self) -> int:
            """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
            # 1 diagonal element + 2 for every coupling
            return self._max_conn

        def get_conn_padded(self, x):
            xp, mels, _ = self._get_conn_padded(x)
            return xp, mels

        def _get_conn_padded(self, x):
            return _bose_hubbard_kernel_jax(self._max_conn, self._edges, self._U, self._V, self._J, self._mu, self._n_max)

        def to_numba_operator(self) -> "BoseHubbard":  # noqa: F821
            """
            Returns the standard numba version of this operator, which is an
            instance of :class:`netket.experimental.operator.FermionOperator2nd`.
            """
            from ._bose_hubbard_numba import BoseHubbard
            return self.copy(cls=BoseHubbard)
