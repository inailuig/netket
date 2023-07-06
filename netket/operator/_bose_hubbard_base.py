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
from numba import jit

import numpy as np
import jax.numpy as jnp
import math

from netket.graph import AbstractGraph, Graph
from netket.hilbert import Fock
from netket.utils.types import DType
from netket.utils.numbers import dtype as _dtype
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from . import boson
from ._local_operator import LocalOperator
from ._hamiltonian import SpecialHamiltonian


class BoseHubbardBase(SpecialHamiltonian):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    @property
    def is_hermitian(self):
        return True

    @property
    def dtype(self):
        return self._dtype

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def U(self):
        """The strength of on-site interaction term."""
        return self._U

    @property
    def V(self):
        """The strength of density-density interaction term."""
        return self._V

    @property
    def J(self):
        """The hopping amplitude."""
        return self._J

    @property
    def mu(self):
        """The chemical potential."""
        return self._mu

    def copy(self, cls=None):
        if cls is None:
            cls = type(self)
        graph = Graph(edges=[list(edge) for edge in self.edges])
        return cls(
            hilbert=self.hilbert,
            graph=graph,
            J=self.J,
            U=self.U,
            V=self.V,
            mu=self.mu,
            dtype=self.dtype,
        )

    def to_local_operator(self):
        # The hamiltonian
        ha = LocalOperator(self.hilbert, dtype=self.dtype)

        if self.U != 0 or self.mu != 0:
            for i in range(self.hilbert.size):
                n_i = boson.number(self.hilbert, i)
                ha += (self.U / 2) * n_i * (n_i - 1) - self.mu * n_i

        if self.J != 0:
            for (i, j) in self.edges:
                ha += self.V * (
                    boson.number(self.hilbert, i) * boson.number(self.hilbert, j)
                )
                ha -= self.J * (
                    boson.destroy(self.hilbert, i) * boson.create(self.hilbert, j)
                    + boson.create(self.hilbert, i) * boson.destroy(self.hilbert, j)
                )

        return ha

    def _iadd_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._mu += other.mu
        self._U += other.U
        self._J += other.J
        self._V += other.V

    def _isub_same_hamiltonian(self, other):
        if self.hilbert != other.hilbert:
            raise NotImplementedError(
                "Cannot add hamiltonians on different hilbert spaces"
            )

        self._mu -= other.mu
        self._U -= other.U
        self._J -= other.J
        self._V -= other.V
