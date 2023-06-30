# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import List, Union, Optional

import numpy as np
from jax.tree_util import tree_map
import copy

from netket.utils.types import DType
from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._pauli_strings.base import _count_of_locations
from netket.hilbert.abstract_hilbert import AbstractHilbert
from netket.utils.numbers import is_scalar, dtype as _dtype
from netket.errors import concrete_or_error, NumbaOperatorGetConnDuringTracingError

from netket.experimental.hilbert import SpinOrbitalFermions

from ._fermion_operator_2nd_utils import (
    _convert_terms_to_spin_blocks,
    _collect_constants,
    _canonicalize_input,
    _check_hermitian,
    _herm_conj,
    _is_diag_term,
    _make_tuple_tree,
    _remove_dict_zeros,
    _verify_input,
    OperatorDict,
)


# TODO is this a reasonable cutoff?
def _isclose(a, b, cutoff=1e-6):  # pragma: no cover
    return np.abs(a - b) < cutoff


class FermionOperator2ndBase(DiscreteOperator):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        terms: Union[List[str], List[List[List[int]]]],
        weights: Optional[List[Union[float, complex]]] = None,
        constant: Union[float, complex] = 0.0,
        dtype: DType = None,
    ):
        r"""
        Constructs a fermion operator given the single terms (set of
        creation/annihilation operators) in second quantization formalism.

        This class can be initialized in the following form:
        `FermionOperator2nd(hilbert, terms, weights ...)`.
        The term contains pairs of `(idx, dagger)`, where `idx ∈ range(hilbert.size)`
        (it identifies an orbital) and `dagger` is a True/False flag determining if the
        operator is a creation or destruction operator.
        A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
        `((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
        represents :math:`\hat{a}_2`.
        To split up per spin, use the creation and annihilation operators to build the
        operator.

        Args:
            hilbert: hilbert of the resulting FermionOperator2nd object
            terms: single term operators (see
                example below)
            weights: corresponding coefficients of the single term operators
                (defaults to a list of 1)
            constant: constant contribution, corresponding to the
                identity operator * constant (default = 0)

        Returns:
            A FermionOperator2nd object.

        Example:
            Constructs the fermionic hamiltonian in :math:`2^{nd}` quantization
            :math:`(0.5-0.5j)*(a_0^\dagger a_1) + (0.5+0.5j)*(a_2^\dagger a_1)`.

            >>> import netket.experimental as nkx
            >>> terms, weights = (((0,1),(1,0)),((2,1),(1,0))), (0.5-0.5j,0.5+0.5j)
            >>> hi = nkx.hilbert.SpinOrbitalFermions(3)
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> terms = ("0^ 1", "2^ 1")
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> op.hilbert
            SpinOrbitalFermions(n_orbitals=3)
            >>> op.hilbert.size
            3

        """
        super().__init__(hilbert)

        # bring terms, weights into consistent form, autopromote dtypes if necessary
        _operators, _constant, dtype = _canonicalize_input(
            terms, weights, constant, dtype
        )
        _verify_input(hilbert, _operators, raise_error=True)
        self._dtype = dtype

        # we keep the input, in order to be able to add terms later
        self._operators = _operators
        self._constant = _constant

        self._initialized = False
        self._is_hermitian = None  # set when requested
        self._max_conn_size = None

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None
        self._max_conn_size = None

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        raise NotImplementedError

    @classmethod
    def from_openfermion(
        cls,
        hilbert: AbstractHilbert,
        of_fermion_operator=None,  # : "openfermion.ops.FermionOperator" type
        *,
        n_orbitals: Optional[int] = None,
        convert_spin_blocks: bool = False,
    ) -> "FermionOperator2ndBase":
        r"""
        Converts an openfermion FermionOperator into a netket FermionOperator2nd.

        The hilbert first argument can be dropped, see __init__ for details and default
        value.
        Warning: convention of openfermion.hamiltonians is different from ours: instead
        of strong spin components as subsequent hilbert state outputs (i.e. the 1/2 spin
        components of spin-orbit i are stored in locations (2*i, 2*i+1)), we concatenate
        blocks of definite spin (i.e. locations (i, n_orbitals+i)).

        Args:
            hilbert: (optional) hilbert of the resulting FermionOperator2nd object
            of_fermion_operator (openfermion.ops.FermionOperator):
                `FermionOperator object <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`_ .
                More information about those objects can be found in
                `OpenFermion's documentation <https://quantumai.google/reference/python/openfermion>`_
            n_orbitals: (optional) total number of orbitals in the system, default
                None means inferring it from the FermionOperator2nd. Argument is
                ignored when hilbert is given.
            convert_spin_blocks: whether or not we need to convert the FermionOperator
                to our convention. Only works if hilbert is provided and if it has
                spin != 0

        """
        from openfermion.ops import FermionOperator

        if hilbert is None:
            raise ValueError(
                "The first argument `from_openfermion` must either be an "
                "openfermion operator or an Hilbert space, followed by "
                "an openfermion operator"
            )

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, of_fermion_operator = None, hilbert

        if not isinstance(of_fermion_operator, FermionOperator):
            raise NotImplementedError()

        if convert_spin_blocks and hilbert is None:
            raise ValueError("if convert_spin_blocks, the hilbert must be specified")

        terms = list(of_fermion_operator.terms.keys())
        weights = list(of_fermion_operator.terms.values())
        terms, weights, constant = _collect_constants(terms, weights)

        if hilbert is not None:
            # no warning, just overwrite
            n_orbitals = hilbert.n_orbitals

            if convert_spin_blocks:
                if not hasattr(hilbert, "spin") or hilbert.spin is None:
                    raise ValueError(
                        f"cannot convert spin blocks for hilbert space {hilbert} without spin"
                    )
                n_spin = hilbert._n_spin_states
                terms = _convert_terms_to_spin_blocks(terms, n_orbitals, n_spin)
        if n_orbitals is None:
            # we always start counting from 0, so we only determine the maximum location
            n_orbitals = _count_of_locations(of_fermion_operator)
        if hilbert is None:
            hilbert = SpinOrbitalFermions(n_orbitals)  # no spin splitup assumed

        return cls(hilbert, terms, weights=weights, constant=constant)

    def __repr__(self):
        return (
            f"{type(self).__name__}(hilbert={self.hilbert}, "
            f"n_operators={len(self._operators)}, dtype={self.dtype})"
        )

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    def copy(self, *, dtype: Optional[DType] = None):
        if dtype is None:
            dtype = self.dtype
        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")
        op = type(self)(self.hilbert, [], [], constant=self._constant, dtype=dtype)
        # careful to make sure we propagate the correct dtype
        terms = copy.deepcopy(list(self._operators.keys()))
        weights = np.array(list(self._operators.values()), dtype=dtype)
        op._operators = dict(zip(terms, weights))
        return op

    def _remove_zeros(self):
        """Reduce the number of operators by removing unnecessary zeros"""
        op_dict = _remove_dict_zeros(self._operators)
        terms = list(op_dict.keys())
        weights = list(op_dict.values())
        op = type(self)(
            self.hilbert, terms, weights, constant=self._constant, dtype=self.dtype
        )
        return op

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        if self._max_conn_size is None:
            self._setup()
        return self._max_conn_size

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        if self._is_hermitian is None:  # only compute when needed, is expensive
            terms = list(self._operators.keys())
            weights = list(self._operators.values())
            self._is_hermitian = _check_hermitian(terms, weights)
        return self._is_hermitian

    def operator_string(self) -> str:
        """Return a readable string describing all the operator terms"""
        op_string = []
        if not _isclose(self._constant, 0.0):
            op_string.append(f"{self._constant} []")
        for term, weight in self._operators.items():
            s = []
            for idx, dag in term:
                dag_string = "^" if bool(dag) else ""
                s.append(f"{int(idx)}{dag_string}")
            s = " ".join(s)
            s = f"{weight} [{s}]"
            op_string.append(s)
        return " +\n".join(op_string)

    def _op__imatmul__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        if not self.hilbert == other.hilbert:
            raise ValueError(
                "Can only multiply identical hilbert spaces (got A@B, "
                f"A={self.hilbert}, B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )

        terms = []
        weights = []
        for t, w in self._operators.items():
            for to, wo in other._operators.items():
                terms.append(tuple(t) + tuple(to))
                weights.append(w * wo)
        if not _isclose(other._constant, 0.0):
            for t, w in self._operators.items():
                terms.append(tuple(t))
                weights.append(w * other._constant)
        if not _isclose(self._constant, 0.0):
            for t, w in other._operators.items():
                terms.append(tuple(t))
                weights.append(w * self._constant)
        constant = self._constant * other._constant

        self._operators = _remove_dict_zeros(dict(zip(terms, weights)))
        self._constant = constant
        self._reset_caches()
        return self

    def _op__matmul__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        dtype = np.promote_types(self.dtype, other.dtype)
        op = self.copy(dtype=dtype)
        return op._op__imatmul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        dtype = np.promote_types(self.dtype, _dtype(other))
        op = self.copy(dtype=dtype)
        return op.__iadd__(other)

    def __iadd__(self, other):
        if is_scalar(other):
            if not _isclose(other, 0.0):
                self._constant += other
            return self
        if not isinstance(other, type(self)):
            raise NotImplementedError(
                f"In-place addition not implemented for {type(self)} "
                f"and {type(other)}"
            )
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only add identical hilbert spaces (got A+B, A={self.hilbert}, "
                f"B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )
        for t, w in other._operators.items():
            if t in self._operators.keys():
                self._operators[t] += w
            else:
                self._operators[t] = w
        self._constant += other._constant
        self._operators = _remove_dict_zeros(self._operators)
        self._reset_caches()
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self.__mul__(np.array(-1, dtype=self.dtype))

    def __rmul__(self, scalar):
        return self * scalar

    def __imul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            self._op__imatmul__(scalar)
        if not np.can_cast(_dtype(scalar), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace scalar with dtype {type(scalar)} "
                f"to operator with dtype {self.dtype}"
            )
        scalar = np.array(scalar, dtype=self.dtype).item()
        _operators = tree_map(lambda x: x * scalar, self._operators)
        self._operators = _remove_dict_zeros(_operators)
        self._constant *= scalar
        self._reset_caches()
        return self

    def __mul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            return self._op__matmul__(scalar)
        dtype = np.promote_types(self.dtype, _dtype(scalar))
        op = self.copy(dtype=dtype)
        return op.__imul__(scalar)

    def conjugate(self, *, concrete=False):
        r"""Returns the complex conjugate of this operator."""

        terms = list(self._operators.keys())
        weights = list(self._operators.values())
        terms, weights = _herm_conj(terms, weights)  # changes also the terms
        terms = _make_tuple_tree(terms)

        new = type(self)(
            self.hilbert,
            [],
            [],
            constant=np.conjugate(self._constant),
            dtype=self.dtype,
        )
        new._operators = dict(zip(terms, weights))
        return new


def _pack_internals(operators: OperatorDict, dtype: DType):
    """
    Create the internal structures to compute the matrix elements
    Processes and adds a single term such that we can compute its matrix elements, in tuple format ((1,1), (2,0))
    """
    # properties of single-fermion operators, e.g. "0^"
    orb_idxs = []
    daggers = []
    # properties of multi-body operators, e.g. "0^ 1"
    weights = []
    # herm_term = []
    diag_idxs = []
    off_diag_idxs = []
    # below connect the second type to the first type (used to split single-fermion lists)
    term_split_idxs = []

    term_counter = 0
    single_op_counter = 0
    for term, weight in operators.items():
        if len(term) == 0:
            raise ValueError("terms cannot be size 0")
        if not all(len(t) == 2 for t in term):
            raise ValueError(f"terms must contain (i, dag) pairs, but received {term}")

        # fill some info about the term
        weights.append(weight)
        is_diag = _is_diag_term(term)
        if is_diag:
            diag_idxs.append(term_counter)
        else:
            off_diag_idxs.append(term_counter)

        # single-fermion operators
        for orb_idx, dagger in reversed(term):
            # orb_idxs: holds the hilbert index of the orbital
            orb_idxs.append(orb_idx)
            # daggers: stores whether operator is creator or annihilator
            daggers.append(bool(dagger))
            single_op_counter += 1

        term_split_idxs.append(single_op_counter)
        term_counter += 1

    orb_idxs = np.array(orb_idxs, dtype=np.intp)
    daggers = np.array(daggers, dtype=bool)
    weights = np.array(weights, dtype=dtype)
    # term_ends = np.array(term_ends, dtype=bool)
    # herm_term = np.array(herm_term, dtype=bool)
    diag_idxs = np.array(diag_idxs, dtype=np.intp)
    off_diag_idxs = np.array(off_diag_idxs, dtype=np.intp)
    term_split_idxs = np.array(term_split_idxs, dtype=np.intp)

    return orb_idxs, daggers, weights, diag_idxs, off_diag_idxs, term_split_idxs
