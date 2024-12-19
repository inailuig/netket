from functools import partial, wraps

import numpy as np
import sparse
import itertools

import jax
import jax.numpy as jnp

from flax import struct

from netket.jax import reduce_xor, COOTensor
from netket.operator import DiscreteJaxOperator
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import PyTree

from ._fermion_operator_2nd_jax import FermionOperator2ndJax
from ._normal_order_utils import to_normal_order
from ._pyscf_utils import TV_from_pyscf_molecule

from netket.experimental.operator._pyscf_utils import to_desc_order_sparse

def _prepare_data_helper(sites_destr, sites_create, weights, n_orbitals, sparse_=True):
    # we encode sites_create==sites_destr by passing sites_create=None
    is_diagonal = sites_create is None

    n_terms, half_n_ops = sites_destr.shape
    assert weights.shape == (n_terms,)
    if not is_diagonal:
        assert sites_destr.shape == sites_create.shape

    if half_n_ops == 0:  # constant
        assert n_terms == 1
        index_array = jnp.zeros((), dtype=np.int32)
        create_array = jnp.zeros((1, 1, 0), dtype=np.int32)
        weight_array = jnp.array(weights, dtype=weights.dtype)
    elif is_diagonal:
        assert sites_destr.max() < n_orbitals
        index_array = None
        create_array = None
        # use sparse.COO to sort since COOTensor expects sorted indices
        # TODO do it inside COOTensor
        tmp = sparse.COO(sites_destr.T, weights, (n_orbitals,) * (half_n_ops))
        weight_array = COOTensor(
            jnp.asarray(tmp.coords.T),
            jnp.asarray(tmp.data),
            (n_orbitals,) * (half_n_ops),
        )
        if not sparse_:
            weight_array = weight_array.todense()
    else:
        assert sites_destr.max() < n_orbitals
        assert sites_create.max() < n_orbitals

        ### simple, inefficient version
        # destr_unique = np.unique(sites_destr, axis=0)
        # nper = np.zeros(len(destr_unique), dtype=int)
        # for i, d in enumerate(destr_unique):
        #     nper[i] = (d[None] == sites_destr).all(axis=-1).sum()
        ###
        A = sparse.COO(np.concatenate([sites_destr, sites_create], axis=1).T, weights)
        axes_create = tuple(range(A.ndim // 2, A.ndim))
        n_destr = (A != 0).sum(axes_create)
        destr_unique = n_destr.coords.T
        nper = n_destr.data
        ###

        # we pad with zeros, so we take create_array and weight_array of size nunique+1
        # (where the 0th element is the padding)
        # and put zeros in the index_array, for terms which dont exist

        ### simple, inefficient version
        # nmax = int(nper.max())
        # nunique = len(destr_unique)
        # create_array = jnp.zeros((1 + nunique, nmax, half_n_ops), dtype=np.int32)
        # weight_array = jnp.zeros((1 + nunique, nmax), dtype=weights.dtype)
        # for i, d in enumerate(destr_unique):
        #     mask = (sites_destr == d).all(axis=-1)
        #     weight_array = weight_array.at[i + 1, : nper[i]].set(weights[mask])
        #     create_array = create_array.at[i + 1, : nper[i]].set(sites_create[mask])
        ###
        # select only nonzero destr rows
        B = A[tuple(destr_unique.T)]
        # create an arange for every row
        row_ind, *create_indices = B.coords
        row_start = np.where(np.diff(row_ind, prepend=-1))
        a = np.arange(B.nnz)
        compressed_col_ind = a - np.repeat(a[row_start], nper)
        # +1 because of the padding
        new_coords = np.vstack([row_ind + 1, compressed_col_ind])
        weight_array = jnp.asarray(sparse.COO(new_coords, B.data).todense())
        create_array = jnp.concatenate(
            [sparse.COO(new_coords, c).todense()[..., None] for c in create_indices],
            axis=-1,
        )
        ###
        # destr_unique should be already sorted at this point (in np.unique / sparse.COO)
        index_array = COOTensor(
            jnp.asarray(destr_unique),
            jnp.arange(1, len(destr_unique) + 1),
            (n_orbitals,) * (half_n_ops),
        )
        if not sparse_:
            index_array = index_array.todense()

    return index_array, create_array, weight_array


def prepare_data_diagonal(sites_destr, weights, n_orbitals, **kwargs):
    """

    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax, for the diagonal part of the operator
    of strings of a fixed length

    Assume we are given a sequence of equal-length normal ordered strings \sum_i w_i c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{a_i1} ... c_{a_iN}
    with 2N fermionic operators, with larger indices to the left: a_i1 >=...>=a_iN

    Please refer to the docstring of prepare_data for a more complete explanation of the storage format.

    We can treat the diagonal opeators separately to the non-diagonal ones in a more efficient way:
        index_array: None
        create_array: None
        weight_array: shape (n,)*N  is indexed directly with the list of destruction operators (given by b above)

    Args:
        sites_destr: a matrix containing the indices of c^dagger/c for every string [[a_i1, ..., a_iN]] ()
        weights: array of the corresponding weights [w_i]
        n_orbitals: number of orbitals n
        sparse_: whether to store weight_array in dense or sparse
    Returns:
        A tuple (index_array, create_array, weight_array) as defined above

    """
    return _prepare_data_helper(sites_destr, None, weights, n_orbitals, **kwargs)


def prepare_data(sites, weights, n_orbitals, **kwargs):
    """
    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax
    of strings of a fixed length

    It is given by a 3-tuple for every length N of string in normal ordering (containg 2N fermionic operators),

    Assume we are given a sequence of equal-length normal ordered strings \sum_i w_i c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{b_i1} ... c_{b_iN}
    with 2N fermionic operators, with larger indices to the left: a_i1 >=...>=a_iN, b_i1 >=...>=b_iN

    Here a \in {1..n} contains the indices of the c^\dagger, b the indices of the c operators and w the corresponding weight of every string.
    where n is the number of orbitals.

    The 3-tuple for these operators is given by
        index_array: shape (n,)*N+(n_max,) contains list of integer indices for all the strings for a given list of destruction operators (given by b above)
                     can be stored either in a dense, or sparse format; is padded to the maximum number of operators n_max for a given sequence of destruction ops (b)
                     Since the operators are assumed to be in normal order (larger sites to the left) only the lower triangular part is used.
        create_array: shape (n_ops+1,)+(N,) for every index from index_array contains creation operators of the corresponding term (given by a above)
        weight_array: shape (n_ops+1,)+(1,) for every index from index_array contains the weight of the corresponding term
                      The 0 weight for the padding is stored as the last element.

    Then for a given basis state |b_1,...,b_m> (where b_j indicates the occupied orbitals, for a fixed number of electrons m) we find all the connected elements
    by taking all m choose N combinations of occupied orbitals to be destroyed as index for index_array, excluding the strings which try to destroy an empty orbital.


    Args:
        sites: a matrix containing the indices of c^dagger and c for every string [[a_i1, ..., a_iN, b_i1, ..., b_iN]]
        weights: array of the corresponding weights [w_i]
        n_orbitals: number of orbitals n
        sparse_: wether to store index_array in dense or sparse
    Returns:
        A tuple (index_array, create_array, weight_array) as defined above

    """

    # sites is an array (n_terms, n_ops) containing the sites
    # of terms in normal order (daggers to left, desc order)
    #
    # weights is of shape (n_terms,)
    n_terms, n_ops = sites.shape
    assert n_ops % 2 == 0
    sites_destr = sites[:, : n_ops // 2]
    sites_create = sites[:, n_ops // 2 :]
    return _prepare_data_helper(sites_destr, sites_create, weights, n_orbitals, **kwargs)


def split_diag_offdiag(sites, weights):
    n_terms, n_ops = sites.shape
    assert weights.shape == (n_terms,)
    assert n_ops % 2 == 0

    idestr = sites[:, : (n_ops // 2)]
    icreate = sites[:, (n_ops // 2) :]

    is_diag = (idestr == icreate).all(axis=-1)
    diag_sites = idestr[is_diag]
    diag_weights = weights[is_diag]
    offdiag_sites = sites[~is_diag]
    offdiag_weights = weights[~is_diag]
    return (diag_sites, diag_weights), (offdiag_sites, offdiag_weights)


def  prepare_operator_data_from_coords_data_dict(
coords_data_dict, n_orbitals, **kwargs
):
"""
    Prepare the custom sparse internal data for ParticleNumberConservingFermioperator2ndJax

    of a string of operators \sum_N \sum_i w_i^{(N)} c_{a_i1^{(N)}}^\dagger ... c_{a_iN^{(N)}}^\dagger c_{b_i1^{(N)}} ... c_{b_iN^{(N)}}
    in descending order a_i1 >=...>=a_iN, b_i1 >=...>=b_iN.

    Please refer to the docstring of prepare_data, prepare_data_diagonal for a more complete explanation of the storage format.

    Args:
        coords_data_dict: A dictionary {N: (sites, weights)}
            where for every length N
                sites is a matrix containing the stacked indices [[a_i1^{(N)}, ... a_iN^{(N)}, b_i1^{(N)}, ..., b_iN^{(N)}]]
                of the c^\dagger and c
                weights is a vector containing the corresponding weights [w_i^{(N)}]
        n_orbitals: number of orbitals
    Returns:
        A dictionary {'diag': { N:  (None, None, weight_array)}, 'offdiag': { N : (index_array, create_array, weight_array)}}
        containing the sparse representation for every lenght N of strings c_{a_i1}^\dagger ... c_{a_iN}^\dagger c_{b_i1} ... c_{b_iN}
"""
    data_offdiag = {}
    data_diag = {}
    for k, v in coords_data_dict.items():
        sw_diag, sw_offdiag = split_diag_offdiag(*v)
        if len(sw_diag[-1]) > 0:
            data_diag[k] = prepare_data_diagonal(*sw_diag, n_orbitals, **kwargs)
        if len(sw_offdiag[-1]) > 0:
            data_offdiag[k] = prepare_data(*sw_offdiag, n_orbitals, **kwargs)
    data = {'diag': data_diag, 'offdiag':data_offdiag}
    return data


def _comb(kl, n):
    """
    compute all combinations of n elements from kl
    """
    if len(kl) < n:
        return jnp.zeros((n, 0), dtype=kl.dtype)
    c = list(itertools.combinations(np.arange(len(kl)), n))
    return kl[np.array(c, dtype=kl.dtype).T[::-1]]


def _jw_kernel(k_destroy, l_create, x):
    # destroy
    xd = jax.vmap(lambda i: x.at[i].set(0))(k_destroy.T)
    # create
    xp = jax.vmap(jax.vmap(lambda x, i: x.at[i].set(1), in_axes=(None, 0)))(
        xd, l_create
    )

    m = jnp.arange(x.shape[-1], dtype=k_destroy.dtype)

    # we apply the destruction operators in descending order,
    # the jordan-wigner sign of an operator does not depend on sites larger than it, therefore,
    # given it is in normal order, we can compute it all in terms of the initial state.
    # (sum the axis which is the one of the indices we destroy/create (size number of operators//2))
    jw_mask_destroy = reduce_xor(k_destroy[..., None] > m, axes=0)

    # same for when we create again, except then have to apply it to the state where we already destroyed
    jw_mask_create = reduce_xor(l_create[..., None] > m, axes=2)

    create_was_empty = jax.vmap(jax.vmap(lambda x, i: ~x[i].any(), in_axes=(None, 0)))(
        xd, l_create
    )

    sgn_destroy = reduce_xor(jw_mask_destroy * x[None], axes=-1)
    sgn_create = reduce_xor(jw_mask_create * xd[:, None], axes=-1)
    sgn = sgn_create + sgn_destroy[:, None]
    sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
    sign = 1 - 2 * sgn.astype(np.int8)

    return xp, sign, create_was_empty


@partial(jax.jit, static_argnums=0)
@partial(jnp.vectorize, signature="(n)->(m,n),(m)", excluded=(0, 2, 3, 4))
def _get_conn_padded(n_fermions, x, index_array, create_array, weight_array):
    # if create_array=None is passed we assume it's diagonal: index_array==create_array
    assert x.ndim == 1
    if index_array is not None:
        half_n_ops = index_array.ndim
    else:  # diagonal
        half_n_ops = weight_array.ndim

    if half_n_ops == 0:  # constant
        xp = x[None, :]
        mels = weight_array.reshape(xp.shape[:-1])
    else:
        dtype = x.dtype

        (l_occupied,) = jnp.where(x, size=n_fermions)
        k_destroy = _comb(l_occupied, half_n_ops)

        if index_array is None:  # diagonal
            weight = weight_array[tuple(k_destroy)]
            xp = x[None, :]
            # we first destroy in desc order, then create
            # sites not acted on cancel by the create/destroy pair of the same site,
            # so we can assume they are not there.
            # When we create all smaller sites acted on are 0,
            # therefore the jw sign is determined just from the signs from destroy.
            # Then it' is easy to see that only every other site counts (the rest cancel),
            # and the sign is given by (+1 if there is an even number of other sites, -1 if odd)
            # sign = [+,+,-,-,+,+,-,-,+,+,-,-,...][half_n_ops]
            sgn = (half_n_ops // 2) % 2
            sign = 1 - 2 * sgn
            mels = sign * weight.sum()[None]
        else:
            ind = index_array[tuple(k_destroy)]
            weight = weight_array[ind]
            l_create = create_array[ind]

            xp, sign, create_was_empty = _jw_kernel(k_destroy, l_create, x)
            mels = weight * sign * create_was_empty

            # make sure we don't return states w/ wrong number of electrons
            # because of the padding we check if the mel is 0
            # xp = jnp.where(create_was_empty[..., None], xp, x[..., None, None, :])
            xp = jnp.where((mels == 0)[:, :, None], x[None, None, :], xp)

            xp = jax.lax.collapse(xp, 0, xp.ndim - 1).astype(dtype)
            mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp, mels


def _to_fermiop_helper(index_array, create_array, weight_array):
    if index_array is None:  # diagonal
        if weight_array.ndim == 0:  # const
            return np.array([()], dtype=np.int32), np.array(weight_array)
        else:
            if not isinstance(weight_array, COOTensor):
                weight_array = COOTensor.fromdense(weight_array)
            destr = np.array(weight_array.coords.T)
            weights = np.array(weight_array.data)
            sites = np.concatenate([destr, destr], axis=-1)
    else:
        if index_array.ndim == 0:  # const
            return np.array([()], dtype=np.int32), np.array(weight_array)
        if not isinstance(index_array, COOTensor):
            index_array = COOTensor.fromdense(index_array)
        ind = np.array(index_array.data)
        destr = np.array(index_array.coords.T[:, None, :])
        create = create_array[ind]
        destr = np.broadcast_to(destr, create.shape)
        weights = weight_array[ind]
        sites = np.concatenate([destr, create], axis=-1)

    # flatten
    weights = weights.reshape(-1)
    sites = sites.reshape(-1, sites.shape[-1])

    daggers = np.zeros_like(sites)
    daggers[:, : daggers.shape[1] // 2] = 1
    terms = np.concatenate([sites[..., None], daggers[..., None]], axis=-1)

    return terms, weights

# TODO merge this with fermionoperator2nd prepare_terms_list
def _fermiop_terms_to_sites_daggers_weights(terms, weights):
    """
    helper function to turn the python dictionary of FermionOperator2nd/FermionOperator2ndJax


    Args:
        terms: terms as specified in FermionOperator2nd/FermionOperator2ndJax
        weights: a list of weights 
    Returns:
        a dictionary {k: (sites, daggers, weights)}
        where for every set of operators of length k
            sites: (n_terms, k) matrix containing the indices of the c/c^\dagger operators
            daggers: (n_terms,k), matrix storing c/c^\dagger as 0/1
            weights: (n_terms,) vector of corresponding weights
    """
    out = {}
    for t, w in zip(terms, weights):
        if len(t) == 0:  # constant
            out[0] = np.zeros((1, 0), dtype=np.int32), np.zeros((1, 0), dtype=np.int8), np.array([w])
        else:
            sites, daggers = np.array(t).T
            l = len(daggers)
            assert l % 2 == 0
            assert 2 * daggers.sum() == l
            tl, dl, wl= out.get(l, ([], [], []))
            out[l] = tl + [sites,], dl + [daggers,], wl + [w,]  # fmt: skip
    return {k: (jnp.array(v[0], dtype=np.int32), jnp.array(v[1], dtype=np.int8), jnp.array(v[2])) for k, v in out.items()}

def _collect_ops(operators):
    ops = {}
    for A in operators:
        if isinstance(A, sparse.COO):
            k = A.ndim
            if A.shape == ():
                A = A.fill_value
            else:
                assert A.fill_value == 0
        elif jnp.isscalar(A) or (hasattr(A, "__array__") and A.ndim==0):
            A = np.asarray(A)
            k = 0
        elif hasattr(A, "__array__"):
            A = sparse.COO.from_numpy(np.asarray(A))
            k = A.ndim
        else:
            raise NotImplementedError
        Ak = ops.pop(k, None)
        if Ak is not None:
            ops[k] = Ak + A
        else:
            ops[k] = A
    return ops

def sparse__arrays_to_coords_data_dict(ops):
    const = ops.pop(0, None)
    coords_data_dict = {A.ndim: (A.coords.T, A.data) for A in ops.values()}
    if const is not None:
        coords_data_dict[0] = np.zeros((1, 0), dtype=int), np.array([const])
    return coords_data_dict


@struct.dataclass
class ParticleNumberConservingFermioperator2ndJax(DiscreteJaxOperator):
    """
    Particle-number conserving fermionc operator
    H = w + Σ_ij w_ij c_i^† c_j + Σ_ijkl w_ijkl  c_i^† c_j^† c_k c_l + Σ_ijklmn w_ijklmn c_i^† c_j^† c_k^† c_l c_m c_n + ...

    Version without spin.

    To be used with netket.hilbert.SpinOrbitalFermions with a fixed number of fermions.

    It uses a custom sparse internal representation,
    please refer to the docstrings of prepare_data and prepare_data_diagonal for details.

    We provide several factory methods to create this operator:
        - ParticleNumberConservingFermioperator2ndJax.from_fermiop:
               Conversion form FermionOperator2nd/FermionOperator2ndJax
        - ParticleNumberConservingFermioperator2ndJax.from_sparse_arrays_normal_order:
                From sparse arrays (w, w_ij, w_ijkl, w_ijklmn) where i>=j, i>=j>=k>=l etc,
                and only the lower triangular part is nonzero
        - ParticleNumberConservingFermioperator2ndJax.from_coords_data_normal_order:
                From tuples of (sites, daggers, weights) representing w, w_ij, ...
                where only the lower triangular part is nonzero
        - ParticleNumberConservingFermioperator2ndJax.from_sparse_arrays:
                From tuples of matrices (sites, daggers, weights) representing w, w_ij, ...
        - ParticleNumberConservingFermioperator2ndJax.from_pyscf_molecule:
                From pyscf

    Furthermore it can be converted to FermionOperator2nd/FermionOperator2ndJax using the .to_fermiop method.
    """
    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree # custom sparse internal representation

    @jax.jit
    @wraps(DiscreteJaxOperator.get_conn_padded)
    def get_conn_padded(self, x):
        dtype = x.dtype
        if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
            x = x.astype(jnp.int8)

        xp_list = []
        mels_list = []
        xp_diag = None
        mels_diag = 0
        for k, v in self._operator_data['diag'].items():
            xp, mels = _get_conn_padded(self._hilbert.n_fermions, x, *v)
            xp_diag = xp
            mels_diag = mels_diag + mels
            xp_list = [xp_diag]
            mels_list = [mels_diag]
        for k, v in self._operator_data['offdiag'].items():
            xp, mels = _get_conn_padded(self._hilbert.n_fermions, x, *v)
            xp_list.append(xp)
            mels_list.append(mels)
        xp = jnp.concatenate(xp_list, axis=-2)
        mels = jnp.concatenate(mels_list, axis=-1)
        return xp.astype(dtype), mels

    @property
    @jax.jit
    def max_conn_size(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.shape[-1]

    @property
    def dtype(self):
        return NotImplemented
        # return list(self._operator_data.values())[0][2].dtype

    @property
    def is_hermitian(self):
        # TODO actually check it is
        # return True
        return NotImplemented

    @classmethod
    def from_coords_data_normal_order(cls, hilbert, coords_data_dict, **kwargs):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        n_orbitals = hilbert.n_orbitals * hilbert.n_spin_subsectors
        data =  prepare_operator_data_from_coords_data_dict(
            coords_data_dict, n_orbitals, **kwargs
        )
        return cls(hilbert, data)

    @classmethod
    def from_sparse_arrays_normal_order(cls, hilbert, operators, **kwargs):
        terms = sparse__arrays_to_coords_data_dict(_collect_ops(operators))

        for k, v in terms.items():
            if k <= 2:
                pass
            idx = v[0]
            idx_create = idx[:, :idx.shape[1]//2]
            idx_destroy = idx[:, idx.shape[1]//2:]
            for idx_arr in idx_destroy, idx_create:
                if (jnp.diff(idx_arr) > 0).any():
                    raise ValueError('Input arrays are not in normal order')

        return cls.from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_sparse_arrays(cls, hilbert, operators, **kwargs):
        # daggers on the left, but not necessarily desc order
        ops = _collect_ops(operators)
        cutoff = kwargs.get('cutoff', 0)
        ops = jax.tree_util.tree_map(partial(to_desc_order_sparse, cutoff=cutoff), ops)
        terms = sparse__arrays_to_coords_data_dict(ops)
        return cls.from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_fermiop(cls, ha, **kwargs):
        # ha = ha.to_normal_order()
        t = _fermiop_terms_to_sites_daggers_weights(ha.terms, ha.weights)
        t = to_normal_order(t)
        terms = {k: (v[0], v[2]) for k, v in t.items()} # drop daggers
        return cls.from_coords_data_normal_order(ha.hilbert, terms, **kwargs)

    def to_fermiop(self, cls=FermionOperator2ndJax):
        terms = []
        weights = []
        for d in self._operator_data:
            for k, v in d.items():
                t, w = _to_fermiop_helper(*v)
                terms = terms + t.tolist()
                weights = weights + w.tolist()
        return cls(self._hilbert, terms, weights)

    @classmethod
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11, **kwargs):
        n_orbitals = int(mol.nao)
        hi = SpinOrbitalFermions(n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec)
        E_nuc, Tij, Vijkl = TV_from_pyscf_molecule(mol, mo_coeff, cutoff=cutoff)
        return cls.from_sparse_arrays_normal_order(hi, [E_nuc, Tij, 0.5 * Vijkl], **kwargs)
