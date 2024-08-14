from functools import partial

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

from netket.experimental.operator._pyscf_utils import to_desc_order_sparse

def _prepare_data(sites_destr, sites_create, weights, n_orbitals, _sparse=True):
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
        if not _sparse:
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
        if not _sparse:
            index_array = index_array.todense()

    return index_array, create_array, weight_array


def prepare_data_diagonal(sites_destr, weights, n_orbitals, **kwargs):
    # sites_destr needs to contain the sites only once, not twice!!
    return _prepare_data(sites_destr, None, weights, n_orbitals, **kwargs)


def prepare_data(sites, weights, n_orbitals, **kwargs):
    # sites is an array (n_terms, n_ops) containing the sites
    # of terms in normal order (daggers to left, desc order)
    #
    # weights is of shape (n_terms,)
    n_terms, n_ops = sites.shape
    assert n_ops % 2 == 0
    sites_destr = sites[:, : n_ops // 2]
    sites_create = sites[:, n_ops // 2 :]
    return _prepare_data(sites_destr, sites_create, weights, n_orbitals, **kwargs)


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


def _comb(kl, n):
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

def _sparse_arrays_to_coords_data_dict(ops):
    const = ops.pop(0, None)
    coords_data_dict = {A.ndim: (A.coords.T, A.data) for A in ops.values()}
    if const is not None:
        coords_data_dict[0] = np.zeros((1, 0), dtype=int), np.array([const])
    return coords_data_dict


def _prepare_operator_data_from_coords_data_dict(
    coords_data_dict, n_orbitals, **kwargs
):
    # n_fermions = hi.n_fermions
    data_offdiag = {}
    data_diag = {}
    for k, v in coords_data_dict.items():
        sw_diag, sw_offdiag = split_diag_offdiag(*v)
        if len(sw_diag[-1]) > 0:
            data_diag[k] = prepare_data_diagonal(*sw_diag, n_orbitals, **kwargs)
        if len(sw_offdiag[-1]) > 0:
            data_offdiag[k] = prepare_data(*sw_offdiag, n_orbitals, **kwargs)
    data = data_diag, data_offdiag
    return data


@struct.dataclass
class ParticleNumberConservingFermioperator2ndJax(DiscreteJaxOperator):
    """
    H = a + Σ_ij b_ij c_i^† c_j + Σ_ijkl c_ijkl  c_i^† c_j^† c_k c_l + Σ_ijklmn c_ijklmn c_i^† c_j^† c_k^† c_l c_m c_n + ...
    """
    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree

    @jax.jit
    def get_conn_padded(self, x):
        dtype = x.dtype
        if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
            x = x.astype(jnp.int8)

        xp_list = []
        mels_list = []
        xp_diag = None
        mels_diag = 0
        for k, v in self._operator_data[0].items():
            xp, mels = _get_conn_padded(self._hilbert.n_fermions, x, *v)
            xp_diag = xp
            mels_diag = mels_diag + mels
            xp_list = [xp_diag]
            mels_list = [mels_diag]
        for k, v in self._operator_data[1].items():
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
        return NotImplemented
        # return True

    @classmethod
    def from_coords_data_normal_order(cls, hilbert, coords_data_dict, **kwargs):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        n_orbitals = hilbert.n_orbitals * hilbert.n_spin_subsectors
        data = _prepare_operator_data_from_coords_data_dict(
            coords_data_dict, n_orbitals, **kwargs
        )
        return cls(hilbert, data)

    @classmethod
    def from_sparse_arrays_normal_order(cls, hilbert, operators, **kwargs):
        terms = _sparse_arrays_to_coords_data_dict(_collect_ops(operators))

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
        terms = _sparse_arrays_to_coords_data_dict(ops)
        return cls.from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_fermiop(cls, ha, **kwargs):
        ha = ha.to_normal_order()
        terms = {}
        for t, w in zip(ha.terms, ha.weights):
            if len(t) == 0:  # constant
                terms[0] = np.zeros((1, 0), dtype=np.int32), np.array([w])
            else:
                sites, daggers = np.array(t).T
                l = len(daggers)
                assert l % 2 == 0
                assert 2 * daggers.sum() == l
                tl, wl = terms.get(l, ([], []))
                terms[l] = tl + [sites,], wl + [w,]  # fmt: skip
        terms = {
            k: (jnp.array(v[0], dtype=np.int32), jnp.array(v[1]))
            for k, v in terms.items()
        }
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
