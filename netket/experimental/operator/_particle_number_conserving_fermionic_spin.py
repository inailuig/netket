from functools import partial

import sparse

import jax
import jax.numpy as jnp

from flax import struct

from netket.operator import DiscreteJaxOperator
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import PyTree

from ._particle_number_conserving_fermionic import (
    _get_conn_padded,
    _jw_kernel,
    _prepare_operator_data_from_coords_data_dict,
    _sparse_arrays_to_coords_data_dict,
    split_diag_offdiag,
    prepare_data,
    prepare_data_diagonal,
    _collect_ops,
)
from ._pyscf_utils import compute_pyscf_integrals, to_desc_order_sparse


# TODO do this in hilbert
@partial(jax.jit, static_argnames='n_spin_subsectors')
def unpack_du(x, n_spin_subsectors=2):
    assert x.shape[-1] % n_spin_subsectors == 0
    x_ = x.reshape(x.shape[:-1] + (n_spin_subsectors, x.shape[-1]//n_spin_subsectors))
    return tuple(x_[..., i, :] for i in range(n_spin_subsectors))


@jax.jit
def pack_du(*xs):
    xs = jnp.broadcast_arrays(*xs)
    xd = xs[0]
    n_spin_subsectors = len(xs)
    res = jnp.zeros(
        xd.shape[:-1]
        + (
            n_spin_subsectors,
            xd.shape[-1],
        ),
        dtype=xd.dtype,
    )
    for i, xi in enumerate(xs):
        res = res.at[..., i, :].set(xi)
    return jax.lax.collapse(res, res.ndim - 2, res.ndim)


@partial(jax.jit, static_argnums=(0, 1))
@partial(jnp.vectorize, signature="(n),(n)->(m,n),(m,n),(m)", excluded=(0, 1, 4, 5, 6))
def _get_conn_padded_interaction_up_down(
    nelectron_down, nelectron_up, x_down, x_up, index_array, create_array, weight_array
):
    dtype = x_down.dtype
    # n_orbitals = x_down.shape[-1]

    assert x_down.ndim == 1
    if index_array is not None:
        assert index_array.ndim == 2
    else:  # diagonal
        assert weight_array.ndim == 2

    (down_occupied,) = jnp.where(x_down, size=nelectron_down)
    (up_occupied,) = jnp.where(x_up, size=nelectron_up)

    k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)

    if index_array is None:  # diagonal
        weight = weight_array[k_destroy_down, k_destroy_up]
        xp_down = x_down[None, :]
        xp_up = x_up[None, :]
        sign = 1
        mels = sign * weight.sum()[None]
    else:
        ind = index_array[k_destroy_down, k_destroy_up].ravel()
        weight = weight_array[ind]
        l_create = create_array[ind]

        k_destroy_down = k_destroy_down.reshape(1, -1)
        k_destroy_up = k_destroy_up.reshape(1, -1)
        l_create_down = l_create[..., :1]
        l_create_up = l_create[..., 1:]

        xp_down, sign_down, down_create_is_not_occupied = _jw_kernel(
            k_destroy_down, l_create_down, x_down
        )
        xp_up, sign_up, up_create_is_not_occupied = _jw_kernel(
            k_destroy_up, l_create_up, x_up
        )

        up_is_diagonal = k_destroy_up[0][:, None] == l_create_up[..., 0]
        down_is_diagonal = k_destroy_down[0][:, None] == l_create_down[..., 0]
        both_not_occupied = (down_create_is_not_occupied | down_is_diagonal) & (
            up_create_is_not_occupied | up_is_diagonal
        )

        sign = sign_up * sign_down
        mels = weight * both_not_occupied * sign

        xp_down = jnp.where((mels == 0)[:, :, None], x_down[None, None, :], xp_down)
        xp_up = jnp.where((mels == 0)[:, :, None], x_up[None, None, :], xp_up)

        xp_down = jax.lax.collapse(xp_down, 0, xp_down.ndim - 1).astype(dtype)
        xp_up = jax.lax.collapse(xp_up, 0, xp_up.ndim - 1).astype(dtype)
        mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp_down, xp_up, mels


@partial(jax.jit, static_argnames=("nelec", "use_symm"))
def get_conn_padded_pnc_spin(_operator_data, x, nelec, use_symm=True):
    n_spin_subsectors = len(nelec)
    xs = unpack_du(x, n_spin_subsectors)
    xs_diag = tuple(a[..., None, :] for a in xs)
    dtype = xs[0].dtype

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0

    for k, v in _operator_data[0].items():
        for xi, nelectroni in zip(xs, nelec):
            _, melsi = _get_conn_padded(nelectroni, xi, *v)
            mels_diag = mels_diag + melsi
            if k ==0:
                break
        xp_diag = x[..., None, :]
        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for k, v in _operator_data[2].items():
        if k != 4:
            raise NotImplementedError

        for i in range(n_spin_subsectors):
            for j in range(i+1, n_spin_subsectors):
                # here j>i
                xi, xj = xs[i], xs[j]
                nelectroni, nelectronj = nelec[i], nelec[j]

                *_, melsij = _get_conn_padded_interaction_up_down(
                    nelectroni, nelectronj, xi, xj, *v
                )
                xp_diag = x[..., None, :]
                # we use the symmetry in hijkl
                if use_symm:
                    # the udud term is equal to dudu and we can just compute one and multiply with 2
                    mels_diag = mels_diag + 2 * melsij
                else:
                    *_, melsji = _get_conn_padded_interaction_up_down(
                        nelectronj, nelectroni, xj, xi, *v
                    )
                    mels_diag = mels_diag + melsij + melsji

        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for k, v in _operator_data[1].items():
        for i, (xi, nelectroni) in enumerate(zip(xs, nelec)):
            xpi, melsi = _get_conn_padded(nelectroni, xi, *v)
            xpi = pack_du(*xs_diag[:i], xpi, *xs_diag[i+1:])
            xp_list.append(xpi)
            mels_list.append(melsi)

    for k, v in _operator_data[3].items():
        if k != 4:
            raise NotImplementedError
        for i in range(n_spin_subsectors):
            for j in range(i+1, n_spin_subsectors):
                # here j>i
                xi, xj = xs[i], xs[j]
                nelectroni, nelectronj = nelec[i], nelec[j]

                xpi, xpj, melsij = _get_conn_padded_interaction_up_down(
                    nelectroni, nelectronj, xi, xj, *v
                )
                xpij = pack_du(*xs_diag[:i], xpi, *xs_diag[i+1:j], xpj, *xs_diag[j+1:])
                xp_list.append(xpij)
                if use_symm:
                    # we use the symmetry in hijkl
                    # the udud term is equal to dudu and we can just compute one and multiply with 2
                    mels_list.append(2 * melsij)
                else:
                    mels_list.append(melsij)
                    xpj, xpi, melsji = _get_conn_padded_interaction_up_down(
                        nelectronj, nelectroni, xj, xi, *v
                    )
                    xpji = pack_du(*xs_diag[:i], xpi, *xs_diag[i+1:j], xpj, *xs_diag[j+1:])
                    xp_list.append(xpji)
                    mels_list.append(melsji)
    if len(xp_list) > 0:
        xp = jnp.concatenate(xp_list, axis=-2).astype(dtype)
        mels = jnp.concatenate(mels_list, axis=-1)
    else:
        xp = jnp.zeros((*x.shape[:-1], 0,x.shape[-1]), dtype=dtype)
        mels = jnp.zeros(xp.shape[:-1]) # TODO dtype?
    return xp, mels


def _sparse_arrays_to_coords_data_spin(operators, cutoff=1e-11):
    # operators = [const, hij, hijkl]
    # ops = {0: const, 2: hij_sparse, 4: hijkl_sparse}
    ops = _collect_ops(operators)
    for k in ops.keys():
        if k not in (0,2,4):
            raise NotImplementedError
    hijkl_sparse = ops.get(4, None)
    if hijkl_sparse is not None:
        ops[4] = to_desc_order_sparse(hijkl_sparse, cutoff)
        v = _sparse_arrays_to_coords_data_dict({4: hijkl_sparse})[4]
        coords_data_mixed = v[0][:, [0, 1, 3, 2]], *v[1:]  # swap ijkl->ijlk
    else:
        coords_data_mixed = None
    coords_data = _sparse_arrays_to_coords_data_dict(ops)
    return coords_data, coords_data_mixed

def prepare_operator_data_from_coords_data_dict_spin(coords_data, coords_data_mixed, n_orbitals):
    operator_data = _prepare_operator_data_from_coords_data_dict(coords_data, n_orbitals)
    # process mixed terms
    data_diag_mixed = {}
    data_offdiag_mixed = {}
    if coords_data_mixed is not None:
        sw_diag, sw_offdiag = split_diag_offdiag(*coords_data_mixed)
        if len(sw_diag[-1]) > 0:
            data_diag_mixed = {4: prepare_data_diagonal(*sw_diag, n_orbitals, _sparse=False)}
        if len(sw_offdiag[-1]) > 0:
            data_offdiag_mixed = {4: prepare_data(*sw_offdiag, n_orbitals, _sparse=False)}
    operator_data = *operator_data, data_diag_mixed, data_offdiag_mixed
    return operator_data


# TODO generalize it to >4 fermionic operators

@struct.dataclass
class ParticleNumberConservingFermioperator2ndSpinJax(DiscreteJaxOperator):
    """
    H = a + Σ_ijσ b_ij c_iσ^† c_jσ + Σ_ijklσρ c_ijkl  c_iσ^† c_jρ^† c_kρ c_lσ
    """
    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree
    _use_symm: bool = struct.field(pytree_node=False, default=False)

    @property
    def dtype(self):
        return NotImplemented

    @property
    def is_hermitian(self):
        return True

    @property
    @jax.jit
    def max_conn_size(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.shape[-1]

    def get_conn_padded(self, x):
        return get_conn_padded_pnc_spin(
            self._operator_data, x, self._hilbert.n_fermions_per_spin, self._use_symm,
        )

    @classmethod
    def from_coords_data(cls, hilbert, coords_data, coords_data_mixed):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        assert hilbert.n_spin_subsectors >= 2
        n_orbitals = hilbert.n_orbitals
        operator_data = prepare_operator_data_from_coords_data_dict_spin(coords_data, coords_data_mixed, n_orbitals)
        return cls(hilbert, operator_data)

    @classmethod
    def from_sparse_arrays(cls, hilbert, operators, **kwargs):
        coords_data, coords_data_mixed = _sparse_arrays_to_coords_data_spin(operators, **kwargs)
        return cls.from_coords_data(hilbert, coords_data, coords_data_mixed)

    @classmethod
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11):
        # TODO eventually deprecate this in favour of pyscf.py ?
        n_orbitals = int(mol.nao)
        hi = SpinOrbitalFermions(n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec)

        const, hij, hijkl = compute_pyscf_integrals(mol, mo_coeff)  # not in normal order
        hij = hij * (jnp.abs(hij) > cutoff)
        hij_sparse = sparse.COO.from_numpy(hij)
        hijkl = hijkl * (jnp.abs(hijkl) > cutoff)
        hijkl_sparse = 0.5 * sparse.COO.from_numpy(hijkl)

        return cls.from_sparse_arrays(hi, [const, hij_sparse, hijkl_sparse], cutoff=cutoff).replace(_use_symm=True)
