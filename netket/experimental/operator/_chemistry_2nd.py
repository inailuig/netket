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
)
from .pyscf import compute_pyscf_integrals, to_desc_order_sparse


# TODO generalize this to a ParticleNumberConservingFermioperator2ndSpinJax


# TODO use hilbert for this
@jax.jit
def unpack_du(x):
    x_ = x.reshape(x.shape[:-1] + (2, x.shape[-1]//2))
    return x_[..., 0, :], x_[..., 1, :]


@jax.jit
def pack_du(xd, xu):
    xd, xu = jnp.broadcast_arrays(xd, xu)
    res = jnp.zeros(
        xd.shape[:-1]
        + (
            2,
            xd.shape[-1],
        ),
        dtype=xd.dtype,
    )
    res = res.at[..., 0, :].set(xd)
    res = res.at[..., 1, :].set(xu)
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

        # TODO do we need an extra minus sign? because we swapped the center two ops??
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
    x_down, x_up = unpack_du(x)
    dtype = x_down.dtype

    nelectron_down, nelectron_up = nelec

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0

    for k, v in _operator_data[0].items():
        _, mels_uu = _get_conn_padded(nelectron_up, x_up, *v)
        mels_diag = mels_diag + mels_uu
        if k != 0:
            _, mels_dd = _get_conn_padded(nelectron_down, x_down, *v)
            mels_diag = mels_diag + mels_dd

        xp_diag = x[..., None, :]
        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for k, v in _operator_data[2].items():
        if k != 4:
            raise NotImplementedError
        *_, mels_du = _get_conn_padded_interaction_up_down(
            nelectron_down, nelectron_up, x_down, x_up, *v
        )
        xp_diag = x[..., None, :]
        # we use the symmetry in hijkl
        if use_symm:
            # the udud term is equal to dudu and we can just compute one and multiply with 2
            mels_diag = mels_diag + 2 * mels_du
        else:
            *_, mels_ud = _get_conn_padded_interaction_up_down(
                nelectron_up, nelectron_down, x_up, x_down, *v
            )
            mels_diag = mels_diag + mels_du + mels_ud

        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for k, v in _operator_data[1].items():
        xp_dd, mels_dd = _get_conn_padded(nelectron_down, x_down, *v)
        xp_uu, mels_uu = _get_conn_padded(nelectron_up, x_up, *v)
        xp_dd = pack_du(xp_dd, x_up[..., None, :])
        xp_uu = pack_du(x_down[..., None, :], xp_uu)
        xp_list.append(xp_dd)
        xp_list.append(xp_uu)
        mels_list.append(mels_dd)
        mels_list.append(mels_uu)

    for k, v in _operator_data[3].items():
        if k != 4:
            raise NotImplementedError
        *xp_du, mels_du = _get_conn_padded_interaction_up_down(
            nelectron_down, nelectron_up, x_down, x_up, *v
        )
        xp_du = pack_du(*xp_du)
        # we use the symmetry in hijkl
        # the udud term is equal to dudu and we can just compute one and multiply with 2
        xp_list.append(xp_du)
        if use_symm:
            mels_list.append(2 * mels_du)
        else:
            mels_list.append(mels_du)
            *xp_ud, mels_ud = _get_conn_padded_interaction_up_down(
                nelectron_up, nelectron_down, x_up, x_down, *v
            )
            xp_ud = tuple(reversed(xp_ud))
            xp_list.append(xp_ud)
            mels_list.append(mels_ud)

    xp = jnp.concatenate(xp_list, axis=-2)
    mels = jnp.concatenate(mels_list, axis=-1)

    return xp.astype(dtype), mels

def prepare_coords_data(mol, mo_coeff, cutoff=1e-11):
    # TODO make this more modular
    # TODO actually use cutoff everywhere
    n_orbitals = int(mol.nao)

    const, hij, hijkl = compute_pyscf_integrals(
        mol, mo_coeff
    )  # not in normal order
    hij = hij * (jnp.abs(hij) > cutoff)
    hijkl = hijkl * (jnp.abs(hijkl) > cutoff)

    hijkl_sparse = 0.5 * sparse.COO.from_numpy(hijkl)
    hij_sparse = sparse.COO.from_numpy(hij)

    arrays_desc_order = (
        const,
        hij_sparse,
        to_desc_order_sparse(hijkl_sparse, cutoff),
    )
    coords_data_dict = _sparse_arrays_to_coords_data_dict(arrays_desc_order)


    v = _sparse_arrays_to_coords_data_dict([hijkl_sparse])[4]
    coords_data_mixed = v[0][:, [0, 1, 3, 2]], *v[1:]  # swap ijkl->ijlk
    return coords_data_dict, coords_data_mixed

def prepare_operator_data_from_coords_data_dict_spin(coords_data_dict, coords_data_mixed, n_orbitals):
    operator_data = _prepare_operator_data_from_coords_data_dict(coords_data_dict, n_orbitals)
    # process mixed terms
    sw_diag, sw_offdiag = split_diag_offdiag(*coords_data_mixed)
    data_offdiag_mixed = prepare_data(*sw_offdiag, n_orbitals, _sparse=False)
    data_diag_mixed = prepare_data_diagonal(*sw_diag, n_orbitals, _sparse=False)
    operator_data = *operator_data, {4: data_diag_mixed}, {4: data_offdiag_mixed}
    return operator_data

@struct.dataclass
class Chemistry2ndJax(DiscreteJaxOperator):
    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree
    _use_symm: bool = struct.field(pytree_node=False, default=True)

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
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11):
        # TODO actually use cutoff everywhere
        n_orbitals = int(mol.nao)
        coords_data_dict, coords_data_mixed = prepare_coords_data(mol, mo_coeff, cutoff=cutoff)
        operator_data = prepare_operator_data_from_coords_data_dict_spin(coords_data_dict, coords_data_mixed, n_orbitals)
        hi = SpinOrbitalFermions(n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec)
        return cls(hi, operator_data)
