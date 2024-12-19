# utilities to convert from / to FermionOperator2nd/FermionOperator2ndJax

from functools import partial

import numpy as np
import sparse

import jax
import jax.numpy as jnp

from netket.jax import COOTensor

from ._operator_data import len_helper
from ._normal_order_utils import split_spin_sectors


# TODO merge this with fermionoperator2nd prepare_terms_list
def fermiop_terms_to_sites_daggers_weights(terms, weights):
    r"""
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

def to_fermiop_helper(index_array, create_array, weight_array):
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





def fermiop_terms_to_sites_sectors_daggers_weights(terms, weights, n_orbitals, n_spin_subsectors):
    # output: { size : (sites, sectors, daggers, weights) }
    return split_spin_sectors(fermiop_terms_to_sites_daggers_weights(terms, weights), n_orbitals, n_spin_subsectors)


def sites_daggers_weights_to_sparse(sites, daggers, weights, n_orbitals):
    n = daggers.shape[-1]
    assert n%2 == 0
    assert (daggers[:, :n//2] == 1).all()
    assert (daggers[:, n//2:] == 0).all()
    # TODO cutoff?
    return sparse.COO(sites.T, weights, shape=(n_orbitals,)*n)


def _insert_append_helper(d, k, s, o, cutoff):
    # check if an element with the same matrix but different sectors exist
    # if yes append to the list of sectors
    # else insert new element into the dict
    for (k2, s2), o2 in d.items():
        # and same number of sectors, same number of fermionic operators, same matrix (up to cutoff)
        if ((s==() and s2 == ()) or (len(s2)>0 and len(s)>0 and  len_helper(s2[0]) == len_helper(s[0]))) and k==k2 and sparse.abs(o-o2).max() < cutoff :
            d[k, s2+s] = d.pop((k2, s2))
            break
    else:
        d[k, s] = o

def to_operators_sector(tno_sector, n_spin_subsectors, n_orbitals, cutoff=1e-11):
    r"""
    Args:
        tno_sector: a list of tuples [(sites, sectors, daggers, weights)]
                    of terms in normal order with higher sectors on the left

    Returns: a dict {(k, sectors) : data}
             where k is the number of c/c^\dagger,
             sectors are the spin sectors acted on
             and data is a sparse matrix of size (n_orbitals,)*2k
    """

    operators_sector = {}

    for k, (sites, sectors, daggers, weights) in tno_sector.items():
        for i in range(n_spin_subsectors):
            if not (((2*daggers-1)*(sectors==i)).sum(axis=-1) == 0).all():
                raise ValueError # does not conserve particle number per sector

        sector_count = jax.vmap(partial(jnp.bincount, length=n_spin_subsectors))(sectors)

        # merge sectors which have same sparse matrix

        if k == 0:
            operators_sector[0, ()] = weights.reshape(())
        elif k == 2:
            # at this point we know there is only one sector this acts on
            sector = sectors[:, 0]  # = sectors[:, 1]
            for i in np.unique(sector):
                m = sector==i
                o = sites_daggers_weights_to_sparse(sites[m], daggers[m], weights[m], n_orbitals=n_orbitals)
                _insert_append_helper(operators_sector, k, (i,), o, cutoff)
        elif k == 4:
            # at this point we know that n_sectors_acting_on \in 1,2
            n_sectors_acting_on = np.count_nonzero(sector_count, axis=-1)

            # all same sector
            m_same = n_sectors_acting_on==1
            sector = sectors[:, 0]
            for i in np.unique(sector[m_same]):
                m = (sector == i) & m_same
                o = sites_daggers_weights_to_sparse(sites[m], daggers[m], weights[m], n_orbitals=n_orbitals)
                _insert_append_helper(operators_sector, k, (i,), o, cutoff)

            m_different = ~m_same
            sector = sectors[:, :2]
            # i > j because we made it normal order (with site shifted by N*spin) above
            for ij in np.unique(sector[m_different], axis=0):
                m = (sector == ij[None]).all(axis=-1) & m_different
                # minus sign because in the operator (_get_conn_padded_interaction_up_down) we assume it's swaped to (assuming σ>ρ)
                # cσ^† cσ cρ^† cρ = - cσ^† cρ^† cσ cρ
                o = - sites_daggers_weights_to_sparse(sites[m], daggers[m], weights[m], n_orbitals=n_orbitals)
                _insert_append_helper(operators_sector, k, (tuple(ij),), o, cutoff)
        else:
            raise NotImplementedError
    return operators_sector
