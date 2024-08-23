import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from netket.experimental.operator._pyscf_utils import _parity

def _split_spin_sectors(sites, n_orbitals, n_spin_subsectors):
    n_ops = sites.shape[1]
    if n_ops == 0:
        return sites, jnp.zeros_like(sites)
    L = np.arange(n_spin_subsectors)*n_orbitals
    R = np.arange(1, n_spin_subsectors+1)*n_orbitals
    sectors_mask = ((sites[...,None] >= L) & (sites[...,None] < R)) # n_terms x n_ops x n_spin_subsectors
    sectors = jnp.einsum('...i,i', sectors_mask, jnp.arange(n_spin_subsectors)).astype(np.int32)
    sites = sites - sectors * n_orbitals
    return sites, sectors

def _fermiop_terms_to_arrays_spin(terms, weights, n_orbitals, n_spin_subsectors):
    d = _fermiop_terms_to_arrays(terms, weights)
    # { size : (sites, sectors, daggers, weights) }
    return {k: (*_split_spin_sectors(v[0], n_orbitals, n_spin_subsectors), *v[1:]) for k, v in d.items()}


def prune(sites, daggers, weights):
    # remove ci ci and ci+ci+ on the same site i
    mask = ~((np.diff(daggers, axis=-1) == 0) & (np.diff(sites, axis=-1) == 0)).any(axis=-1)
    return sites[mask], daggers[mask], weights[mask]

def move(i, j, x, mask=None):
    n = x.shape[-1]
    a = jnp.arange(n)[None]
    # move i after j
    masklr = (a<i) | (a>j)
    maskj = a == j
    mask_middle = ~(masklr | maskj)
    x1 = np.roll(x, -1, axis=-1)
    xi = jnp.take_along_axis(x, i, 1)
    res = masklr * x + mask_middle * x1 + maskj * xi
    if mask is None:
        return res
    return res * mask + (~mask)*x

def remove(i, j, x):
    n = x.shape[-1]
    a = jnp.arange(n)[None]
    # remove i and j
    maskl = a<i
    maskr = a>j-2
    mask_middle = ~(maskl | maskr)
    x1 = np.roll(x, -1, axis=-1)
    x2 = np.roll(x, -2, axis=-1)
    return (maskl * x + mask_middle * x1 + maskr * x2)[..., :-2]


import sparse

def _move_daggers_left(sites_, daggers_, weights_):
    n = daggers_.shape[-1]
    if n == 0:
        return (sites_, daggers_, weights_),
    new_sites_smaller = []
    new_daggers_smaller = []
    new_weights_smaller = []
    while True:
        a = jnp.arange(n)[None]
        # find leftmost c
        i = np.argmin(daggers_,axis=1, keepdims=True)
        # find next dagger
        j = np.argmax((i < a) & daggers_,axis=1, keepdims=True)

        # now move the c after the dagger if necessary

        do_move = j > i
        if ~do_move.any():
            break

        sign = 1-2*(((j-i)*do_move)%2).ravel()

        si = jnp.take_along_axis(sites_, i, 1)
        sj = jnp.take_along_axis(sites_, j, 1)

        new_sites = move(i, j, sites_, mask=do_move)
        new_daggers = move(i, j, daggers_, mask=do_move)
        new_weights = weights_ * sign

        same = ((si==sj) & do_move).ravel()
        new_sites2 = remove(i[same], j[same], sites_[same])
        new_daggers2 = remove(i[same], j[same], daggers_[same])
        new_weights2 = -(weights_ * sign)[same]
        new_sites2, new_daggers2, new_weights2 = prune(new_sites2, new_daggers2, new_weights2)
        if len(new_sites2) > 0:
            new_sites_smaller.append(new_sites2)
            new_daggers_smaller.append(new_daggers2)
            new_weights_smaller.append(new_weights2)
        # set var for next iteration
        sites_, daggers_, weights_ = prune(new_sites, new_daggers, new_weights)

    if len(new_sites_smaller) > 0:
        new_sites_smaller = np.concatenate(new_sites_smaller, axis=0)
        new_daggers_smaller = np.concatenate(new_daggers_smaller, axis=0)
        new_weights_smaller = np.concatenate(new_weights_smaller, axis=0)
        # recursion; TODO collapse first and only run once for each size instead?
        return (sites_, daggers_, weights_), *_move_daggers_left(new_sites_smaller, new_daggers_smaller, new_weights_smaller)
    else:
        return (sites_, daggers_, weights_),

def move_daggers_left(t):
    d = {}
    for sdw in [x for v in t.values() for x in _move_daggers_left(*v)]:
        k = sdw[0].shape[-1]
        if sdw[-1].size > 0: # not empty
            di = d.pop(k, None)
            if di is None:
                d[k] = sdw
            else:
               d[k] = tuple(np.concatenate([a,b], axis=0) for a, b in zip(di,sdw))
    return d

def swd_to_sparse(sites, daggers, weights, n_orbitals):
    n = daggers.shape[-1]
    assert n%2 == 0
    assert (daggers[:, :n//2] == 1).all()
    assert (daggers[:, n//2:] == 0).all()
    # TODO cutoff?
    return sparse.COO(sites.T, weights, shape=(n_orbitals,)*n)

def extract_operators_normal_order(*swd, n_orbitals):
    operators = []
    while True:
        swd_daggers_left, swd = move_daggers_left(*swd)
        swd_daggers_left = to_desc_order(*swd_daggers_left)
        o = swd_to_sparse(*swd_daggers_left, n_orbitals)
        operators.append(o)
        if swd[0] is None:
            break
        elif swd[0].shape[-1] == 0:
            operators.append(swd[2].sum())
            break
    return operators



def _to_desc_order(sites_, daggers_, weights_):
    n = daggers_.shape[-1]
    if n == 0:
        return sites_, daggers_, weights_
    # check min and max do not over/underflow
    # TODO promote to signed / bigger dtype if necessary
    xl = sites_.min()-1
    xr = sites_.max()+1
    assert (xl < sites_.min()).all()
    assert (xr > sites_.max()).all()

    # minus because we order descending
    s0 = -daggers_ * xr - (1-daggers_) * sites_
    s1 = - daggers_ * sites_ - (1-daggers_)*xl

    perm0 = np.argsort(s0, axis=-1)
    perm1 = np.argsort(s1, axis=-1)
    a = np.arange(len(sites_))[:,None]
    sites_desc = sites_[a, perm0] * (1-daggers_) + sites_[a, perm1] * daggers_
    weights_desc = weights_ * (1-2*(_parity(perm0) ^ _parity(perm1)))

    # TODO also merge duplicates
    return prune(sites_desc, daggers_, weights_desc)

def to_desc_order(t):
    # assumes daggers are already left
    # TODO sum duplicates
    return {k: _to_desc_order(*v) for k,v in t.items()}


def arrays_to_fermiop_terms(t):
    terms = []
    weights = []
    for s,d,w in t.values():
        terms = terms + np.concatenate([s[..., None],d[..., None]], axis=-1).tolist()
        weights = weights + w.tolist()
    return terms, weights

def to_normal_order(t):
    return to_desc_order(move_daggers_left(t))

# test:
# t = _fermiop_terms_to_arrays(ha.terms, ha.weights)
# ha1 = FermionOperator2nd(hi, *arrays_to_fermiop_terms(t))
# np.allclose(ha.to_dense(), ha1.to_dense())
# t_left = move_daggers_left(t)
# ha2 = FermionOperator2nd(hi, *arrays_to_fermiop_terms(t_left))
# np.allclose(ha.to_dense(), ha2.to_dense())
# t_normal = to_desc_order(t_left)
# ha3 = FermionOperator2nd(hi, *arrays_to_fermiop_terms(t_normal))
# np.allclose(ha.to_dense(), ha3.to_dense())
