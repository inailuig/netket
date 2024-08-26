import numpy as np
import jax
import jax.numpy as jnp
from netket.experimental.operator import ParticleNumberConservingFermioperator2ndJax, FermionOperator2nd
from netket.experimental.hilbert import SpinOrbitalFermions

import pytest


def _cast_normal_order(A):
    idx = jnp.array(jnp.where(A)).T
    idx_create = idx[:, :idx.shape[1]//2]
    idx_destroy = idx[:, idx.shape[1]//2:]
    mask = (jnp.diff(idx_destroy) > 0).any(axis=1) | (jnp.diff(idx_create) > 0).any(axis=1)
    return A.at[idx[mask].T].set(0)


def test_pnc():

    N = 6
    n = 3
    cutoff = 0.1
    key = np.random.randint(2**32)

    k0, k1, k2, k3 = jax.random.split(jax.random.key(key), 4)
    c = jax.random.normal(k0)
    hij = jax.random.normal(k1, shape=(N,)*2)
    hijkl = _cast_normal_order(jax.random.normal(k2, shape=(N,)*4))
    hijklmn = _cast_normal_order(jax.random.normal(k3, shape=(N,)*6))

    hi = SpinOrbitalFermions(N, n_fermions=n)

    terms = []
    weights = []
    terms = terms + [""]
    weights = weights + [c]
    ij = jnp.where(jnp.abs(hij)>cutoff)
    terms = terms + [f"{i}^ {j}" for i,j in list(zip(*ij))]
    weights = weights + list(hij[ij])
    ijkl = jnp.where(jnp.abs(hijkl)>cutoff)
    terms = terms + [f"{i}^ {j}^ {k} {l}" for i,j,k,l in list(zip(*ijkl))]
    weights = weights + list(hijkl[ijkl])
    ijklmn = jnp.where(jnp.abs(hijklmn)>cutoff)
    terms = terms + [f"{i}^ {j}^ {k}^ {l} {m} {n}" for i,j,k,l,m,n in list(zip(*ijklmn))]
    weights = weights + list(hijklmn[ijklmn])
    ha = FermionOperator2nd(hi, terms=terms, weights=weights)

    ha2 = ParticleNumberConservingFermioperator2ndJax.from_sparse_arrays_normal_order(hi, [c, hij*(jnp.abs(hij) > cutoff), hijkl*(jnp.abs(hijkl) > cutoff), hijklmn*(jnp.abs(hijklmn) > cutoff)])

    np.testing.assert_allclose(ha.to_dense(), ha2.to_dense())


