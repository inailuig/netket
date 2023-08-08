import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(jax.jit, static_argnames=('local_size', 'dtype'))
def packuints(x, local_size = 2, dtype=jax.dtypes.canonicalize_dtype(jnp.uint64)):
    # here we assume little endian order
    # we only support axis=-1

    assert jnp.issubdtype(dtype, jnp.unsignedinteger)
    x = x.astype(dtype)
    n_bits = 8*jnp.zeros(0, dtype).itemsize
    n_sites = x.shape[-1]
    n_bits_per_site = int(np.ceil(np.log2(local_size)))
    padded_local_size = 2**n_bits_per_site
    n_sites_per_uint, _ = divmod(n_bits, n_bits_per_site) # round down

    if n_sites <= n_sites_per_uint:
        local_basis = jnp.flip(padded_local_size ** jnp.arange(n_sites, dtype=dtype))
        return x.dot(local_basis)
    else:
        n, r = divmod(n_sites, n_sites_per_uint)
        lengths = (n_sites_per_uint,) * n
        if r != 0:
            lengths = (r,) + lengths
        start = np.cumsum((0,) + lengths[:-1])
        end = np.cumsum(lengths)
        # recursion
        return jnp.moveaxis(jnp.array([packuints(x[..., l:r], local_size, dtype=dtype) for l, r in zip(start, end)]), 0, -1)

@partial(jax.jit, static_argnames=('n_sites', 'local_size', 'out_dtype'))
def unpackuints(i, n_sites, local_size = 2, out_dtype=jnp.uint8):
    dtype = i.dtype
    assert jnp.issubdtype(dtype, jnp.unsignedinteger)
    n_bits = 8*jnp.zeros(0, dtype).itemsize
    n_bits_per_site = int(np.ceil(np.log2(local_size)))
    padded_local_size = 2**n_bits_per_site
    n_sites_per_uint, _ = divmod(n_bits, n_bits_per_site) # round down

    if n_sites <= n_sites_per_uint:
        local_basis = jnp.flip(padded_local_size ** jnp.arange(n_sites, dtype=dtype))
        # TODO special case for local_size == 2
        if local_size == 2:
            res = jax.lax.bitwise_and(jnp.expand_dims(i, -1), local_basis[(jnp.newaxis,) * i.ndim]) != 0
        else:
            local_mask = jnp.array(2**(n_bits_per_site)-1, dtype=dtype)
            res = jax.lax.bitwise_and(jnp.expand_dims(i, -1) // local_basis[(jnp.newaxis,) * i.ndim], local_mask)
        return res.astype(out_dtype)
    else:
        n, r = divmod(n_sites, n_sites_per_uint)
        lengths = (n_sites_per_uint,) * n
        if r != 0:
            lengths = (r,) + lengths
        # recursion
        return jnp.concatenate([unpackuints(ii, N, local_size, out_dtype=out_dtype) for ii, N in zip(jnp.moveaxis(i, -1, 0), lengths)], axis=-1)


packbits = partial(packuints, local_size = 2)
unpackbits = partial(unpackuints, local_size = 2)

@partial(jax.jit, static_argnames=('dtype',))
def spin_to_qubit(x, dtype=jnp.uint8):
    return (x.astype(dtype) + 1) // 2

@partial(jax.jit, static_argnames=('dtype',))
def qubit_to_spin(x, dtype=jnp.int8):
    return (2 * x - 1).astype(dtype)
