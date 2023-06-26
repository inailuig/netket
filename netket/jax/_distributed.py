import jax
from functools import partial, wraps
import numpy as np


from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map


@partial(jax.jit, static_argnums=0)
def _replicate_shmap_callback(f, x):
    # TODO how to get mesh and axes?
    mesh = Mesh(jax.devices(), axis_names=("i"))

    @partial(shard_map, mesh=mesh, in_specs=(P("i")), out_specs=P("i"))
    def _f(x):
        # here we do eval shape by hand
        # TODO better way?
        dummy_x = np.zeros((1,) * (x.ndim - 1) + x.shape[-1:], x.dtype)
        dummy_xp, dummy_mels = f(dummy_x)
        xp_shape = jax.ShapeDtypeStruct(
            x.shape[:-1] + dummy_xp.shape[x.ndim - 1 :], dummy_xp.dtype
        )
        mels_shape = jax.ShapeDtypeStruct(
            x.shape[:-1] + dummy_mels.shape[x.ndim - 1 :], dummy_mels.dtype
        )
        result_shape = (xp_shape, mels_shape)
        return jax.pure_callback(f, result_shape, x, vectorized=True)

    return _f(x)


def replicate_sharding_shmap(f):
    return partial(_replicate_shmap_callback, f)


replicate_sharding = replicate_sharding_shmap


_identity = lambda x: x


def put_global(inp_data, axis=0):
    # distribute a local array along an axis to all (local and global) devices
    # each process needs to have the whole array; parts not belonging to it can be filled with garbage
    shape = [
        1,
    ] * inp_data.ndim
    shape[axis] = -1
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(shape)
    return jax.jit(_identity, out_shardings=sharding)(inp_data)


def extract_replicated(t):
    # extract the content of a fully replicated global device array
    def _extract_replicated(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            assert x.is_fully_replicated
            return x.addressable_data(0)
        else:
            return x

    return jax.tree_map(_extract_replicated, t)
