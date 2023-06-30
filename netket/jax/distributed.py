import jax
from functools import partial
import numpy as np


from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map


@partial(jax.jit, static_argnums=0)
def _replicate_shmap_callback(f, x):
    # TODO where to get mesh and axes from without hardcoding them here?
    mesh = Mesh(jax.devices(), axis_names=("i"))

    @partial(shard_map, mesh=mesh, in_specs=(P("i")), out_specs=P("i"))
    def _f(x):
        # here we infer the output shape by doing eval_shape by hand
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


# TODO eventually make this a bit more univesal
# supporting more fancy sharding and multiple input/output args
def replicate_sharding_shmap(f):
    """
    Wrapper for python get_conn_padded to make it work with shared/global device arrays.
    Calls f on every shard, and puts the results back on the devices with the correct sharding.
    The input to f is assumed to have PositionalSharding (or equivalent) along a single batch axis.

    version which uses pure_callback and jax.experimental.shard_map internally

    Args:
        f: a python get_conn_padded (which takes x and maps it to (xp,mels))
    """
    return partial(_replicate_shmap_callback, f)


replicate_sharding = replicate_sharding_shmap


_identity = lambda x: x


def put_global(inp_data, axis=0):
    """
    distribute a local array equally along an axis to all (local and global) devices
    The size of the axis needs to be divisible by the number of devices.
    each process needs to have the whole array (parts not belonging to it can be filled with garbage)
    Args:
        inp_data: the full array (on every process)
        axis: (optional) axis alogn which to distribute
    returns:
        a distributed jax.Array
    """
    shape = [
        1,
    ] * inp_data.ndim
    shape[axis] = -1
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(shape)
    return jax.jit(_identity, out_shardings=sharding)(inp_data)


def extract_replicated(t):
    """
    Extract the value of a fully replicated global device array.
    Args:
        t: a jax Array (or a pytree of jax Arrays)
    Returns:
        A locally adressable representation of t
    """

    def _extract_replicated(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            assert x.is_fully_replicated
            return x.addressable_data(0)
        else:
            return x

    return jax.tree_map(_extract_replicated, t)
