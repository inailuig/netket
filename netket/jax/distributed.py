import math
from functools import partial, wraps
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from netket.utils import config


# @partial(jax.jit, static_argnums=0)
# def _replicate_shmap_callback(f, x):
#     # TODO where to get mesh and axes from without hardcoding them here?
#     mesh = Mesh(jax.devices(), axis_names=("i"))
#
#     @partial(shard_map, mesh=mesh, in_specs=(P("i")), out_specs=P("i"))
#     def _f(x):
#         # here we infer the output shape by doing eval_shape by hand
#         # TODO better way?
#         dummy_x = np.zeros((1,) * (x.ndim - 1) + x.shape[-1:], x.dtype)
#         dummy_xp, dummy_mels = f(dummy_x)
#         xp_shape = jax.ShapeDtypeStruct(
#             x.shape[:-1] + dummy_xp.shape[x.ndim - 1 :], dummy_xp.dtype
#         )
#         mels_shape = jax.ShapeDtypeStruct(
#             x.shape[:-1] + dummy_mels.shape[x.ndim - 1 :], dummy_mels.dtype
#         )
#         result_shape = (xp_shape, mels_shape)
#         return jax.pure_callback(f, result_shape, x, vectorized=True)
#
#     return _f(x)


# TODO eventually make this a bit more univesal
# supporting more fancy sharding and multiple input/output args
def replicate_sharding(f):
    """
    Wrapper for python get_conn_padded to make it work with shared/global device arrays.
    Calls f on every shard, and puts the results back on the devices with the correct sharding.
    The input to f is assumed to have PositionalSharding (or equivalent) along a single batch axis.

    version which uses pure_callback and jax.experimental.shard_map internally

    Args:
        f: a python get_conn_padded (which takes self, x and maps it to (xp,mels))
    """
    # ideally I would like to use a simple shard map with callback, however
    # for that we need to know the shape a priori which would require an extra n_conn call.

    if config.netket_experimental_pjit:
        @wraps(f)
        def _f(self, x):
            xp_mels_np = []
            n_conn_dev = []
            for s in x.addressable_shards:
                xp, mels = f(self, s.data)
                xp_mels_np.append((xp, mels))
                n_conn_dev.append(jax.device_put(np.array([mels.shape[-1],]), s.device))
            # numba might pad every x differently, so here we pad all to the common max over devices and all processes
            n_conn = jax.make_array_from_single_device_arrays((len(x.devices()),), jax.sharding.PositionalSharding(list(x.devices())), n_conn_dev)
            n_conn_max = int(jax.jit(lambda x: x.max())(n_conn))
            xp_dev = []
            mels_dev = []
            for (xp, mels), s in zip(xp_mels_np, x.addressable_shards):
                npad = n_conn_max - mels.shape[-1]
                if npad > 0:
                    mels = np.pad(mels, pad_width=((0,0),)*(mels.ndim-1)+((0, npad),))
                    xp = np.pad(xp, pad_width=((0,0),)*(mels.ndim-1)+((0, npad),)+((0,0),))
                    xp[..., -npad:, :] = xp[..., :1, :]
                xp_dev.append(jax.device_put(xp, s.device))
                mels_dev.append(jax.device_put(mels, s.device))
            shape = x.shape[:-1]+(n_conn_max,)
            xp = jax.make_array_from_single_device_arrays(shape+x.shape[-1:], x.sharding.reshape(x.sharding.shape[:-1]+(1,)+x.sharding.shape[-1:]), xp_dev)
            mels = jax.make_array_from_single_device_arrays(shape, x.sharding, mels_dev)
            return xp, mels
        return _f
    else:
        return f


_identity = lambda x: x


def _prepare_mask(n, n_pad):
    return jnp.ones(n + n_pad, dtype=bool).at[-n_pad:].set(0)


def put_global(inp_data, axis=0, pad=False, pad_value=None):
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
    if pad:
        n = inp_data.shape[0]
        # pad to the next multiple of device_count
        device_count = jax.device_count()
        n_pad = math.ceil(inp_data.shape[0] / device_count) * device_count - n
        inp_data = jnp.pad(inp_data, ((0, n_pad), (0, 0)))
        if pad_value is not None:
            inp_data = inp_data.at[-n_pad:].set(pad_value)

    shape = [
        1,
    ] * inp_data.ndim
    shape[axis] = -1
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(shape)
    out_data = jax.jit(_identity, out_shardings=sharding)(inp_data)
    if pad:
        if n_pad > 0:
            mask = jax.jit(
                _prepare_mask, out_shardings=sharding.reshape(-1), static_argnums=(0, 1)
            )(n, n_pad)
        else:
            mask = None
        return out_data, mask
    else:
        return out_data


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


def gather(x):
    return jax.jit(_identity, out_shardings=x.sharding.replicate())(x)


def sharding_decorator(f, sharded_argnums, reduction_op=None):
    # sharded_args: list of indices indicating that the input is sharded on axis 0, (assumed to be replicated otherwise)
    # reduction_op: function to reduce the output (assumed to be sharded otherwise), e.g. jax.lax.psum
    # only supports 1 output for now

    if config.netket_experimental_pjit:
        @wraps(f)
        def _fun(*args):
            n_args = len(args)

            # workaround for shard_map not supporting non-array args part 1/2
            nonarray_argnums = tuple(i for i, a in enumerate(args) if not hasattr(a, 'dtype') )
            for i in nonarray_argnums: assert i not in sharded_argnums
            nonarray_args = tuple(a for i,a in enumerate(args) if i in nonarray_argnums)
            args = tuple(a for i,a in enumerate(args) if i not in nonarray_argnums)

            mesh = Mesh(jax.devices(), axis_names=("i"))
            in_specs = tuple(P("i") if i in sharded_argnums else P() for i in range(n_args))
            in_specs = tuple(s for i, s in enumerate(in_specs) if i not in nonarray_argnums)
            out_specs = P("i") if reduction_op is None else P()

            _reduction = None
            if reduction_op is not None:
                _reduction = partial(jax.tree_map, partial(reduction_op, axis_name="i"))

            @partial(shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
            def _f(*args):

                # workaround for shard_map not supporting non-array args part 2/2
                it = iter(args)
                it_nonarray = iter(nonarray_args)
                args = tuple(next(it_nonarray) if i in nonarray_argnums else next(it) for i in range(n_args))

                res = f(*args)
                if _reduction is not None:
                    res = _reduction(res)
                return res
            return _f(*args)
        return _fun

    return f
