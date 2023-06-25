import jax
from functools import partial, wraps
import numpy as np

_identity = lambda x: x

def put_global(inp_data):
    # TODO rename
    # each rank has the whole thing; parts not belonging to it can be filled with garbage
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape((-1,)+(1,)*(inp_data.ndim-1))
    return jax.jit(_identity, out_shardings=sharding)(inp_data)


# TODO put it somewhere
def _multimap(f, *args):
    try:
        return tuple(map(lambda a: f(*a), zip(*args)))
    except TypeError:
        return f(*args)

def _make_array(old_shape, old_sharding, xs):
    xs = list(xs)
    # assumes all xs have same shape in all axes which are not shared
    is_shared = tuple(a>1 for a in old_sharding.shape)
    x0 = xs[0]
    def _reshape(t, fill_value):
        # extend/shorten the tuple to x0.ndim
        return t[:x0.ndim] + (fill_value,)*(x0.ndim-len(t))
    old_shape = _reshape(old_shape, None)
    old_sharding_shape = _reshape(old_sharding.shape, None)
    is_shared = _reshape(is_shared, False)
    new_shape = _multimap(lambda c, t1, t2: t1 if c else t2, is_shared, old_shape, x0.shape)
    new_sharding_shape = _multimap(lambda c, t: t if c else 1, is_shared, old_sharding_shape)
    new_sharding = old_sharding.reshape(new_sharding_shape)
    return jax.make_array_from_single_device_arrays(new_shape, new_sharding, xs)

# TODO use jax._tree_transpose and is_leaf in the tree map instead
class _fake_list(list): pass # not a leave

def _tree_transpose(list_of_trees):
    return jax.tree_map(lambda *xs: _fake_list(xs), *list_of_trees)

def _f(f, x):
    if isinstance(x, jax.Array) and not isinstance(x.sharding, jax.sharding.SingleDeviceSharding):
        # here we make a list so that below we can use tuple to find the leaves
        y = _tree_transpose([jax.device_put(f(s.data), s.data.device()) for s in x.addressable_shards])
        return jax.tree_map(partial(_make_array, x.shape, x.sharding), y)
    else:
        return f(x)


# do it by hand

def replicate_sharding1(f):
    # wrapper for a python function to act on a jax.Array, putting back the output with the infered sharding
    # assumes only a single argument
    # assumes the function acts element-wise on all shared axes (those with sharding.shape > 1)
    # assumes no axes are inserted or deleted before the last shared axis
    # does not yet support pytrees / multiple arguments for the input, but does support it for the output
    return partial(_f, f)

def replicate_sharding_cls(f):
    @wraps(f)
    def __f(self, x):
        return partial(_f, partial(f, self))(x)
    return __f



# simpler alternative using shmap and callback
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

@partial(jax.jit, static_argnums=0)
def test(f, x):
    # TODO how to get mesh and axes?
    mesh = Mesh(jax.devices(), axis_names=('i'))
    @partial(shard_map, mesh=mesh, in_specs=(P('i')), out_specs=P('i'))
    def _test(x):
        # here we do eval shape by hand
        # TODO better way?
        dummy_x = np.zeros((1,)*(x.ndim-1)+x.shape[-1:], x.dtype)
        dummy_xp, dummy_mels = f(dummy_x)
        xp_shape = jax.ShapeDtypeStruct(x.shape[:-1]+dummy_xp.shape[x.ndim-1:], dummy_xp.dtype)
        mels_shape = jax.ShapeDtypeStruct(x.shape[:-1]+dummy_mels.shape[x.ndim-1:], dummy_mels.dtype)
        result_shape = (xp_shape, mels_shape)
        return jax.pure_callback(f, result_shape, x, vectorized=True)

    return _test(x)

def replicate_sharding_shmap(f):
    return partial(test, f)

replicate_sharding = replicate_sharding_shmap

def _extract_replicated(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        assert x.is_fully_replicated
        return x.addressable_data(0)
    else:
        return x

def extract_replicated(t):
    return jax.tree_map(_extract_replicated, t)
