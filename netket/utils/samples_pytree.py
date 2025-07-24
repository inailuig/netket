import abc
import numpy as np

import jax
from jax.interpreters import batching

from flax import struct

@struct.dataclass
class SampleWrapper():
    # how many batch dimensions; same role as n_batch in jax sparse arrays
    n_batch : int = struct.field(pytree_node=False)

    @property
    def dtype(self):
        # TODO figure out what / if to return here
        # we can't return None since that uses a few wrong code paths in netket
        return NotImplemented

    @property
    def ndim(self):
        return self.n_batch + 1

    @property
    def _batch_shape(self):
        # shape of the batch dims; assume the same for all leaves
        return jax.tree.leaves(self)[0].shape[: self.n_batch]

    @property
    def _samp_size(self):
        # sum of all sizes
        return sum(
            int(np.prod(x.shape[self.n_batch :])) for x in jax.tree.leaves(self)
        )

    @property
    def shape(self):
        return self._batch_shape + (self._samp_size,)

    def reshape(self, *shape):
        if len(shape) == 1 and hasattr(shape[0], "len"):
            (shape,) = shape
        # only support reshape of the batch dims
        assert len(shape) >= 1
        if shape[-1] == -1:
            assert np.prod(shape[:-1]) == np.prod(self.shape[:-1])
            shape = shape[:-1] + (self._samp_size,)
        else:
            assert shape[-1] == self._samp_size
        return jax.tree.map(
            lambda x: x.reshape(shape[:-1] + x.shape[self.n_batch :]), self
        )

    def swapaxes(self, axis1, axis2):
        assert axis1 < self.n_batch
        assert axis2 < self.n_batch
        return jax.tree.map(lambda x: x.swapaxes(axis1, axis2), self)

    def __getitem__(self, *idx):
        # only supports indexing of the batch dims
        dummy_leaf = jax.tree.leaves(self)[0]
        n_dim_removed = (
            dummy_leaf.ndim - jax.eval_shape(lambda x: x[*idx], dummy_leaf).ndim
        )
        assert n_dim_removed <= self.n_batch
        return jax.tree.map(lambda x: x[*idx], self)






from netket.utils.samples_pytree import SampleWrapper, SampleWrapperExample
from jax.interpreters import batching

# # vmappable handlers
# def _sw_to_elt(cont, _, val, axis):
#     if axis is None:
#         return val
#     if axis >= val.n_batch:
#         raise ValueError(f"Cannot map in_axis={axis} for SampleWrapper array with n_batch={val.n_batch}. "
#                           "in_axes for batched SampleWrapper operations must correspond to a batch dimension.")
#     return jax.tree.map(lambda v: cont(v, axis), val).replace(n_batch=val.n_batch-1)

# def _sw_from_elt(cont, axis_size, elt, axis):
#     if axis is None:
#         return elt
#     if axis > elt.n_batch:
#         raise ValueError(f"SampleWrapper: cannot add out_axis={axis} for BCOO array with n_batch={elt.n_batch}. "
#                      "SampleWrapper batch axes must be a contiguous block of leading dimensions.")

#     return jax.tree.map(lambda v: cont(axis_size, v, axis), elt).replace(n_batch=elt.n_batch+1)

def _sw_to_elt(cont, _, val, axis):
    if axis is None:
        return val
    if axis >= val.n_batch:
        raise ValueError(f"Cannot map in_axis={axis} for SampleWrapper array with n_batch={val.n_batch}.")
    return jax.tree.map(lambda v: cont(v, axis), val).replace(n_batch=val.n_batch-1)

def _sw_from_elt(cont, axis_size, elt, axis):
    if axis is None:
        return elt
    if axis > elt.n_batch:
        raise ValueError(f"SampleWrapper: cannot add out_axis={axis} for BCOO array with n_batch={elt.n_batch}.")
    return jax.tree.map(lambda v: cont(axis_size, v, axis), elt).replace(n_batch=elt.n_batch+1)


batching.register_vmappable(SampleWrapper, int, int, _sw_to_elt, _sw_from_elt, None)


# example implementation using a tuple of sub states for TensorHilbert
# (would work for any pytree of sub_states)
@struct.dataclass
class SampleWrapperExample(SampleWrapper):
    sub_states: tuple[jax.Array]  # pytree of substates

    @property
    def dtype(self):
        return jax.tree.map(lambda x: x.dtype, self.sub_states)

batching.register_vmappable(SampleWrapperExample, int, int, _sw_to_elt, _sw_from_elt, None)
