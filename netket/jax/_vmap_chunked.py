from typing import Callable, Optional

import jax
from jax import custom_batching


def tree_split(mask, tree):
    lhs = jax.tree_map(lambda l, x: x if l else None, mask, tree)
    rhs = jax.tree_map(lambda l, x: None if l else x, mask, tree)
    return lhs, rhs


def tree_merge(mask, lhs_tree, rhs_tree):
    return jax.tree_map(lambda l, x_l, x_r: x_l if l else x_r, mask, lhs_tree, rhs_tree)


def batched_vmap(f, batch_size, fun_is_already_vectorised=False):

    f_orig = f
    f = custom_batching.custom_vmap(f)

    @f.def_vmap
    def rule(axis_size, in_batched, *args):
        # del axis_size

        mapped_args, bcast_args = tree_split(in_batched, list(args))

        def to_map(mapped_args):
            args = tree_merge(in_batched, mapped_args, bcast_args)
            return f_orig(*args)

        if not fun_is_already_vectorised:
            to_map = jax.vmap(to_map)

        # TODO for simplicity we do padding to the next multiple
        # TODO special case axis_size < batch_size
        # TODO split & do remainder at end
        n_batches, n_rest = divmod(axis_size, batch_size)

        if n_rest != 0:

            def _pad(x):
                pad_width = ((0, batch_size - n_rest),) + ((0, 0),) * (x.ndim - 1)
                return jax.numpy.pad(x, pad_width, mode="wrap")

            mapped_args = jax.tree_map(_pad, mapped_args)

        def _batch(x):
            return x.reshape(
                (
                    -1,
                    batch_size,
                )
                + x.shape[1:]
            )

        out_shape = jax.eval_shape(to_map, mapped_args)
        # this contains a somewhat ugly workaround to figure out which is the correct out_axis
        # since the function might add new axes at the beginnig and so we don't know which one to merge with
        # TODO any ideas?
        def _unbatch(x, expected):
            s1 = expected.shape
            s2 = x.shape[1:]
            different = np.where(np.array(s1) != np.array(s2))
            assert len(different) == 1
            axis = int(different[0]) + 1
            x = jnp.moveaxis(x, axis, 1)
            return x.reshape((-1,) + x.shape[2:])

        out = jax.tree_multimap(
            _unbatch, jax.lax.map(to_map, jax.tree_map(_batch, mapped_args)), out_shape
        )

        if n_rest != 0:
            out = jax.tree_map(lambda x: x[:axis_size], out)

        out_batched = jax.tree_map(lambda _: True, out)
        return [out], [out_batched]

    return f


def vmap_chunked(f, *args, **kwargs):
    """
    Behaves like jax.vmap but uses a custom vmap to chunk the computations in smaller chunks.
    """
    chunk_size = kwargs.pop("chunk_size")
    if chunk_size is None:
        return jax.vmap(f, *args, **kwargs)
    else:
        f = batched_vmap(f, chunk_size)
        return jax.vmap(f, *args, **kwargs)
