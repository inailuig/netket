import jax
import jax.numpy as jnp
from flax import struct
from functools import partial, reduce, singledispatchmethod
from typing import Any, Sequence, Callable, Collection, Union

from functools import reduce
from operator import mul

import math

PyTree = Any
# Scalar = Union[float, int, complex]


@struct.dataclass
class PyTreeArrayT:
    tree: PyTree
    treedef_l: Any = struct.field(pytree_node=False)
    treedef_r: Any = struct.field(pytree_node=False)
    axes_l: Any = struct.field(pytree_node=False)
    axes_r: Any = struct.field(pytree_node=False)

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        # TODO also transpose the arrays / move tensor axes
        t_trans = self.tree_trans()
        is_leaf = lambda l: jax.tree_structure(l) == self.treedef_l

        def mva(x, n):
            axes = tuple(range(jnp.ndim(x)))
            return jnp.transpose(x, axes[n:] + axes[:n])

        t_trans = jax.tree_map(
            lambda l: jax.tree_multimap(mva, l, self.axes_l), t_trans, is_leaf=is_leaf
        )
        return PyTreeArrayT(
            t_trans, self.treedef_r, self.treedef_l, self.axes_r, self.axes_l
        )

    def tree_trans(self):
        # just transpose the treedef
        return jax.tree_transpose(self.treedef_l, self.treedef_r, self.tree)

    # TODO @singledispatchmethod
    def __add__(self, t: PyTree):
        if jnp.isscalar(t):
            res = jax.tree_map(lambda x: x + t, self.tree)
            return self.replace(tree=res)
        elif isinstance(t, PyTreeArrayT):
            return self + t.tree
        else:  # PyTree
            assert jax.tree_structure(self.tree) == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.add, self.tree, t)
            return self.replace(tree=res)

    def __rmul__(self, t):
        return self * t

    def __radd__(self, t):
        return self + t

    def __rsub__(self, t):
        return (-self) + t

    def __neg__(self):
        return (-1) * self

    def _elementwise(self, f):
        return self.replace(tree=jax.tree_map(f, self.tree))

    def __mul__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x * t)
        elif isinstance(t, PyTreeArrayT):
            # TODO check equal treedef_l and treedef_r, axes
            return self * t.tree
        else:  # PyTree
            assert jax.tree_structure(self.tree) == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.multiply, self.tree, t)
            return self.replace(tree=res)

    def __truediv__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x / t)
        elif isinstance(t, PyTreeArrayT):
            # TODO check equal treedef_l and treedef_r, axes
            return self / t.tree
        else:  # PyTree
            assert jax.tree_structure(self.tree) == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.divide, self.tree, t)
            return self.replace(tree=res)

    def __sub__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x - t)
        elif isinstance(t, PyTreeArrayT):
            return self - t.tree
        else:  # PyTree
            assert jax.tree_structure(self.tree) == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.subtract, self.tree, t)
            return self.replace(tree=res)

    def __pow__(self, t):
        assert jnp.isscalar(t)
        return self._elementwise(lambda x: x ** t)

    def __getitem__(self, *args, **kwargs):
        return self.tree.__getitem__(*args, **kwargs)

    def __matmul__(pt1, pt2):
        if not isinstance(pt2, PyTreeArrayT):
            # assume its a pytree vector
            pt2 = PyTreeArray(pt2)
        assert pt1.treedef_r == pt2.treedef_l
        # assert pt1.axes_r == pt2.axes_l  # TODO
        def tree_dot(t1, t2, axes_tree):
            res = jax.tree_util.tree_reduce(
                jax.lax.add, jax.tree_multimap(jnp.tensordot, t1, t2, axes_tree)
            )
            return res

        is_leaf = lambda l: jax.tree_structure(l) == pt1.treedef_r
        tree = jax.tree_map(
            lambda t1: jax.tree_map(
                lambda t2: tree_dot(t1, t2, pt1.axes_r),
                pt2.tree_trans(),
                is_leaf=is_leaf,
            ),
            pt1.tree,
            is_leaf=is_leaf,
        )
        return PyTreeArrayT(tree, pt1.treedef_l, pt2.treedef_r, pt1.axes_l, pt2.axes_r)

    def conjugate(self):
        return self._elementwise(jnp.conj)

    def conj(self):
        return self.conjugate()

    @property
    def imag(self):
        return self._elementwise(jnp.imag)

    @property
    def real(self):
        return self._elementwise(jnp.real)

    @property
    def H(self):
        return self.T.conj()

    def _l_map(self, f):
        return jax.tree_map(
            f, self.tree, is_leaf=lambda x: jax.tree_structure(x) == self.treedef_r
        )

    def _lr_map(self, f):
        return self._l_map(lambda r: jax.tree_map(f, r))

    def _lr_amap(self, f):
        return self._l_map(lambda r: jax.tree_multimap(f, r, self.axes_r))

    def _flatten_tensors(self):
        def rs(x, ndim_r):
            _prod = lambda x: (reduce(mul, x, 1),)
            s = x.shape
            ndim_l = x.ndim - ndim_r
            sl = _prod(s[:ndim_l])
            sr = _prod(s[ndim_l:])
            return x.reshape(sl + sr)

        t_flat = self._lr_amap(rs)
        _set1 = lambda x: jax.tree_map(lambda _: 1, x)
        return self.replace(
            tree=t_flat, axes_l=_set1(self.axes_l), axes_r=_set1(self.axes_r)
        )

    def to_dense(self):
        self_flat = self._flatten_tensors()
        tree_dense_r = self_flat._l_map(
            lambda r: jax.vmap(
                lambda x: jax.flatten_util.ravel_pytree(x)[0], in_axes=0, out_axes=0
            )(r)
        )
        tree_dense_lr = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0], in_axes=1, out_axes=1
        )(tree_dense_r)
        return tree_dense_lr

    def add_diag_scalar(self, a):
        assert self.treedef_l == self.treedef_r
        nl = self.treedef_l.num_leaves

        def _is_diag(i):
            return i % (nl + 1) == 0

        def _tree_map_diag(f, tree, is_diag):
            leaves, treedef = jax.tree_flatten(tree)
            return treedef.unflatten(
                f(l) if is_diag(i) else l for i, l in enumerate(leaves)
            )

        def _add_diag_tensor(x, a):
            # TODO simpler ?
            s = x.shape
            n = x.ndim
            assert n % 2 == 0
            assert s[: n // 2] == s[n // 2 :]
            sl = s[: n // 2]
            _prod = lambda x: reduce(mul, x, 1)
            il = jnp.unravel_index(jnp.arange(_prod(sl)), sl)
            i = il + il
            return jax.ops.index_add(x, i, a)

        tree = _tree_map_diag(partial(_add_diag_tensor, a=a), self.tree, _is_diag)
        return self.replace(tree=tree)

    def sum(self, axis=0, keepdims=None):
        # for vectors only for now
        assert self.treedef_l == _arr_treedef
        tree = jax.tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims), self.tree)
        if keepdims:
            n_ax = 0
        else:
            n_ax = 1 if isinstance(axis, int) else len(axis)
        axes_l = self.axes_l - n_ax
        return self.replace(tree=tree, axes_l=axes_l)

    def astype(self, dtype_tree):
        if isinstance(dtype_tree, PyTreeArrayT):
            dtype_tree = dtype_tree.tree
        tree = jax.tree_multimap(lambda x, y: x.astype(y.dtype), self.tree, dtype_tree)
        return self.replace(tree=tree)


_arr_treedef = jax.tree_structure(jnp.zeros(0))  # TODO proper way to get * ??

# for a vector
def PyTreeArray(t):
    treedef_l = jax.tree_structure(t)
    treedef_r = _arr_treedef
    axes_l = jax.tree_map(jnp.ndim, t)
    axes_r = 0
    return PyTreeArrayT(t, treedef_l, treedef_r, axes_l, axes_r)


# for the oks
def PyTreeArray2(t):
    treedef_l = _arr_treedef
    treedef_r = jax.tree_structure(t)
    axes_l = 1
    axes_r = jax.tree_map(lambda x: x - axes_l, jax.tree_map(jnp.ndim, t))
    return PyTreeArrayT(t, treedef_l, treedef_r, axes_l, axes_r)


def tree_allclose(t1, t2):
    return jax.tree_structure(t1) == jax.tree_structure(
        t2
    ) and jax.tree_util.tree_reduce(
        lambda x, y: x and y, jax.tree_multimap(jnp.allclose, t1, t2)
    )


# TODO eye_like / lazy add to diagonal
# TODO ignore flax FrozenDict in treedef comparison
# TODO ndim attr if treedefs are both * to emulate array
# TODO make it work with the solvers, like FrozenDict does; why doesnt pytree_node=False work???
