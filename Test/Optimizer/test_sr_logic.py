# test sr_onthefly_logic with inhomogeneous parameters

import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from jax.scipy.sparse.linalg import cg
from netket.optimizer.sr import sr_onthefly_logic, sr_treemv_logic
from functools import partial
import itertools

import pytest

from .. import common

pytestmark = common.skipif_mpi

# TODO move the transformation and tree utils out of the test


# transform inhomogeneous parameters into a flattened array of reals parameters
# without needing to pad


def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat


def toreal(x):
    if jnp.iscomplexobj(x):
        return jnp.array([x.real, x.imag])
    else:
        return x


def tree_toreal(x):
    return jax.tree_map(toreal, x)


def tree_toreal_flat(x):
    return flatten(tree_toreal(x))


# invert the transformation tree_toreal_flat using linear_transpose (AD)
def reassemble_complex(x, target, fun=tree_toreal_flat):
    # target: a tree with the expected shape and types of the result
    (res,) = jax.linear_transpose(fun, target)(x)
    res = sr_onthefly_logic.tree_conj(res)
    # fix the dtypes:
    return sr_onthefly_logic.tree_cast(res, target)


def tree_allclose(t1, t2):
    t = jax.tree_multimap(jnp.allclose, t1, t2)
    return all(jax.tree_util.tree_flatten(t)[0])


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_multimap(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


class Example:
    def f_real_flat(self, p, samples):
        return self.f(reassemble_complex(p, target=self.params), samples)

    def f_real_flat_scalar(self, params, x):
        return self.f_real_flat(params, jnp.expand_dims(x, 0))[0]

    # same as in nk.machine.Jax R->C
    @partial(jax.vmap, in_axes=(None, None, 0))
    def grads_real(self, params, x):
        r = jax.grad(lambda pars, v: self.f_real_flat_scalar(pars, v).real)(params, x)
        i = jax.grad(lambda pars, v: self.f_real_flat_scalar(pars, v).imag)(params, x)
        return jax.lax.complex(r, i)

    @property
    def n_samp(self):
        return len(self.samples)

    def __init__(self, n_samp, seed, outdtype, pardtype, holomorphic):

        self.dtype = outdtype

        self.target = {
            "a": jnp.array([[[0j], [0j]]], dtype=jnp.complex128),
            "b": jnp.array(0, dtype=jnp.float64),
            "c": jnp.array(0j, dtype=jnp.complex64),
        }

        if pardtype is None:  # mixed precision as above
            pass
        else:
            self.target = jax.tree_map(lambda x: x.astype(pardtype), self.target)

        k = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)

        self.samples = jax.random.normal(k1, (n_samp, 2))
        self.w = jax.random.normal(k2, (n_samp,), self.dtype).astype(
            self.dtype
        )  # TODO remove astype once its fixed in jax
        self.params = tree_random_normal_like(k3, self.target)
        self.v = tree_random_normal_like(k4, self.target)
        self.grad = tree_random_normal_like(k5, self.target)

        if holomorphic:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return (
                    params["a"][0][0][0] * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"] + 1j)
                    * params["c"]
                ).astype(self.dtype)

        else:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return (
                    params["a"][0][0][0].conjugate() * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"].conjugate() + 1j)
                    * params["c"].conjugate()
                ).astype(self.dtype)

        self.f = f

        self.params_real_flat = tree_toreal_flat(self.params)
        self.grad_real_flat = tree_toreal_flat(self.grad)
        self.v_real_flat = tree_toreal_flat(self.v)
        self.ok_real = self.grads_real(self.params_real_flat, self.samples)
        self.okmean_real = self.ok_real.mean(axis=0)
        self.dok_real = self.ok_real - self.okmean_real
        self.S_real = (
            self.dok_real.conjugate().transpose() @ self.dok_real / n_samp
        ).real


@pytest.fixture
def e(n_samp, outdtype, pardtype, holomorphic, seed=123):
    return Example(n_samp, seed, outdtype, pardtype, holomorphic)


rt = [jnp.float32, jnp.float64]
ct = [jnp.complex64, jnp.complex128]
nct = [None] + ct  # None means inhomogeneous
r_r_test_types = list(itertools.product(rt, rt))
c_c_test_types = list(itertools.product(ct, ct))
r_c_test_types = list(itertools.product(ct, rt))
rc_c_test_types = list(itertools.product(ct, [None]))
# c_r_test_types = list(itertools.product(rt, ct))
test_types = r_r_test_types + c_c_test_types + r_c_test_types + rc_c_test_types

# tests


@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [0])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_reassemble_complex(e):
    assert tree_allclose(
        e.params, reassemble_complex(tree_toreal_flat(e.params), target=e.target)
    )


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_vjp(e):
    actual = sr_onthefly_logic.O_vjp(e.f, e.params, e.samples, e.w)
    expected = sr_onthefly_logic.tree_conj(
        reassemble_complex(
            (e.w @ e.ok_real).real.astype(e.params_real_flat.dtype), target=e.target
        )
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", r_r_test_types + c_c_test_types)
def test_mean(e):
    actual = sr_onthefly_logic.O_mean(e.f, e.params, e.samples)
    expected = sr_onthefly_logic.tree_conj(
        reassemble_complex(e.okmean_real.real, target=e.target)
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_OH_w(e):
    actual = sr_onthefly_logic.OH_w(
        e.f,
        e.params,
        e.samples,
        e.w,
    )
    expected = reassemble_complex(
        (e.ok_real.conjugate().transpose() @ e.w).real.astype(e.params_real_flat.dtype),
        target=e.target,
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_jvp(e):
    actual = sr_onthefly_logic.O_jvp(e.f, e.params, e.samples, e.v)
    expected = e.ok_real @ e.v_real_flat
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_Odagger_O_v(e):
    actual = sr_onthefly_logic.Odagger_O_v(e.f, e.params, e.samples, e.v)
    expected = reassemble_complex(
        (e.ok_real.conjugate().transpose() @ e.ok_real @ e.v_real_flat).real / e.n_samp,
        target=e.target,
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_Odagger_DeltaO_v(e):
    actual = sr_onthefly_logic.Odagger_DeltaO_v(e.f, e.params, e.samples, e.v)
    expected = reassemble_complex(e.S_real @ e.v_real_flat, target=e.target)
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_DeltaOdagger_DeltaO_v(e, holomorphic):
    actual = sr_onthefly_logic.DeltaOdagger_DeltaO_v(
        e.f, e.params, e.samples, e.v, holomorphic
    )
    expected = reassemble_complex(e.S_real @ e.v_real_flat, target=e.target)
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp, batchsize", [(25, None), (25, 5), (1024, 128)])
@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_matvec(e, centered, jit, holomorphic, batchsize):
    diag_shift = 0.01
    mv = sr_onthefly_logic.mat_vec
    if jit:
        mv = jax.jit(mv, static_argnums=(1, 5, 6, 7))
    actual = mv(
        e.v, e.f, e.params, e.samples, diag_shift, centered, holomorphic, batchsize
    )
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp, batchsize", [(25, None), (25, 5), (1024, 128)])
@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_matvec_linear_transpose(e, centered, jit, holomorphic, batchsize):
    def mvt(v, params, samples, w):
        mv = lambda v_: sr_onthefly_logic.mat_vec(
            v_, e.f, params, samples, 0.0, centered, holomorphic, batchsize
        )
        (res,) = jax.linear_transpose(mv, v)(w)
        return res

    if jit:
        mvt = jax.jit(mvt)

    w = e.v
    actual = mvt(e.v, e.params, e.samples, w)

    # use that S is hermitian:
    # S^T = (O^H O)^T = O^T O* = (O^H O)* = S*
    # S^T w = S* w = (S w*)*
    expected = sr_onthefly_logic.tree_conj(
        sr_onthefly_logic.mat_vec(
            sr_onthefly_logic.tree_conj(w), e.f, e.params, e.samples, 0.0, centered
        )
    )
    # (expected,) = jax.linear_transpose(lambda v_: reassemble_complex(S_real @ tree_toreal_flat(v_), target=e.target), v)(v)
    assert tree_allclose(actual, expected)


# TODO separate test for prepare_doks
@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [25, 1024])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize(
    "outdtype, pardtype", r_r_test_types + c_c_test_types + r_c_test_types
)
def test_matvec_treemv(e, jit, holomorphic):
    diag_shift = 0.01
    mv = sr_treemv_logic.mat_vec
    pdoks = sr_treemv_logic.prepare_doks
    if jit:
        mv = jax.jit(mv)
        pdoks = jax.jit(pdoks, static_argnums=0)

    doks = pdoks(e.f, e.params, e.samples)
    actual = mv(e.v, doks, diag_shift)
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert tree_allclose(actual, expected)


# TODO test with MPI
