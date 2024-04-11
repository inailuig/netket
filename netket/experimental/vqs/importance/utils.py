import jax
import jax.numpy as jnp
import netket as nk

from netket.utils import mpi


def logsumexp_mpi(*args, token=None, **kwargs):
    if mpi.n_nodes > 1:
        raise NotImplementedError
    return jax.scipy.special.logsumexp(*args, **kwargs), token


def mpi_tree_sum_jax(x, *, token=None):
    if mpi.n_nodes > 1:
        raise NotImplementedError
    return x, token


def mean_mpi(x):
    assert x.ndim == 1
    return mpi.mpi_sum_jax(x.sum() / (len(x) * mpi.n_nodes))


def mean_mpi_log(logx):
    assert logx.ndim == 1
    return logsumexp_mpi(logx - jnp.log(len(logx) * mpi.n_nodes))


# TODO move this stuff to nk.stats


def stats_ratio_generic(f_X, f_Y, expect_fn):
    # <X>/<Y>
    # error prop for a ratio, same samples in both nominator and denominator
    # lets hope DCE will take care of the repeated computations

    μx = expect_fn(f_X)
    μy = expect_fn(f_Y)
    μxsq = expect_fn(nk.jax.compose(jax.lax.square, f_X))
    μysq = expect_fn(nk.jax.compose(jax.lax.square, f_Y))
    μxy = expect_fn(
        nk.jax.compose(
            lambda *args, **kwargs: f_X(*args, **kwargs) * f_Y(*args, **kwargs)
        )
    )

    def _div(x, y):
        res = x / y
        return jax.lax.select(x == 0, jnp.zeros_like(res), res)

    ev = _div(μx, μy)
    var = ev**2 * (_div(μxsq, μx**2) - 2 * _div(μxy, μx * μy) + _div(μysq, μy**2))
    return ev, var


def stats_ratio_generic_log(logf_X, logf_Y, expect_fn_log):
    # expect_fn needs to act on log-values and return the log

    μx = expect_fn_log(logf_X)
    μy = expect_fn_log(logf_Y)
    μxsq = expect_fn_log(nk.jax.compose(lambda x: 2 * x, logf_X))
    μysq = expect_fn_log(nk.jax.compose(lambda x: 2 * x, logf_Y))
    μxy = expect_fn_log(
        nk.jax.compose(
            lambda *args, **kwargs: logf_X(*args, **kwargs) + logf_Y(*args, **kwargs)
        )
    )
    ev = jnp.exp(μx - μy)
    var = ev**2 * (
        jnp.exp(μxsq - 2 * μx) - 2 * jnp.exp(μxy - (μx + μy)) + jnp.exp(μysq - 2 * μy)
    )
    return ev, var


@jax.jit
def stats_ratio(X, Y):
    # for vectors
    ev, var = stats_ratio_generic(lambda: X, lambda: Y, lambda f: mean_mpi(f())[0])
    Ns = X.shape[0] * mpi.n_nodes
    stderr = jnp.sqrt(var / Ns)
    # return Stats(mean=ev, variance=var, error_of_mean=stderr)
    return ev, var, stderr


@jax.jit
def stats_ratio_log(logX, logY):
    # for vectors
    ev, var = stats_ratio_generic_log(
        lambda: logX, lambda: logY, lambda f: mean_mpi_log(f())[0]
    )
    Ns = logX.shape[0] * mpi.n_nodes
    stderr = jnp.sqrt(var / Ns)
    # return Stats(mean=ev, variance=var, error_of_mean=stderr)
    return ev, var, stderr
