import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial
import netket as nk
import numpy as np

from netket.utils.mpi import mpi_sum_jax
from sequa.utils.stats import log_stats_importance
from netket.utils import mpi

from netket.stats import Stats
from netket.jax import tree_conj

def logsumexp_mpi(*args, token=None, **kwargs):
    if mpi.n_nodes > 1:
        raise NotImplementedError
    return jax.scipy.special.logsumexp(*args, **kwargs), token

def mpi_tree_sum_jax(x, token):
    if mpi.n_nodes > 1:
        raise NotImplementedError
    return x, token


def stats_ratio_generic(f_X, f_Y, expect_fn):
    # <X>/<Y>
    # error prop for a ratio, same samples in both nominator and denominator
    # lets hope DCE will take care of the repeated computations

    μx = expect_fn(f_X)
    μy = expect_fn(f_Y)
    μxsq = expect_fn(nk.jax.compose(jax.lax.square, f_X))
    μysq = expect_fn(nk.jax.compose(jax.lax.square, f_Y))
    μxy =  expect_fn(nk.jax.compose(lambda *args, **kwargs : f_X(*args, **kwargs)*f_Y(*args, **kwargs)))
    def _div(x,y):
        res = x/y
        return jax.lax.select(x==0, jnp.zeros_like(res), res)
    ev = _div(μx, μy)
    var = ev**2 * (_div(μxsq, μx**2) -2*_div(μxy, μx*μy)+_div(μysq, μy**2))
    return ev, var


@jax.jit
def stats_ratio(X, Y):
    # for vectors
    ev, var = stats_ratio_generic(lambda: X, lambda: Y, lambda f: jnp.mean(f()))
    Ns = X.shape[0]
    stderr = jnp.sqrt(var/Ns)
    #return Stats(mean=ev, variance=var, error_of_mean=stderr)
    return ev, var, stderr


def log_stats_importance(eloc, lw, token=None):
    # compute std error of ⟨Eloc p/q⟩_q/⟨p/q⟩_q
    # using error propagation
    # lw: log (p/q)

    Ns = eloc.shape[0] * mpi.n_nodes
    # TODO error propagation

    # we omit /Ns because it cancels anyway
    log_denom_mean, token = logsumexp_mpi(lw, token=token)
    E, token = mpi.mpi_sum_jax((eloc * jnp.exp(lw-log_denom_mean)).sum(), token=token)

    log_denom_squared_mean, token = logsumexp_mpi(2 * lw, token=token)
    var, token = mpi.mpi_sum_jax(((eloc-E)**2 * jnp.exp(2*lw-log_denom_squared_mean)).sum(), token=token)
    err = jnp.sqrt(var/Ns)

    # effective sample size
    # logESS1, token = logsumexp_mpi(lw, token=token)
    # logESS2, token = logsumexp_mpi(2 * lw, token=token)
    # logESS = 2 * logESS1 - logESS2
    # ESS = jnp.exp(logESS)

    #return (E, err, ESS), token
    return (E, err, var), token


# modified expect and grad for importance sampling #
@jax.jit
def forces_expect_hermitian_importance(logeloc_fn, logw_fn, samples_q, apply_fun, variables, token):

    # we do not yet support batch/chain dim
    assert samples_q.ndim == 2
    Ns = samples_q.shape[0] * mpi.n_nodes

    leloc = logeloc_fn(samples_q)
    lw = logw_fn(samples_q)  # log p/q
    nom, token = logsumexp_mpi(leloc + lw, token=token)
    nom = nom - jnp.log(Ns)  # log ⟨Eloc * p/q⟩
    denom, token = logsumexp_mpi(lw, token=token)
    denom = denom - jnp.log(Ns)  # log ⟨p/q⟩
    # todo block variance, chains, etc
    lE = nom - denom
    E = jnp.exp(lE)  # ⟨Eloc * p/q⟩ / ⟨p/q⟩

    # TODO use the energy computed in log_stats_importance and avoid computing it twice
    (_, err, var), token = log_stats_importance(jnp.exp(leloc), lw, token)

    logw = leloc + lw - denom
    tmp, token = logsumexp_mpi(lw, token=token)
    logw_mean, token = logsumexp_mpi(leloc + lw - tmp, token=token)
    w = jnp.exp(logw)
    w_mean = jnp.exp(logw_mean)  # TODO take just w.mean() as it shoul be the same now
    w_centered = w - w_mean  # TODO check if we need to conj w (not for real op...)
    _, vjp_fun = jax.vjp(apply_fun, variables, samples_q)
    force_est, _ = vjp_fun(jax.lax.conj(w_centered) / Ns)
    force_est, token = mpi_tree_sum_jax(force_est, token=token)
    E = Stats(E, err, var)
    return (E, tree_conj(force_est["params"])), token


def local_value_kernel_log(logpsi: Callable, pars: PyTree, x: Array, args: PyTree):
    xp, mels = args
    logpsi_x = logpsi(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])
    logpsi_xp = logpsi(xp.reshape(-1, xp.shape[-1])).reshape(xp.shape[:-1])
    token = None
    leloc, token = logsumexp_mpi(a=logpsi_xp - logpsi_x[..., None], b=mels, axis=-1, token=token)
    return leloc

def local_value_kernel_log_jax(logpsi: Callable, pars: PyTree, x: Array, O: DiscreteJaxOperator):
    return local_value_kernel_log(logpsi, pars, x, O.get_conn_padded(x))

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel_importance(vstate: MCStateImportance, ha: AbstractOperator):  # noqa: F811
    return local_value_kernel_log

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel_importance(vstate: MCStateImportance, ha: DiscreteJaxOperator):  # noqa: F811
    return local_value_kernel_log_jax

def local_estimator_fun_wrapper(local_estimator_fun, logpsi, pars, *args, x)
    return local_estimator_fun(logpsi, pars, x, *args)

@nk.vqs.expect_and_forces.dispatch
def expect_and_forces_importance(  # noqa: F811
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    x: jnp.ndarray,
    local_value_args: PyTree
    ):

    if samples_q.ndim > 2:
        samples_q = samples_q.reshape(-1, samples_q.shape[-1])

    samples_q, args = get_local_kernel_arguments(vstate, ha)
    local_estimator_fun = get_local_kernel(vstate, ha)

    variables = vs._variables
    apply_fun = Partial(vs._apply_fun)
    logpsi_fn = Partial(vs._apply_fun, vs._variables)
    logw_fn = vs.log_w_fun
    logeloc_fn = Partial(local_estimator_fun_wrapper, Partial(local_estimator_fun), apply_fun, variables, *args)
    (E, force), _ = expect_and_grad_importance(logeloc_fn, logw_fn, samples_q, apply_fun, variables, None)
    return E, force
