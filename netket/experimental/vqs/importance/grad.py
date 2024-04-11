from typing import Callable

from flax.core.scope import CollectionFilter

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import netket as nk

from netket.utils import mpi

from netket.utils.types import PyTree, Array

from netket.operator import DiscreteOperator, DiscreteJaxOperator, AbstractOperator

from netket.stats import Stats
from netket.jax import tree_conj


from .state import MCStateImportance
from .utils import mpi_tree_sum_jax, logsumexp_mpi, stats_ratio_log, mean_mpi


@jax.jit
def forces_expect_hermitian_importance(
    logeloc_fn, logw_fn, samples_q, apply_fun, variables
):
    leloc = logeloc_fn(samples_q)

    # in the following we do not yet support batch/chain dim
    samples_q = jax.lax.collapse(samples_q, 0, samples_q.ndim - 1)
    leloc = jax.lax.collapse(leloc, 0, leloc.ndim)
    Ns = samples_q.shape[0] * mpi.n_nodes

    lw = logw_fn(samples_q)  # log p/q
    nom, _ = logsumexp_mpi(leloc + lw)
    nom = nom - jnp.log(Ns)  # log ⟨Eloc * p/q⟩
    denom, _ = logsumexp_mpi(lw)
    denom = denom - jnp.log(Ns)  # log ⟨p/q⟩
    # todo block variance, chains, etc
    lE = nom - denom
    E = jnp.exp(lE)  # ⟨Eloc * p/q⟩ / ⟨p/q⟩

    _, var, err = stats_ratio_log(leloc, lw)

    eloc = jnp.exp(leloc)
    pdf = jnp.exp(lw - denom)
    w_centered = pdf * (eloc - mean_mpi(pdf * eloc)[0])
    # here we turn it back to real if necessary
    w_centered = w_centered.astype(
        jax.eval_shape(apply_fun, variables, samples_q).dtype
    )
    _, vjp_fun = jax.vjp(apply_fun, variables, samples_q)
    force_est, _ = vjp_fun(w_centered.conj() / Ns)
    force_est, _ = mpi_tree_sum_jax(force_est)

    E = Stats(E, err.real, var.real)
    return E, tree_conj(force_est["params"])


def local_value_kernel_log(logpsi: Callable, pars: PyTree, x: Array, args: PyTree):
    xp, mels = args
    logpsi_x = logpsi(pars, x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])
    logpsi_xp = logpsi(pars, xp.reshape(-1, xp.shape[-1])).reshape(xp.shape[:-1])
    # here we make it complex
    leloc, _ = logsumexp_mpi(a=logpsi_xp - logpsi_x[..., None] + 0.0j, b=mels, axis=-1)
    return leloc


def local_value_kernel_log_jax(
    logpsi: Callable, pars: PyTree, x: Array, O: DiscreteJaxOperator
):
    return local_value_kernel_log(logpsi, pars, x, O.get_conn_padded(x))


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel_importance(
    vstate: MCStateImportance, ha: DiscreteOperator
):  # noqa: F811
    return local_value_kernel_log


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel_importance_jax(
    vstate: MCStateImportance, ha: DiscreteJaxOperator
):  # noqa: F811
    return local_value_kernel_log_jax


@nk.vqs.expect_and_forces.dispatch
def expect_and_forces_importance(  # noqa: F811
    vs: MCStateImportance,
    ha: AbstractOperator,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
):
    if mutable or chunk_size is not None:
        raise NotImplementedError

    samples_q, args = nk.vqs.get_local_kernel_arguments(vs, ha)
    local_estimator_fun = nk.vqs.get_local_kernel(vs, ha)

    variables = vs.variables
    apply_fun = Partial(vs._apply_fun)
    logw_fn = vs.log_w_fun
    logeloc_fn = Partial(local_estimator_fun, apply_fun, variables, args=args)
    return forces_expect_hermitian_importance(
        logeloc_fn, logw_fn, samples_q, apply_fun, variables
    )
