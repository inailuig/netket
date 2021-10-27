from functools import partial
from typing import Callable

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    DiscreteOperator,
    Squared,
)

from .base import expect
from .mc_state import MCState
from .mc_mixed_state import MCMixedState

from .mc_expect import _check_hilbert


def local_value_kernel_batched(logpsi, pars, σ, σp, mel, *, batch_size=None):
    """
    local_value kernel for MCState and generic operators
    """
    logpsi_batched = nkjax.vmap_batched(
        partial(logpsi, pars), in_axes=0, batch_size=batch_size
    )
    N = σ.shape[-1]

    logpsi_σ = logpsi_batched(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
    logpsi_σp = logpsi_batched(σp.reshape((-1, N))).reshape(σp.shape[:-1])

    return jnp.sum(mel * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)


def local_value_squared_kernel_batched(logpsi, pars, σ, σp, mel, *, batch_size=None):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return (
        jnp.abs(
            local_value_kernel_batched(logpsi, pars, σ, σp, mel, batch_size=batch_size)
        )
        ** 2
    )


def local_value_op_op_cost_batched(logpsi, pars, σ, σp, mel, *, batch_size=None):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σ_σp = jax.vmap(
        lambda σpi, σi: jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(
            σpi, σi
        ),
        in_axes=(0, 0),
        out_axes=0,
    )
    σ_σ = jax.vmap(lambda σi: jnp.hstack((σi, σi)), in_axes=0)(σ)

    return local_value_kernel_batched(
        logpsi, pars, σ_σ, σ_σp, mel, batch_size=batch_size
    )


# If batch_size is None, ignore it and remove it from signature
@expect.dispatch
def expect_nominibatch(vstate: MCState, operator: DiscreteOperator, batch_size: None):
    return expect(vstate, operator)


# if no implementation exists for batched, run the code unbatched
@expect.dispatch
def expect_fallback(vstate: MCState, operator: DiscreteOperator, batch_size):
    if config.FLAGS["NETKET_DEBUG"]:
        print(
            "Ignoring `batch_size={batch_size}` because no implementation supporting:"
            "batching exists."
        )

    return expect(vstate, operator)


@expect.dispatch
def expect_mcstate_ao_b(
    vstate: MCState, Ô: DiscreteOperator, batch_size: int
) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    σ = vstate.samples

    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect_minibatches(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_kernel_batched,
        batch_size,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@dispatch.multi((MCState, Squared, int), (MCMixedState, Squared, int))
def expect(vstate: MCState, Ô: Squared, batch_size: int) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    σ = vstate.samples

    σp, mels = Ô.parent.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect_minibatches(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_squared_kernel_batched,
        batch_size,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@expect.dispatch
def expect(
    vstate: MCMixedState, Ô: DiscreteOperator, batch_size: int
) -> Stats:  # noqa: F811
    _check_hilbert(vstate.diagonal, Ô)

    σ = vstate.diagonal.samples

    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect_minibatches(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_op_op_cost_batched,
        batch_size,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def _expect_minibatches(
    machine_pow: int,
    model_apply_fun: Callable,
    local_value_kernel: Callable,
    batch_size: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    local_value_vmap = partial(local_value_kernel, logpsi, batch_size=batch_size)

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, σ, σp, mels, n_chains=σ_shape[0]
    )

    return Ō_stats
