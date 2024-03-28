# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial, wraps

import jax
import jax.numpy as jnp

from netket.jax import tree_conj
from netket.stats import Stats, statistics
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils.types import PyTree, Optimizer


from .abstract_variational_driver import AbstractVariationalDriver


@partial(jax.jit, static_argnames=("N", "unique", "log", "uniform"), inline=True)
def _get_samples_jax(rng_key, X, Y, N, *, unique=False, uniform=False, log=True):
    if uniform:
        i = jax.random.randint(rng_key, (N,), 0, len(X))
    else:
        if log:
            Y_ = jnp.exp(Y)
        else:
            Y_ = Y
        pdf = (Y_ * Y_.conj()).real
        # normalize
        pdf = pdf / pdf.sum()
        i = jax.random.choice(rng_key, len(pdf), (N,), p=pdf, replace=not unique)
    return X[i], Y[i]


@partial(jax.jit, static_argnames=("loss", "apply_fun"), inline=True)
def _loss_value_and_grad(loss, apply_fun, params, model_state, x_train, y_train):
    def _loss(p):
        l = loss(apply_fun, {"params": p, **model_state}, x_train, y_train)
        if isinstance(l, Stats):
            return l.mean.real, l
        else:
            return l.real, l

    (_, val), grad = jax.value_and_grad(_loss, has_aux=True)(params)
    return val, tree_conj(grad)


sa = ExactSampler()

def _prepare_target_state(samples, targets, batch_size, uniform=False):
    if batch_size is None:
        raise ValueError('batch_size is required')
    hi = LookupTableHilbert(samples)
    #samples_sorted = hi.all_states()
    targets_sorted = samples[jnp.argsort(hi.states_to_numbers(samples))]
    if uniform:
        
    else:
        sa = nk.sampler.ExactSampler(hi)
    ma = nk.models.LogStateVector(hi)
    var = {'params': {'logstate': targets_sorted}}
    return  MCState(sa, ma, variables=var, n_samples=batch_size)

class Supervised(AbstractVariationalDriver):
    """
    Supervised learning scheme to learn data.
    """

    def __init__(
        self,
        variational_state : MCState,
        loss : Callable,
        optimizer : Optimizer,
        samples: Array = None,
        targets : Array = None,
        batch_size : int = None,
        *args,
        preconditioner: PreconditionerT = identity_preconditioner,
        target_state : MCState = None,
        sample_uniform: bool = False,
        seed: Optional[SeedT] = None,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            loss: A loss function # TODO
            ... TODO
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

        # TODO deduct name from the loss function or allow to specify custom one?
        super().__init__(variational_state, optimizer, minimized_quantity_name="Loss")

        if target_state is None:
            target_state = _prepare_target_state(samples, targets)


        self.preconditioner = preconditioner

        self._dp = None
        self._S = None
        self._sr_info = None
        self._loss_fn = loss
        self._samples = samples
        self._targets = targets
        self._sample_uniform = sample_uniform
        self._batch_size = batch_size
        self._rng_key = rng_key

    def _forward_and_backward(self):
        """
        Performs a number of Supervised optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        k, self._rng_key = jax.random.split(self._rng_key)
        x_train, logy_train = _get_samples_jax(
            k,
            self._samples,
            self._targets,
            self._batch_size,
            uniform=self._sample_uniform,
            log=True,
        )

        self._loss_stats, self._loss_grad = _loss_value_and_grad(
            self._loss_fn,
            self.state._apply_fun,
            self.state.parameters,
            self.state.model_state,
            x_train,
            logy_train,
        )

        # TODO less hacky way of doing SR?
        self.state._samples = jnp.expand_dims(x_train, 0)
        self._dp = self.preconditioner(self.state, self._loss_grad)
        return self._dp

    @property
    def loss(self) -> Stats:
        """
        Return MCMC statistics for the expectation loss in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "Supervised("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
