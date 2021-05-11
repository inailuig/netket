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

import math

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from netket.operator import Squared
from netket.stats import Stats

from netket.variational import MCMixedState

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver


class SteadyState(AbstractVariationalDriver):
    """
    Steady-state driver minimizing L^†L.
    """

    def __init__(
        self,
        lindbladian,
        optimizer,
        *args,
        variational_state=None,
        preconditioner=None,
        preconditioner_restart: bool = None,
        sr=None,
        sr_restart=False,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            lindbladian: The Lindbladian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no preconditioner
                is used and the bare gradient is passed to the optimizer.
            preconditioner_restart: Whever to use information from the last preconditioning
                to speed up the process at the following iteration.

        """
        if variational_state is None:
            variational_state = MCMixedState(*args, **kwargs)

        if sr is not None:
            if preconditioner is not None:
                raise ValueError(
                    "sr is deprecated in favour of preconditioner kwarg. You should not pass both"
                )
            else:
                preconditioner = sr
        if sr_restart is not None:
            if preconditioner_restart is not None:
                raise ValueError(
                    "sr_restart is deprecated in favour of preconditioner_restart kwarg. You should not pass both"
                )
            else:
                preconditioner_restart = sr_restart

        # default value. After deprecation move as kwarg
        if preconditioner_restart is None:
            preconditioner_restart = False

        super().__init__(variational_state, optimizer, minimized_quantity_name="LdagL")

        self._lind = lindbladian
        self._ldag_l = Squared(lindbladian)

        self.preconditioner = preconditioner
        self.preconditioner_restart = preconditioner_restart

        self._dp = None
        self._S = None
        self._sr_info = None

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._ldag_l)

        if self.sr is not None:
            self._S = self.preconditioner[0](self.state)

            # use the previous solution as an initial guess to speed up the solution of the linear system
            x0 = self._dp if self.preconditioner_restart is False else None
            self._dp, self._sr_info = self._S.solve(
                self.preconditioner[1], self._loss_grad, x0=x0
            )
        else:
            # tree_map(lambda x, y: x if is_ccomplex(y) else x.real, self._grads, self.state.parameters)
            self._dp = self._loss_grad

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_multimap(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    @property
    def ldagl(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    #    def reset(self):
    #        super().reset()

    def __repr__(self):
        return (
            "SteadyState("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Lindbladian ", self._lind),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
