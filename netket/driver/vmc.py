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

from netket.stats import Stats
from netket.variational import MCState
from netket.utils import warn_deprecation

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver


class VMC(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    # TODO docstring
    def __init__(
        self,
        hamiltonian,
        optimizer,
        *args,
        variational_state=None,
        preconditioner=None,
        preconditioner_restart: bool = None,
        sr=None,
        sr_restart: bool = None,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
            preconditioner_restart: Whever to use information from the last preconditioning
                to speed up the process at the following iteration.

        """
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

        if variational_state.hilbert != hamiltonian.hilbert:
            raise TypeError(
                f"""the variational_state has hilbert space {variational_state.hilbert} 
                                (this is normally defined by the hilbert space in the sampler), but
                                the hamiltonian has hilbert space {hamiltonian.hilbert}. 
                                The two should match."""
            )

        if sr is not None:
            if preconditioner is not None:
                raise ValueError(
                    "sr is deprecated in favour of preconditioner kwarg. You should not pass both"
                )
            else:
                preconditioner = sr
                warn_deprecation(
                    (
                        "The `sr` keyword argument is deprecated in favour of `preconditioner`."
                        "Please update your code to `VMC(.., precondioner=your_sr)`"
                    )
                )
        if sr_restart is not None:
            if preconditioner_restart is not None:
                raise ValueError(
                    "sr_restart is deprecated in favour of preconditioner_restart kwarg. You should not pass both"
                )
            else:
                preconditioner_restart = sr_restart
                warn_deprecation(
                    (
                        "The `sr_restart` keyword argument is deprecated in favour of `preconditioner_restart`."
                        "Please update your code to `VMC(.., preconditioner_restart=True/False)`"
                    )
                )

        # default value. After deprecation move as kwarg
        if preconditioner_restart is None:
            preconditioner_restart = False

        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._ham = hamiltonian.collect()  # type: AbstractOperator

        self.preconditioner = preconditioner  # type: SR
        self.preconditioner_restart = preconditioner_restart

        self._dp = None  # type: PyTree
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
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._ham)

        if self.preconditioner is not None:
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
    def energy(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "Vmc("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
                ("State       ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
