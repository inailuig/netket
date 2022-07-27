import math
import jax
import numpy as np
import netket as nk

from netket.optimizer import identity_preconditioner

from netket.driver import AbstractVariationalDriver
from ._Fest_Gest import Fest_and_Gest_nk_vs_batched

from ._batch_utils import *


class Fidopt(AbstractVariationalDriver):
    def __init__(
        self,
        target_variational_state,
        variational_state,
        optimizer,
        sr=None,
        logfid=False,
        fidbatchsize=None,
        nonzero_wavefun=True,
        apply_fun_is_logwf=True,
        reuse_target_samples=False,
        F_and_G_fun=Fest_and_Gest_nk_vs_batched,
    ):

        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Fidelity"
        )

        self._target_variational_state = target_variational_state

        if sr is not None:
            self.preconditioner = sr  # this calls the setter
        else:
            self.preconditioner = identity_preconditioner

        self._fidbatchsize = fidbatchsize
        self._logfid = logfid  # compute the gradient of the log of the fideiltiy (NB: grad F = F grad log F), instead of the gradient of the fidelity
        self._nonzero_wavefun = nonzero_wavefun  # calculate the gradient by only sampling from the target wavefunction, which is only correct without zeros (uses formula from 2009.01760)
        self._apply_fun_is_logwf = apply_fun_is_logwf  # True: variational_state and  target_variational_state are logpsi; False: they are psi
        self._reuse_target_samples = (
            reuse_target_samples  # don't sample from target_variational_state again
        )
        self._phi_samplesphi = None
        self._F_and_G_fun = F_and_G_fun

    def _forward_and_backward(self):
        self.state.reset()  #  triggers new sample generation
        if not self._reuse_target_samples:
            self._target_variational_state.reset()
        else:
            if (
                self._phi_samplesphi is None
                or self._target_variational_state._samples is None
            ):
                self._phi_samplesphi = batch(
                    self._target_variational_state.log_value(
                        unbatch(self._target_variational_state.samples)
                    ),
                    self._target_variational_state.samples.shape[1],
                )

        # this accesses variational_state.samples which we just generated
        F, G = self._F_and_G_fun(
            self._target_variational_state,
            self.state,
            self._logfid,
            self._nonzero_wavefun,
            self._apply_fun_is_logwf,
            self._fidbatchsize,
            self._phi_samplesphi,
        )

        # TODO error propatation for the product
        self._loss_stats = nk.stats.Stats(F.real, np.nan, np.nan, np.nan, np.nan)
        self._loss_grad = jax.tree_map(
            jax.lax.neg, G
        )  # negative because its a maximization

        self._dp = self.preconditioner(self.state, self._loss_grad)

        return self._dp

    def info(self):
        pass
