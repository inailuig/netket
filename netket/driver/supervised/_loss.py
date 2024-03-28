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


from functools import wraps
import jax.numpy as jnp
from netket.stats import statistics

# This file contains some example Loss functions for the Supervised driver

def simple_loss(loss):
    """
    Decorator to make it easier to write loss functions which only depend on the
    training targets and the output of the model at the training configurations.

    Example:

        @simple_loss
        def loss(logpsi_vec, logtarget_vec):
            return jnp.linalg.norm(logpsi_vec-logtarget_vec)
    """
    @wraps(loss)
    def _loss(apply_fn, variables, x_train, logy_train, **kwargs):
        logy = apply_fn(variables, x_train)
        s = statistics(loss(logy, logy_train, **kwargs))
        return s
    return _loss


@simple_loss
def loss_mse_log(logpsi, logtarget):
    """
    Mean-squared error of the log-amplitudes of the wavefunction.
    L = 0.5 ∑ᵢ|logΨ(xᵢ)-log(yᵢ)|²
    """
    # 0.5 * (log(psi) - log(target)) * (log(psi) - log(target)).conj()
    dy = logpsi - logtarget
    res = 0.5 * dy * dy.conj()
    return res.real


def loss_log_overlap(apply_fn, variables, x_train, logy_train):
    """
    Fidelity computed in the subspace spanned by the training samples.
    L = |∑ᵢ Ψ*(xᵢ) yᵢ|² / ( ∑ᵢ|Ψ(xᵢ)|²  ∑ᵢ|yᵢ|² )
    """
    value = apply_fn(variables, x_train)
    t = jnp.exp(logy_train)

    value = jnp.exp(value - value.real.max())
    t = jnp.exp(t - t.real.max())

    num1 = (value.conj() * t).sum()
    num2 = (value * t.conj()).sum()
    num3 = (value * value.conj()).sum()
    num4 = (t * t.conj()).sum()
    complex_log_overlap = -(
        jnp.log(num1) + jnp.log(num2) - jnp.log(num3) - jnp.log(num4)
    )
    return complex_log_overlap.real
