import netket as nk
import numpy as np
import jax
from jax.experimental.optimizers import sgd as JaxSgd
import cProfile

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)


# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

ha = nk.operator.Ising(h=1.0, hilbert=hi)

alpha = 1
ma = nk.machine.JaxRbm(hi, alpha, dtype=complex)
ma.init_random_parameters(seed=1232)

# Jax Sampler
# use same key so that we get the same samples
rng_key = jax.random.PRNGKey(123)
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=2, rng_key=rng_key)
sa2 = nk.sampler.MetropolisLocal(machine=ma, n_chains=2, rng_key=rng_key)

# Using Sgd
op = nk.optimizer.Sgd(ma, 0.01)


# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma, diag_shift=0.1)

# Create the optimization driver
gs = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, sr=sr, n_discard=0
)

gs2 = nk.Vmc(
    hamiltonian=ha, sampler=sa2, optimizer=op, n_samples=1000, sr=sr, n_discard=0,
    sronthefly=True
)

# The first iteration is slower because of start-up jit times
gs.run(out="test", n_iter=2)
gs2.run(out="test", n_iter=2)

gs.run(out="test", n_iter=300)
gs2.run(out="test", n_iter=300)
