#!/usr/bin/env python3
import jax
import os


# the default of distributed.initialize is to only use one gpu
# in practice its probably best to have one process per gpu
# but to test hybrid configurations we force it to use all CUDA_VISIBLE_DEVICES here

#ldi = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES").split(",")))
jax.distributed.initialize()
#print(f"p{jax.process_index()} | local devices :", jax.local_devices())
#print(f"p{jax.process_index()} | global devices:", jax.devices())

os.environ["NETKET_EXPERIMENTAL_PJIT"] = "1"

print('dev', jax.devices())
print('localdev', jax.local_devices())

import jax
import netket as nk
import jax.numpy as jnp
from functools import partial

L = 32
n_chains_per_device = 32
n_samples_per_device = 512
n_discard = 0  # to be fair comparison we set discard to 0, as it's per chain

n_chains = n_chains_per_device * jax.device_count()
n_samples = n_samples_per_device * jax.device_count()

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

ma = nk.models.RBM(alpha=8, param_dtype=complex)
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, dtype=jnp.int8)

op = nk.optimizer.Sgd(learning_rate=0.1)
sr = nk.optimizer.SR(
    diag_shift=0.01,
    qgt=nk.optimizer.qgt.QGTOnTheFly,
    solver=partial(jax.scipy.sparse.linalg.cg, tol=0, maxiter=100),
)

print('prevs')

vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=n_discard, seed=123)

print('pregs')
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
print('prer2')
gs.run(2, show_progress=(jax.process_index() == 0))
gs.run(100, show_progress=(jax.process_index() == 0))

jax.distributed.shutdown()
