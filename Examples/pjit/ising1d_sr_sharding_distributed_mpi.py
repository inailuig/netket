import jax
jax.config.update('jax_threefry_partitionable', True)
import netket as nk
import jax.numpy as jnp
from functools import partial


L = 40
n_chains_per_device = 512
#n_samples = 64*1024
#n_samples = 8192*mpi
n_discard = 0 # to be fair comparison we set discard to 0, as it's per chain
# TODO later increase Ns and chains beyond what cuda can handle in paralell, and turn back on discard

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=8, param_dtype=complex)
sa2 = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=n_chains_per_device)
op = nk.optimizer.Sgd(learning_rate=0.1)
sr = nk.optimizer.SR(diag_shift=0.01, qgt=nk.optimizer.qgt.QGTOnTheFly, solver=partial(jax.scipy.sparse.linalg.cg, tol=0, maxiter=100))
srp = nk.optimizer.SR(diag_shift=0.01, qgt=partial(nk.optimizer.qgt.QGTJacobianPyTree, holomorphic=True), solver=partial(jax.scipy.sparse.linalg.cg, tol=0, maxiter=100))

vs = nk.vqs.MCState(sa2, ma, n_samples_per_rank=8192, n_discard_per_chain=n_discard)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=srp)
gs.run(2)
gs.run(100)
