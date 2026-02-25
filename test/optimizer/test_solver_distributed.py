import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import netket as nk

from test import common  # noqa: F401




import netket as nk
from netket import experimental as nkx
import optax

def test_ntk_disparax():
    L = 20
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
    ma = nk.models.RBM(alpha=1, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))
    vs = nk.vqs.MCState(sa, ma, n_samples=1024, n_discard_per_chain=10)
    gs = nk.driver.VMC_SR(
        ha,
        op,
        variational_state=vs,
        diag_shift=0.01,
        use_ntk=True,
        on_the_fly =True,
        _bs = 128,
    )
    gs.run(n_iter=10)
