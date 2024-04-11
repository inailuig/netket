import matplotlib.pyplot as plt
from jax.tree_util import Partial
import netket as nk
from netket.experimental.vqs.importance import MCStateImportance
from netket.utils import HashablePartial

L = 8
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=1, use_visible_bias=False, param_dtype=float)
sa = nk.sampler.ExactSampler(hi)
op = nk.optimizer.Sgd(learning_rate=0.1)
sr = nk.optimizer.SR(diag_shift=0.1, qgt=nk.optimizer.qgt.QGTJacobianPyTree())
vs = nk.vqs.MCState(sa, ma, n_samples=32 * 1024)
p0 = vs.parameters
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
l = nk.logging.RuntimeLog()

gs.run(n_iter=100, out=l)


def q(logpsi, params, x):
    # sample from sqrt(|Ψ|)
    return 0.5 * logpsi(params, x).real


# def q(logpsi, params, x):
#     # sample from |Ψ|^2
#     return 2 * logpsi(params, x).real

# def q(logpsi, params, x):
#     # sample unif
#     return jnp.zeros_like(jax.eval_shape(logpsi, params, x))


class MCStateImportanceTest(MCStateImportance):
    @property
    def log_q_fun(self):
        return Partial(HashablePartial(q, self.model.apply), self.variables)


l2 = nk.logging.RuntimeLog()

sa_importance = nk.sampler.ExactSampler(hi, machine_pow=1)
vs_importance = MCStateImportanceTest(sa_importance, ma, n_samples=32 * 1024)
vs_importance.parameters = p0
gs_importance = nk.VMC(ha, op, variational_state=vs_importance, preconditioner=sr)

gs_importance.run(100, out=l2)


plt.plot(l["Energy"]["Mean"])
plt.plot(l2["Energy"]["Mean"])
plt.show()
