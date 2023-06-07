import jax
jax.config.update('jax_threefry_partitionable', True)
import netket as nk


L = 32
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
#ma = nk.models.RBM(alpha=8, param_dtype=complex)
ma = nk.models.GCNN(g, features=8, layers=4, param_dtype=complex, mode='fft')
sa = nk.sampler.MetropolisLocal(hi, n_chains=16 * 8)
op = nk.optimizer.Sgd(learning_rate=0.1)

sr = nk.optimizer.SR(diag_shift=0.01, qgt=nk.optimizer.qgt.QGTOnTheFly)


vs = nk.vqs.MCState(sa, ma, n_samples=2048, n_discard_per_chain=0)
sharding = jax.sharding.PositionalSharding(jax.devices())
vs.sampler_state = vs.sampler_state.replace(σ=jax.device_put(vs.sampler_state.σ, sharding.reshape(-1, 1)))
# 
# x = vs.samples
# p = vs.variables
# f = vs._apply_fun
# lowered = jax.jit(jax.vmap(f, in_axes=(None, 0))).lower(p, x)
# compiled = lowered.compile()
# ca = compiled.cost_analysis()
# intensity = ca[0]['flops'] / ca[0]['bytes accessed']
# print('Comp. intensity fwd pass:', intensity, 'flops/byte')
#


# S = vs.quantum_geometric_tensor(partial(nk.optimizer.qgt.QGTJacobianPyTree, holomorphic=True))
# S = vs.quantum_geometric_tensor(nk.optimizer.qgt.QGTOnTheFly)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
#gs.run(n_iter=300, out=None)
