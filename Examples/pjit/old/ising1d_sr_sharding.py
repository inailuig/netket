# #%env XLA_PYTHON_CLIENT_MEM_FRACTION=.25
# %env JAX_LOG_COMPILES=1

# +
import jax
jax.config.update('jax_threefry_partitionable', True)
import netket as nk

import netket.experimental
from functools import partial

# from jax.config import config
# config.update("jax_enable_x64", False)
# del config
# -


jax.devices()

# +
L = 32

n_chains_per_device = 512

n_discard = 0 # to be fair comparison we set discard to 0, as it's per chain
# TODO later increase Ns and chains beyond what cuda can handle in paralell, and turn back on discard

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=8, param_dtype=complex)
#ma = nk.models.GCNN(g, features=8, layers=4, param_dtype=complex, mode='fft')
sa1 = nk.sampler.MetropolisLocal(hi, n_chains=n_chains_per_device)
sa2 = nk.sampler.MetropolisLocal(hi, n_chains=n_chains_per_device*jax.local_device_count())

op = nk.optimizer.Sgd(learning_rate=0.1)
# -

sr = nk.optimizer.SR(diag_shift=0.01, qgt=nk.optimizer.qgt.QGTOnTheFly)
srp = nk.optimizer.SR(diag_shift=0.01, qgt=partial(nk.optimizer.qgt.QGTJacobianPyTree, holomorphic=True))


# we create 2 vstates:
# - vs1 using 1 gpu
# - vs2 using 2 local gpus
#
# note that so far this most likely only works with local devices, as we don't sync the PRNG for the non pjit stuff yet

# +
vs1 = nk.vqs.MCState(sa1, ma, n_samples=8192, n_discard_per_chain=n_discard)
sampler_state1 =  vs1.sampler_state

sharding = jax.sharding.PositionalSharding(jax.devices())
vs2 = nk.vqs.MCState(sa2, ma, n_samples=8192, n_discard_per_chain=n_discard)
sampler_state2 = vs2.sampler_state.replace(σ=jax.device_put(vs2.sampler_state.σ, sharding.reshape(-1, 1)))
vs2.sampler_state = sampler_state2
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
# -




# ### benchmark the sampler

from netket.sampler.metropolis import MetropolisLocal
from functools import partial


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(sampler, machine, parameters, state, chain_length):
    state, samples = jax.lax.scan(
        lambda state, _: sampler.sample_next(machine, parameters, state),
        state,
        xs=None,
        length=chain_length,
    )

    return samples, state



x1 = jax.block_until_ready(_sample_chain(sa1, ma, vs1.variables, sampler_state1, 128))

x2 = jax.block_until_ready(_sample_chain(sa2, ma, vs2.variables, sampler_state2, 128/jax.local_device_count()))

# %timeit _ = jax.block_until_ready(_sample_chain(sa1, ma, vs1.variables, sampler_state1, 128))

# %timeit _ = jax.block_until_ready(_sample_chain(sa2, ma, vs2.variables, sampler_state2, 128/jax.local_device_count()))

sap = netket.experimental.sampler.MetropolisSamplerPmap(hi, nk.sampler.rules.LocalRule(), n_chains=n_chains_per_device*jax.local_device_count())

sampler_statep = sap.init_state(ma, vs1.variables)

xp = jax.block_until_ready(_sample_chain(sap, ma, vs1.variables, sampler_statep, 128/jax.local_device_count()))

# %timeit _ = jax.block_until_ready(_sample_chain(sap, ma, vs1.variables, sampler_statep, 128/jax.local_device_count()))

x1[0].shape, x2[0].shape, xp[0].shape

x1[0].sharding, x2[0].sharding, xp[0].sharding

x2[0].sharding.shape

x1 = jax.block_until_ready(vs1.sample())
x2 = jax.block_until_ready(vs2.sample())
x1 = jax.block_until_ready(vs1.sample())
x2 = jax.block_until_ready(vs2.sample())

# %timeit jax.block_until_ready(vs1.sample())

# %timeit jax.block_until_ready(vs2.sample())

# ### benchmark gradients (includes numba operator on the cpu)

eg1 = jax.block_until_ready(vs1.expect_and_grad(ha))
eg2 = jax.block_until_ready(vs2.expect_and_grad(ha))

# %timeit eg1 = jax.block_until_ready(vs1.expect_and_grad(ha))

# %timeit eg2 = jax.block_until_ready(vs2.expect_and_grad(ha))

# ### benchmark SR

S1 = vs1.quantum_geometric_tensor(sr.qgt_constructor)
S1p = vs1.quantum_geometric_tensor(srp.qgt_constructor)
S2 = vs2.quantum_geometric_tensor(sr.qgt_constructor)
S2p = vs2.quantum_geometric_tensor(srp.qgt_constructor)

jax.tree_util.tree_leaves(S1p.O)[1].sharding

jax.tree_util.tree_leaves(S2p.O)[1].sharding.shape

jax.tree_util.tree_leaves(S1._mat_vec)[0].sharding

jax.tree_util.tree_leaves(S2._mat_vec)[0].sharding.shape


# #### otf

@jax.jit
def mv(S, v):
    return S@v


_ = jax.block_until_ready(mv(S1, vs1.parameters))
_ = jax.block_until_ready(mv(S1p, vs1.parameters))

# %timeit jax.block_until_ready(mv(S1, vs1.parameters))

# %timeit jax.block_until_ready(mv(S2, vs2.parameters))

jvp_fn, = S1._mat_vec.args

# +
from netket.optimizer.qgt.qgt_onthefly_logic import *

def _O_jvp(forward_fn, params, samples, v):
    _, res = jax.jvp(lambda p: forward_fn(p, samples), (params,), (v,))
    return res


def _O_vjp(forward_fn, params, samples, w):
    _, vjp_fun = jax.vjp(forward_fn, params, samples)
    res, _ = vjp_fun(w)
    return res

def _OH_w(forward_fn, params, samples, w):
    return tree_conj(_O_vjp(forward_fn, params, samples, w.conjugate()))


def _Odagger_DeltaO_v(forward_fn, params, samples, v):
    w = _O_jvp(forward_fn, params, samples, v)
    w = w * (1.0 / (samples.shape[0] * samples.shape[1] * mpi.n_nodes))
    #w_mean = w.sum(axis=(0,1), keepdims=True) / (samples.shape[0] * samples.shape[1] * mpi.n_nodes)
    #w_mean, _ = mpi.mpi_sum_jax(w_mean)
    #w = w - w_mean
    res = _OH_w(forward_fn, params, samples, w)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI



# -

@partial(jax.jit, static_argnums=0)
def mv(forward_fn, params, samples, v, diag_shift):
    f = lambda p, x: jax.vmap(lambda x: forward_fn({'params': p},x))(x)
    res = _Odagger_DeltaO_v(f, params, samples, v)
    return tree_axpy(diag_shift, v, res)


y1 = jax.block_until_ready(mv(vs1._apply_fun, vs1.parameters, vs1.samples, vs1.parameters, 0.))
y2 = jax.block_until_ready(mv(vs2._apply_fun, vs2.parameters, vs2.samples, vs2.parameters, 0.))


# %timeit _ = jax.block_until_ready(mv(vs1._apply_fun, vs1.parameters, vs1.samples, vs1.parameters, 0.))

# %timeit _ = jax.block_until_ready(mv(vs2._apply_fun, vs2.parameters, vs2.samples, vs2.parameters, 0.))

# - otf is much slower; figure out why

# #### pytree

@jax.jit
def mv(S, v):
    return S@v


_  =jax.block_until_ready(mv(S1p, vs1.parameters))
_  =jax.block_until_ready(mv(S2p, vs2.parameters))

# %timeit jax.block_until_ready(mv(S1p, vs1.parameters))

# %timeit jax.block_until_ready(mv(S2p, vs2.parameters))

# - pytree is faster

# ### vmc
#
# #### pytree

gs1 = nk.VMC(ha, op, variational_state=vs1, preconditioner=srp)
gs2 = nk.VMC(ha, op, variational_state=vs2, preconditioner=srp)

gs1.run(2)
gs2.run(2)

gs1.run(100)

gs2.run(100)

1.17 * 1.60 # speedup


