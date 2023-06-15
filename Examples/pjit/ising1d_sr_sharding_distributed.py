import jax
jax.config.update('jax_threefry_partitionable', True)
import netket as nk
import jax.numpy as jnp
from functools import partial


# TODO call with args if needed
# jax.distributed.initialize()

L = 32
n_chains_per_device = 512
n_chains = n_chains_per_device*jax.device_count()
n_local_chains = n_chains_per_device*jax.local_device_count()
n_samples = 16384
n_discard = 0 # to be fair comparison we set discard to 0, as it's per chain
# TODO later increase Ns and chains beyond what cuda can handle in paralell, and turn back on discard



def put_global1(local_array):
    # each rank has its own part
    # TODO avoid copying if local_array is already shared
    local_shape = local_array.shape
    global_shape = (local_shape[0]*jax.process_count(),)+ local_shape[1:]
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape((-1,)+(1,)*(local_array.ndim-1))
    arrays = jax.device_put(jnp.split(local_array, jax.local_device_count(), axis = 0), jax.local_devices())
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

def put_global2(inp_data):
    # each rank has the whole thing; parts not belonging to it can be filled with garbage
    # TODO avoid copying if local_array is already shared
    global_shape = inp_data.shape
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape((-1,)+(1,)*(global_array.ndim-1))
    arrays = [jax.device_put(inp_data[index], d) for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
ma = nk.models.RBM(alpha=8, param_dtype=complex)
# TODO iirc netket should already divide by the correct Ns everywhere, as the shape is global; check!
sa2 = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
op = nk.optimizer.Sgd(learning_rate=0.1)
sr = nk.optimizer.SR(diag_shift=0.01, qgt=nk.optimizer.qgt.QGTOnTheFly)
srp = nk.optimizer.SR(diag_shift=0.01, qgt=partial(nk.optimizer.qgt.QGTJacobianPyTree, holomorphic=True))

# we divide by jax.process_count() so that nk determines the correct chain length so that overall we get desired Ns
# this is necessary as nk sees only the chains per rank; TODO fix it eventually

# we set the chain length as it is for sure correct; eventually use netket
vs = nk.vqs.MCState(sa2, ma, n_samples=n_chains, n_discard_per_chain=n_discard)
vs.chain_length = n_samples // n_chains

# TODO here we assume every rank has different seed; split manually?
vs.sampler_state = vs.sampler_state.replace(σ=put_global2(vs.sampler_state.σ))
# broadcast params from root
vs.parameters = jax.experimental.multihost_utils.broadcast_one_to_all(vs.parameters)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=srp)
gs.run(2)
gs.run(100)

