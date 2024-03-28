import netket as nk
import jax.numpy as jnp
import jax

from netket.driver.supervised import loss_mse_log


L = 10

hi = nk.hilbert.Spin(1 / 2, L)
g = nk.graph.Chain(L)
ha = nk.operator.Ising(hi, g, 1)
sa = nk.sampler.MetropolisLocal(hi)
ma = nk.models.RBM(param_dtype=complex, alpha=4)
# TODO use exactstate?
vs = nk.vqs.MCState(sa, ma)
opt = nk.optimizer.Adam(0.01)

# generate training data
E0, psi = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)

print(f"E0={E0}")

x_train = hi.all_states()
logpsi_train = jnp.log(psi.astype(complex)).ravel()
batch_size = 128
k = jax.random.PRNGKey(123)


# use uniform=True w/ MSE
s = nk.Supervised(
    vs, loss_mse_log, x_train, logpsi_train, batch_size, k, opt, sample_uniform=True
)

s.run(1000)

print(s.estimate(ha))
