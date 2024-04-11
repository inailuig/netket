import jax
import jax.numpy as jnp

from .utils import logsumexp_mpi

@jax.jit
def importance_weight_normalized(log_w_fun, samples_q):
    samples_q = samples_q.reshape(-1, samples_q.shape[-1])
    lpqinv = log_w_fun(samples_q)  # log p/q
    denom, _ = logsumexp_mpi(lpqinv)  # <p/q>
    return jnp.exp(lpqinv - denom).reshape(samples_q.shape[:-1])
