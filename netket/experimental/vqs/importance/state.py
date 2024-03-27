import netket as nk
import jax
import jax.numpy as jnp

from .grad import expect_and_grad
from jax.tree_util import Partial

from .utils import logsumexp2

from netket.vqs.mc.mc_state.state import compute_chain_length
from netket.vqs.mc.common import force_to_grad as _force_to_grad

#@jax.jit
def fake_model(variables, x):
    logpsi = variables["logpsi"]
    return logpsi(x)

@jax.jit
def default_logp_fun(logpsi_fn, machine_pow, x):
    # p = |Psi|^machine_pow
    return machine_pow * logpsi_fn(x).real

@jax.jit
def default_logw_fun(logq_fun, logp_fn, x):
    # w = p/q
    return logp_fn(x) - logq_fun(x)

@jax.jit
def default_logq_fun(logw_fun, logp_fn, x):
    # q = p/w
    return logp_fn(x) - logw_fun(x)


class MCStateImportance(nk.vqs.MCState):
    def __init__(self, *args, machine_pow=2, **kwargs):
        super().__init__(*args, **kwargs)
        # machine pow of the model (used for p)
        self._machine_pow = machine_pow
        # make sure machine pow for q is 1
        assert self.sampler.machine_pow == 1

    @property
    def log_p_fun(self):
        logpsi_fn = Partial(self._apply_fun, self.variables)
        return Partial(default_logp_fun, logpsi_fn, self._machine_pow)

    @property
    def log_q_fun(self):
        # implement / override this according to the prob you want to sample from
        # self.variables are overridden, use self._variables instead if needed
        # you need to return a jax.tree_util.Partial
        raise NotImplementedError

    # @property
    # def log_q_fun(self):
    #     # optionally override this, e.g. when q contains p which will cancel
    #     logp_fn = self.log_p_fun
    #     logw_fn = self.log_w_fun
    #     return Partial(default_logq_fun, logw_fn, logp_fn)

    @property
    def log_w_fun(self):
        # optionally override this, e.g. when q contains p which will cancel
        logp_fn = self.log_p_fun
        logq_fn = self.log_q_fun
        return Partial(default_logw_fun, logq_fn, logp_fn)

    # we override the model and variables which are the only quantities used for sampling
    @property
    def model(self):
        return fake_model

    @property
    def variables(self):
        # params are not used but apparently we need to pass them for flax not to complain
        fake_var = {"logpsi": self.log_q_fun, "params": None}
        return fake_var

    # fix users of self.variables 1/2
    def log_value(self, x):
        return jit_evaluate(self._apply_fun, self._variables, x)

    # fix users of self.variables 2/2
    def to_array(self, normalize=True):
        return nn.to_array(
            self.hilbert,
            self._apply_fun,
            self._variables,
            normalize=normalize,
            chunk_size=self.chunk_size,
        )
