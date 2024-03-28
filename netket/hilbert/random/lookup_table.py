import jax

from netket.hilbert import LookupTableHilbert
from netket.utils.dispatch import dispatch

@dispatch
def random_state(hilb: LookupTableHilbert, key, batches: int, *, dtype=np.float32):
    shape = (batches,)
    i = jax.random.randint(key, shape, 0, hilb.n_states)
    return hi.numbers_to_states(i)
