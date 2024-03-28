from netket.hilbert import DiscreteHilbert
from netket.hilbert.index import LookupTableHilbertIndex

class LookupTableHilbert(DiscreteHilbert):
    """
    Hilbert space represented by a list of states.
    """
    def __init__(self, all_states):
        self._hilbert_index = LookupTableHilbertIndex(all_states)

    @property
    def size(self):
        return self._hilbert_index.all_states().shape[-1]

    @property
    def _attrs(self):
        return (self._hilbert_index,)

    def _numbers_to_states(self, numbers: np.ndarray) -> np.ndarray:
        return self._hilbert_index.numbers_to_states(numbers)

    def _states_to_numbers(self, states: np.ndarray):
        return self._hilbert_index.states_to_numbers(states)

    def all_states(self) -> np.ndarray:
        return self._hilbert_index.all_states()

    @property
    def is_finite(self):
        return True

    @property
    def is_indexable(self):
        return True

    @property
    def n_states(self):
        return self._hilbert_index.n_states
