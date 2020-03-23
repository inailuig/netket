import numpy as _np
from numba import jitclass, types, typed
from numba import int64, float64
import numba


spec = [
    ('_local_states', float64[:]),
    ('_local_size', int64),
    ('_size', int64),
    ('_basis', int64[:]),
]


@jitclass(spec)
class HilbertIndex():
    def __init__(self, local_states, local_size, size):
        self._local_states = _np.sort(local_states)
        self._local_size = local_size
        self._size = size

        self._basis = _np.zeros(size, dtype=_np.int64)
        ba = 1
        for s in range(size):
            self._basis[s] = ba
            ba *= local_size

    def _local_state_number(self, x):
        return _np.searchsorted(self._local_states, x)

    @property
    def n_states(self):
        return self._local_size**self._size

    def state_to_number(self, state):
        # Converts a vector of quantum numbers into the unique integer identifier
        number = 0

        for i in range(self._size):
            number += self._local_state_number(state[self._size -
                                                     i - 1]) * self._basis[i]

        return number

    def number_to_state(self, number, out=None):
        if(out is None):
            out = _np.empty(self._size)
        else:
            assert(out.size == self._size)

        out.fill(self._local_states[0])

        ip = number
        k = self._size - 1
        while(ip > 0):
            out[k] = self._local_states[ip % self._local_size]
            ip = ip // self._local_size
            k -= 1

        return out

    def states_to_numbers(self, states, out=None):

        if(out is None):
            out = _np.empty(states.shape[0], _np.int64)
        else:
            assert(out.size == states.shape[0])

        for i in range(states.shape[0]):
            out[i] = 0
            for j in range(self._size):
                out[i] += self._local_state_number(
                    states[i, self._size - j - 1]) * self._basis[j]

        return out

    def numbers_to_states(self, numbers, out=None):
        if(out is None):
            out = _np.empty((numbers.shape[0], self._size))
        else:
            assert(out.shape == (numbers.shape[0], self._size))

        for i in range(numbers.shape[0]):
            out[i] = self.number_to_state(numbers[i])

        return out

    def all_states(self, out=None):
        numbers = _np.arange(0, self.n_states, dtype=_np.int64)
        return self.numbers_to_states(numbers, out)
