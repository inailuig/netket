from netket.operator import AbstractOperator
import abc

class JaxOperator(AbstractOperator):
    def get_conn_padded(self, x):
        return self.get_get_conn_padded_closure()(x)

    @abc.abstractmethod
    def get_get_conn_padded_closure(self):
        pass
