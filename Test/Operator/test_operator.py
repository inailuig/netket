import netket as nk
import networkx as nx
import numpy as np

import pytest
import jax

operators = {}

# Ising 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
operators["Ising 1D"] = nk.operator.Ising(hi, g, h=1.321)

# Heisenberg 1D
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi, graph=g)

# Bose Hubbard
g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
hi = nk.hilbert.Boson(n_max=3, n_bosons=6, N=g.n_nodes)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi, graph=g)

# Graph Hamiltonian
N = 20
sigmax = np.asarray([[0, 1], [1, 0]])
mszsz = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
edges = [[i, i + 1] for i in range(N - 1)] + [[N - 1, 0]]

g = nk.graph.Graph(edges=edges)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian"] = nk.operator.GraphOperator(
    hi, g, site_ops=[sigmax], bond_ops=[mszsz]
)

# Graph Hamiltonian with colored edges
edges_c = [(i, j, i % 2) for i, j in edges]
g = nk.graph.Graph(edges=edges_c)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
operators["Graph Hamiltonian (colored edges)"] = nk.operator.GraphOperator(
    hi,
    g,
    site_ops=[sigmax],
    bond_ops=[1.0 * mszsz, 2.0 * mszsz],
    bond_ops_colors=[0, 1],
)

# Custom Hamiltonian
sx = [[0, 1], [1, 0]]
sy = [[0, -1.0j], [1.0j, 0]]
sz = [[1, 0], [0, -1]]
g = nk.graph.Graph(edges=[[i, i + 1] for i in range(20)])
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)


def _loc(*args):
    return nk.operator.LocalOperator(hi, *args)


sx_hat = _loc([sx] * 3, [[0], [1], [5]])
sy_hat = _loc([sy] * 4, [[2], [3], [4], [9]])
szsz_hat = _loc(sz, [0]) @ _loc(sz, [1])
szsz_hat += _loc(sz, [4]) @ _loc(sz, [5])
szsz_hat += _loc(sz, [6]) @ _loc(sz, [8])
szsz_hat += _loc(sz, [7]) @ _loc(sz, [0])

operators["Custom Hamiltonian"] = sx_hat + sy_hat + szsz_hat
operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + (2.0 * sy_hat)

operators["Pauli Hamiltonian"] = nk.operator.PauliStrings(
    ["XX", "YZ", "IZ"], [0.1, 0.2, -1.4]
)


op_special = {}
for name, op in operators.items():
    if hasattr(op, "to_local_operator"):
        op_special[name] = op


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_produce_elements_in_hilbert(op):
    hi = op.hilbert
    assert len(hi.local_states) == hi.local_size
    assert hi.size > 0
    rstate = np.zeros(hi.size)

    local_states = hi.local_states

    max_conn_size = op.max_conn_size

    for i in range(1000):
        hi.random_state(out=rstate)

        rstatet, mels = op.get_conn(rstate)

        assert np.all(np.isin(rstatet, local_states))
        assert len(mels) <= max_conn_size


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_is_hermitean(op):
    hi = op.hilbert
    assert len(hi.local_states) == hi.local_size

    rstate = np.zeros(hi.size)

    local_states = hi.local_states

    for i in range(100):
        hi.random_state(out=rstate)
        rstatet, mels = op.get_conn(rstate)

        for k, state in enumerate(rstatet):

            invstates, mels1 = op.get_conn(state)

            found = False
            for kp, invstate in enumerate(invstates):
                if np.array_equal(rstate, invstate.flatten()):
                    found = True
                    assert mels1[kp] == np.conj(mels[k])
                    break

            assert found


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_repr(op):
    assert type(op).__name__ in repr(op)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_get_conn_numpy_closure(op):
    hi = op.hilbert
    closure = op._get_conn_flattened_closure()
    # work around Numba#6980
    v = np.asarray(hi.random_state(jax.random.PRNGKey(0), 120))
    conn = np.empty(v.shape[0], dtype=np.intp)
    conn2 = np.empty(v.shape[0], dtype=np.intp)

    vp, mels = closure(v, conn)
    vp2, mels2 = op.get_conn_flattened(v, conn, pad=False)

    np.testing.assert_equal(vp, vp2)
    np.testing.assert_equal(mels, mels2)


@pytest.mark.parametrize(
    "op", [pytest.param(op, id=name) for name, op in op_special.items()]
)
def test_to_local_operator(op):
    op_local = op.to_local_operator()
    # TODO check dense representaiton.


def test_no_segfault():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

    lo = nk.operator.LocalOperator(hi, [[1, 0], [0, 1]], [0])
    lo = lo.transpose()

    hi = None

    lo = lo @ lo

    assert True


def test_deduced_hilbert_pauli():
    op = nk.operator.PauliStrings(["XXI", "YZX", "IZX"], [0.1, 0.2, -1.4])
    assert op.hilbert.size == 3
    assert len(op.hilbert.local_states) == 2
    assert np.allclose(op.hilbert.local_states, (0, 1))


def test_Heisenberg():
    g = nk.graph.Hypercube(8, 1)
    hi = nk.hilbert.Spin(0.5) ** 8

    def gs_energy(ham):
        return nk.exact.lanczos_ed(ham)

    ha1 = nk.operator.Heisenberg(hi, graph=g)
    ha2 = nk.operator.Heisenberg(hi, graph=g, J=2.0)

    assert 2 * gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    ha1 = nk.operator.Heisenberg(hi, graph=g, sign_rule=True)
    ha2 = nk.operator.Heisenberg(hi, graph=g, sign_rule=False)

    assert gs_energy(ha1) == pytest.approx(gs_energy(ha2))

    with pytest.raises(
        ValueError, match=r"sign_rule=True specified for a non-bipartite lattice"
    ):
        g = nk.graph.Hypercube(7, 1)
        hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

        assert not g.is_bipartite()

        ha = nk.operator.Heisenberg(hi, graph=g, sign_rule=True)


def test_pauli():
    op = nk.operator.PauliStrings(["XX", "YZ", "IZ"], [0.1, 0.2, -1.4])

    op_l = (
        0.1
        * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmax(op.hilbert, 1)
    )
    op_l += (
        0.2
        * nk.operator.spin.sigmay(op.hilbert, 0)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
    )
    op_l -= 1.4 * nk.operator.spin.sigmaz(op.hilbert, 1)

    assert np.allclose(op.to_dense(), op_l.to_dense())

    assert op.to_sparse().shape == op_l.to_sparse().shape
