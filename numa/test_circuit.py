import pytest

from mps_toolkit.core import evaluate, two_qubit_depth, _cut_positions, _HAVE_QUIMB
from mps_toolkit.circuit import _HAVE_QISKIT

pytestmark = pytest.mark.skipif(not _HAVE_QISKIT, reason="qiskit not installed")

if _HAVE_QISKIT:
    from qiskit import transpile
    from mps_toolkit.circuit import build_circuit, circuit_fidelity

needs_quimb = pytest.mark.skipif(not _HAVE_QUIMB, reason="quimb not installed")


@pytest.mark.parametrize("L,k,m,phi_max", [
    (3, 1, 1.0, 1.5),
    (4, 2, 1.0, 1.5),
    (6, 3, 1.0, 1.5),
    (4, 1, 2.0, 1.5),
    (4, 1, 1.0, 1.0),
])
def test_circuit_fidelity_matches_the_prediction(L, k, m, phi_max):
    """The test that closes the loop, from tensors to a runnable circuit. A
    qubit-ordering mistake produces a perfectly valid circuit that prepares the
    wrong state, and nothing else in the package would notice."""
    qc = build_circuit(L=L, m=m, k=k, phi_max=phi_max)
    predicted = evaluate(L=L, m=m, k=k, phi_max=phi_max)["fidelity"]
    assert circuit_fidelity(qc, L=L, m=m, phi_max=phi_max) \
        == pytest.approx(predicted, abs=1e-9)


@pytest.mark.parametrize("L,k", [(6, 1), (6, 3)])
def test_transpiled_cost_matches_the_metrics(L, k):
    """Confronts the advertised metrics with what Qiskit actually produces
    after KAK decomposition, rather than trusting the formulas."""
    qc = build_circuit(L=L, m=1.0, k=k)
    tq = transpile(qc, basis_gates=["cx", "u"], optimization_level=1,
                   initial_layout=list(range(2 * L)))
    assert tq.count_ops().get("cx", 0) == evaluate(L=L, m=1.0, k=k)["n_cnot"]
    assert tq.depth(lambda i: i.operation.num_qubits == 2) == two_qubit_depth(L, k)


@pytest.mark.parametrize("L,k", [(6, 3), (10, 5)])
def test_no_gate_crosses_a_cut(L, k):
    """Structural guarantee behind the parallelism: one gate straddling a cut
    and the depth stops being divided."""
    qc = build_circuit(L=L, m=1.0, k=k)
    bounds = [0] + list(_cut_positions(L, k)) + [2 * L]
    seg_of = {q: s for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))
              for q in range(a, b)}
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        assert len({seg_of[q] for q in qs}) == 1, \
            f"gate {inst.operation.name} on {qs} crosses a segment boundary"


@needs_quimb
def test_flagship_configuration_L50():
    """The circuit quoted in the report: 100 qubits, 282 CNOT, depth 51."""
    qc = build_circuit(L=50, m=1.0, k=6)
    assert qc.num_qubits == 100
    tq = transpile(qc, basis_gates=["cx", "u"], optimization_level=1,
                   initial_layout=list(range(100)))
    assert tq.count_ops().get("cx", 0) == 282
    assert tq.depth(lambda i: i.operation.num_qubits == 2) == 51


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
