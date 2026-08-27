"""MPS to Qiskit circuit. See README for the qubit-ordering convention."""
from __future__ import annotations
import numpy as np

from .core import (vacuum_mps, _truncate, _cut_positions, segment_sizes,
                   two_qubit_depth, PHI_MAX_DEFAULT)

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    _HAVE_QISKIT = True
except ImportError:
    _HAVE_QISKIT = False

_CHI = 2  # bond dimension the staircase is built for


def _require_qiskit():
    """Raise a helpful ImportError if qiskit is missing (the rest works without)."""
    if not _HAVE_QISKIT:
        raise ImportError(
            "build_circuit needs qiskit (`pip install qiskit`). The rest of "
            "mps_toolkit (fidelity, depth, optimal_cuts) works without it.")

def _complete_unitary(V):
    """Complete an isometry into a unitary, filling the missing columns with an
    orthonormal basis of the complement (inputs the staircase never produces)."""
    d, r = V.shape
    if r == d:
        return V
    P = np.eye(d) - V @ V.conj().T
    U, _, _ = np.linalg.svd(P)
    G = np.hstack([V, U[:, :d - r]])
    if not np.allclose(G.conj().T @ G, np.eye(d), atol=1e-9):
        raise RuntimeError("isometry completion failed (input not canonical?)")
    return G


def _staircase_gates(tensors):
    """Turn one segment's tensors into [(qa, qb, G), ...] by completing each
    isometry; qb is None for the closing 1-qubit gate, G is in basis |qa qb>."""
    n = len(tensors)
    if n < 2:
        raise ValueError(f"a segment needs at least 2 qubits, got {n}")
    for T in tensors:
        if max(T.shape[0], T.shape[2]) > _CHI:
            raise ValueError("bond dimension above 2: the staircase would not "
                             "fit in two-qubit gates")
    gates = []

    # last site: (chi_l, 2, 1) -> a two-qubit state preparation
    A = tensors[-1]
    V = np.zeros((4, 1), dtype=complex)
    for b in range(A.shape[0]):
        for s in range(2):
            V[2 * b + s, 0] = A[b, s, 0]
    gates.append((n - 2, n - 1, _complete_unitary(V)))

    # middle sites, right to left: isometry (4, chi_r)
    for k in range(n - 2, 0, -1):
        A = tensors[k]
        chi_l, _, chi_r = A.shape
        V = np.zeros((4, chi_r), dtype=complex)
        for b in range(chi_r):
            for bp in range(chi_l):
                for s in range(2):
                    V[2 * bp + s, b] = A[bp, s, b]
        gates.append((k - 1, k, _complete_unitary(V)))

    # first site: a one-qubit gate
    A = tensors[0]
    chi_r = A.shape[2]
    V1 = np.zeros((2, chi_r), dtype=complex)
    for b in range(chi_r):
        for s in range(2):
            V1[s, b] = A[0, s, b]
    gates.append((0, None, _complete_unitary(V1)))
    return gates


def _blocks(ts, cuts):
    """Split the truncated MPS at the cuts into independent segments; rank-1
    boundary bonds mean each block is already normalised and left-canonical."""
    bounds = [0] + list(cuts) + [len(ts)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        block = [T.copy() for T in ts[a:b]]
        if block[0].shape[0] != 1 or block[-1].shape[2] != 1:
            raise RuntimeError(f"segment [{a},{b}) is not decoupled "
                               "(boundary bond dimension is not 1)")
        out.append((a, block))
    return out



# public api
def build_circuit(L, m=1.0, k=1, phi_max=PHI_MAX_DEFAULT, backend="auto",
                  chi_ref=32, name=None):
    """QuantumCircuit on 2L qubits preparing the vacuum, one parallel staircase
    per segment. Same arguments as evaluate(), which predicts its fidelity."""
    _require_qiskit()
    ts_ref, _ = vacuum_mps(L, m, phi_max, backend, chi_ref)
    cuts = _cut_positions(L, k)
    ts = _truncate(ts_ref, cuts, chi_inner=_CHI)

    n = 2 * L
    qc = QuantumCircuit(n, name=name or f"vacuum_L{L}_m{m}_k{k}")
    for offset, block in _blocks(ts, cuts):
        for qa, qb, G in _staircase_gates(block):
            if qb is None:
                qc.unitary(G, [offset + qa], label="V")
            else:
                # Qiskit reads a 2-qubit matrix with the LAST qarg as the most
                # significant bit, and G is written with qa most significant.
                qc.unitary(G, [offset + qb, offset + qa], label="G")
    return qc


def circuit_fidelity(qc, L, m=1.0, phi_max=PHI_MAX_DEFAULT, backend="auto",
                     chi_ref=32):
    """Simulate the circuit and compare it to the reference vacuum, handling the
    little-endian conversion. Needs the statevector, so 2L <= ~28 qubits."""
    _require_qiskit()
    from .core import _statevector_reference
    psi = _statevector_reference(L, m, phi_max, backend, chi_ref)
    got = Statevector(qc).reverse_qargs().data
    return float(abs(np.vdot(psi, got)) ** 2)


def circuit_report(L, m=1.0, k=1, phi_max=PHI_MAX_DEFAULT, **kw):
    """build_circuit() plus its size and depth metrics, in one call."""
    qc = build_circuit(L, m, k, phi_max, **kw)
    return qc, dict(L=L, m=m, k=k,
                    qubits=2 * L,
                    twoq_depth=two_qubit_depth(L, k),
                    segment_qubits=max(segment_sizes(L, k)))
