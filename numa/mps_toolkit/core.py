from __future__ import annotations
import numpy as np
from functools import reduce
from scipy.sparse import kron as skron, eye as seye, csr_matrix
from scipy.sparse.linalg import eigsh

# quimb is only needed for the DMRG backend (large L)
try:
    import quimb.tensor as qtn
    _HAVE_QUIMB = True
except ImportError:
    _HAVE_QUIMB = False

PHI_MAX_DEFAULT = 1.5
J1 = J3 = 0.5
EXACT_MAX_L = 8  # above this, the exact backend is too slow so we switch to DMRG


def _local_ops(phi_max):
    """Single-site phi, pi^2 and phi^2 on 4 field values."""
    N = 4
    delta = 2 * phi_max / 3
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    PHI = np.diag([phi_max - k * delta for k in range(N)])
    PI = F.conj().T @ np.diag(
        [k * np.pi / (4 * delta) for k in [3, 1, -1, -3]]) @ F
    return PHI, PI @ PI, PHI @ PHI


def build_H_sparse(L, m=1.0, phi_max=PHI_MAX_DEFAULT):
    """Sparse (CSR) Hamiltonian of the L-site scalar field, mass m."""
    J2 = 0.5 * m ** 2
    PHI, PI2, PHI2 = _local_ops(phi_max)
    PI2_s, PHI2_s, PHI_s = csr_matrix(PI2), csr_matrix(PHI2), csr_matrix(PHI)
    ID = seye(4, format="csr")

    def embed(op, j):
        ops = [ID] * L; ops[j] = op
        return reduce(lambda a, b: skron(a, b, format="csr"), ops)

    def embed2(oj, ok, j, k):
        ops = [ID] * L; ops[j] = oj; ops[k] = ok
        return reduce(lambda a, b: skron(a, b, format="csr"), ops)

    H = J1 * sum(embed(PI2_s, j) for j in range(L))
    H += J2 * sum(embed(PHI2_s, j) for j in range(L))
    for j in range(L - 1):
        H += J3 * embed(PHI2_s, j) + J3 * embed(PHI2_s, j + 1)
        H -= 2 * J3 * embed2(PHI_s, PHI_s, j, j + 1)
    return H



def _mps_from_statevector(psi, n):
    """Exact left-canonical MPS (list of (chi_l, 2, chi_r) tensors) from a
    full statevector of n qubits."""
    tensors, rest, chi_l = [], psi.reshape(1, -1).astype(complex), 1
    for _ in range(n - 1):
        U, S, Vd = np.linalg.svd(rest.reshape(chi_l * 2, -1), full_matrices=False)
        keep = np.sum(S > 1e-14)
        tensors.append(U[:, :keep].reshape(chi_l, 2, keep))
        rest = np.diag(S[:keep]) @ Vd[:keep]
        chi_l = keep
    tensors.append(rest.reshape(chi_l, 2, 1))
    return tensors


def _vacuum_exact(L, m, phi_max):
    """Vacuum by sparse diagonalisation, then split into a qubit-level MPS."""
    H = build_H_sparse(L, m, phi_max)
    val, vec = eigsh(H, k=1, which="SA")
    psi = vec[:, 0].astype(complex)
    psi /= np.linalg.norm(psi)
    # split each 4-dim site into two qubits
    return _mps_from_statevector(psi, 2 * L), float(val[0])


# DMRG for large L > EXACT_MAX_L
def _build_mpo(L, m, phi_max):
    """Build the MPO for the Hamiltonian of the L-site scalar field, mass m."""
    PHI, PI2, PHI2 = _local_ops(phi_max)
    J2 = 0.5 * m ** 2
    Id = np.eye(4)

    def A(j):
        nb = (j > 0) + (j < L - 1)      # bonds touching site j
        return J1 * PI2 + (J2 + nb * J3) * PHI2

    arrays = []
    W0 = np.zeros((3, 4, 4), dtype=complex)
    W0[0], W0[1], W0[2] = A(0), -2 * J3 * PHI, Id
    arrays.append(W0)
    for j in range(1, L - 1):
        W = np.zeros((3, 3, 4, 4), dtype=complex)
        W[0, 0] = Id; W[1, 0] = PHI
        W[2, 0], W[2, 1], W[2, 2] = A(j), -2 * J3 * PHI, Id
        arrays.append(W)
    WL = np.zeros((3, 4, 4), dtype=complex)
    WL[0], WL[1], WL[2] = Id, PHI, A(L - 1)
    arrays.append(WL)
    arrays = [np.swapaxes(a, -2, -1) for a in arrays]   # physical-index convention
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def _quimb_to_list(psi):
    """Convert a quimb MPS to a list of numpy arrays, each with shape (Dl, d, Dr)."""
    out = []
    for i in range(psi.L):
        t = psi[i]; phys = psi.site_ind(i); order = []
        if i > 0:
            (left,) = set(t.inds) & set(psi[i - 1].inds); order.append(left)
        order.append(phys)
        if i < psi.L - 1:
            (right,) = set(t.inds) & set(psi[i + 1].inds); order.append(right)
        data = np.asarray(t.transpose(*order, inplace=False).data)
        if i == 0: data = data[None, :, :]
        if i == psi.L - 1: data = data[:, :, None]
        out.append(data)
    return out


def _split_sites_to_qubits(ts4):
    """Each site tensor (Dl,4,Dr) -> two qubit tensors (Dl,2,x)(x,2,Dr)."""
    out = []
    for T in ts4:
        Dl, _, Dr = T.shape
        M = T.reshape(Dl, 2, 2, Dr).reshape(Dl * 2, 2 * Dr)
        U, S, Vd = np.linalg.svd(M, full_matrices=False)
        keep = np.sum(S > 1e-14)
        out.append(U[:, :keep].reshape(Dl, 2, keep))
        out.append((np.diag(S[:keep]) @ Vd[:keep]).reshape(keep, 2, Dr))
    return out


def _vacuum_dmrg(L, m, phi_max, chi_ref=32):
    """Vacuum by DMRG on the Hamiltonian MPO, for sizes exact diag cannot reach."""
    if L < 2:
        raise ValueError("DMRG needs at least 2 sites; use backend='exact' for L=1.")
    if not _HAVE_QUIMB:
        raise ImportError("The DMRG backend needs quimb (`pip install quimb`).")
    mpo = _build_mpo(L, m, phi_max)
    dmrg = qtn.DMRG2(mpo, bond_dims=[4, 8, 16, chi_ref], cutoffs=1e-12)
    dmrg.solve(tol=1e-10, verbosity=0)
    return _split_sites_to_qubits(_quimb_to_list(dmrg.state)), float(np.real(dmrg.energy))


def vacuum_mps(L, m=1.0, phi_max=PHI_MAX_DEFAULT, backend="auto", chi_ref=32):
    """Vacuum as a list of qubit tensors, plus its energy.
    backend "auto" picks exact diagonalisation up to L=8, DMRG beyond."""
    if backend == "auto":
        backend = "exact" if L <= EXACT_MAX_L else "dmrg"
    if backend == "exact":
        return _vacuum_exact(L, m, phi_max)
    if backend == "dmrg":
        return _vacuum_dmrg(L, m, phi_max, chi_ref)
    raise ValueError(f"unknown backend {backend!r}")



def _right_canonicalize(ts):
    """Right-canonicalise a copy of the MPS by sweeping SVDs from the right."""
    ts = [t.copy() for t in ts]
    for k in range(len(ts) - 1, 0, -1):
        chi_l, d, chi_r = ts[k].shape
        U, S, Vd = np.linalg.svd(ts[k].reshape(chi_l, d * chi_r), full_matrices=False)
        ts[k] = Vd.reshape(-1, d, chi_r)
        ts[k - 1] = np.einsum("abc,cd->abd", ts[k - 1], U @ np.diag(S))
    ts[0] /= np.linalg.norm(ts[0])
    return ts


def _truncate(ts, cut_qubits, chi_inner=2):
    """Left-to-right sweep: bond truncated to rank 1 at each cut (segment
    boundary), to chi_inner elsewhere. Returns a left-canonical MPS."""
    ts = _right_canonicalize(ts)
    cutset = set(cut_qubits)
    out, carry = [], np.eye(ts[0].shape[0], dtype=complex)
    for k, T in enumerate(ts):
        T = np.einsum("ab,bsc->asc", carry, T)
        chi_l, d, chi_r = T.shape
        if k == len(ts) - 1:
            out.append(T / np.linalg.norm(T)); break
        chi = 1 if (k + 1) in cutset else chi_inner
        U, S, Vd = np.linalg.svd(T.reshape(chi_l * d, chi_r), full_matrices=False)
        keep = min(chi, np.sum(S > 1e-14))
        out.append(U[:, :keep].reshape(chi_l, d, keep))
        carry = np.diag(S[:keep]) @ Vd[:keep]
    return out


def _overlap(a, b):
    """Inner product of two left-canonical MPS (lists of (Dl,2,Dr) tensors)."""
    E = np.ones((1, 1), dtype=complex)
    for A, B in zip(a, b):
        E = np.einsum("ab,asc,bsd->cd", E, A.conj(), B)
    return complex(E[0, 0])


def _statevector_reference(L, m=1.0, phi_max=PHI_MAX_DEFAULT, backend="auto",
                           chi_ref=32):
    """Reference vacuum as a full statevector, ordered like build_H_sparse.
    Contracts the MPS, so it is exponential in L: validation only."""
    ts, _ = vacuum_mps(L, m, phi_max, backend, chi_ref)
    ref = _truncate(ts, [], chi_inner=max(4, chi_ref))
    psi = ref[0]
    for T in ref[1:]:
        psi = np.tensordot(psi, T, axes=([-1], [0]))
    psi = psi.reshape(-1)
    return psi / np.linalg.norm(psi)


def _cut_positions(L, k):
    """Qubit indices of the k-1 inter-site boundaries splitting L sites into
    k roughly equal segments."""
    if not 1 <= k <= L:
        # beyond L the boundaries would collide and the caller would silently
        # get fewer segments than it asked for
        raise ValueError(f"k must be between 1 and L={L}, got {k}")
    if k == 1:
        return []
    cuts = sorted({2 * (s * L // k) for s in range(1, k)})
    return [c for c in cuts if 0 < c < 2 * L]


def segment_sizes(L, k):
    """Sizes (in qubits) of the k segments."""
    n = 2 * L
    pts = [0] + _cut_positions(L, k) + [n]
    return [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]


def two_qubit_depth(L, k, cnots_per_block=3):
    """CNOT depth of the segmented staircase (sequential within a segment,
    segments in parallel): (max segment qubits - 1) * cnots_per_block."""
    return (max(segment_sizes(L, k)) - 1) * cnots_per_block


def evaluate(L, m=1.0, k=1, phi_max=PHI_MAX_DEFAULT, backend="auto",
             chi_ref=32, _cache=None):
    """Fidelity and cost of the k-segment circuit, by overlapping the segmented
    MPS with the untruncated one. Returns a dict of metrics."""
    key = (L, round(m, 6), phi_max, backend, chi_ref)
    if _cache is not None and key in _cache:
        ts_ref, E0, ref = _cache[key]
    else:
        ts_ref, E0 = vacuum_mps(L, m, phi_max, backend, chi_ref)
        # ref does not depend on k, so cache it too: recomputing it on every
        # call cost a third of optimal_cuts runtime at L=50.
        ref = _truncate(ts_ref, [], chi_inner=max(4, chi_ref))
        if _cache is not None:
            _cache[key] = (ts_ref, E0, ref)

    seg = _truncate(ts_ref, _cut_positions(L, k), chi_inner=2)
    F = abs(_overlap(ref, seg)) ** 2
    # n_blocks counts two-qubit unitaries, n_cnot counts CNOT after KAK: at
    # L=50, k=1 that is 99 and 297. twoq_depth is a CNOT depth, compare it to n_cnot.
    sizes = segment_sizes(L, k)
    n_blocks = sum(max(0, s - 1) for s in sizes)
    return dict(L=L, m=m, k=k, fidelity=F,
                twoq_depth=two_qubit_depth(L, k),
                n_blocks=n_blocks, n_cnot=n_blocks * 3,
                n_segments=len(sizes), segment_qubits=max(sizes),
                vacuum_energy=E0)      # energy of the target, not of the circuit


def optimal_cuts(L, m=1.0, target_fidelity=0.90, phi_max=PHI_MAX_DEFAULT,
                 backend="auto", chi_ref=32, k_max=None, _cache=None):
    """Shallowest circuit still meeting `target_fidelity`, by scanning every k.
    Returns an evaluate() dict plus a 'feasible' flag."""
    if k_max is None:
        k_max = L                       # at most one site per segment
    best = None
    # Depth plateaus once segments equalise (at L=50, k=10..12 all give 27)
    # while fidelity keeps dropping, so rank on depth then fidelity, never on k.
    # Scan every k: cut positions shift with k, so fidelity is not monotone.
    for k in range(1, k_max + 1):
        r = evaluate(L, m, k, phi_max, backend, chi_ref, _cache=_cache)
        if r["fidelity"] < target_fidelity:
            continue
        if best is None or (r["twoq_depth"], -r["fidelity"]) \
                         < (best["twoq_depth"], -best["fidelity"]):
            best = r
    if best is None:                    # even k=1 fails
        r = evaluate(L, m, 1, phi_max, backend, chi_ref, _cache=_cache)
        r["feasible"] = False
        return r
    best["feasible"] = True
    return best
