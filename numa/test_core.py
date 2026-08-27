import numpy as np
import pytest

from mps_toolkit.core import (
    build_H_sparse, vacuum_mps, evaluate, optimal_cuts,
    segment_sizes, _cut_positions, _mps_from_statevector,
    _HAVE_QUIMB, PHI_MAX_DEFAULT, EXACT_MAX_L,
)
from scipy.sparse.linalg import eigsh

needs_quimb = pytest.mark.skipif(not _HAVE_QUIMB, reason="quimb not installed")


@pytest.mark.parametrize("L,continuum", [(1, 0.500000), (4, 3.046167)])
def test_energy_approaches_the_continuum(L, continuum):
    """Validates _local_ops and build_H_sparse against the analytic answer
    E0 = 1/2 sum_n sqrt(m^2 + 4 sin^2(pi n / 2L)). The 4-value truncation is
    variational, so the energy must sit above it, and not far above."""
    E = vacuum_mps(L, m=1.0, backend="exact")[1]
    assert E > continuum
    assert E - continuum < 0.07


def test_fidelity_increases_with_mass():
    """Heavier field, shorter correlation length, easier to prepare. Catches a
    sign or factor error on the mass term, which nothing else exercises."""
    fids = [evaluate(L=6, m=m, k=1)["fidelity"] for m in (0.5, 1.0, 2.0)]
    assert fids == sorted(fids)


def test_mps_reproduces_the_exact_vacuum():
    """An untruncated MPS must rebuild the eigenvector to machine precision."""
    L = 4
    H = build_H_sparse(L, m=1.0, phi_max=PHI_MAX_DEFAULT)
    psi = eigsh(H, k=1, which="SA")[1][:, 0].astype(complex)
    psi /= np.linalg.norm(psi)

    ts = _mps_from_statevector(psi, 2 * L)
    back = ts[0]
    for T in ts[1:]:
        back = np.tensordot(back, T, axes=([-1], [0]))
    assert abs(abs(np.vdot(psi, back.reshape(-1))) ** 2 - 1.0) < 1e-12


@pytest.mark.parametrize("L,k", [(3, 2), (6, 3), (10, 6), (50, 1), (50, 6), (50, 50)])
def test_segments_are_valid(L, k):
    """Segments must tile the register exactly, be as many as requested, and
    never cut inside a site: an intra-site cut severs a 0.91-ebit bond instead
    of a 0.13-ebit one, costing ~30% fidelity instead of ~1.5%."""
    sizes = segment_sizes(L, k)
    cuts = _cut_positions(L, k)
    assert sum(sizes) == 2 * L
    assert len(sizes) == k
    assert all(c % 2 == 0 for c in cuts)


@pytest.mark.parametrize("L,k", [(5, 6), (6, 0)])
def test_invalid_k_is_rejected(L, k):
    """Beyond L the boundaries would collide and the caller would silently get
    fewer segments than requested."""
    with pytest.raises(ValueError):
        _cut_positions(L, k)


@pytest.mark.parametrize("L,k,expected", [(6, 1, 0.99714), (6, 3, 0.96942),
                                          (8, 1, 0.99599)])
def test_reference_fidelities(L, k, expected):
    r = evaluate(L=L, m=1.0, k=k, backend="exact")
    assert r["fidelity"] == pytest.approx(expected, abs=5e-5)


@needs_quimb
def test_reference_point_20_qubits():
    """The 99.5% at 20 qubits quoted in the report, through the auto backend."""
    r = evaluate(L=10, m=1.0, k=1)
    assert r["fidelity"] == pytest.approx(0.9948, abs=5e-4)
    assert r["twoq_depth"] == 57


@needs_quimb
@pytest.mark.parametrize("L", [2, 5])
def test_mpo_matches_the_sparse_hamiltonian(L):
    """The DMRG path rebuilds the Hamiltonian as an MPO by hand. Nothing
    guarantees a priori that it is the same operator: the on-site bond count is
    easy to get wrong, and did use to be wrong at L=1."""
    from mps_toolkit.core import _vacuum_exact, _vacuum_dmrg
    assert _vacuum_dmrg(L, 1.0, PHI_MAX_DEFAULT)[1] == \
        pytest.approx(_vacuum_exact(L, 1.0, PHI_MAX_DEFAULT)[1], abs=1e-8)


@needs_quimb
def test_dmrg_backend_agrees_with_exact():
    r_exact = evaluate(L=EXACT_MAX_L, m=1.0, k=1, backend="exact")
    r_dmrg = evaluate(L=EXACT_MAX_L, m=1.0, k=1, backend="dmrg")
    assert r_dmrg["fidelity"] == pytest.approx(r_exact["fidelity"], abs=1e-4)
    assert r_dmrg["vacuum_energy"] == pytest.approx(r_exact["vacuum_energy"], abs=1e-6)


@needs_quimb
def test_dmrg_refuses_a_single_site():
    """A 1-site chain has no bond, and quimb's MPO is degenerate there."""
    from mps_toolkit.core import _vacuum_dmrg
    with pytest.raises(ValueError):
        _vacuum_dmrg(1, 1.0, PHI_MAX_DEFAULT)


def test_optimal_cuts_returns_an_undominated_config():
    """No feasible k may beat the answer on depth, nor on fidelity at equal
    depth. The solver used to keep the largest k, returning k=12 at L=50 where
    k=10 gave the same depth and 2.3 more points of fidelity."""
    L, m, target = 6, 1.0, 0.90
    cache = {}
    best = optimal_cuts(L, m, target_fidelity=target, _cache=cache)
    assert best["feasible"]
    for k in range(1, L + 1):
        r = evaluate(L, m, k, _cache=cache)
        if r["fidelity"] >= target:
            assert (best["twoq_depth"], -best["fidelity"]) \
                <= (r["twoq_depth"], -r["fidelity"])


def test_optimal_cuts_flags_infeasible_targets():
    r = optimal_cuts(6, m=1.0, target_fidelity=0.9999)
    assert r["feasible"] is False


def test_cache_does_not_change_results():
    """evaluate shares one reference MPS through the cache. Calls with other k
    must not perturb it."""
    plain = evaluate(L=6, m=1.0, k=2, backend="exact")
    cache = {}
    evaluate(L=6, m=1.0, k=3, backend="exact", _cache=cache)
    again = evaluate(L=6, m=1.0, k=2, backend="exact", _cache=cache)
    assert again["fidelity"] == pytest.approx(plain["fidelity"], abs=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
