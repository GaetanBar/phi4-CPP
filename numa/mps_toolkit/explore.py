from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import evaluate, optimal_cuts, _cut_positions, PHI_MAX_DEFAULT

_PALETTE = ["#24185E", "#1C7293", "#0E7C86", "#E71D73", "#F0A202", "#7B2D8E"]


def depth_fidelity_tradeoff(L, m=1.0, k_values=None, target_fidelity=0.90,
                            phi_max=PHI_MAX_DEFAULT, backend="auto",
                            cache=None, ax=None):
    """Fidelity against two-qubit depth as k grows, at fixed (L, m).
    Plots one point per k, annotated, with the target line."""
    if cache is None: cache = {}
    if k_values is None: k_values = [k for k in range(1, min(L, 7) + 1)]
    rows = [evaluate(L, m, k, phi_max=phi_max, backend=backend, _cache=cache)
            for k in k_values]

    if ax is None: _, ax = plt.subplots(figsize=(8, 5))
    depths = [r["twoq_depth"] for r in rows]
    fids = [r["fidelity"] for r in rows]
    ax.plot(depths, fids, "o-", color=_PALETTE[0], lw=2, ms=8,
            mfc="white", mew=2, zorder=3)
    for r in rows:
        ax.annotate(f"k={r['k']}", (r["twoq_depth"], r["fidelity"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=_PALETTE[0])
    if target_fidelity:
        ax.axhline(target_fidelity, color="gray", ls="--", lw=1.3,
                   label=f"target {target_fidelity:.0%}")
        ax.legend()
    ax.set_xlabel("2-qubit depth (CNOT)")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"depth vs fidelity, L={L}, m={m}", fontweight="bold")
    ax.grid(alpha=0.3)
    return rows, ax


def scaling_vs_size(L_values, masses, k=1, segment_qubits=None,
                    target_fidelity=0.90, phi_max=PHI_MAX_DEFAULT,
                    backend="auto", cache=None, ax=None):
    """Fidelity against L, one curve per mass. If segment_qubits is given, k is
    picked per L to keep segments that size, which holds the depth constant."""
    if cache is None: cache = {}
    if ax is None: _, ax = plt.subplots(figsize=(8, 5))
    results = {}
    for i, m in enumerate(masses):
        rows = []
        for L in L_values:
            kk = (min(L, max(1, round(2 * L / segment_qubits)))
                  if segment_qubits else k)
            rows.append(evaluate(L, m, kk, phi_max=phi_max, backend=backend,
                                 _cache=cache))
        results[m] = rows
        ax.plot(L_values, [r["fidelity"] for r in rows], "o-",
                color=_PALETTE[i % len(_PALETTE)], lw=2, ms=6, label=f"m={m}")
    if target_fidelity:
        ax.axhline(target_fidelity, color="gray", ls="--", lw=1.3,
                   label=f"target {target_fidelity:.0%}")
    sub = (f"segments of ~{segment_qubits} qubits"
           if segment_qubits else f"k={k} segment(s)")
    ax.set_xlabel("L (sites)")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"scaling, {sub}", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return results, ax


def critical_mass(masses, L, k=1, target_fidelity=0.90,
                  phi_max=PHI_MAX_DEFAULT, backend="auto",
                  cache=None, ax=None):
    """Fidelity against mass at fixed (L, k), scanning upward to report the
    lightest mass the circuit still prepares above target."""
    if cache is None: cache = {}
    masses = sorted(masses)         # the scan below assumes ascending masses
    rows = [evaluate(L, m, k, phi_max=phi_max, backend=backend, _cache=cache)
            for m in masses]
    fids = [r["fidelity"] for r in rows]

    m_crit = None
    for m, F in zip(masses, fids):
        if F >= target_fidelity:
            m_crit = m; break          # masses assumed ascending

    if ax is None: _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(masses, fids, "o-", color=_PALETTE[3], lw=2, ms=7, zorder=3)
    if target_fidelity:
        ax.axhline(target_fidelity, color="gray", ls="--", lw=1.3,
                   label=f"target {target_fidelity:.0%}")
    if m_crit is not None:
        ax.axvline(m_crit, color=_PALETTE[0], ls=":", lw=1.5,
                   label=f"critical mass ≈ {m_crit:g}")
    ax.set_xlabel("Mass m")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"critical mass, L={L}, k={k}", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return dict(rows=rows, m_crit=m_crit), ax


def optimal_cuts_table(L_values, masses, target_fidelity=0.90,
                       phi_max=PHI_MAX_DEFAULT, backend="auto", cache=None):
    """Run optimal_cuts over an (L, m) grid; returns one dict per cell."""
    if cache is None: cache = {}
    table = []
    for L in L_values:
        for m in masses:
            r = optimal_cuts(L, m, target_fidelity, phi_max=phi_max,
                             backend=backend, _cache=cache)
            table.append(dict(L=L, m=m, k=r["k"], fidelity=r["fidelity"],
                              twoq_depth=r["twoq_depth"],
                              feasible=r.get("feasible", True)))
    return table


def plot_optimal_cuts_heatmap(table, L_values, masses, value="k", ax=None):
    """Heatmap of a column of optimal_cuts_table over the (L, m) grid."""
    if ax is None: _, ax = plt.subplots(figsize=(8, 5))
    grid = np.full((len(masses), len(L_values)), np.nan)
    lut = {(r["L"], r["m"]): r for r in table}
    for i, m in enumerate(masses):
        for j, L in enumerate(L_values):
            r = lut.get((L, m))
            if r is not None and r["feasible"]:
                grid[i, j] = r[value]
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(L_values)), L_values)
    ax.set_yticks(range(len(masses)), masses)
    ax.set_xlabel("L (sites)"); ax.set_ylabel("mass m")
    lbl = {"k": "optimal segment k", "twoq_depth": "CNOT depth",
           "fidelity": "fidelity"}.get(value, value)
    ax.set_title(lbl, fontweight="bold")
    plt.colorbar(im, ax=ax, label=lbl)
    # annotate cells
    for i in range(len(masses)):
        for j in range(len(L_values)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.3g}", ha="center", va="center",
                        color="white", fontsize=8)
    return ax
