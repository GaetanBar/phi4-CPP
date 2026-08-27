"""Four sweeps driven by the CONFIG block, written as PNG and CSV into
results/."""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mps_toolkit import explore

# to modify as you like
CONFIG = {
    "output_dir": "results",
    "target_fidelity": 0.90,
    "phi_max": 1.5,

    # depth/fidelity tradeoff: one curve per k, at fixed (L, m). The target line is drawn.
    "tradeoff": {
        "enabled": True,
        "L": 50, "m": 1.0,
        "k_values": [1, 2, 3, 4, 5, 6],
    },

    # scaling vs size (fidelity vs L, one curve per mass, fixed k or segment size)
    "scaling": {
        "enabled": True,
        "L_values": [3, 4, 5, 6, 7, 8],
        "masses": [0.5, 1.0, 1.5, 2.0],
        "segment_qubits": None,       # None -> fixed k below; else constant-depth study
        "k": 1,
    },

    # critical mass: the mass at which fidelity crosses the target, for a given L and k
    "critical_mass": {
        "enabled": True,
        "L": 10,
        "k": 3,
        "masses": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    },

    # optimal cuts + heatmap
    "optimal_cuts": {
        "enabled": True,
        "L_values": [5, 8, 10, 20, 50],
        "masses": [0.5, 1.0, 1.5, 2.0],
    },
}


def _save(fig, path):
    """Write the figure to disk and close it to free the memory."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _write_csv(path, rows):
    """Write a list of dicts as CSV, taking the columns from the first row."""
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  -> {path}")


def main():
    """Run the analyses enabled in CONFIG, sharing one vacuum cache across them."""
    out = CONFIG["output_dir"]
    os.makedirs(out, exist_ok=True)
    tgt = CONFIG["target_fidelity"]
    pm = CONFIG["phi_max"]
    cache = {}  # shared vacuum cache across all analyses

    if CONFIG["tradeoff"]["enabled"]:
        print("[1] depth/fidelity tradeoff ...")
        c = CONFIG["tradeoff"]
        rows, ax = explore.depth_fidelity_tradeoff(
            c["L"], c["m"], c["k_values"], target_fidelity=tgt,
            phi_max=pm, cache=cache)
        _save(ax.figure, f"{out}/tradeoff_L{c['L']}_m{c['m']}.png")
        _write_csv(f"{out}/tradeoff_L{c['L']}_m{c['m']}.csv", rows)

    if CONFIG["scaling"]["enabled"]:
        print("[2] scaling vs size ...")
        c = CONFIG["scaling"]
        res, ax = explore.scaling_vs_size(
            c["L_values"], c["masses"], k=c["k"],
            segment_qubits=c["segment_qubits"], target_fidelity=tgt,
            phi_max=pm, cache=cache)
        _save(ax.figure, f"{out}/scaling.png")
        flat = [r for rows in res.values() for r in rows]
        _write_csv(f"{out}/scaling.csv", flat)

    if CONFIG["critical_mass"]["enabled"]:
        print("[3] critical mass ...")
        c = CONFIG["critical_mass"]
        res, ax = explore.critical_mass(
            c["masses"], c["L"], k=c["k"], target_fidelity=tgt,
            phi_max=pm, cache=cache)
        _save(ax.figure, f"{out}/critical_mass_L{c['L']}.png")
        _write_csv(f"{out}/critical_mass_L{c['L']}.csv", res["rows"])
        print(f"    critical mass (F>={tgt:.0%}): {res['m_crit']}")

    if CONFIG["optimal_cuts"]["enabled"]:
        print("[4] optimal-cuts table + heatmap ...")
        c = CONFIG["optimal_cuts"]
        table = explore.optimal_cuts_table(
            c["L_values"], c["masses"], target_fidelity=tgt,
            phi_max=pm, cache=cache)
        _write_csv(f"{out}/optimal_cuts.csv", table)
        fig, ax = plt.subplots(figsize=(8, 5))
        explore.plot_optimal_cuts_heatmap(table, c["L_values"], c["masses"],
                                          value="k", ax=ax)
        _save(fig, f"{out}/optimal_cuts_heatmap.png")

    print("Done.")


if __name__ == "__main__":
    main()
