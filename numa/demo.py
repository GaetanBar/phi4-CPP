from mps_toolkit.core import evaluate

L = 50
m = 1.0
k = 5
phi_max = 1.5


r = evaluate(L=L, m=m, k=k, phi_max=phi_max)

print()
print(f"  Sites L .................. {r['L']}   ({2*r['L']} qubits)")
print(f"  Mass m ................... {r['m']}")
print(f"  Segments k ............... {r['k']}   (segments of {r['segment_qubits']} qubits)")
print(f"  phi_max .................. {phi_max}")
print("  " + "-" * 45)
print(f"  Fidelity ................. {r['fidelity']:.4f}")
print(f"  Depth 2-qubit ............ {r['twoq_depth']} CNOT")
print(f"  2-qubit blocks ........... {r['n_blocks']}   (SU(4) gates)")
print(f"  CNOT (total) ............. {r['n_cnot']}   (= 3 per block)")
print(f"  Energy of the vacuum ..... {r['vacuum_energy']:.6f}")
print()

try:
    from mps_toolkit import build_circuit
except ImportError as exc:                      # qiskit missing
    print(f"  (circuit not built: {exc})\n")
else:
    qc_init = build_circuit(L=L, m=m, k=k, phi_max=phi_max)
    print(f"  Circuit .................. {qc_init.name}")
    print(f"                             {qc_init.num_qubits} qubits, "
          f"{len(qc_init.data)} unitary blocks")
    print("  To use it:                 qc = qc_init.compose(your_evolution)")
    print()
