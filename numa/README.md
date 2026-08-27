# mps_toolkit

Prepares the vacuum of the discretised scalar field as a quantum circuit.

The circuit is read off a Matrix Product State of the vacuum: each MPS tensor is
an isometry, completing it into a unitary gives a staircase of two-qubit gates
that maps |0...0> onto the state. Every gate is computed classically by SVD, so
there is nothing to optimise and no variational loop.

The entanglement across an inter-site cut is only 0.13 ebit, so the chain can be
cut into `k` pieces prepared in parallel. That divides the two-qubit depth by
about `k` and costs around 1.5% fidelity per cut.

## Install

```bash
pip install -r requirements.txt
```

numpy, scipy and matplotlib are always needed. quimb is used for `L > 8` (DMRG),
qiskit only to emit the circuit.

cotengra is pinned to 0.8.0 on purpose: quimb 1.14 calls an API that 0.8.2
removed, and DMRG then fails with `'numpy.ndarray' object has no attribute 'get_function'`.

## Usage

```python
from mps_toolkit import build_circuit

qc_init = build_circuit(L=50, m=1.0, k=6)      # 100 qubits, 2-qubit depth 51
qc = qc_init.compose(your_time_evolution)
```

To size a circuit before building it:

```python
from mps_toolkit import evaluate, optimal_cuts

evaluate(L=50, m=1.0, k=5)                        # fidelity 0.9196, depth 57
optimal_cuts(L=50, m=1.0, target_fidelity=0.90)   # k=6, fidelity 0.9069, depth 51
```

Both take the same arguments as `build_circuit`, so `evaluate` tells you in
advance what you will get. `k` must satisfy `1 <= k <= L`: beyond that the cuts
would collide and you would silently get fewer segments than you asked for.

## Qubit ordering

Circuit qubit `q` is physical qubit `q`: site `j` sits on qubits `2j` and
`2j+1`, same order as `build_H_sparse`. Qiskit's `Statevector` is little-endian,
so a numpy reference vector needs one reversal before you compare it.
`circuit_fidelity(qc, L, m)` handles that, up to about 28 qubits.

## Reference numbers

At `L=50`, `m=1`, `phi_max=1.5`, computed by DMRG and exact tensor contraction
rather than extrapolated:

| k | fidelity | 2-qubit depth | CNOT |
| - | -------- | ------------- | ---- |
| 1 | 0.9720   | 297           | 297  |
| 3 | 0.9454   | 99            | 291  |
| 5 | 0.9196   | 57            | 285  |
| 6 | 0.9069   | 51            | 282  |

Segmentation cuts the depth, not the gate count. The gain comes from segments
running in parallel.

## Files

```
mps_toolkit/core.py       Hamiltonian, vacuum (exact or DMRG), fidelity, depth
mps_toolkit/circuit.py    MPS to QuantumCircuit
mps_toolkit/explore.py    parameter sweeps and plots
demo.py                   one configuration, parameters at the top of the file
run_analysis.py           four sweeps driven by a CONFIG block, writes to results/
test_core.py              tests
test_circuit.py           tests
```

## Tests

```bash
pytest -q
```

The one that matters simulates the circuit and checks its fidelity against what
`evaluate` predicted. That is what catches a qubit-ordering mistake, which
otherwise produces a circuit that looks fine and prepares the wrong state.
