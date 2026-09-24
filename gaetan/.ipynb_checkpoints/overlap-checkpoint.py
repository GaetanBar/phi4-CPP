import numpy as np
from qiskit_aer import AerSimulator
from qiskit import transpile

def state_overlap(circuit1, circuit2, simulator=None):
    """
    Compute the complex overlap <psi2|psi1> between two quantum states.

    Parameters
    ----------
    circuit1 : qiskit.QuantumCircuit
        Circuit preparing |psi1>.
    circuit2 : qiskit.QuantumCircuit
        Circuit preparing |psi2>.
    simulator : optional
        Qiskit Aer simulator. If None, an AerSimulator is created.

    Returns
    -------
    complex
        The overlap <psi2|psi1>.
    """

    if simulator is None:
        simulator = AerSimulator(method="statevector")

    # Add save instruction so Aer returns the statevector
    circuit1_sv = circuit1.copy()
    circuit2_sv = circuit2.copy()

    circuit1_sv.save_statevector()
    circuit2_sv.save_statevector()

    # Transpile for simulator
    circuit1_sv = transpile(circuit1_sv, simulator)
    circuit2_sv = transpile(circuit2_sv, simulator)

    # Run
    result1 = simulator.run(circuit1_sv).result()
    result2 = simulator.run(circuit2_sv).result()

    # Extract statevectors
    psi1 = np.asarray(result1.get_statevector(circuit1_sv))
    psi2 = np.asarray(result2.get_statevector(circuit2_sv))

    # <psi2|psi1>
    overlap = np.vdot(psi2, psi1)

    return overlap

    def fidelity(c1,c2, simu=None):
        return abs(state_overlap(c1,c2,simu)**2)
