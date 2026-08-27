"""Segmented MPS preparation of the scalar-field vacuum. See README.md."""
from .core import (build_H_sparse, vacuum_mps, evaluate, optimal_cuts,
                   two_qubit_depth, segment_sizes)
from . import explore
from .circuit import build_circuit, circuit_fidelity, circuit_report

__all__ = ["build_H_sparse", "vacuum_mps", "evaluate", "optimal_cuts",
           "two_qubit_depth", "segment_sizes", "explore",
           "build_circuit", "circuit_fidelity", "circuit_report"]
