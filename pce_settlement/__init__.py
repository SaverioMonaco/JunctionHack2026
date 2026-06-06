"""Pauli Correlation Encoding (PCE) for financial transaction settlement.

Qubit-efficient solver for the transaction settlement / netting problem.
See PCE_settlement_implementation_plan.md (v1) and PCE_settlement_plan_v2.md.

This package ships the v1 NumPy + COBYLA core (numpy/scipy/matplotlib only) and
the v2 JAX/PennyLane/Braket hardware layer (device/hea/pce_jax/hardware), both
over the same instance/qubo/baseline/repair foundation. The v2 modules are not
imported here so the core stays usable without jax/pennylane installed.
"""

__all__ = [
    # shared foundation + v1 core
    "config",
    "instance",
    "qubo",
    "pauli",
    "ansatz",
    "pce_solve",
    "repair",
    "baseline",
    "compare",
    "plots",
    "results",
    # v2 (import explicitly; needs jax + pennylane)
    "device",
    "hea",
    "pce_jax",
    "hardware",
]
