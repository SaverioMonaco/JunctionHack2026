"""Topology-aware hardware-efficient ansatz in PennyLane (plan v2 §5).

Identical structure to the NumPy ansatz.py (Ry layers + CZ entangler, p layers
plus a final Ry), so parameters trained on one backend transfer to the other
and onto hardware. theta shape: (p+1, n).

- 'linear': CZ chain (sim / IonQ all-to-all).
- 'square': two-pass brick-layer mapping to IQM Garnet/Emerald native edges.

IQM Garnet/Emerald on Braket support ry and cz natively -> no decomposition.
"""
from __future__ import annotations

import pennylane as qml


def build_hea(theta, n: int, p: int, topology: str = "linear") -> None:
    """Apply the p-layer HEA in-place inside a QNode (no return)."""
    def ry_layer(layer):
        for i in range(n):
            qml.RY(theta[layer, i], wires=i)

    def entangler():
        if topology == "linear":
            for i in range(n - 1):
                qml.CZ(wires=[i, i + 1])
        elif topology == "square":
            for i in range(0, n - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, n - 1, 2):
                qml.CZ(wires=[i, i + 1])
        else:
            raise ValueError(f"unknown topology '{topology}'")

    for layer in range(p):
        ry_layer(layer)
        entangler()
    ry_layer(p)  # final Ry; total params = n*(p+1)
