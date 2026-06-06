"""Hardware-efficient ansatz (HEA) on a NumPy statevector (plan §2, §5).

Ry rotation layers interleaved with CZ entanglers. Parameters theta have shape
(p+1, n): p entangling layers, each preceded by an Ry layer, plus a final Ry
layer. Total parameters = n * (p + 1). Circuit depth is p, independent of the
QUBO density -- the whole point of PCE vs QAOA (plan §0).

Topology mirrors v2's hea.py so parameters transfer to PennyLane/hardware:
- 'linear': CZ chain (0,1),(1,2),... (sim default / IonQ all-to-all).
- 'square': two-pass brick-layer mapping to IQM square-lattice native edges.
"""
from __future__ import annotations

import numpy as np


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)


def _apply_cz(psi: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    idx = [slice(None)] * n
    idx[q1] = 1
    idx[q2] = 1
    psi[tuple(idx)] *= -1.0
    return psi


def _entangler(psi: np.ndarray, n: int, topology: str) -> np.ndarray:
    if topology == "linear":
        for i in range(n - 1):
            psi = _apply_cz(psi, i, i + 1, n)
    elif topology == "square":
        for i in range(0, n - 1, 2):
            psi = _apply_cz(psi, i, i + 1, n)
        for i in range(1, n - 1, 2):
            psi = _apply_cz(psi, i, i + 1, n)
    else:
        raise ValueError(f"unknown topology '{topology}'")
    return psi


def run_hea(theta: np.ndarray, n: int, p: int, topology: str = "linear") -> np.ndarray:
    """Run the p-layer HEA and return the flat (2^n,) statevector.

    theta: (p+1, n) rotation angles.
    """
    theta = np.asarray(theta, dtype=float).reshape(p + 1, n)
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1.0  # |0...0>

    for layer in range(p):
        for q in range(n):
            psi = _apply_1q(psi, _ry(theta[layer, q]), q)
        psi = _entangler(psi, n, topology)
    for q in range(n):  # final Ry layer
        psi = _apply_1q(psi, _ry(theta[p, q]), q)

    return psi.ravel()


def n_params(n: int, p: int) -> int:
    return n * (p + 1)
