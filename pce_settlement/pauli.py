"""Pauli-string enumeration and 3-basis expectation values (plan §1.5).

PCE encodes m binary variables into n qubits by assigning each variable a
distinct k-body Pauli string. We use the homogeneous family from the seminal
paper: for each k-subset of qubits and each Pauli letter P in {X, Y, Z}, the
string is P on every qubit in the subset, identity elsewhere. This gives
3 * C(n, k) available strings (the qubit-sizing bound) and -- crucially -- lets
all same-letter expectations be read from a single measurement basis (the
"3-basis trick"): X-basis, Y-basis, Z-basis measurements cover every string.

A Pauli string is represented as ``(frozenset[int], str)`` = (qubit subset,
letter in {'X','Y','Z'}).
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

PauliString = tuple[frozenset, str]


def qubits_for(m: int, k: int) -> int:
    """Smallest n with m <= 3 * C(n, k) (plan §1.5 sizing recap)."""
    n = k
    while 3 * comb(n, k) < m:
        n += 1
    return n


def layers_for(m: int, n: int) -> int:
    """HEA depth p = floor(m / n), at least 1 (plan §1.5)."""
    return max(1, m // n)


def alpha_for(n: int, k: int) -> float:
    """Sharpening factor alpha = n^floor(k/2) (= n for k in {2,3})."""
    return float(n ** (k // 2))


def enumerate_pauli_strings(n: int, k: int, m: int) -> list[PauliString]:
    """Return the first ``m`` distinct k-body Pauli strings on n qubits.

    Order: all X strings (over k-subsets in combinatorial order), then Y, then Z.
    This grouping by letter is what the 3-basis decoder exploits.
    """
    capacity = 3 * comb(n, k)
    if m > capacity:
        raise ValueError(f"need m={m} strings but only 3*C({n},{k})={capacity} available")
    strings: list[PauliString] = []
    for letter in ("X", "Y", "Z"):
        for subset in combinations(range(n), k):
            strings.append((frozenset(subset), letter))
            if len(strings) == m:
                return strings
    return strings


# ---- 3-basis expectation values (statevector simulator) -------------------

_H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
# Basis-change V with V P V^dagger = Z, applied to the state before a Z-parity
# readout:  <P> = <V psi | Z | V psi>.
_BASIS_ROT = {
    "Z": np.eye(2, dtype=complex),
    "X": _H,
    "Y": _H @ _SDG,
}


def _apply_1q(psi: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply single-qubit gate U to qubit q of an n-qubit statevector (nd form)."""
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)


def _basis_probabilities(psi_flat: np.ndarray, letter: str, n: int) -> np.ndarray:
    """Computational-basis probabilities after rotating to measure ``letter``."""
    psi = psi_flat.reshape([2] * n)
    U = _BASIS_ROT[letter]
    for q in range(n):
        psi = _apply_1q(psi, U, q, n)
    return np.abs(psi) ** 2  # shape (2,)*n


def expectations(psi_flat: np.ndarray, strings: list[PauliString], n: int) -> np.ndarray:
    """Expectation <Pi_i> for every string, via the 3-basis trick.

    Rotates the state once per letter, then reads each string's expectation as
    a Z-parity moment over its qubit subset. O(3 * 2^n + m * 2^n).
    """
    probs = {letter: _basis_probabilities(psi_flat, letter, n) for letter in ("X", "Y", "Z")}
    out = np.empty(len(strings), dtype=float)
    sgn = np.array([1.0, -1.0])
    for i, (subset, letter) in enumerate(strings):
        p = probs[letter]
        w = np.ones([2] * n)
        for q in subset:
            shape = [1] * n
            shape[q] = 2
            w = w * sgn.reshape(shape)
        out[i] = float(np.sum(p * w))
    return out


# ---- Direct-apply oracle (for tests only) ---------------------------------

_PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def expectation_oracle(psi_flat: np.ndarray, string: PauliString, n: int) -> float:
    """Reference <psi|Pi|psi> by directly applying the Pauli operator."""
    subset, letter = string
    psi = psi_flat.reshape([2] * n)
    out = psi.copy()
    for q in subset:
        out = _apply_1q(out, _PAULI[letter], q, n)
    return float(np.real(np.vdot(psi.ravel(), out.ravel())))
