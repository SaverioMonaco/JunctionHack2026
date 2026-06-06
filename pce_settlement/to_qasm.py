"""OpenQASM 2.0 export for PCE hardware inference (plan v2 §7).

PCE needs no variational loop on the device: train theta* offline, then run a
single forward pass = exactly 3 static circuits (one per Pauli basis X/Y/Z,
since every PCE string is homogeneous). This module bakes a trained theta* into
those 3 circuits as literal OpenQASM 2.0, for QASM-submission platforms (Qmill,
etc.) that don't need programmatic access.

Workflow:
    train offline -> export_three(theta, n, p, topology) -> submit 3 QASM,
    N shots each -> download counts -> decode_counts() -> signs -> repair.

Gate set: ry, cz, h, sdg, measure -- all standard qelib1, no decomposition.
"""
from __future__ import annotations

import numpy as np

# Basis-change gates applied AFTER the ansatz, BEFORE measurement, so that a
# Z-measurement reads the chosen Pauli (see pauli.py / hardware.py):
#   Z: none ;  X: h ;  Y: sdg then h.
_BASIS_GATES = {"Z": [], "X": ["h"], "Y": ["sdg", "h"]}


def _fmt(angle: float) -> str:
    return f"{float(angle):.12g}"


def _hea_lines(theta: np.ndarray, n: int, p: int, topology: str) -> list[str]:
    """HEA gate lines (Ry layers + CZ entanglers + final Ry), theta=(p+1, n)."""
    theta = np.asarray(theta, dtype=float).reshape(p + 1, n)
    lines: list[str] = []

    def ry_layer(layer: int):
        for i in range(n):
            lines.append(f"ry({_fmt(theta[layer, i])}) q[{i}];")

    def entangler():
        if topology == "linear":
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif topology == "square":
            pairs = ([(i, i + 1) for i in range(0, n - 1, 2)]
                     + [(i, i + 1) for i in range(1, n - 1, 2)])
        else:
            raise ValueError(f"unknown topology '{topology}'")
        for a, b in pairs:
            lines.append(f"cz q[{a}],q[{b}];")

    for layer in range(p):
        ry_layer(layer)
        entangler()
    ry_layer(p)  # final Ry
    return lines


def circuit_qasm(theta: np.ndarray, n: int, p: int, basis: str,
                 topology: str = "linear") -> str:
    """Full OpenQASM 2.0 for one basis-measurement circuit."""
    if basis not in _BASIS_GATES:
        raise ValueError(f"basis must be X, Y, or Z; got '{basis}'")
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";',
           f"qreg q[{n}];", f"creg c[{n}];", "", f"// --- PCE ansatz (theta* baked in) ---"]
    out += _hea_lines(theta, n, p, topology)
    out += ["", f"// --- rotate to {basis} basis ---"]
    for g in _BASIS_GATES[basis]:
        out += [f"{g} q[{i}];" for i in range(n)]
    out += ["", "measure q -> c;"]
    return "\n".join(out) + "\n"


def export_three(theta: np.ndarray, n: int, p: int,
                 topology: str = "linear") -> dict[str, str]:
    """The 3 circuits (X, Y, Z) to submit for one inference run."""
    return {b: circuit_qasm(theta, n, p, b, topology) for b in ("X", "Y", "Z")}


def calibration_qasm(n: int, flip_qubit: int = 0) -> str:
    """Tiny endianness check: flip one qubit, measure. Confirm which classical
    bit goes high in the device's returned counts before trusting the decode."""
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];",
           f"creg c[{n}];", "", f"x q[{flip_qubit}];", "", "measure q -> c;"]
    return "\n".join(out) + "\n"


def decode_counts(counts: dict[str, dict[str, int]], strings, n: int,
                  bit_order: str = "big") -> np.ndarray:
    """Decode Pauli expectations from device count histograms.

    counts: {'X': {'0101': 37, ...}, 'Y': {...}, 'Z': {...}} as returned by a
            QASM platform (bitstring -> shot count).
    bit_order: 'big' if the leftmost char is qubit 0 (c[0]); 'little' to reverse.
    Returns (m,) expectations <Pi_i> = sum_shots (-1)^parity / n_shots.
    """
    def bits_of(s: str) -> np.ndarray:
        b = np.array([int(ch) for ch in s], dtype=int)
        return b if bit_order == "big" else b[::-1]

    exps = np.empty(len(strings), dtype=float)
    for i, (subset, letter) in enumerate(strings):
        hist = counts[letter]
        total = sum(hist.values())
        acc = 0.0
        for s, c in hist.items():
            bits = bits_of(s)
            parity = int(bits[list(subset)].sum() % 2)
            acc += c * (1.0 if parity == 0 else -1.0)
        exps[i] = acc / total
    return exps
