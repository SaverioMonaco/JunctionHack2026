r"""qff_pricing.quantum.primitives
==================================

Distribution-loading circuits -- the ``U_path`` half of QMCI.

Two layers are provided.

1.  **Realisable loader** (:func:`qsample_loader`).  Given a discretised
    probability vector ``probs`` of length ``N = 2**n``, it builds a circuit
    whose computational-basis measurement distribution is exactly ``probs``
    (its amplitudes are ``sqrt(probs)``).  This is the *discrete qsample* of the
    paper (Definition "Discrete Qsample") and is what the end-to-end pricer
    executes.  It is implemented with Qiskit's :class:`StatePreparation`, which
    is the concrete stand-in for the paper's QET / black-box amplitude encoder
    (Lemma "Polynomial Approximation State Preparation").

2.  **Structured primitive loaders** (:func:`gaussian_loader`,
    :func:`chi2_loader`).  The fast-forwarding scheme builds the path qsample by
    composing *one-dimensional primitives* (standard Gaussian + central
    chi-square for CIR; plus integral-of-CIR for Heston) and a coherent
    arithmetic "jump".  These helpers load a single truncated primitive onto a
    grid and exist to make the per-step structure explicit and to feed the
    resource counting in :mod:`qff_pricing.scaling`.

Why this matters for scaling
----------------------------
A *naive* amplitude encoding of the full ``T``-step path would need a state over
``T * n`` qubits and ``O(2**(T n))`` gates -- **exponential** in ``T``.
Fast-forwarding instead loads ``O(T)`` cheap one-dimensional primitives and
applies ``O(T)`` arithmetic jumps, giving the ``poly(T, log(1/eps))`` gate count
of Theorem "CIR Discrete-Sum Loading".  :func:`qsample_loader` realises the
*marginal* needed for a European payoff; the structured loaders expose the
per-primitive cost that the scaling module extrapolates.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


# ---------------------------------------------------------------------------
# Realisable qsample loader
# ---------------------------------------------------------------------------


def amplitudes_from_probs(probs: np.ndarray) -> np.ndarray:
    """Return a normalised amplitude vector ``sqrt(probs)`` of length ``2**n``."""
    probs = np.asarray(probs, dtype=float)
    n = int(round(np.log2(len(probs))))
    if 2 ** n != len(probs):
        raise ValueError(f"probs length {len(probs)} is not a power of two.")
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        raise ValueError("probs must have positive sum.")
    return np.sqrt(probs / s)


def qsample_loader(probs: np.ndarray, label: str = "U_path") -> QuantumCircuit:
    r"""Build ``U_path`` such that  U_path|0> = sum_k sqrt(probs[k]) |k>.

    The qubit ordering is Qiskit-standard little-endian: basis index ``k``
    corresponds to qubit 0 being the least-significant bit.
    """
    amps = amplitudes_from_probs(probs)
    n = int(round(np.log2(len(amps))))
    qc = QuantumCircuit(n, name=label)
    qc.append(StatePreparation(amps), range(n))
    return qc


# ---------------------------------------------------------------------------
# Structured one-dimensional primitive loaders
# ---------------------------------------------------------------------------


def _grid_pmf(dist, lo: float, hi: float, n_qubits: int) -> np.ndarray:
    """Renormalised cell masses of a frozen scipy distribution over [lo, hi]."""
    N = 2 ** n_qubits
    edges = np.linspace(lo, hi, N + 1)
    pmf = np.diff(dist.cdf(edges))
    total = pmf.sum()
    if total <= 0:
        raise ValueError("Distribution puts no mass in the truncation window.")
    return pmf / total


def gaussian_loader(n_qubits: int, a: float = 5.0) -> QuantumCircuit:
    r"""Load the standard-Gaussian primitive ``Z ~ N(0,1)`` truncated to [-a, a].

    Truncation radius ``a = O(sqrt(log(1/eps_trunc)))`` (paper Sec. 6.1.2).
    """
    pmf = _grid_pmf(stats.norm(), -a, a, n_qubits)
    return qsample_loader(pmf, label="U_Gaussian")


def chi2_loader(df: float, n_qubits: int, b_lo: float = 1e-3,
                b_hi: float | None = None) -> QuantumCircuit:
    r"""Load the central chi^2_{df} primitive ``Y`` truncated to [b_lo, b_hi].

    Used with ``df = eta - 1`` for CIR.  A small region near zero is excluded
    (``b_lo > 0``) exactly as in the discretisation analysis (paper Sec. 6.1.2),
    which truncates ``(0, b_L)`` to avoid singularities in the chi^2 derivative.
    """
    d = stats.chi2(df)
    if b_hi is None:
        b_hi = float(d.mean() + 8.0 * d.std())
    pmf = _grid_pmf(d, b_lo, b_hi, n_qubits)
    return qsample_loader(pmf, label="U_chi2")


__all__ = [
    "amplitudes_from_probs",
    "qsample_loader",
    "gaussian_loader",
    "chi2_loader",
]
