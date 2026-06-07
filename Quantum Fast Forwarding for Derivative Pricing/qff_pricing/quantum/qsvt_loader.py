r"""qff_pricing.quantum.qsvt_loader
===================================

The **polylogarithmic** distribution loader -- a real, gate-level implementation
of the polynomial-approximation / Quantum Eigenvalue Transformation (QET / QSVT)
state-preparation of McArdle, Gilyen & Berta (2022), which is the method the
paper invokes in Section 3.2 ("Quantum Eigenvalue Transformation for State
Preparation") and uses throughout Appendices C and D to load the CIR/Heston
primitives.

Unlike :func:`qff_pricing.quantum.primitives.qsample_loader` (which uses Qiskit's
``StatePreparation`` and therefore costs ``Theta(N)`` gates -- the *naive*
encoder), this loader prepares

    |psi> = (1/Z) sum_{k=0}^{N-1} sqrt(p_k) |k>

using only ``O(n * d)`` one- and two-qubit gates, where ``n = log2 N`` and
``d = polylog(1/eps)`` is the degree of a polynomial approximating ``sqrt(p)``.
That is **polylogarithmic in the number of data points ``N``** -- the headline
claim, now realised as an actual circuit whose gate count you can measure.

Construction (verified end-to-end against the statevector)
----------------------------------------------------------
1.  **Block-encode** ``A = diag(sin(k/N))`` with a single ancilla using an
    ``O(n)``-gate reflection ``U_sin`` (a bank of controlled ``RX`` rotations).
    Each register basis state ``|k>`` rotates the ancilla by
    ``arccos(sin(k/N)) = pi/2 - k/N`` about ``X``.

2.  **QSP / QSVT.** Interleave ``d`` applications of ``U_sin`` with ancilla
    ``Z``-phase rotations ``e^{i phi_j Z}``.  For a *symmetric* phase sequence
    ``Phi`` this realises ``Re<0|U_Phi|0> = P(sin(k/N))`` for a real,
    even-degree-``d`` polynomial ``P`` determined by ``Phi``.

3.  **Phase finding.** ``Phi`` is fitted (Levenberg-Marquardt) so that
    ``P(sin(k/N)) propto sqrt(p_k)`` on the grid.  Starting from all-zero phases
    (``P = T_d``, a Chebyshev polynomial) the fit converges reliably.

4.  **Real-part extraction.** ``<0|U_Phi|0>`` is generally complex
    ``P + iQ``; a one-ancilla LCU of ``U_Phi`` and its negated-phase copy
    ``U_{-Phi}`` (whose ``(0,0)`` block is the complex conjugate) projects onto
    ``(P + conj) / 2 = P`` -- exactly the real polynomial.  Costs one extra
    ancilla and ``d`` controlled-``RZ`` gates.

The loader uses ``n + 2`` qubits: ``n`` register qubits + the signal ancilla +
the LCU (Hadamard-test) ancilla.  The prepared state lives on the register when
**both** ancillas are measured ``0`` (post-selection success probability equals
the squared ``L2`` filling fraction; in the full algorithm this is removed by
``O(1/F)`` rounds of amplitude amplification, which preserves the polylog gate
count).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# QSP phase finding (self-contained; no external angle-finder dependency)
# ---------------------------------------------------------------------------


def qsp_response(full_phases: np.ndarray, s: np.ndarray) -> np.ndarray:
    r"""Return ``Re<0|U|0>`` for the Wx-convention QSP unitary, vectorised over ``s``.

    ``U = e^{i phi_0 Z} prod_{k>=1} [ W(s) e^{i phi_k Z} ]`` with
    ``W(s) = [[s, i sqrt(1-s^2)], [i sqrt(1-s^2), s]]``.  For a symmetric phase
    sequence this real part is a definite-parity polynomial ``P(s)``.
    """
    s = np.asarray(s, float)
    rt = np.sqrt(np.clip(1 - s * s, 0.0, 1.0))
    W = np.empty((len(s), 2, 2), complex)
    W[:, 0, 0] = s
    W[:, 1, 1] = s
    W[:, 0, 1] = 1j * rt
    W[:, 1, 0] = 1j * rt
    U = np.zeros((len(s), 2, 2), complex)
    e0 = np.exp(1j * full_phases[0])
    U[:, 0, 0] = e0
    U[:, 1, 1] = np.conj(e0)
    for ph in full_phases[1:]:
        U = U @ W
        U = U @ np.array([[np.exp(1j * ph), 0], [0, np.exp(-1j * ph)]], complex)
    return U[:, 0, 0].real


def _symmetric(reduced: np.ndarray) -> np.ndarray:
    """Even-parity symmetric phase sequence from its reduced half."""
    reduced = np.asarray(reduced, float)
    return np.concatenate([reduced[::-1], reduced[1:]])


def fit_qsp_phases(s_nodes: np.ndarray, target: np.ndarray, degree: int,
                   max_nfev: int = 8000) -> np.ndarray:
    r"""Fit a symmetric (even) phase sequence so ``Re<0|U|0>(s) ~ target(s)``.

    ``degree`` must be even.  Returns the full (symmetric) phase array of length
    ``degree + 1``.  Initialised at all-zero phases (``P = T_degree``).
    """
    if degree % 2 != 0:
        degree += 1
    reduced0 = np.zeros(degree // 2 + 1)

    def residual(reduced):
        return qsp_response(_symmetric(reduced), s_nodes) - target

    sol = least_squares(residual, reduced0, method="lm", max_nfev=max_nfev)
    return _symmetric(sol.x)


# ---------------------------------------------------------------------------
# The QSVT loader circuit
# ---------------------------------------------------------------------------


@dataclass
class QSVTLoaderInfo:
    n_qubits: int                 # register qubits (data points = 2**n)
    degree: int                   # polynomial degree d
    signal_ancilla: int           # qubit index of the block-encoding ancilla
    lcu_ancilla: int              # qubit index of the real-part (LCU) ancilla
    phases: np.ndarray            # the fitted symmetric phase sequence
    filling_sq: float = 0.0       # F^2 = (1/N) sum_k P(sin k/N)^2 (post-select prob)


def default_degree(n_qubits: int) -> int:
    """A degree schedule that keeps the loader fidelity high (even number)."""
    d = 2 * n_qubits + 4
    return d if d % 2 == 0 else d + 1


def build_qsvt_loader(probs: np.ndarray, degree: int | None = None,
                      target_scale: float = 0.9) -> tuple[QuantumCircuit, QSVTLoaderInfo]:
    r"""Build the polylog QSVT loader for the distribution ``probs`` (length ``2**n``).

    Returns ``(circuit, info)``.  The circuit acts on ``n + 2`` qubits; the
    target state ``sum_k sqrt(probs_k) |k>`` is prepared on the ``n`` register
    qubits conditioned on both ancillas (``info.signal_ancilla`` and
    ``info.lcu_ancilla``) being measured in ``|0>``.
    """
    probs = np.clip(np.asarray(probs, float), 0.0, None)
    N = len(probs)
    n = int(round(np.log2(N)))
    if 2 ** n != N:
        raise ValueError("len(probs) must be a power of two.")
    if degree is None:
        degree = default_degree(n)

    g = np.sqrt(probs / probs.sum())
    s_all = np.sin(np.arange(N) / N)             # block-encoding eigenvalues
    # scale the target into (-1, 1); only proportionality matters (state is
    # renormalised), so a global scale is irrelevant to the loaded distribution.
    target_all = g / g.max() * target_scale
    # A degree-d polynomial is fixed by O(d) points -- subsample the grid for the
    # fit so phase finding stays fast even when N is large (keeps it polylog).
    n_fit = min(N, max(8 * degree, 64))
    idx = np.unique(np.linspace(0, N - 1, n_fit).astype(int))
    phases = fit_qsp_phases(s_all[idx], target_all[idx], degree)

    qc = QuantumCircuit(n + 2, name="U_qsvt")
    a = n            # signal ancilla
    h = n + 1        # LCU / Hadamard-test ancilla
    d = len(phases) - 1

    qc.h(range(n))   # uniform input on the register
    qc.h(h)

    def phase(phi: float) -> None:
        qc.rz(-2.0 * phi, a)          # e^{i phi Z} on the signal ancilla
        qc.crz(4.0 * phi, h, a)       # if h=1, net e^{-i phi Z}  (negated-phase copy)

    def signal() -> None:
        qc.rx(-np.pi, a)
        for j in range(n):
            qc.crx(2.0 * (2 ** j) / (2 ** n), j, a)

    phase(phases[d])
    for k in range(d - 1, -1, -1):
        signal()
        phase(phases[k])
    qc.h(h)

    # F^2 = squared L2 filling fraction = post-selection success probability,
    # computed classically from the realised polynomial on the full grid.
    filling_sq = float(np.mean(qsp_response(phases, s_all) ** 2))
    info = QSVTLoaderInfo(n_qubits=n, degree=d, signal_ancilla=a,
                          lcu_ancilla=h, phases=phases, filling_sq=filling_sq)
    return qc, info


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def loaded_statevector(circuit: QuantumCircuit, n_qubits: int) -> np.ndarray:
    """Post-selected (both ancillas = 0), renormalised amplitudes on the register."""
    from qiskit.quantum_info import Statevector

    N = 2 ** n_qubits
    amp = Statevector.from_instruction(circuit).data[:N]   # ancillas are the MSBs
    nrm = np.linalg.norm(amp)
    if nrm == 0:
        raise RuntimeError("Post-selection amplitude is zero.")
    return amp / nrm


def loader_fidelity(probs: np.ndarray, degree: int | None = None) -> float:
    """Fidelity between the QSVT-loaded state and the exact ``sqrt(probs)`` target."""
    probs = np.clip(np.asarray(probs, float), 0.0, None)
    g = np.sqrt(probs / probs.sum())
    qc, info = build_qsvt_loader(probs, degree=degree)
    loaded = loaded_statevector(qc, info.n_qubits)
    loaded = loaded * np.exp(-1j * np.angle(loaded[np.argmax(np.abs(loaded))]))
    return float(abs(np.vdot(loaded, g)))


def success_probability(circuit: QuantumCircuit, n_qubits: int) -> float:
    """Post-selection success probability = squared L2 filling fraction."""
    from qiskit.quantum_info import Statevector

    N = 2 ** n_qubits
    amp = Statevector.from_instruction(circuit).data[:N]
    return float(np.sum(np.abs(amp) ** 2))


__all__ = [
    "qsp_response",
    "fit_qsp_phases",
    "QSVTLoaderInfo",
    "default_degree",
    "build_qsvt_loader",
    "loaded_statevector",
    "loader_fidelity",
    "success_probability",
]
