r"""qff_pricing.quantum.qmci
=============================

Quantum Monte-Carlo Integration: estimate ``a = P(flag = 1)`` for the amplitude
encoder ``A = U_f . U_path`` and convert it to a price.

The "good" amplitude is

    a = sum_k p_k f~(x_k) = E[f] / M ,

so the (undiscounted) price is ``M * a``.  Quantum Amplitude Estimation (QAE)
returns ``a`` to additive error ``eps`` with ``O(1/eps)`` calls to ``A`` -- the
quadratic speedup over the ``O(1/eps^2)`` of classical Monte-Carlo.

Two estimators:

* :func:`estimate_amplitude_exact` -- reads ``a`` exactly from the statevector
  (the ``eps -> 0`` / infinite-shot limit of QAE).  Pure Qiskit, no Aer needed;
  always available, used for validating the pipeline.

* :func:`estimate_amplitude_iqae` -- genuine Iterative Amplitude Estimation via
  ``qiskit_algorithms`` if installed; this is the routine that delivers the
  ``O(1/eps)`` query complexity.  Falls back to the exact estimator otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class AmplitudeEstimate:
    amplitude: float           # estimated a = P(flag = 1)
    method: str                # "exact_statevector" | "iqae"
    epsilon_target: float | None = None
    confidence_interval: tuple[float, float] | None = None
    oracle_queries: int | None = None


def build_amplitude_encoder(u_path: QuantumCircuit,
                            u_f: QuantumCircuit) -> tuple[QuantumCircuit, int]:
    """Compose ``A = U_f . U_path`` and return ``(circuit, objective_qubit)``.

    ``u_path`` acts on ``n`` index qubits; ``u_f`` on ``n + 1`` qubits
    (index + flag).  The flag is qubit ``n``.
    """
    n = u_path.num_qubits
    if u_f.num_qubits != n + 1:
        raise ValueError("u_f must act on (u_path qubits + 1) qubits.")
    a = QuantumCircuit(n + 1, name="A")
    a.compose(u_path, qubits=range(n), inplace=True)
    a.compose(u_f, qubits=range(n + 1), inplace=True)
    return a, n


def estimate_amplitude_exact(encoder: QuantumCircuit,
                             objective_qubit: int) -> AmplitudeEstimate:
    """Exact ``a = P(objective = 1)`` from the statevector of ``encoder``."""
    sv = Statevector.from_instruction(encoder)
    p0, p1 = sv.probabilities([objective_qubit])
    return AmplitudeEstimate(amplitude=float(p1), method="exact_statevector")


def build_amplitude_encoder_qsvt(u_qsvt: QuantumCircuit, u_f: QuantumCircuit,
                                 n: int) -> tuple[QuantumCircuit, int, list[int]]:
    r"""Compose the QSVT loader with the payoff oracle.

    ``u_qsvt`` acts on ``n + 2`` qubits (register ``0..n-1`` + signal ancilla
    ``n`` + LCU ancilla ``n+1``); ``u_f`` acts on ``n + 1`` qubits (register +
    flag).  The combined circuit uses ``n + 3`` qubits: register ``0..n-1``,
    ancillas ``n, n+1``, flag ``n+2``.

    Returns ``(circuit, flag_qubit, ancilla_qubits)``.
    """
    qc = QuantumCircuit(n + 3, name="A_qsvt")
    qc.compose(u_qsvt, qubits=range(n + 2), inplace=True)
    qc.compose(u_f, qubits=list(range(n)) + [n + 2], inplace=True)
    return qc, n + 2, [n, n + 1]


def estimate_amplitude_exact_conditional(encoder: QuantumCircuit, flag_qubit: int,
                                         ancilla_qubits: list[int]) -> AmplitudeEstimate:
    r"""Exact QMCI amplitude when the loader post-selects ancillas.

    With the QSVT loader the encoder produces

        A|0> = F |psi_target> |0...0>_anc (flag part) + ... (ancillas != 0),

    so the bare ``P(flag=1)`` is scaled by the squared filling fraction ``F^2``.
    Conditioning on the ancillas being ``|0>`` removes it exactly:

        a = P(flag = 1 AND anc = 0) / P(anc = 0).
    """
    sv = Statevector.from_instruction(encoder)
    p_load = float(sv.probabilities(ancilla_qubits)[0])           # P(anc = 0)
    qargs = list(ancilla_qubits) + [flag_qubit]
    idx_good = 1 << len(ancilla_qubits)                           # anc=0, flag=1
    p_good = float(sv.probabilities(qargs)[idx_good])
    if p_load <= 0:
        raise RuntimeError("Loader success probability is zero.")
    return AmplitudeEstimate(amplitude=p_good / p_load,
                             method="exact_statevector_conditional")


def _reflect_good(nq: int, flag: int, ancillas: list[int]) -> QuantumCircuit:
    """Phase flip on the good subspace: ``flag = 1`` and all ``ancillas = 0``."""
    qc = QuantumCircuit(nq, name="S_chi")
    for q in ancillas:
        qc.x(q)
    qc.h(flag)
    qc.mcx(ancillas, flag)     # with the X's above: marks ancillas=0, flag=1
    qc.h(flag)
    for q in ancillas:
        qc.x(q)
    return qc


def _reflect_zero(nq: int) -> QuantumCircuit:
    """Phase flip about the all-zero state on ``nq`` qubits."""
    qc = QuantumCircuit(nq, name="S_0")
    qc.x(range(nq))
    qc.h(nq - 1)
    qc.mcx(list(range(nq - 1)), nq - 1)
    qc.h(nq - 1)
    qc.x(range(nq))
    return qc


def grover_operator_for_good(encoder: QuantumCircuit, flag: int,
                             ancillas: list[int]) -> QuantumCircuit:
    r"""Grover/AE operator ``Q = A . S_0 . A^dagger . S_chi`` for the good subspace.

    ``S_chi`` flips the phase of states with ``flag = 1`` and all ``ancillas = 0``;
    ``S_0`` reflects about the all-zero state.  Applying ``Q^m`` after ``A`` makes
    the good probability ``sin^2((2m+1) theta)`` with ``sin^2(theta)`` the good
    amplitude of ``A|0>`` -- the basis of (shot-based) amplitude estimation.
    """
    nq = encoder.num_qubits
    q = QuantumCircuit(nq, name="Q")
    q.compose(_reflect_good(nq, flag, ancillas), inplace=True)
    q.compose(encoder.inverse(), inplace=True)
    q.compose(_reflect_zero(nq), inplace=True)
    q.compose(encoder, inplace=True)
    return q


def estimate_amplitude_mlae(encoder: QuantumCircuit, flag: int,
                            ancillas: list[int], filling_sq: float,
                            shots: int = 2048,
                            powers: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13),
                            seed: int = 1234) -> AmplitudeEstimate:
    r"""**Shot-based** Maximum-Likelihood Amplitude Estimation through the loader.

    Runs ``A . Q^m`` on the Aer simulator for each ``m in powers``, counts the
    good outcomes (``flag=1`` & ancillas ``0``), and maximum-likelihood-fits the
    good amplitude ``a_good = sin^2(theta)`` using ``P(good|m) = sin^2((2m+1)theta)``.

    The QSVT loader post-selects its ancillas with probability ``F^2 =
    filling_sq``; the QMCI amplitude is therefore ``a = a_good / F^2``.  This is a
    genuine amplitude-estimation run -- ``O(1/eps)`` oracle calls -- not a
    statevector read-out.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    nq = encoder.num_qubits
    Q = grover_operator_for_good(encoder, flag, ancillas)
    sim = AerSimulator(seed_simulator=seed)

    def run(m: int) -> tuple[int, int]:
        qc = QuantumCircuit(nq, nq)
        qc.compose(encoder, inplace=True)
        for _ in range(m):
            qc.compose(Q, inplace=True)
        qc.measure(range(nq), range(nq))
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        good = 0
        for bitstr, c in counts.items():
            b = bitstr[::-1]            # qiskit is little-endian in the string
            if b[flag] == "1" and all(b[a] == "0" for a in ancillas):
                good += c
        return good, shots

    data = [(m, *run(m)) for m in powers]

    thetas = np.linspace(1e-4, np.pi / 2 - 1e-4, 30000)
    loglik = np.zeros_like(thetas)
    for m, g, s in data:
        p = np.clip(np.sin((2 * m + 1) * thetas) ** 2, 1e-12, 1 - 1e-12)
        loglik += g * np.log(p) + (s - g) * np.log(1 - p)
    a_good = float(np.sin(thetas[int(np.argmax(loglik))]) ** 2)

    queries = int(sum(2 * m + 1 for m in powers))
    a = a_good / filling_sq if filling_sq > 0 else a_good
    return AmplitudeEstimate(amplitude=a, method="mlae_shots",
                             oracle_queries=queries)


def estimate_amplitude_iqae(encoder: QuantumCircuit,
                            objective_qubit: int,
                            epsilon_target: float = 0.01,
                            alpha: float = 0.05,
                            shots: int = 4096) -> AmplitudeEstimate:
    """Iterative Amplitude Estimation (real QAE) if ``qiskit_algorithms`` exists."""
    try:
        from qiskit_algorithms import (
            EstimationProblem,
            IterativeAmplitudeEstimation,
        )
        from qiskit.primitives import Sampler
    except Exception:
        # Graceful fallback -- keeps the package usable without the extra dep.
        est = estimate_amplitude_exact(encoder, objective_qubit)
        est.epsilon_target = epsilon_target
        return est

    problem = EstimationProblem(
        state_preparation=encoder,
        objective_qubits=[objective_qubit],
    )
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,
        alpha=alpha,
        sampler=Sampler(options={"shots": shots}),
    )
    result = iae.estimate(problem)
    ci = tuple(result.confidence_interval) if result.confidence_interval else None
    return AmplitudeEstimate(
        amplitude=float(result.estimation),
        method="iqae",
        epsilon_target=epsilon_target,
        confidence_interval=ci,
        oracle_queries=int(getattr(result, "num_oracle_queries", 0)) or None,
    )


__all__ = [
    "AmplitudeEstimate",
    "build_amplitude_encoder",
    "build_amplitude_encoder_qsvt",
    "grover_operator_for_good",
    "estimate_amplitude_exact",
    "estimate_amplitude_exact_conditional",
    "estimate_amplitude_mlae",
    "estimate_amplitude_iqae",
]
