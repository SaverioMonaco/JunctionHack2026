r"""qff_pricing.scaling.resources
=================================

Resource models for amplitude-encoding the discretised price.

All "theory" functions evaluate the **asymptotic formulas of the paper with the
hidden big-O constants set to 1**.  They are meant to reveal *trends* (slopes on
a log plot), not exact gate tallies.  The qualitative conclusion is robust:

  * **Naive amplitude encoding** of the full ``T``-step, ``d``-asset path
    distribution loads an arbitrary state on ``Q = T * d * n`` qubits, costing
    ``O(2**Q) = O(N**(T d))`` gates -- **exponential in T**.

  * **Fast-forwarding** (CIR Thm; Heston Thm) loads ``O(T)`` one-dimensional
    primitives plus ``O(T)`` arithmetic jumps, costing only
    ``poly(T, d, log(1/eps))`` gates -- **polynomial in T, polylog in 1/eps**.

  * The number of grid points needed per primitive, ``N``, scales only
    *polynomially* in ``1/eps`` (CIR) / with a controlled exponential prefactor
    in ``T`` (Heston), so the qubit count ``n = log2(N)`` is ``polylog(1/eps)``.

Naming
------
``T``  monitoring points (integration dimension);   ``d`` number of assets;
``n``  qubits per primitive (``N = 2**n`` grid points);
``B``  payoff Lipschitz slope;   ``r`` ~ chi-square dof (``eta``);
``b``  truncation radius;        ``eps`` target additive error;   ``Nf`` payoff gate cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Naive baseline -- exponential amplitude encoding
# ---------------------------------------------------------------------------


def naive_encoding_gates(T: int, n: int, d: int = 1) -> float:
    r"""Gate cost of directly amplitude-encoding the full path state.

    A general state on ``Q = T*d*n`` qubits needs ``Theta(2**Q)`` gates
    (Shende-Bullock-Markov).  This is the exponential cost fast-forwarding avoids.
    Returned in log10 units would overflow for large T, so we return the value
    in **log2 form is the exponent**; here we return the raw (possibly huge) float
    via ``2**Q`` guarded against overflow.
    """
    Q = T * d * n
    # guard: cap to avoid OverflowError; callers usually plot log2 = Q directly.
    return math.inf if Q > 1000 else float(2 ** Q)


def naive_encoding_log2_gates(T: int, n: int, d: int = 1) -> float:
    """``log2`` of :func:`naive_encoding_gates` = ``T*d*n`` (linear in T -> slope)."""
    return float(T * d * n)


# ---------------------------------------------------------------------------
# CIR fast-forwarding (paper Sec. 6.1.1)
# ---------------------------------------------------------------------------


def cir_required_bits(T: int, B: float, eta: float,
                      eps_disc: float, eps_trunc: float) -> float:
    r"""Bits per primitive distribution (CIR Discretisation Thm).

        n = O( log( eta B T / (eps_disc eps_trunc) ) ).
    """
    return math.log2(max(eta * B * T / (eps_disc * eps_trunc), 2.0))


def cir_ff_gates(T: int, n: int, B: float = 1.0, r: float = 5.0,
                 b: float = 6.0, eps: float = 1e-3, Nf: float = 0.0) -> float:
    r"""CIR fast-forwarding gate count (CIR Discrete-Sum Loading Thm).

        O( T^2 log^2(N) log(T)
           + T (b + r) log(N) log(B T b / eps)
           + log^3(B T b N)
           + Nf ).
    """
    N = 2 ** n
    logN = math.log2(N)
    logT = math.log2(max(T, 2))
    term1 = T ** 2 * logN ** 2 * logT
    term2 = T * (b + r) * logN * math.log2(max(B * T * b / eps, 2.0))
    term3 = math.log2(max(B * T * b * N, 2.0)) ** 3
    return float(term1 + term2 + term3 + Nf)


def cir_state_qubits(T: int, n: int) -> float:
    r"""Qubits the qsample lives on:  O(T log N) = T * n."""
    return float(T * n)


def cir_total_qubits(T: int, n: int, B: float = 1.0) -> float:
    r"""Total working qubits:  O(T log(B T N))."""
    N = 2 ** n
    return float(T * math.log2(max(B * T * N, 2.0)))


# ---------------------------------------------------------------------------
# Heston fast-forwarding (paper Sec. 6.2.1)
# ---------------------------------------------------------------------------


def heston_required_qubits_total(T: int, d: int, eps_disc: float,
                                 eps_trunc: float) -> float:
    r"""Total (qu)bits for Heston (Heston Discretisation Thm):

        O( T^4 d^2 log^{3/2}(T d / eps_trunc) log(T d / eps_disc) ).
    """
    a = math.log2(max(T * d / eps_trunc, 2.0))
    bb = math.log2(max(T * d / eps_disc, 2.0))
    return float(T ** 4 * d ** 2 * (a ** 1.5) * bb)


def heston_ff_gates(T: int, n: int, d: int = 1, B: float = 1.0,
                    b: float = 6.0, Nf: float = 0.0) -> float:
    r"""Representative Heston fast-forwarding gate count.

    The theorem states the cost is ``O(poly(T, d, b, log B) + Nf)`` without
    fixing the polynomial degree (the proof's degree is large but finite).  We
    use a concrete representative polynomial that is faithful to the *structure*
    (it reduces to the CIR cost at ``d = 1`` up to the integral-of-CIR loader):

        O( T^4 d^2 log^2(N) + T d (b) log(N) + Nf ).

    The key point -- polynomial in ``T`` and ``d`` -- is what matters versus the
    exponential naive baseline.
    """
    N = 2 ** n
    logN = math.log2(N)
    return float(T ** 4 * d ** 2 * logN ** 2 + T * d * b * logN + Nf)


def heston_state_qubits(T: int, n: int, d: int = 1) -> float:
    r"""Qubits the Heston qsample lives on:  O(d T log N) = d * T * n."""
    return float(d * T * n)


# ---------------------------------------------------------------------------
# EUROPEAN OPTIONS (T = 1): scaling in the number of DATA POINTS N = 2**n
# ---------------------------------------------------------------------------
#
# A European payoff only sees the terminal value, so there is a single
# monitoring point (T = 1) and the path collapses to one distribution-loading
# problem: amplitude-encode the terminal law on a grid of ``N = 2**n`` points.
#
#   * Naive amplitude encoding of an arbitrary N-amplitude state costs
#     ``Theta(N)`` gates (Shende-Bullock-Markov)  ->  LINEAR in the number of
#     data points, i.e. EXPONENTIAL in the qubit count ``n = log2 N``.
#
#   * The paper's polynomial-approximation / QET loader (Lemma "Polynomial
#     Approximation State Preparation", Theorem "Arithmetic-Free Loading")
#     costs ``O(log(N) * d_delta)`` gates, where ``d_delta = polylog(1/eps)`` is
#     the degree of the polynomial approximating sqrt(p)  ->  POLYLOGARITHMIC in
#     the number of data points N.
#
#   * The black-box (Grover/Sanders) loader costs ``O(log^3(Lambda/delta))``
#     gates -- also polylog, with no explicit N dependence.
#
# These are the curves that answer "polylog of the number of data points instead
# of exponential amplitude encoding" for European options.


def european_naive_loading_gates(n: int) -> float:
    r"""Naive amplitude-encoding gate cost: ``Theta(N) = Theta(2**n)``."""
    return float(2 ** n)


def _poly_degree(eps: float, span: float = 12.0) -> float:
    r"""Representative polynomial degree ``d_delta = polylog(1/eps)``.

    The paper guarantees a degree ``O(polylog(1/delta))`` approximation of
    ``sqrt(p(.))`` for every primitive it loads (Gaussian, chi^2, ...), with
    ``delta = eps / (b-a)^2``.  We use ``d ~ (log2((b-a)^2 / eps))^2`` as a
    concrete polylog stand-in (constants set to 1).
    """
    delta = eps / (span ** 2)
    return max(math.log2(1.0 / max(delta, 1e-300)), 1.0) ** 2


def european_poly_loading_gates(n: int, eps: float = 1e-3,
                                span: float = 12.0) -> float:
    r"""Polynomial-approximation (QET) loader: ``O(log(N) * polylog(1/eps))``.

    This is the realisation the paper uses; it is **polylogarithmic in the
    number of data points** ``N = 2**n``.
    """
    logN = float(n)  # log2(N)
    return logN * _poly_degree(eps, span)


def european_blackbox_loading_gates(eps: float = 1e-3, span: float = 12.0,
                                    lam: float = 1.0) -> float:
    r"""Black-box (Grover) loader: ``O(log^3(Lambda/delta))`` -- polylog, no N."""
    delta = eps ** 2 / (lam ** 3 * span ** 4)
    return max(math.log2(lam / max(delta, 1e-300)), 1.0) ** 3


def european_qubits_for_accuracy(eps: float, max_deriv: float = 1.0,
                                 span: float = 12.0) -> float:
    r"""Qubits ``n = log2 N`` needed for discretisation error ``eps`` (T = 1).

    The number of grid points must satisfy ``N = Omega((b-a) max|p'| / eps)``
    (Theorem "Arithmetic-Free Loading"), so ``n = log2 N = O(log(1/eps))`` --
    polylogarithmic accuracy scaling.
    """
    N_required = span * max_deriv / eps
    return math.log2(max(N_required, 2.0))


def european_data_points_for_accuracy(eps: float, max_deriv: float = 1.0,
                                      span: float = 12.0) -> float:
    r"""Number of data points ``N`` needed for error ``eps``: ``O(1/eps)``."""
    return span * max_deriv / eps


def qubits_for_data_points(N: float) -> float:
    r"""Qubits needed to amplitude-encode ``N`` data points: ``n = log2(N)``.

    The defining feature of amplitude encoding: ``N`` data points are packed into
    only ``log2(N)`` qubits (exponential compression).  Independent of the model.
    """
    return math.log2(max(N, 2.0))


# --- model-specific European loaders (same polylog(N) scaling, diff. constants) ---


def cir_european_loading_gates(n: int, eps: float = 1e-3,
                               span: float = 12.0) -> float:
    r"""CIR terminal loader (European, T=1).

    The terminal law is a scaled non-central chi-square loaded with the QET /
    polynomial-approximation state-prep:  ``O(log(N) * polylog(1/eps))`` gates.
    Same as :func:`european_poly_loading_gates`; named for clarity in plots.
    """
    return float(n) * _poly_degree(eps, span)


def heston_european_loading_gates(n: int, eps: float = 1e-3, span: float = 12.0,
                                  max_v: float = 1.0) -> float:
    r"""Heston terminal loader (European, T=1).

    Loading the terminal price requires the **integral-of-CIR** primitive, whose
    black-box loader costs (Corollary "Integral of CIR Loading")

        O( (max_t V)^4 [ log^2(N) log^2(max_t V * N/(b-a)) + log^3(N) ] )

    gates -- a larger prefactor and a ``log^3 N`` term than CIR, but **still
    polylogarithmic in the number of data points** ``N = 2**n``.
    """
    logN = float(n)
    inner = math.log2(max(max_v * (2 ** n) / span, 2.0))
    return (max_v ** 4) * (logN ** 2 * inner ** 2 + logN ** 3)


# ---------------------------------------------------------------------------
# Sampling complexity: classical MC vs quantum QMCI (the quadratic speedup)
# ---------------------------------------------------------------------------


def classical_mc_samples(eps: float) -> float:
    r"""Classical Monte-Carlo sample complexity ~ ``1/eps^2``."""
    return 1.0 / eps ** 2


def quantum_qmci_queries(eps: float) -> float:
    r"""Quantum QMCI (amplitude estimation) query complexity ~ ``1/eps``."""
    return 1.0 / eps


# ---------------------------------------------------------------------------
# Measured resources from an actual Qiskit circuit
# ---------------------------------------------------------------------------


@dataclass
class MeasuredResources:
    n_qubits: int
    depth: int
    size: int
    gate_counts: dict


def measure_resources(circuit, basis_gates=("cx", "u")) -> MeasuredResources:
    """Transpile ``circuit`` to a basic gate set and count gates / depth."""
    from qiskit import transpile

    t = transpile(circuit, basis_gates=list(basis_gates), optimization_level=0)
    return MeasuredResources(
        n_qubits=t.num_qubits,
        depth=t.depth(),
        size=t.size(),
        gate_counts=dict(t.count_ops()),
    )


__all__ = [
    "naive_encoding_gates",
    "naive_encoding_log2_gates",
    "cir_required_bits",
    "cir_ff_gates",
    "cir_state_qubits",
    "cir_total_qubits",
    "heston_required_qubits_total",
    "heston_ff_gates",
    "heston_state_qubits",
    "european_naive_loading_gates",
    "european_poly_loading_gates",
    "european_blackbox_loading_gates",
    "european_qubits_for_accuracy",
    "european_data_points_for_accuracy",
    "qubits_for_data_points",
    "cir_european_loading_gates",
    "heston_european_loading_gates",
    "classical_mc_samples",
    "quantum_qmci_queries",
    "MeasuredResources",
    "measure_resources",
]
