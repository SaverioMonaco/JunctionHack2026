r"""qff_pricing.scaling.advantage
=================================

``T``-dependent resource models for *path-dependent* pricing, and the end-to-end
quantum advantage.  This complements :mod:`qff_pricing.scaling.resources` (which
focuses on the European ``T=1`` data-point scaling) by making the **monitoring
points ``T``** the asymptotic axis -- the regime that matters once the payoff is
path-dependent (Asian, barrier, lookback).

Three stories, all from the paper:

1. **Fast-forwarding vs naive (gates vs T).**  Naively amplitude-encoding the
   joint law of the ``T``-step path on a grid of ``N = 2^n`` points per step is
   an arbitrary state on ``T n`` qubits -> ``Theta(N^T) = Theta(2^{T n})`` gates,
   **exponential in T**.  Fast-forwarding loads ``O(T)`` one-dimensional
   primitives + ``O(T)`` arithmetic jumps -> ``poly(T)`` gates (CIR Thm
   ``cir_qsample_resources``; Heston Thm ``heston_loading_resource``).

2. **Quantum advantage (cost vs accuracy eps).**  Classical Monte-Carlo needs
   ``O(1/eps^2)`` paths; quantum QMCI needs ``O(1/eps)`` queries -- and because
   the loading overhead is only ``poly(T)``, the quadratic advantage survives
   end-to-end (Thm ``cir-main``, ``heston-main``).

3. **The ``Omega(T)`` barrier (qubits vs T).**  A path-dependent payoff sees all
   ``T`` monitoring points, so *any* coherent scheme needs ``Omega(T)`` qubits
   and gates (Contribution 5 -- no sublinear-in-T simulation).  The runnable
   circuit in :mod:`qff_pricing.quantum.fast_forward` realises exactly the linear
   ``T * (2 n_inc + n_v)`` qubit growth.

Big-O constants are set to 1 (these are *trends*), matching the convention of
:mod:`qff_pricing.scaling.resources`.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
#  (1) Gates vs T  --  fast-forwarding (poly) vs naive (exp)
# --------------------------------------------------------------------------- #
def naive_path_log2_gates(T: int, n: int, d: int = 1) -> float:
    r"""log2 of the naive joint-path amplitude-encoding cost = ``T d n``.

    The cost itself is ``2^{T d n}`` (arbitrary state on ``T d n`` qubits); we
    return the exponent so it can be plotted without overflowing.  Linear in T
    on a log-axis == exponential in T.
    """
    return float(T * d * n)


def cir_ff_gates(T: int, n: int, B: float = 1.0, r: float = 5.0,
                 b: float = 6.0, eps: float = 1e-3) -> float:
    r"""CIR fast-forwarding gate count (Thm ``cir_qsample_resources``), poly(T).

        O( T^2 log^2 N log T + T (b+r) log N log(B T b / eps) + log^3(B T b N) ).
    """
    N = 2.0 ** n
    logN = math.log2(N)
    logT = math.log2(max(T, 2))
    return float(T ** 2 * logN ** 2 * logT
                 + T * (b + r) * logN * math.log2(max(B * T * b / eps, 2.0))
                 + math.log2(max(B * T * b * N, 2.0)) ** 3)


def heston_ff_gates(T: int, n: int, d: int = 1, b: float = 6.0) -> float:
    r"""Heston fast-forwarding gate count (Thm ``heston_loading_resource``), poly(T,d).

        O( T^4 d^2 log^2 N + T d b log N ).  Larger polynomial than CIR (nested
        step + black-box integrated-variance loader) but still polynomial in T.
    """
    N = 2.0 ** n
    logN = math.log2(N)
    return float(T ** 4 * d ** 2 * logN ** 2 + T * d * b * logN)


def naive_log2_minus_ff(T: int, n: int) -> float:
    """How many orders of magnitude (log2) naive exceeds CIR fast-forwarding."""
    return naive_path_log2_gates(T, n) - math.log2(cir_ff_gates(T, n))


# --------------------------------------------------------------------------- #
#  (2) Quantum advantage  --  cost vs accuracy eps (with poly(T) prefactor)
# --------------------------------------------------------------------------- #
def classical_mc_cost(eps: float, T: int = 1, n: int = 10) -> float:
    r"""Classical MC end-to-end cost ~ (path work) * O(1/eps^2).

    Path work per sample is ``O(T)`` (one fast-forward step chain); samples
    needed ``O(1/eps^2)``.
    """
    return T / eps ** 2


def quantum_qmci_cost(eps: float, T: int = 1, n: int = 10) -> float:
    r"""Quantum QMCI end-to-end cost ~ (loading work) * O(1/eps).

    Loading work is ``poly(T)`` (here the CIR FF gate model) and QAE uses
    ``O(1/eps)`` queries -- the quadratic improvement that *survives* because the
    loading overhead is only polynomial in T.
    """
    return cir_ff_gates(T, n) / eps


def advantage_factor(eps: float, T: int = 1, n: int = 10) -> float:
    """Classical / quantum end-to-end cost ratio (grows like ~1/eps)."""
    return classical_mc_cost(eps, T, n) / quantum_qmci_cost(eps, T, n)


# --------------------------------------------------------------------------- #
#  (3) Qubits vs T  --  the Omega(T) barrier (Contribution 5)
# --------------------------------------------------------------------------- #
def ff_qubits(T: int, n_inc: int, n_v: int) -> int:
    r"""Qubits used by the runnable CIR FF circuit: ``T (2 n_inc + n_v) + 1``.

    Two increment registers (Y, Z) + one variance register per step, + a flag.
    Linear in T -- the realisation of the ``Omega(T)`` lower bound.
    """
    return T * (2 * n_inc + n_v) + 1


def lower_bound_qubits(T: int, n: int = 1) -> int:
    """The unavoidable ``Omega(T)`` floor: at least one register per monitoring point."""
    return T * max(n, 1)


# --------------------------------------------------------------------------- #
#  Measured anchors from the actual runnable circuit
# --------------------------------------------------------------------------- #
def measured_circuit_resources(T: int, n_inc: int = 2, n_v: int = 2,
                               strike: float = 0.038, with_gates: bool = True) -> dict:
    """Build the real CIR FF circuit and report measured qubits / transpiled gates.

    NOTE: the demo's ``U_jump`` is the exact (non-scalable) lookup table, so its
    *gate* count is NOT the poly(T) FF figure -- only the *qubit* count is the
    honest linear-in-T resource.  Returned here so plots can anchor the qubit
    axis on a real circuit and clearly separate measured vs modelled.
    """
    from qff_pricing.data import make_synthetic_dataset
    from dataclasses import replace
    from qff_pricing.models.cir import CIRModel
    from qff_pricing.quantum.fast_forward import build_cir_ff_path, add_asian_payoff

    ds = make_synthetic_dataset("cir", seed=0)
    model = CIRModel(ds.cir, ds.contract.maturity / T)
    ff = build_cir_ff_path(model, T, n_inc=n_inc, n_v=n_v)
    add_asian_payoff(ff, strike=strike)
    out = {"T": T, "qubits": ff.circuit.num_qubits}
    if with_gates:
        from qiskit import transpile
        out["gates_lookup_demo"] = transpile(
            ff.circuit, basis_gates=["cx", "u"], optimization_level=0).size()
    return out


__all__ = [
    "naive_path_log2_gates", "cir_ff_gates", "heston_ff_gates",
    "naive_log2_minus_ff", "classical_mc_cost", "quantum_qmci_cost",
    "advantage_factor", "ff_qubits", "lower_bound_qubits",
    "measured_circuit_resources",
]
