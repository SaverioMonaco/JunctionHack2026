"""Comparison & headline numbers (plan §6, Phase 6).

The demo's spine: PCE qubit count vs QAOA (n = m), and PCE settled value vs the
exact ILP/brute-force optimum. Plus the pure-arithmetic projection table that
carries the scaling pitch.
"""
from __future__ import annotations

from math import comb

import numpy as np

from . import pauli
from .instance import Instance


def qubit_counts(m: int, k: int) -> dict:
    """PCE qubit count n vs QAOA's n = m for m binary variables."""
    n_pce = pauli.qubits_for(m, k)
    return {
        "m": m,
        "k": k,
        "qaoa_qubits": m,
        "pce_qubits": n_pce,
        "compression": m / n_pce if n_pce else float("inf"),
        "pce_layers": pauli.layers_for(m, n_pce),
    }


def projection_table(k: int = 3) -> list[dict]:
    """Pure-arithmetic reach table for the pitch (plan §6)."""
    rows = []
    for n in (12, 17, 50, 100, 200):
        rows.append({"qubits": n, "addressable_vars": 3 * comb(n, k)})
    return rows


def quality(pce_value: float, optimum: float) -> float:
    """Approximation ratio: PCE settled value / exact optimum (aim >= 0.95)."""
    if optimum == 0:
        return 1.0 if pce_value == 0 else 0.0
    return pce_value / optimum


def greek_metrics(inst: Instance, x_pce: np.ndarray, x_naive: np.ndarray,
                  targets: np.ndarray | None = None) -> dict:
    """Net-Greek residuals for the PCE vs the Greek-naive settlement.

    Returns per-Greek residuals and residual norms for both, plus the
    neutralization ratio 1 - ||resid_pce|| / ||resid_naive|| ("PCE cut residual
    Greek exposure by X%"). Residual = net_greeks(x) - targets.
    """
    G = len(inst.greek_names)
    tgt = np.zeros(G) if targets is None else np.asarray(targets, dtype=float)
    resid_pce = inst.net_greeks(x_pce) - tgt
    resid_naive = inst.net_greeks(x_naive) - tgt
    norm_pce = float(np.linalg.norm(resid_pce))
    norm_naive = float(np.linalg.norm(resid_naive))
    neutralization = (1.0 - norm_pce / norm_naive) if norm_naive > 1e-12 else 0.0
    return {
        "greek_names": inst.greek_names,
        "resid_pce": resid_pce,
        "resid_naive": resid_naive,
        "norm_pce": norm_pce,
        "norm_naive": norm_naive,
        "neutralization": neutralization,
    }


def format_greek_summary(inst: Instance, x_pce: np.ndarray, x_naive: np.ndarray,
                         targets: np.ndarray | None = None) -> str:
    """Human-readable net-Greek before/after block for the demo."""
    gm = greek_metrics(inst, x_pce, x_naive, targets)
    names = gm["greek_names"]
    lines = [
        "-" * 56,
        "GREEK NEUTRALITY (net exposure, target = 0)",
        "-" * 56,
        f"{'greek':<10}{'naive settle':>16}{'Greek-aware PCE':>20}",
    ]
    for g, nm in enumerate(names):
        lines.append(f"{nm:<10}{gm['resid_naive'][g]:>16.3f}"
                     f"{gm['resid_pce'][g]:>20.3f}")
    lines += [
        "-" * 56,
        f"residual norm  naive={gm['norm_naive']:.3f}  "
        f"PCE={gm['norm_pce']:.3f}",
        f"neutralization ratio         : {gm['neutralization']:.1%} "
        f"(residual cut)",
        "-" * 56,
    ]
    return "\n".join(lines)


def format_summary(m: int, k: int, pce_value: float, optimum: float,
                   baseline_name: str) -> str:
    qc = qubit_counts(m, k)
    ratio = quality(pce_value, optimum)
    lines = [
        "=" * 56,
        "PCE SETTLEMENT - HEADLINE",
        "=" * 56,
        f"binary variables m           : {m}",
        f"PCE qubits (k={k})            : {qc['pce_qubits']}  (layers p={qc['pce_layers']})",
        f"QAOA qubits (n=m)            : {qc['qaoa_qubits']}",
        f"qubit compression            : {qc['compression']:.1f}x",
        "-" * 56,
        f"PCE settled value            : {pce_value:.1f}",
        f"{baseline_name} optimum{' ' * (21 - len(baseline_name))}: {optimum:.1f}",
        f"approximation ratio          : {ratio:.3f}",
        "=" * 56,
    ]
    return "\n".join(lines)
