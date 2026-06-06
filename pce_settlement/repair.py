"""Feasibility check and classical repair (plan §1, Phase 4).

Penalty-based readouts can violate a balance constraint. A demo that "settles"
an infeasible batch is worthless (plan §6), so we always repair before
reporting: drop the lowest-value transaction from the most-violated party until
feasible, then greedily re-add any transaction that now fits to recover value.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance


def feasible(x: np.ndarray, inst: Instance) -> bool:
    """True iff decision vector x in {0,1}^M keeps every party solvent.

    Feasibility is *solvency only*. Greek neutrality is a soft objective (a
    penalty in the QUBO), not a hard constraint, so it never gates feasibility
    or repair -- see greek_residual for reporting net-Greek exposure.
    """
    x = np.asarray(x).astype(int)[: inst.M]
    return bool((inst.net_flow(x) <= inst.balances).all())


def greek_residual(x: np.ndarray, inst: Instance,
                   targets: np.ndarray | None = None) -> np.ndarray:
    """Residual net Greek exposure net_greeks(x) - targets (reporting only).

    Returns a (G,) array. ``targets`` defaults to zero (full neutrality). Used
    by compare.py / plots.py to show how far the settled book is from neutral.
    """
    net = inst.net_greeks(x)
    if targets is None:
        return net
    return net - np.asarray(targets, dtype=float)


def repair(x: np.ndarray, inst: Instance,
           lambdas: np.ndarray | None = None,
           targets: np.ndarray | None = None) -> np.ndarray:
    """Return a feasible decision vector close to x (plan §1 repair).

    1. While infeasible: from the most-violated party, drop the settled
       transaction it *sends* with the lowest score-gain.
    2. Greedy re-add pass: turn back on any currently-off transaction that
       still leaves the batch feasible and improves the objective.

    Objective is settled value when ``lambdas`` is None (cash-only settlement),
    or the Greek-aware augmented score ``value - sum_g lambda_g (net_g - tgt)^2``
    when ``lambdas`` is given. Greek-awareness matters: a value-greedy re-add
    would add high-value trades that push net Greeks away from neutral, undoing
    the QUBO's neutrality steering. With ``lambdas`` set, a trade is re-added
    only if it raises the augmented score.
    """
    x = np.asarray(x).astype(int)[: inst.M].copy()
    use_greeks = lambdas is not None and inst.greeks is not None

    def score(xv: np.ndarray) -> float:
        v = inst.value(xv)
        if not use_greeks:
            return v
        G = inst.greeks.shape[1]
        tgt = np.zeros(G) if targets is None else np.asarray(targets, dtype=float)
        resid = inst.net_greeks(xv) - tgt
        return v - float(np.asarray(lambdas) @ (resid ** 2))

    # ---- Phase 1: drop until feasible -------------------------------------
    while True:
        violation = inst.net_flow(x) - inst.balances  # >0 means over budget
        if (violation <= 0).all():
            break
        p = int(np.argmax(violation))  # most-violated party
        # Candidate drops: transactions party p SENDS that are currently on.
        # Dropping a sent tx reduces p's outflow. (Dropping a received tx would
        # raise p's net flow, never helps.)
        cand = [t for t in range(inst.M)
                if x[t] == 1 and inst.senders[t] == p]
        if not cand:
            # No outgoing tx to drop for the worst party: drop any settled tx.
            cand = [t for t in range(inst.M) if x[t] == 1]
            if not cand:
                break
        # Drop the tx whose removal hurts the objective least (i.e. gives the
        # highest score after dropping). Cheap: one score eval per candidate.
        def score_without(t: int) -> float:
            x[t] = 0
            s = score(x)
            x[t] = 1
            return s
        t_drop = max(cand, key=score_without)
        x[t_drop] = 0

    # ---- Phase 2: greedy re-add to improve the objective ------------------
    # Try each off transaction; keep it only if it stays feasible AND raises the
    # objective. Repeat to a fixed point so additions enabled by earlier ones
    # are considered.
    improved = True
    while improved:
        improved = False
        base = score(x)
        best_t, best_gain = None, 1e-9
        for t in range(inst.M):
            if x[t] == 1:
                continue
            x[t] = 1
            if feasible(x, inst):
                gain = score(x) - base
                if gain > best_gain:
                    best_t, best_gain = t, gain
            x[t] = 0
        if best_t is not None:
            x[best_t] = 1
            improved = True

    return x
