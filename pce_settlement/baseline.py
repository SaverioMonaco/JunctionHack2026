"""Classical baselines: the honest competitors (plan §5, Phase 5).

- brute_force: exact, exponential, for M <= ~20. Defines "truth".
- greedy: value-sorted feasible insertion. Fast, suboptimal.
- ilp_optimal: exact ILP via PuLP if installed (medium M). Optional.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .instance import Instance


def _feasible(x: np.ndarray, inst: Instance) -> bool:
    return bool((inst.net_flow(x) <= inst.balances).all())


def _augmented_objective(inst: Instance, x: np.ndarray,
                         lambdas: np.ndarray | None,
                         targets: np.ndarray | None) -> float:
    """Settled value minus the Greek-neutrality penalty (the maximized score).

    score(x) = sum_t w_t x_t - sum_g lambda_g (net_g(x) - target_g)^2

    With ``lambdas=None`` this is just the settled value (the cash-only optimum).
    """
    v = inst.value(x)
    if lambdas is None or inst.greeks is None:
        return v
    G = inst.greeks.shape[1]
    tgt = np.zeros(G) if targets is None else np.asarray(targets, dtype=float)
    resid = inst.net_greeks(x) - tgt
    return v - float(np.asarray(lambdas, dtype=float) @ (resid ** 2))


def brute_force(inst: Instance, lambdas: np.ndarray | None = None,
                targets: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Exact optimum by enumerating all 2^M cash-feasible subsets (M <= ~20).

    Maximizes the augmented objective ``value(x) - Greek penalty`` (see
    ``_augmented_objective``). With ``lambdas=None`` this is the pure settled-
    value optimum (identical to the original behaviour). The returned value is
    always the *settled value* inst.value(best_x) so callers compare like with
    like; the Greek penalty only steers which subset wins.
    """
    if inst.M > 24:
        raise ValueError(f"brute_force infeasible for M={inst.M} (> 24)")
    best_x = np.zeros(inst.M, dtype=int)
    best_score = _augmented_objective(inst, best_x, lambdas, targets)
    for bits in product((0, 1), repeat=inst.M):
        x = np.array(bits, dtype=int)
        if _feasible(x, inst):
            s = _augmented_objective(inst, x, lambdas, targets)
            if s > best_score:
                best_score, best_x = s, x
    return best_x, inst.value(best_x)


def greedy(inst: Instance) -> tuple[np.ndarray, float]:
    """Greedy value-sorted feasible insertion."""
    order = np.argsort(-inst.weights)  # high value first
    x = np.zeros(inst.M, dtype=int)
    for t in order:
        x[t] = 1
        if not _feasible(x, inst):
            x[t] = 0
    return x, inst.value(x)


def ilp_optimal(inst: Instance) -> tuple[np.ndarray, float]:
    """Exact ILP via PuLP (optional dependency).

    Raises ImportError if PuLP is not installed -- callers should fall back to
    brute_force for small M.
    """
    try:
        import pulp
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("PuLP not installed; `pip install pulp` or use brute_force") from e

    prob = pulp.LpProblem("settlement", pulp.LpMaximize)
    xs = [pulp.LpVariable(f"x_{t}", cat="Binary") for t in range(inst.M)]
    prob += pulp.lpSum(float(inst.weights[t]) * xs[t] for t in range(inst.M))

    for p in range(inst.N):
        expr = []
        for t in range(inst.M):
            if inst.senders[t] == p:
                expr.append(int(inst.amounts[t]) * xs[t])
            elif inst.receivers[t] == p:
                expr.append(-int(inst.amounts[t]) * xs[t])
        if expr:
            prob += pulp.lpSum(expr) <= int(inst.balances[p])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    x = np.array([int(round(pulp.value(v) or 0)) for v in xs], dtype=int)
    return x, inst.value(x)


def best_known(inst: Instance) -> tuple[np.ndarray, float, str]:
    """Best exact optimum available: ILP if PuLP present, else brute force."""
    try:
        x, v = ilp_optimal(inst)
        return x, v, "ilp"
    except ImportError:
        x, v = brute_force(inst)
        return x, v, "brute_force"
