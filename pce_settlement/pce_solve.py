"""Generalized PCE solve: loss, COBYLA training, readout, bit-swap (plan §1.5, Phase 3).

The seminal PCE loss is MaxCut-only (no field term). General QUBO has fields, so
we use the extended loss (plan §1.5):

    alpha = n^floor(k/2)
    t_i   = tanh(alpha * <Pi_i>)
    L     = sum_{i<j} J_ij t_i t_j + sum_i h_i t_i + L_reg
    L_reg = beta * nu * [ (1/m) sum_i t_i^2 ]^2

Minimizing L drives each z_i = sgn(<Pi_i>) toward the configuration lowering the
Ising energy. Readout: z_i = sgn(<Pi_i>), then x_i = (1 + z_i)/2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import pauli, qubo as qubo_mod, repair as repair_mod
from .ansatz import run_hea, n_params
from .config import Config
from .instance import Instance
from .qubo import ising_energy


@dataclass
class SolveResult:
    z: np.ndarray            # spin solution in {-1,+1}^m (after optional bit-swap)
    z_raw: np.ndarray        # spin solution straight from PCE readout
    energy: float            # Ising energy of z
    loss: float              # best PCE loss achieved
    n: int                   # qubits used
    p: int                   # HEA layers
    m: int                   # binary variables
    k: int
    loss_history: list[float]
    theta: np.ndarray | None = None  # trained params (p+1, n); for hardware export


def _nu(J: np.ndarray, h: np.ndarray) -> float:
    """Objective-magnitude scale nu = sum_{i<j}|J_ij| + sum_i|h_i| (plan §1.5)."""
    return float(np.abs(np.triu(J, k=1)).sum() + np.abs(h).sum())


def pce_loss(theta: np.ndarray, J: np.ndarray, h: np.ndarray, strings, n: int,
             p: int, k: int, beta: float, nu: float, topology: str = "linear",
             alpha: float | None = None) -> float:
    """Evaluate the generalized PCE loss for parameters theta (plan §1.5).

    ``alpha`` is the tanh sharpness; defaults to the paper's alpha = n^floor(k/2).
    Iterative-alpha training (arXiv:2602.17479v2) passes an increasing schedule.
    """
    m = len(strings)
    if alpha is None:
        alpha = pauli.alpha_for(n, k)
    psi = run_hea(theta, n, p, topology)
    exps = pauli.expectations(psi, strings, n)
    t = np.tanh(alpha * exps)
    quad = float(t @ np.triu(J, k=1) @ t)
    field = float(h @ t)
    reg = beta * nu * (np.mean(t ** 2) ** 2)
    return quad + field + reg


def _readout(theta: np.ndarray, J: np.ndarray, h: np.ndarray, strings, n: int,
             p: int, topology: str) -> np.ndarray:
    psi = run_hea(theta, n, p, topology)
    exps = pauli.expectations(psi, strings, n)
    z = np.sign(exps)
    z[z == 0] = 1.0  # tie-break: map exactly-zero expectation to +1
    return z.astype(int)


def bit_swap(z: np.ndarray, J: np.ndarray, h: np.ndarray, const: float = 0.0) -> np.ndarray:
    """One greedy Ising local-search to convergence (plan §3 bit_swap).

    Repeatedly flips the single spin that most lowers the energy until no flip
    helps. Cheap polish on the PCE readout.
    """
    z = np.asarray(z, dtype=float).copy()
    Jsym = np.triu(J, k=1)
    Jsym = Jsym + Jsym.T
    while True:
        # Energy change for flipping spin i: dE_i = -2 z_i (sum_j Jsym_ij z_j + h_i)
        local = Jsym @ z + h
        dE = -2.0 * z * local
        i = int(np.argmin(dE))
        if dE[i] < -1e-12:
            z[i] = -z[i]
        else:
            break
    return z.astype(int)


def solve(J: np.ndarray, h: np.ndarray, k: int, cfg: Config,
          const: float = 0.0) -> SolveResult:
    """Solve an Ising problem with PCE (plan Phase 3).

    Returns a SolveResult with z in {-1,+1}^m.
    """
    m = J.shape[0]
    n = pauli.qubits_for(m, k)
    p = pauli.layers_for(m, n)
    strings = pauli.enumerate_pauli_strings(n, k, m)
    nu = _nu(J, h)
    nparam = n_params(n, p)
    rng = np.random.default_rng(cfg.seed)

    history: list[float] = []

    def run_restarts(alpha: float, warm: np.ndarray | None):
        """COBYLA restarts at a fixed alpha; warm-start one restart from ``warm``.
        Returns (best_theta, best_loss_at_this_alpha)."""
        def objective(theta: np.ndarray) -> float:
            val = pce_loss(theta, J, h, strings, n, p, k, cfg.beta, nu,
                           cfg.topology, alpha=alpha)
            history.append(val)
            return val

        bt, bl = None, np.inf
        for r in range(cfg.n_restarts):
            theta0 = (warm if (warm is not None and r == 0)
                      else rng.uniform(0.0, 2.0 * np.pi, size=nparam))
            res = minimize(
                objective, theta0, method="COBYLA",
                options={"maxiter": cfg.maxiter, "rhobeg": cfg.rhobeg, "tol": cfg.tol},
            )
            if res.fun < bl:
                bl, bt = float(res.fun), res.x
        return bt, bl

    alpha0 = pauli.alpha_for(n, k)
    if cfg.iterative_alpha:
        # Ramp alpha across rounds, warm-starting theta from the previous round
        # (arXiv:2602.17479v2 Algorithm 1). Readout uses the final (sharpest)
        # parameters; the reported loss is the last round's best.
        best_theta = None
        best_loss = np.inf
        for j in range(cfg.alpha_rounds):
            alpha = alpha0 * (cfg.alpha_growth ** j)
            best_theta, best_loss = run_restarts(alpha, warm=best_theta)
    else:
        best_theta, best_loss = run_restarts(alpha0, warm=None)

    z_raw = _readout(best_theta, J, h, strings, n, p, cfg.topology)
    z = bit_swap(z_raw, J, h, const) if cfg.bit_swap else z_raw.copy()
    energy = ising_energy(z, J, h, const)

    return SolveResult(
        z=z, z_raw=z_raw, energy=energy, loss=best_loss,
        n=n, p=p, m=m, k=k, loss_history=history,
        theta=best_theta.reshape(p + 1, n),
    )


@dataclass
class SettlementResult:
    x: np.ndarray             # feasible settlement decisions in {0,1}^M
    value: float              # settled value of x
    P: float                  # penalty weight that produced this result
    feasible: bool
    solve: SolveResult        # the underlying PCE solve at the chosen P


def solve_settlement(inst: Instance, cfg: Config,
                     penalties: list[float] | None = None) -> SettlementResult:
    """End-to-end settlement solve with a penalty sweep (plan §1.3, §6).

    Penalty weight P dominates behaviour: too small => infeasible, too large =>
    flat landscape. The plan prescribes sweeping it and reporting the value
    used. For each P we build the QUBO, solve with PCE, repair to feasibility,
    and keep the highest-value feasible result.
    """
    base = qubo_mod.default_penalty(inst) / 10.0  # = sum_t w_t
    if penalties is None:
        penalties = [0.5 * base, base, 2.0 * base, 5.0 * base]

    # Greek-neutrality penalty (risk-aware settlement). Active only when the
    # instance carries greeks; lambdas auto-default if not supplied.
    lambdas = cfg.lambdas
    if inst.greeks is not None and lambdas is None:
        lambdas = cfg.risk_aversion * qubo_mod.default_lambdas(inst)
    use_greeks = inst.greeks is not None and lambdas is not None

    def score(x: np.ndarray) -> float:
        """Selection criterion: settled value, less Greek penalty when active.
        Steers candidate choice toward neutral books the same way the QUBO does.
        """
        v = inst.value(x)
        if not use_greeks:
            return v
        resid = repair_mod.greek_residual(x, inst, cfg.greek_targets)
        return v - float(np.asarray(lambdas) @ (resid ** 2))

    # Backend dispatch. Both backends share the same loss, readout, bit-swap,
    # and SolveResult; only the optimizer/simulator differ. Lazy import keeps
    # the numpy core usable without jax/pennylane installed.
    if cfg.backend == "jax":
        from . import pce_jax
        solve_fn = pce_jax.solve
    else:
        solve_fn = solve

    best: SettlementResult | None = None
    best_score = -np.inf
    for P in penalties:
        Q, const, _ = qubo_mod.build_qubo(
            inst, P, lambdas=lambdas if use_greeks else None,
            greek_targets=cfg.greek_targets)
        J, h, ic = qubo_mod.qubo_to_ising(Q, const)
        res = solve_fn(J, h, cfg.k, cfg, const=ic)
        x = ((1 + res.z) // 2).astype(int)[: inst.M]
        if cfg.repair:
            x = repair_mod.repair(
                x, inst, lambdas=lambdas if use_greeks else None,
                targets=cfg.greek_targets)
        val = inst.value(x)
        feas = repair_mod.feasible(x, inst)
        cand = SettlementResult(x=x, value=val, P=float(P), feasible=feas, solve=res)
        # Prefer feasible, then higher score (value, less Greek penalty).
        cand_key = (feas, score(x))
        if best is None or cand_key > (best.feasible, best_score):
            best, best_score = cand, score(x)
    return best
