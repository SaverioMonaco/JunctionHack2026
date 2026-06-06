"""Greek-constrained settlement tests (plan "Tests").

Covers the Black-Scholes Greeks, the Greek-penalty QUBO expansion (exact vs
direct, energy equivalence, no new variables, solvency unchanged), and the
end-to-end claim that Greek-aware PCE beats Greek-naive settlement on residual
net-Greek exposure.
"""
import numpy as np

from pce_settlement import baseline, compare, qubo, repair
from pce_settlement.config import Config
from pce_settlement.greeks import (bs_greeks, greek_gridlock_demo,
                                   option_settlement_instance)
from pce_settlement.instance import random_instance


# --- 1. Black-Scholes Greeks sanity --------------------------------------
def test_bs_greeks_sanity():
    S = K = 100.0
    T, r, sigma = 1.0, 0.0, 0.2
    cd, cg, cv = bs_greeks(S, K, T, r, sigma, "call")
    pd, pg, pv = bs_greeks(S, K, T, r, sigma, "put")
    # ATM call delta ~ 0.5-0.6, put delta = call delta - 1.
    assert 0.5 <= cd <= 0.6
    assert abs(pd - (cd - 1.0)) < 1e-9
    # gamma, vega positive and identical for call/put.
    assert cg > 0 and cv > 0
    assert abs(cg - pg) < 1e-9 and abs(cv - pv) < 1e-9
    # finite-difference check of delta.
    eps = 1e-4
    from scipy.stats import norm
    price = lambda s: s * norm.cdf((np.log(s / K) + 0.5 * sigma**2 * T)
                                   / (sigma * np.sqrt(T))) - K * norm.cdf(
        (np.log(s / K) - 0.5 * sigma**2 * T) / (sigma * np.sqrt(T)))
    fd = (price(S + eps) - price(S - eps)) / (2 * eps)
    assert abs(fd - cd) < 1e-3


# --- 2. Greek penalty expands exactly to the direct form -----------------
def test_greek_penalty_expansion_matches_direct():
    inst = option_settlement_instance(N=4, M=8, seed=3)
    lambdas = qubo.default_lambdas(inst)
    Q0, c0, _ = qubo.build_qubo(inst, P=500.0)                  # no greeks
    Q1, c1, _ = qubo.build_qubo(inst, P=500.0, lambdas=lambdas)  # with greeks
    m = Q1.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(1000):
        x = rng.integers(0, 2, size=m)
        delta_energy = qubo.qubo_energy(x, Q1, c1) - qubo.qubo_energy(x, Q0, c0)
        # direct: sum_g lambda_g (a_g . x)^2, a_g = greek col on tx vars, target 0.
        direct = 0.0
        for g in range(inst.greeks.shape[1]):
            a = np.zeros(m)
            a[: inst.M] = inst.greeks[:, g]
            direct += lambdas[g] * (a @ x) ** 2
        assert abs(delta_energy - direct) < 1e-6, (delta_energy, direct)


# --- 3. Solvency QUBO unchanged by the refactor / Greek-off path ---------
def test_solvency_unchanged_when_lambdas_none():
    inst = option_settlement_instance(N=4, M=8, seed=3)
    Qa, ca, va = qubo.build_qubo(inst, P=500.0)
    Qb, cb, vb = qubo.build_qubo(inst, P=500.0, lambdas=None)  # greeks off
    assert np.allclose(Qa, Qb) and abs(ca - cb) < 1e-12
    assert va.m == vb.m and va.slack_bits == vb.slack_bits


# --- 4. QUBO -> Ising still energy-matches with Greek terms ---------------
def test_qubo_to_ising_matches_with_greeks():
    inst = option_settlement_instance(N=4, M=8, seed=7)
    lambdas = qubo.default_lambdas(inst)
    Q, const, _ = qubo.build_qubo(inst, P=500.0, lambdas=lambdas)
    J, h, ic = qubo.qubo_to_ising(Q, const)
    m = Q.shape[0]
    rng = np.random.default_rng(1)
    for _ in range(1000):
        x = rng.integers(0, 2, size=m)
        z = 2 * x - 1
        assert abs(qubo.qubo_energy(x, Q, const)
                   - qubo.ising_energy(z, J, h, ic)) < 1e-6


# --- 7. Greek penalty adds NO new variables ------------------------------
def test_no_new_vars():
    inst = option_settlement_instance(N=4, M=8, seed=3)
    lambdas = qubo.default_lambdas(inst)
    _, _, v0 = qubo.build_qubo(inst, P=500.0)
    _, _, v1 = qubo.build_qubo(inst, P=500.0, lambdas=lambdas)
    assert v0.m == v1.m and v0.slack_bits == v1.slack_bits


# --- 5. Greek-aware PCE beats Greek-naive on the risk-adjusted objective ---
def test_pce_greek_aware_beats_naive_on_augmented_score():
    # PCE is a heuristic on a dense QUBO, so we don't require it to hit the exact
    # Greek-aware optimum. The honest, robust claim is that running PCE *with*
    # the Greek penalty produces a feasible settlement whose risk-adjusted score
    # (value - Greek penalty) is at least as good as the Greek-naive (cash-only)
    # optimum's -- i.e. accounting for Greeks does not hurt and typically helps.
    from pce_settlement import pce_solve
    inst = option_settlement_instance(N=3, M=7, seed=11, tightness=1.6)
    cfg = Config(seed=0, n_restarts=5, maxiter=800)
    lambdas = qubo.default_lambdas(inst)

    def score(x):
        return inst.value(x) - float(lambdas @ inst.net_greeks(x) ** 2)

    sr = pce_solve.solve_settlement(inst, cfg)          # Greek-aware
    x_naive, _ = baseline.brute_force(inst)             # cash-only optimum
    assert repair.feasible(sr.x, inst)
    assert score(sr.x) >= score(x_naive) - 1e-9


# --- 6. Greek-aware PCE cuts residual net-Greek vs cash-only settlement --
def test_greek_residual_beats_naive():
    from pce_settlement import pce_solve
    inst = greek_gridlock_demo()
    # Risk-averse desk: weight neutrality heavily enough to drop a leg for it.
    cfg = Config(seed=0, n_restarts=5, maxiter=800, risk_aversion=10.0)
    sr = pce_solve.solve_settlement(inst, cfg)
    x_naive, _ = baseline.brute_force(inst)              # cash-only optimum
    gm = compare.greek_metrics(inst, sr.x, x_naive)
    assert repair.feasible(sr.x, inst)
    # The cash-only optimum (settle all 5) leaves residual delta -0.6; the
    # Greek-aware PCE settlement flattens it (drops one short-delta leg).
    assert gm["norm_naive"] > 0.1                        # naive is exposed
    assert gm["norm_pce"] < gm["norm_naive"] - 1e-9      # PCE strictly cuts it


# --- 8. Iterative-alpha PCE also solves the hero (arXiv:2602.17479v2) ------
def test_iterative_alpha_neutralizes_hero():
    from pce_settlement import pce_solve
    inst = greek_gridlock_demo()
    cfg = Config(seed=0, n_restarts=3, maxiter=500, risk_aversion=10.0,
                 iterative_alpha=True, alpha_rounds=3)
    sr = pce_solve.solve_settlement(inst, cfg)
    x_naive, _ = baseline.brute_force(inst)
    gm = compare.greek_metrics(inst, sr.x, x_naive)
    assert repair.feasible(sr.x, inst)
    assert gm["norm_pce"] < gm["norm_naive"] - 1e-9


# --- existing instances still construct (greeks default None) ------------
def test_plain_instance_has_no_greeks():
    inst = random_instance(N=4, M=8, seed=0)
    assert inst.greeks is None
    assert np.allclose(inst.net_greeks(np.ones(inst.M, dtype=int)),
                       np.zeros(len(inst.greek_names)))
