"""End-to-end solver tests (plan §5): PCE + bit-swap + repair recovers a
high-quality settlement on small instances, and repair always yields
feasibility. Optimizer budgets are kept modest so the suite stays fast."""
import numpy as np

from pce_settlement import baseline, pce_solve, qubo, repair
from pce_settlement.config import Config
from pce_settlement.instance import gridlock_cycle, random_instance


# Results are deterministic given (instance, cfg): COBYLA is deterministic and
# the restart angles come from cfg.seed. Budgets below are chosen so the
# deterministic outcome clears each threshold with margin.


def test_bit_swap_never_raises_energy():
    inst = random_instance(N=4, M=8, tightness=1.5, seed=3)
    Q, const, _ = qubo.build_qubo(inst, P=500.0)
    J, h, ic = qubo.qubo_to_ising(Q, const)
    rng = np.random.default_rng(0)
    m = J.shape[0]
    z0 = rng.choice([-1, 1], size=m)
    z1 = pce_solve.bit_swap(z0, J, h, ic)
    assert qubo.ising_energy(z1, J, h, ic) <= qubo.ising_energy(z0, J, h, ic) + 1e-9


def test_repair_always_feasible():
    inst = random_instance(N=4, M=10, tightness=2.0, seed=7)
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.integers(0, 2, size=inst.M)
        xr = repair.repair(x, inst)
        assert repair.feasible(xr, inst)


def test_pce_matches_bruteforce_gridlock():
    # Hero instance: the netting win is all-or-nothing (settle the whole cycle).
    inst = gridlock_cycle(3, amount=10)
    cfg = Config(seed=0, n_restarts=4, maxiter=500)
    r = pce_solve.solve_settlement(inst, cfg)
    _, opt = baseline.brute_force(inst)
    assert r.feasible
    assert r.value >= 0.95 * opt   # fully settles the cycle (deterministic 1.0)


def test_pce_quality_small_random():
    # PCE is a heuristic; with a penalty sweep + repair it lands near the exact
    # optimum on a small instance (deterministic ~0.88 here).
    inst = random_instance(N=4, M=8, tightness=1.5, seed=11)
    cfg = Config(seed=0, n_restarts=5, maxiter=800)
    r = pce_solve.solve_settlement(inst, cfg)
    _, opt = baseline.brute_force(inst)
    assert r.feasible
    assert r.value >= 0.80 * opt
