"""QUBO/Ising tests (plan §5): energy equivalence on 1000 random assignments,
and the penalty mechanics (feasible => zero penalty; infeasible penalty grows
with P). Slack bits make m large, so we verify via embedded assignments rather
than exhaustive 2^m search."""
import numpy as np

from pce_settlement import baseline, qubo
from pce_settlement.instance import gridlock_cycle, random_instance


def test_qubo_to_ising_energy_matches():
    inst = random_instance(N=4, M=8, tightness=1.5, seed=2)
    Q, const, _ = qubo.build_qubo(inst, P=500.0)
    J, h, ic = qubo.qubo_to_ising(Q, const)
    m = Q.shape[0]
    rng = np.random.default_rng(0)
    for _ in range(1000):
        x = rng.integers(0, 2, size=m)
        z = 2 * x - 1
        e_qubo = qubo.qubo_energy(x, Q, const)
        e_ising = qubo.ising_energy(z, J, h, ic)
        assert abs(e_qubo - e_ising) < 1e-6, (e_qubo, e_ising)


def test_feasible_assignment_has_zero_penalty():
    # The exact optimum, embedded with correct slack bits, has penalty 0 so its
    # QUBO energy equals -(settled value), independent of P.
    inst = random_instance(N=4, M=8, tightness=1.5, seed=2)
    x_opt, opt = baseline.brute_force(inst)
    for P in (100.0, 1000.0, 50000.0):
        Q, const, vidx = qubo.build_qubo(inst, P)
        full = qubo.embed_solution(inst, x_opt, vidx)
        assert abs(qubo.qubo_energy(full, Q, const) - (-opt)) < 1e-6


def test_infeasible_penalty_grows_with_P():
    # An over-budget assignment carries a positive penalty that scales with P.
    inst = random_instance(N=3, M=6, tightness=1.8, seed=5)
    x_all = np.ones(inst.M, dtype=int)
    assert not bool((inst.net_flow(x_all) <= inst.balances).all())  # infeasible
    energies = []
    for P in (10.0, 100.0, 1000.0):
        Q, const, vidx = qubo.build_qubo(inst, P)
        full = np.zeros(vidx.m, dtype=int)
        full[: inst.M] = x_all  # slack bits 0 (best case for an infeasible point)
        energies.append(qubo.qubo_energy(full, Q, const))
    assert energies[0] < energies[1] < energies[2]


def test_gridlock_full_settlement_is_optimum():
    # On the cyclic gridlock, settling ALL transactions nets every party to zero
    # (feasible, penalty 0) and is the lowest-energy feasible assignment.
    inst = gridlock_cycle(3, amount=100)
    Q, const, vidx = qubo.build_qubo(inst, P=None)
    x_all = np.ones(inst.M, dtype=int)
    full_all = qubo.embed_solution(inst, x_all, vidx)
    e_all = qubo.qubo_energy(full_all, Q, const)
    assert abs(e_all - (-inst.value(x_all))) < 1e-6   # penalty 0

    # Each single transaction alone is infeasible -> higher energy than all.
    for t in range(inst.M):
        x = np.zeros(inst.M, dtype=int)
        x[t] = 1
        full = np.zeros(vidx.m, dtype=int)
        full[: inst.M] = x
        assert qubo.qubo_energy(full, Q, const) > e_all
