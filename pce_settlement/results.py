"""Generate the headline results for the pitch (plan §6).

Run:  python -m pce_settlement.results

Prints two things:
  1. A measured table: for several instances, PCE qubits vs QAOA qubits, PCE
     settled value vs the exact ILP/brute-force optimum (approximation ratio),
     and feasibility. PCE never sees the optimum -- it is computed separately
     only to report the ratio.
  2. The pure-arithmetic projection table (qubit reach at scale) for the slides.
"""
from __future__ import annotations

from . import baseline, compare, pce_solve, qubo
from .config import Config
from .instance import gridlock_cycle, random_instance


def measured_table(cfg: Config | None = None):
    cfg = cfg or Config(seed=0, n_restarts=5, maxiter=800)
    rows = []
    instances = [
        gridlock_cycle(3, amount=10),
        gridlock_cycle(4, amount=10),
        random_instance(N=4, M=8, tightness=1.5, seed=7),
        random_instance(N=4, M=8, tightness=1.5, seed=11),
        random_instance(N=5, M=10, tightness=1.6, seed=3),
    ]
    print(f"\n{'instance':<22}{'m':>4}{'PCE n':>7}{'QAOA n':>8}"
          f"{'compr':>8}{'PCE val':>9}{'opt':>8}{'ratio':>7}{'feas':>6}")
    print("-" * 78)
    for inst in instances:
        _, _, vidx = qubo.build_qubo(inst)
        r = pce_solve.solve_settlement(inst, cfg)
        _, opt, base = baseline.best_known(inst)
        qc = compare.qubit_counts(vidx.m, cfg.k)
        ratio = compare.quality(r.value, opt)
        print(f"{inst.name:<22}{vidx.m:>4}{qc['pce_qubits']:>7}{vidx.m:>8}"
              f"{qc['compression']:>7.1f}x{r.value:>9.0f}{opt:>8.0f}"
              f"{ratio:>7.2f}{str(r.feasible):>6}")
        rows.append((inst.name, vidx.m, qc["pce_qubits"], r.value, opt, ratio))
    return rows


def projection():
    print("\nProjection table (pure arithmetic, k=3) - qubit reach at scale:")
    print(f"{'qubits n':>10}{'addressable vars 3*C(n,3)':>28}")
    print("-" * 40)
    for row in compare.projection_table(k=3):
        print(f"{row['qubits']:>10}{row['addressable_vars']:>28,}")


if __name__ == "__main__":
    measured_table()
    projection()
