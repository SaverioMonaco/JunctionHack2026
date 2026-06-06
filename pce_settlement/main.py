"""End-to-end PCE settlement demo (plan §7 build order step 6).

Run:  python -m pce_settlement.main            # hero gridlock instance
      python -m pce_settlement.main --random   # a random gridlock-prone batch

Pipeline: instance -> QUBO -> Ising -> PCE solve -> bit-swap -> repair ->
compare against the exact classical optimum, print the headline.
"""
from __future__ import annotations

import argparse

from . import baseline, compare, pce_solve, qubo
from .config import Config
from .greeks import greek_gridlock_demo, option_settlement_instance
from .instance import gridlock_cycle, random_instance


def run(inst, cfg: Config, P: float | None = None, verbose: bool = True):
    # P=None => sweep the penalty weight (plan §1.3); else solve at the given P.
    penalties = None if P is None else [P]
    sr = pce_solve.solve_settlement(inst, cfg, penalties=penalties)
    result = sr.solve
    x_repaired = sr.x
    pce_value = sr.value
    is_feasible = sr.feasible

    # Exact baseline.
    x_opt, opt_value, base_name = baseline.best_known(inst)

    if verbose:
        print(f"\nInstance: {inst.name}  (N={inst.N} parties, M={inst.M} transactions)")
        print(f"balances : {inst.balances.tolist()}")
        print(f"QUBO vars m = {result.m}  (M={inst.M} tx + {result.m - inst.M} slack bits)")
        print(f"PCE: n={result.n} qubits, p={result.p} layers, k={cfg.k}, "
              f"chosen P={sr.P:.0f}, best loss={result.loss:.4f}")
        print(f"after repair        : {x_repaired.tolist()}  feasible={is_feasible}")
        print(f"exact ({base_name}) : {x_opt.tolist()}")
        print()
        print(compare.format_summary(result.m, cfg.k, pce_value, opt_value, base_name))

    return {
        "result": result,
        "x_repaired": x_repaired,
        "pce_value": pce_value,
        "feasible": is_feasible,
        "chosen_P": sr.P,
        "x_opt": x_opt,
        "opt_value": opt_value,
        "baseline": base_name,
        "ratio": compare.quality(pce_value, opt_value),
    }


def run_greeks(inst, cfg: Config, verbose: bool = True):
    """Risk-aware settlement demo: net Greeks before (naive) vs after (PCE).

    The Greek-naive competitor is the cash-only optimum (ignores Greeks); the
    Greek-aware solution is PCE driven by the neutrality penalty. We report the
    settled value, the net-Greek residual for each, and the neutralization
    ratio (how much PCE cut residual Greek exposure).
    """
    lambdas = cfg.lambdas
    if lambdas is None:
        lambdas = cfg.risk_aversion * qubo.default_lambdas(inst)

    # Greek-aware PCE settlement (penalty active via solve_settlement).
    sr = pce_solve.solve_settlement(inst, cfg)
    x_pce = sr.x

    # Greek-naive competitor: cash-only optimum (lambdas=None).
    x_naive, _ = baseline.brute_force(inst)

    # Greek-aware exact optimum for an honest quality check.
    x_opt, opt_value = baseline.brute_force(inst, lambdas=lambdas,
                                            targets=cfg.greek_targets)

    if verbose:
        print(f"\nInstance: {inst.name}  (N={inst.N} parties, M={inst.M} trades)")
        print(f"balances : {inst.balances.tolist()}")
        print(f"lambdas  : {[f'{v:.3g}' for v in lambdas]} "
              f"(greeks: {inst.greek_names})")
        print(f"QUBO vars m = {sr.solve.m}  (M={inst.M} tx + "
              f"{sr.solve.m - inst.M} slack bits; Greeks add 0)")
        print(f"naive settle (cash-only)   : {x_naive.tolist()}  "
              f"value={inst.value(x_naive):.1f}")
        print(f"Greek-aware PCE settlement : {x_pce.tolist()}  "
              f"value={sr.value:.1f}  feasible={sr.feasible}")
        print(f"Greek-aware exact optimum  : {x_opt.tolist()}  "
              f"value={opt_value:.1f}")
        print()
        print(compare.format_summary(sr.solve.m, cfg.k, sr.value, opt_value,
                                     "greek-opt"))
        print(compare.format_greek_summary(inst, x_pce, x_naive,
                                           cfg.greek_targets))

    return {
        "x_pce": x_pce, "x_naive": x_naive, "x_opt": x_opt,
        "pce_value": sr.value, "opt_value": opt_value,
        "greek_metrics": compare.greek_metrics(inst, x_pce, x_naive,
                                               cfg.greek_targets),
    }


def main():
    ap = argparse.ArgumentParser(description="PCE settlement demo")
    ap.add_argument("--random", action="store_true", help="use a random instance")
    ap.add_argument("--greeks", action="store_true",
                    help="risk-aware (Greek-constrained) settlement demo")
    ap.add_argument("--book", action="store_true",
                    help="with --greeks: use a random option book instead of "
                         "the delta-gridlock hero")
    ap.add_argument("--risk", type=float, default=10.0,
                    help="with --greeks: risk-aversion multiplier on the Greek "
                         "penalty (>=5 flips the hero to the neutral subset)")
    ap.add_argument("--iterative-alpha", action="store_true",
                    help="ramp the PCE sharpness alpha across rounds "
                         "(arXiv:2602.17479v2 Alg 1); helps on dense QUBOs")
    ap.add_argument("--parties", type=int, default=4, help="gridlock cycle size")
    ap.add_argument("--amount", type=int, default=10, help="gridlock obligation amount")
    ap.add_argument("--N", type=int, default=6, help="parties (random)")
    ap.add_argument("--M", type=int, default=14, help="transactions (random)")
    ap.add_argument("--tightness", type=float, default=1.5)
    ap.add_argument("--P", type=float, default=None, help="penalty weight override")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=6)
    args = ap.parse_args()

    cfg = Config(seed=args.seed, n_restarts=args.restarts,
                 iterative_alpha=args.iterative_alpha)

    if args.greeks:
        cfg.risk_aversion = args.risk
        if args.book:
            inst = option_settlement_instance(args.N, args.M,
                                              tightness=args.tightness,
                                              seed=args.seed)
        else:
            inst = greek_gridlock_demo(amount=max(args.amount, 100))
        run_greeks(inst, cfg)
        return

    if args.random:
        inst = random_instance(args.N, args.M, tightness=args.tightness, seed=args.seed)
    else:
        inst = gridlock_cycle(args.parties, amount=args.amount)

    run(inst, cfg, P=args.P)


if __name__ == "__main__":
    main()
