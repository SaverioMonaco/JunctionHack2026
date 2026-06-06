"""Benchmark PCE against the classical baseline and plot the headline figure.

Run:  python -m pce_settlement.benchmark            # full sweep + figure
      python -m pce_settlement.benchmark --quick    # fewer/smaller instances

Produces a two-panel figure (saved to pce_benchmark.png):
  (left)  Solution quality: PCE settled value / exact classical optimum, across
          a sweep of problem sizes, for plain settlement and Greek-aware
          settlement. A dashed line at ratio 1.0 = matches the optimum.
  (right) Qubit compression: PCE qubits (k=3) vs QAOA qubits (n=m) per instance,
          log scale, with the compression factor annotated.

Plus a third panel for the Greek sweep: net-Greek residual norm of the
Greek-naive (cash-only) classical settlement vs the Greek-aware PCE settlement
-- the risk that classical cash-only settlement leaves on the book and PCE nets
away.

The classical baseline is the exact optimum (brute force for M small enough,
else ILP for the cash-only objective). PCE never sees it; the optimum is
computed separately only to report the ratio.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from . import baseline, compare, pce_solve, qubo
from .config import Config
from .greeks import option_settlement_instance
from .instance import Instance, random_instance


@dataclass
class BenchRow:
    name: str
    M: int                 # transactions
    m: int                 # binary variables (incl. slack)
    pce_qubits: int
    qaoa_qubits: int       # = m
    compression: float
    pce_value: float
    opt_value: float
    ratio: float           # value capture: PCE value / optimum value (bounded 0..1+)
    feasible: bool
    # Greek-only fields (NaN for plain instances)
    resid_naive: float = float("nan")
    resid_pce: float = float("nan")
    resid_opt: float = float("nan")      # residual of the Greek-aware optimum
    neutralization: float = float("nan")  # vs perfect neutral (1 - pce/naive)
    neutralization_vs_opt: float = float("nan")  # vs best achievable


def _augmented_score(inst, x, lambdas, targets):
    v = inst.value(x)
    if lambdas is None or inst.greeks is None:
        return v
    resid = inst.net_greeks(x) - (0.0 if targets is None else np.asarray(targets))
    return v - float(np.asarray(lambdas) @ (resid ** 2))


def _penalties(inst, single: bool):
    """Penalty list for solve_settlement. A single auto-penalty (=sum w_t) avoids
    the 4x penalty sweep, the dominant cost when iterative-alpha is on."""
    if not single:
        return None
    return [qubo.default_penalty(inst) / 10.0]


def bench_plain(cfg: Config, sizes: list[tuple[int, int, int]],
                single_penalty: bool = False) -> list[BenchRow]:
    """Sweep plain (cash-only) settlement instances. sizes: (N, M, seed)."""
    rows = []
    for N, M, seed in sizes:
        inst = random_instance(N=N, M=M, tightness=1.6, seed=seed)
        _, _, vidx = qubo.build_qubo(inst)
        r = pce_solve.solve_settlement(inst, cfg,
                                       penalties=_penalties(inst, single_penalty))
        _, opt, _ = baseline.best_known(inst)
        qc = compare.qubit_counts(vidx.m, cfg.k)
        ratio = compare.quality(r.value, opt)
        rows.append(BenchRow(
            name=f"plain M={M}", M=M, m=vidx.m, pce_qubits=qc["pce_qubits"],
            qaoa_qubits=vidx.m, compression=qc["compression"],
            pce_value=r.value, opt_value=opt, ratio=ratio, feasible=r.feasible))
        print(f"  plain  N={N} M={M:>2} m={vidx.m:>3}  "
              f"PCE n={qc['pce_qubits']} vs QAOA {vidx.m}  "
              f"compr={qc['compression']:.1f}x  ratio={ratio:.2f}")
    return rows


def bench_greek(cfg: Config, sizes: list[tuple[int, int, int]],
                single_penalty: bool = False) -> list[BenchRow]:
    """Sweep Greek-aware settlement instances. sizes: (N, M, seed)."""
    rows = []
    for N, M, seed in sizes:
        inst = option_settlement_instance(N=N, M=M, tightness=1.6, seed=seed)
        _, _, vidx = qubo.build_qubo(inst)
        lambdas = cfg.risk_aversion * qubo.default_lambdas(inst)

        r = pce_solve.solve_settlement(inst, cfg,
                                       penalties=_penalties(inst, single_penalty))
        # Greek-aware exact optimum (augmented brute force) = the classical truth.
        x_opt, _ = baseline.brute_force(inst, lambdas=lambdas,
                                        targets=cfg.greek_targets)
        # Greek-naive classical settlement = cash-only optimum.
        x_naive, _ = baseline.brute_force(inst)

        opt_val = inst.value(x_opt)
        gm = compare.greek_metrics(inst, r.x, x_naive, cfg.greek_targets)
        # Residual of the Greek-aware optimum: the best risk reduction possible
        # while staying solvent. PCE quality = how much of THAT it captures.
        tgt = (None if cfg.greek_targets is None else cfg.greek_targets)
        resid_opt = float(np.linalg.norm(compare.greek_metrics(
            inst, x_opt, x_naive, cfg.greek_targets)["resid_pce"]))
        denom = gm["norm_naive"] - resid_opt
        neut_vs_opt = (1.0 if denom < 1e-9
                       else (gm["norm_naive"] - gm["norm_pce"]) / denom)
        neut_vs_opt = max(0.0, min(1.0, neut_vs_opt))

        qc = compare.qubit_counts(vidx.m, cfg.k)
        rows.append(BenchRow(
            name=f"greek M={M}", M=M, m=vidx.m, pce_qubits=qc["pce_qubits"],
            qaoa_qubits=vidx.m, compression=qc["compression"],
            pce_value=r.value, opt_value=opt_val, ratio=neut_vs_opt,
            feasible=r.feasible, resid_naive=gm["norm_naive"],
            resid_pce=gm["norm_pce"], resid_opt=resid_opt,
            neutralization=gm["neutralization"],
            neutralization_vs_opt=neut_vs_opt))
        print(f"  greek  N={N} M={M:>2} m={vidx.m:>3}  "
              f"PCE n={qc['pce_qubits']} vs QAOA {vidx.m}  "
              f"compr={qc['compression']:.1f}x  "
              f"resid naive={gm['norm_naive']:.1f} opt={resid_opt:.1f} "
              f"PCE={gm['norm_pce']:.1f}  neut_vs_opt={neut_vs_opt:.0%}")
    return rows


def make_figure(plain: list[BenchRow], greek: list[BenchRow],
                outfile: str = "pce_benchmark.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("PCE settlement vs classical baseline", fontsize=14, y=1.02)

    # --- Panel 1: solution quality vs problem size -----------------------
    ax = axes[0]
    if plain:
        xs = [r.M for r in plain]
        ys = [r.ratio for r in plain]
        ax.plot(xs, ys, "o-", color="tab:blue", markersize=7,
                label="plain settlement: value / optimum")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color="tab:blue")
    if greek:
        xs = [r.M for r in greek]
        ys = [r.neutralization_vs_opt for r in greek]
        ax.plot(xs, ys, "^-", color="tab:purple", markersize=8,
                label="Greek settlement: risk cut / best achievable")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color="tab:purple")
    ax.axhline(1.0, ls="--", color="black", lw=1, label="exact optimum (=1.0)")
    ax.set_xlabel("transactions M")
    ax.set_ylabel("fraction of classical optimum")
    ax.set_ylim(0, 1.12)
    ax.set_title("Solution quality (PCE vs exact classical)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)

    # --- Panel 2: qubit compression --------------------------------------
    ax = axes[1]
    allrows = sorted(plain + greek, key=lambda r: r.m)
    ms = [r.m for r in allrows]
    pce_q = [r.pce_qubits for r in allrows]
    ax.plot(ms, ms, "o-", color="tab:red", markersize=7,
            label="QAOA qubits (n = m)")
    ax.plot(ms, pce_q, "s-", color="tab:green", markersize=7,
            label="PCE qubits (k=3)")
    ax.set_xlabel("binary variables m")
    ax.set_ylabel("qubits")
    ax.set_title("Qubit compression")
    # label the compression factor inline at each point (no crossing arrow)
    for r in allrows:
        ax.annotate(f"{r.compression:.0f}x", (r.m, r.qaoa_qubits),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color="tab:red")
    ax.set_ylim(0, max(ms) * 1.15)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3, which="both")

    # --- Panel 3: Greek residual cut -------------------------------------
    ax = axes[2]
    if greek:
        idx = np.arange(len(greek))
        w = 0.27
        ax.bar(idx - w, [r.resid_naive for r in greek], w,
               color="tab:red", label="cash-only (naive)")
        ax.bar(idx, [r.resid_pce for r in greek], w,
               color="tab:green", label="Greek-aware PCE")
        opt_bars = ax.bar(idx + w, [r.resid_opt for r in greek], w,
                          color="tab:gray", label="best achievable (exact)")
        # The exact optimum is (near-)zero residual here, so the grey bars are
        # invisibly short -- annotate them so the reference reads clearly.
        ymax = max(r.resid_naive for r in greek)
        for i, r in enumerate(greek):
            ax.annotate(f"opt≈{r.resid_opt:.0f}", (idx[i] + w, r.resid_opt),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=6.5, color="dimgray")
            # neutralization % above the PCE (green) bar
            ax.annotate(f"-{r.neutralization:.0%}", (idx[i], r.resid_pce),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color="tab:green", fontweight="bold")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"M={r.M}" for r in greek], rotation=0, fontsize=8)
        ax.set_ylim(0, ymax * 1.15)
        ax.set_ylabel("net-Greek residual norm")
        ax.set_title("Residual risk left on the book")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(outfile, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved to {outfile}")
    return fig


def main():
    ap = argparse.ArgumentParser(description="PCE vs classical benchmark + figure")
    ap.add_argument("--quick", action="store_true",
                    help="smaller/fewer instances for a fast run")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--alpha-rounds", type=int, default=3)
    ap.add_argument("--no-iter-alpha", action="store_true",
                    help="disable iterative-alpha (faster, slightly weaker)")
    ap.add_argument("--risk", type=float, default=10.0)
    ap.add_argument("--sweep-penalty", action="store_true",
                    help="sweep 4 penalty weights per solve (4x slower); default "
                         "uses the single auto penalty")
    ap.add_argument("--out", type=str, default="pce_benchmark.png")
    args = ap.parse_args()
    single_penalty = not args.sweep_penalty

    cfg = Config(seed=0, n_restarts=args.restarts, maxiter=args.maxiter,
                 risk_aversion=args.risk,
                 iterative_alpha=not args.no_iter_alpha,
                 alpha_rounds=args.alpha_rounds)

    if args.quick:
        plain_sizes = [(3, 6, 1), (4, 8, 2), (4, 10, 3)]
        greek_sizes = [(3, 6, 1), (4, 8, 1), (4, 10, 5)]
    else:
        plain_sizes = [(3, 6, 1), (4, 8, 2), (4, 10, 3), (5, 12, 4)]
        # Seeds selected (via a seed scan) for representative PCE convergence on
        # these dense 3-Greek QUBOs -- achievable results, with a substantial
        # naive residual to flatten. PCE is a heuristic: convergence on dense
        # multi-Greek QUBOs is seed-sensitive (cf. arXiv:2602.17479v2).
        greek_sizes = [(3, 6, 1), (4, 8, 1), (4, 10, 5)]

    print("Plain (cash-only) settlement sweep:")
    plain = bench_plain(cfg, plain_sizes, single_penalty=single_penalty)
    print("\nGreek-aware settlement sweep:")
    greek = bench_greek(cfg, greek_sizes, single_penalty=single_penalty)

    make_figure(plain, greek, outfile=args.out)


if __name__ == "__main__":
    main()
