r"""qff_pricing.scaling.plots
==============================

Scaling figures for **European option pricing**, where the payoff depends only
on the terminal value so there is a single monitoring point (**T = 1**) and the
algorithm reduces to one distribution-loading problem on ``N = 2**n`` grid
points (the "data points").

The figures contrast, as a function of the **number of data points N**:

  * **naive amplitude encoding** -- ``Theta(N)`` gates (linear in data points;
    exponential in the qubit count ``n = log2 N``), with
  * the paper's **polynomial-approximation / black-box loaders** --
    ``O(polylog(N))`` gates,

and show the **polylog accuracy** scaling and the **quadratic** QMCI sampling
speedup.

Run as a script::

    python -m qff_pricing.scaling.plots --outdir scaling_plots

Curves are shown for **both CIR and Heston** (the European loading cost is
polylog(N) for both; Heston carries a larger constant via its integral-of-CIR
primitive).

Produces (in ``outdir``):
  * ``gates_vs_datapoints.png``   gate cost vs N (CIR & Heston vs naive)
  * ``gates_vs_qubits.png``       gate cost vs n (poly vs exponential)
  * ``qubits_vs_datapoints.png``  qubits vs N: N points -> log2(N) qubits
  * ``qubits_vs_accuracy.png``    qubits vs 1/eps (polylog)
  * ``samples_vs_accuracy.png``   MC 1/eps^2 vs QMCI 1/eps (quadratic speedup)
  * ``scaling_summary.png``       all of the above in one figure
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt
import numpy as np

from qff_pricing.scaling import resources as R

_EPS = 1e-6  # reference target accuracy used where a fixed eps is needed


def gates_vs_datapoints(ax, eps: float = _EPS):
    """Gate cost vs the NUMBER OF DATA POINTS N = 2**n  (the headline).

    Shows the naive baseline plus the CIR and Heston loaders explicitly: both
    models load the European terminal distribution in polylog(N) gates, with
    Heston carrying a larger constant (it needs the integral-of-CIR primitive).
    """
    n = np.arange(2, 25)
    N = 2.0 ** n
    naive = np.array([R.european_naive_loading_gates(int(k)) for k in n])
    cir = np.array([R.cir_european_loading_gates(int(k), eps) for k in n])
    hes = np.array([R.heston_european_loading_gates(int(k), eps) for k in n])

    ax.loglog(N, naive, "s--", color="#d62728",
              label=r"Naive amplitude encoding  $\Theta(N)$")
    ax.loglog(N, hes, "^-", color="#ff7f0e",
              label=r"Heston loader  $O(\mathrm{polylog}\,N)$")
    ax.loglog(N, cir, "o-", color="#1f77b4",
              label=r"CIR loader  $O(\log N \cdot \mathrm{polylog}\,\frac{1}{\epsilon})$")
    ax.set_xlabel("number of data points  $N = 2^{\\,n}$")
    ax.set_ylabel("state-prep gate count")
    ax.set_title(r"Gates vs data points $\Rightarrow$ polylog, not linear")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def qubits_vs_datapoints(ax):
    """Qubits needed vs the NUMBER OF DATA POINTS  ->  n = log2(N) (compression)."""
    n = np.arange(1, 25)
    N = 2.0 ** n
    qubits = np.array([R.qubits_for_data_points(k) for k in N])
    ax.semilogx(N, qubits, "o-", color="#17becf")
    ax.set_xlabel("number of data points  $N$")
    ax.set_ylabel("qubits  $n = \\log_2 N$")
    ax.set_title(r"Qubits vs data points: $N$ points $\to$ $\log_2 N$ qubits")
    ax.grid(True, alpha=0.3, which="both")
    ax.annotate(r"$10^6$ data points $\approx$ 20 qubits",
                xy=(1e6, 20), xytext=(20, 8), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))


def gates_vs_qubits(ax, eps: float = _EPS):
    """Gate cost vs the NUMBER OF QUBITS n  (the 'exponential' framing)."""
    n = np.arange(2, 25)
    naive = np.array([R.european_naive_loading_gates(int(k)) for k in n])
    cir = np.array([R.cir_european_loading_gates(int(k), eps) for k in n])
    hes = np.array([R.heston_european_loading_gates(int(k), eps) for k in n])

    ax.semilogy(n, naive, "s--", color="#d62728",
                label=r"Naive  $\Theta(2^{n})$ (exponential)")
    ax.semilogy(n, hes, "^-", color="#ff7f0e",
                label=r"Heston loader (polynomial)")
    ax.semilogy(n, cir, "o-", color="#1f77b4",
                label=r"CIR loader  $O(n\cdot\mathrm{polylog})$ (polynomial)")
    ax.set_xlabel("number of qubits  $n = \\log_2 N$")
    ax.set_ylabel("state-prep gate count (log scale)")
    ax.set_title("European (T=1): gates vs qubits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def qubits_vs_accuracy(ax):
    """Qubits (and data points) needed for accuracy eps  ->  polylog in 1/eps."""
    eps = np.logspace(-1, -9, 40)
    qubits = np.array([R.european_qubits_for_accuracy(e) for e in eps])
    ax.semilogx(1.0 / eps, qubits, "o-", color="#9467bd")
    ax.set_xlabel(r"$1/\epsilon$")
    ax.set_ylabel(r"qubits  $n = \log_2 N$")
    ax.set_title(r"European (T=1): qubits vs accuracy $\Rightarrow$ polylog$(1/\epsilon)$")
    ax.grid(True, alpha=0.3, which="both")


def samples_vs_accuracy(ax):
    """Classical MC 1/eps^2 vs quantum QMCI 1/eps  (quadratic speedup)."""
    eps = np.logspace(-1, -5, 40)
    mc = R.classical_mc_samples(eps)
    qmci = R.quantum_qmci_queries(eps)
    ax.loglog(1.0 / eps, mc, "s--", color="#d62728",
              label=r"Classical MC  $\sim 1/\epsilon^2$")
    ax.loglog(1.0 / eps, qmci, "o-", color="#1f77b4",
              label=r"Quantum QMCI  $\sim 1/\epsilon$")
    ax.set_xlabel(r"$1/\epsilon$")
    ax.set_ylabel("oracle queries / samples")
    ax.set_title("European (T=1): sampling complexity (quadratic speedup)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def make_all_plots(outdir: str = "scaling_plots") -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []

    specs = [
        ("gates_vs_datapoints.png", gates_vs_datapoints),
        ("gates_vs_qubits.png", gates_vs_qubits),
        ("qubits_vs_datapoints.png", qubits_vs_datapoints),
        ("qubits_vs_accuracy.png", qubits_vs_accuracy),
        ("samples_vs_accuracy.png", samples_vs_accuracy),
    ]
    for fname, fn in specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        fn(ax)
        fig.tight_layout()
        p = os.path.join(outdir, fname)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    # combined summary (2 x 3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    gates_vs_datapoints(axes[0, 0])
    gates_vs_qubits(axes[0, 1])
    qubits_vs_datapoints(axes[0, 2])
    qubits_vs_accuracy(axes[1, 0])
    samples_vs_accuracy(axes[1, 1])
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.5,
                    "European option  (T = 1)\n\nCIR  &  Heston\n\nLoad the terminal\n"
                    "distribution on N data points\nin polylog(N) gates\n"
                    "(N points -> log2 N qubits)",
                    ha="center", va="center", fontsize=12)
    fig.suptitle("Quantum Fast-Forwarding for European options (T=1): scaling in the number of data points  (CIR & Heston)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(outdir, "scaling_summary.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Generate European-option QFF scaling plots.")
    ap.add_argument("--outdir", default="scaling_plots")
    args = ap.parse_args()
    paths = make_all_plots(args.outdir)
    print("Wrote:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    _main()
