r"""qff_pricing.scaling.advantage_plots
========================================

The three ``T``-axis figures for path-dependent fast-forwarding:

  1. ``ff_vs_naive_gates_vs_T``     -- poly(T) fast-forwarding vs exp(T) naive.
  2. ``quantum_advantage_vs_eps``   -- classical O(1/eps^2) vs quantum O(1/eps).
  3. ``qubit_barrier_vs_T``         -- linear Omega(T) qubits, with measured
                                       points from the runnable circuit.

Gate curves are evaluated from the paper's resource theorems; the qubit points in
figure 3 are taken from the actual circuit built in
:mod:`qff_pricing.quantum.fast_forward`.
"""

from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qff_pricing.scaling import advantage as A

_LOG2_10 = math.log10(2.0)


def ff_vs_naive_gates_vs_T(path: str, n: int = 6, T_max: int = 20):
    """Fig 1: gate cost vs T -- fast-forwarding (poly) vs naive (exp)."""
    Ts = np.arange(1, T_max + 1)
    naive = np.array([A.naive_path_log2_gates(T, n) * _LOG2_10 for T in Ts])  # log10
    cir = np.array([math.log10(A.cir_ff_gates(T, n)) for T in Ts])
    hes = np.array([math.log10(A.heston_ff_gates(T, n)) for T in Ts])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(Ts, naive, "s--", color="#d62728",
            label=r"naive joint encoding  $\Theta(2^{Tn})$")
    ax.plot(Ts, cir, "o-", color="#1f77b4",
            label=r"CIR fast-forwarding  $\mathrm{poly}(T)$")
    ax.plot(Ts, hes, "^-", color="#2ca02c",
            label=r"Heston fast-forwarding  $\mathrm{poly}(T,d)$")
    ax.set_xlabel(r"$T$  (monitoring points)")
    ax.set_ylabel(r"$\log_{10}$ state-preparation gate count")
    ax.set_title(f"State-preparation cost vs $T$   ($n = {n}$ qubits/primitive)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def quantum_advantage_vs_eps(path: str):
    """Fig 2: query complexity vs accuracy -- the quadratic speedup."""
    eps = np.logspace(-1, -6, 200)
    classical = 1.0 / eps ** 2
    quantum = 1.0 / eps

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.loglog(eps, classical, "-", color="#d62728", lw=2,
              label=r"classical Monte Carlo  $\mathcal{O}(1/\epsilon^2)$")
    ax.loglog(eps, quantum, "-", color="#1f77b4", lw=2,
              label=r"quantum QMCI  $\mathcal{O}(1/\epsilon)$")
    e0 = 1e-4
    ax.vlines(e0, 1.0 / e0, 1.0 / e0 ** 2, color="#555", ls=":", lw=1.5)
    ax.annotate(r"$1/\epsilon = 10{,}000\times$ at $\epsilon = 10^{-4}$",
                xy=(e0, 1.0 / e0 ** 2), xytext=(3e-4, 3e9),
                fontsize=9, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.fill_between(eps, quantum, classical, color="#1f77b4", alpha=0.08)
    ax.set_xlabel(r"target additive error  $\epsilon$")
    ax.set_ylabel("oracle queries / samples")
    ax.set_title(r"Quantum advantage: $\mathcal{O}(1/\epsilon)$ vs classical "
                 r"$\mathcal{O}(1/\epsilon^2)$")
    ax.invert_xaxis()
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def qubit_barrier_vs_T(path: str, n_inc: int = 2, n_v: int = 2, T_max: int = 20,
                       measured=None):
    """Fig 3: qubits grow linearly in T -- the Omega(T) barrier."""
    Ts = np.arange(1, T_max + 1)
    model = np.array([A.ff_qubits(T, n_inc, n_v) for T in Ts])
    floor = np.array([A.lower_bound_qubits(T, 1) for T in Ts])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(Ts, model, "-", color="#1f77b4", lw=2,
            label=fr"fast-forwarding circuit  $T\,(2n_{{inc}}+n_v)+1$  "
                  fr"($n_{{inc}}={n_inc},\,n_v={n_v}$)")
    ax.plot(Ts, floor, "--", color="#888", label=r"$\Omega(T)$ lower bound")
    if measured is not None:
        mt = [m["T"] for m in measured]
        mq = [m["qubits"] for m in measured]
        ax.plot(mt, mq, "kD", ms=8, zorder=5, label="implemented circuit")
    ax.set_xlabel(r"$T$  (monitoring points)")
    ax.set_ylabel("qubits")
    ax.set_title(r"Qubit count grows linearly in $T$  ($\Omega(T)$ barrier)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path



def implemented_circuit_measured(path: str, T_list=(1, 2, 3, 4, 5, 6)):
    """Fig 4: measured size of the actual runnable circuit vs T.

    Both series are measured on the real verified circuit
    (:mod:`qff_pricing.quantum.fast_forward`): transpiled gate count and qubit
    count. The gate growth reflects the exact lookup ``U_jump`` used so the demo
    runs on a statevector; it is the implemented construction, not the asymptotic
    fast-forwarding gate model in :func:`ff_vs_naive_gates_vs_T`.
    """
    res = [A.measured_circuit_resources(T, with_gates=True) for T in T_list]
    Ts = [r["T"] for r in res]
    gates = [r["gates_lookup_demo"] for r in res]
    qubits = [r["qubits"] for r in res]

    fig, axg = plt.subplots(figsize=(8.2, 5.2))
    axg.set_yscale("log")
    l1 = axg.plot(Ts, gates, "kD-", ms=8, label="transpiled gates (lookup jump)")
    axg.set_xlabel(r"$T$  (monitoring points)")
    axg.set_ylabel("transpiled gate count (log scale)")
    axq = axg.twinx()
    l2 = axq.plot(Ts, qubits, "o--", color="#1f77b4", ms=7, label="qubits")
    axq.set_ylabel("qubits")
    axg.set_title(r"Implemented fast-forwarding circuit: measured size vs $T$")
    lines = l1 + l2
    axg.legend(lines, [ln.get_label() for ln in lines], loc="upper left", framealpha=0.95)
    axg.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def generate_all(outdir: str, measured_T=(1, 2, 3, 4, 5)):
    """Build all three figures; anchor the qubit figure on the real circuit."""
    os.makedirs(outdir, exist_ok=True)
    measured = [A.measured_circuit_resources(T, with_gates=False) for T in measured_T]
    p1 = ff_vs_naive_gates_vs_T(os.path.join(outdir, "ff_vs_naive_gates_vs_T.png"))
    p2 = quantum_advantage_vs_eps(os.path.join(outdir, "quantum_advantage_vs_eps.png"))
    p3 = qubit_barrier_vs_T(os.path.join(outdir, "qubit_barrier_vs_T.png"),
                            measured=measured)
    p4 = implemented_circuit_measured(
        os.path.join(outdir, "implemented_circuit_measured.png"))
    return [p1, p2, p3, p4]


__all__ = ["ff_vs_naive_gates_vs_T", "quantum_advantage_vs_eps",
           "qubit_barrier_vs_T", "implemented_circuit_measured", "generate_all"]
