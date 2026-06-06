"""Plots for the settlement demo (plan §7, Phase 7).

- gridlock_graph: parties = nodes, transactions = directed weighted edges;
  settled = green, dropped = grey.
- qubit_gap_bar: PCE vs QAOA qubit counts (log scale).
- loss_curve: COBYLA loss history.

Uses matplotlib only (no networkx dependency); party nodes are laid out on a
circle, which suits the cyclic gridlock hero instance.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance


def _circle_positions(N: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    return np.column_stack([np.cos(ang), np.sin(ang)])


def gridlock_graph(inst: Instance, x: np.ndarray | None = None, ax=None,
                   title: str | None = None):
    """Draw parties as circle nodes and transactions as directed edges."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    pos = _circle_positions(inst.N)
    x = np.ones(inst.M, dtype=int) if x is None else np.asarray(x).astype(int)

    for t in range(inst.M):
        s, d = inst.senders[t], inst.receivers[t]
        settled = x[t] == 1
        color = "tab:green" if settled else "lightgrey"
        p0, p1 = pos[s], pos[d]
        # offset endpoints toward node edge for arrowhead clarity
        ax.annotate(
            "", xy=p1 * 0.82, xytext=p0 * 0.82,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5 + 2 * settled,
                            shrinkA=12, shrinkB=12,
                            connectionstyle="arc3,rad=0.12"),
        )
        mid = (p0 + p1) / 2 * 0.7
        ax.text(*mid, f"{inst.amounts[t]}", color=color, fontsize=8, ha="center")

    for p in range(inst.N):
        ax.scatter(*pos[p], s=900, c="white", edgecolors="black", zorder=3)
        ax.text(*pos[p], f"P{p}\nB={inst.balances[p]}", ha="center", va="center",
                fontsize=9, zorder=4)

    ax.set_title(title or f"{inst.name}: settled={int(x.sum())}/{inst.M}")
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def qubit_gap_bar(m: int, pce_qubits: int, ax=None):
    """Bar chart: PCE qubits vs QAOA qubits (n=m), log scale."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))
    ax.bar(["PCE", "QAOA"], [pce_qubits, m],
           color=["tab:green", "tab:red"])
    ax.set_yscale("log")
    ax.set_ylabel("qubits (log scale)")
    ax.set_title(f"Qubit count for m={m} variables")
    for i, v in enumerate([pce_qubits, m]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    return ax


def greek_bar(inst: Instance, x_naive: np.ndarray, x_pce: np.ndarray,
              targets: np.ndarray | None = None, ax=None):
    """Net Greek exposure before vs after Greek-aware compression (money shot).

    Grouped bars, one group per Greek (delta/gamma/vega): net exposure for the
    full book, the Greek-naive (cash-only) settlement, and the Greek-aware PCE
    settlement. A dashed line marks the target (0 = neutral). The naive bars are
    tall (residual risk), the PCE bars near the target.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    G = len(inst.greek_names)
    tgt = np.zeros(G) if targets is None else np.asarray(targets, dtype=float)

    full = inst.net_greeks(np.ones(inst.M, dtype=int))
    naive = inst.net_greeks(x_naive)
    pce = inst.net_greeks(x_pce)

    groups = np.arange(G)
    w = 0.25
    ax.bar(groups - w, full, w, label="full book", color="tab:gray")
    ax.bar(groups, naive, w, label="naive settle", color="tab:red")
    ax.bar(groups + w, pce, w, label="Greek-aware PCE", color="tab:green")
    for g in range(G):
        ax.hlines(tgt[g], g - 1.5 * w, g + 1.5 * w, colors="black",
                  linestyles="dashed", lw=1)

    ax.set_xticks(groups)
    ax.set_xticklabels(inst.greek_names)
    ax.set_ylabel("net exposure")
    ax.set_title("Net portfolio Greeks: dashed = target (neutral)")
    ax.legend()
    return ax


def loss_curve(history, ax=None):
    """Plot the COBYLA loss history."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, lw=1)
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("PCE loss L")
    ax.set_title("Training loss")
    return ax
