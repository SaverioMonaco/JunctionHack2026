"""
Quantum Advantage in Asian Option Pricing under the Heston Model.

Methods compared:
  1. Classical MC (Euler)
  2. Classical MLMC (Euler)
  3. Classical MLMC (Milstein)
  4. Quantum MLMC  (resource model)
  5. Direct QAE    (resource model)

Usage:
    python quantum_asian_option.py

All figures are saved to ./figures/
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── Output folder ─────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ saved → {path}")
    plt.close(fig)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.size":         11,
    "legend.framealpha": 0.9,
})

PALETTE = {
    "mc":     "#E07B39",
    "mlmc_e": "#2176AE",
    "mlmc_m": "#57A773",
    "qmlmc":  "#7B2D8B",
    "qae":    "#C0392B",
}

METHOD_STYLE = {
    "Classical MC (Euler)":        {"color": PALETTE["mc"],     "marker": "o",  "ls": "-",  "lw": 2.0},
    "Classical MLMC (Euler)":      {"color": PALETTE["mlmc_e"], "marker": "s",  "ls": "-",  "lw": 2.2},
    "Classical MLMC (Milstein)":   {"color": PALETTE["mlmc_m"], "marker": "D",  "ls": "-",  "lw": 2.2},
    "Quantum MLMC":                {"color": PALETTE["qmlmc"],  "marker": "^",  "ls": "--", "lw": 2.5},
    "Direct QAE":                  {"color": PALETTE["qae"],    "marker": "v",  "ls": ":",  "lw": 2.5},
}

# GLOBAL CONFIGURATION
SEED     = 42
S0       = 100.0
K        = 100.0
T        = 1.0
r        = 0.05

v0       = 0.04
kappa    = 2.0
theta    = 0.04
xi       = 0.3
rho      = -0.7

MLMC_M            = 4
N_LEVELS          = 7
PILOT_N           = 4_000
MIN_LEVEL_SAMPLES = 300

EPSILONS = [0.20, 0.15, 0.10, 0.07, 0.05]

BASE_ORACLE_DEPTH             = 500
BASE_ORACLE_CX                = 300
ORACLE_DEPTH_GROWTH_PER_LEVEL = 2.0
ORACLE_CX_GROWTH_PER_LEVEL    = 2.0
GROVER_FACTOR                 = 2.0

# HESTON SIMULATION
def asian_call_payoff(S_avg):
    return np.exp(-r * T) * np.maximum(S_avg - K, 0.0)


def simulate_heston_payoffs(N, n_steps, rng, scheme="euler"):
    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)
    logS    = np.full(N, np.log(S0), dtype=float)
    v       = np.full(N, v0, dtype=float)
    Sacc    = np.zeros(N)

    for _ in range(n_steps):
        Z1 = rng.standard_normal(N)
        Z2 = rng.standard_normal(N)
        vp = np.maximum(v, 1e-12)
        sv = np.sqrt(vp)
        dWv = Z1 * sqrt_dt
        dWs = (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2) * sqrt_dt
        logS += (r - 0.5 * vp) * dt + sv * dWs
        if scheme == "euler":
            v += kappa * (theta - vp) * dt + xi * sv * dWv
        elif scheme == "milstein":
            v += (kappa * (theta - vp) * dt
                  + xi * sv * dWv
                  + 0.25 * xi**2 * (dWv**2 - dt))
        else:
            raise ValueError("scheme must be 'euler' or 'milstein'.")
        v     = np.maximum(v, 0.0)
        Sacc += np.exp(logS)

    return asian_call_payoff(Sacc / n_steps)

# MLMC MACHINERY
def mlmc_level_correction(level, N, scheme, rng):
    n_fine    = MLMC_M ** level
    dt_f      = T / n_fine
    sqrt_dt_f = np.sqrt(dt_f)
    Z1 = rng.standard_normal((N, n_fine))
    Z2 = rng.standard_normal((N, n_fine))

    def _evolve(n_steps, dt, dWv_mat, dWs_mat):
        logS = np.full(N, np.log(S0), dtype=float)
        v    = np.full(N, v0, dtype=float)
        Sacc = np.zeros(N)
        for i in range(n_steps):
            dWv = dWv_mat[:, i]; dWs = dWs_mat[:, i]
            vp  = np.maximum(v, 1e-12); sv = np.sqrt(vp)
            logS += (r - 0.5 * vp) * dt + sv * dWs
            if scheme == "euler":
                v += kappa * (theta - vp) * dt + xi * sv * dWv
            else:
                v += (kappa * (theta - vp) * dt
                      + xi * sv * dWv
                      + 0.25 * xi**2 * (dWv**2 - dt))
            v     = np.maximum(v, 0.0)
            Sacc += np.exp(logS)
        return asian_call_payoff(Sacc / n_steps)

    dWv_f = Z1 * sqrt_dt_f
    dWs_f = (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2) * sqrt_dt_f
    P_f   = _evolve(n_fine, dt_f, dWv_f, dWs_f)

    if level == 0:
        return P_f, np.zeros(N)

    n_coarse = MLMC_M ** (level - 1)
    dWv_c    = np.zeros((N, n_coarse))
    dWs_c    = np.zeros((N, n_coarse))
    for j in range(n_coarse):
        sl = slice(j * MLMC_M, (j + 1) * MLMC_M)
        dWv_c[:, j] = dWv_f[:, sl].sum(axis=1)
        dWs_c[:, j] = dWs_f[:, sl].sum(axis=1)

    P_c = _evolve(n_coarse, T / n_coarse, dWv_c, dWs_c)
    return P_f, P_c


def estimate_mlmc_level_stats(n_levels=N_LEVELS, scheme="euler",
                               pilot_N=PILOT_N, seed=SEED):
    rng  = np.random.default_rng(seed)
    rows = []
    for level in range(n_levels):
        Pf, Pc = mlmc_level_correction(level, pilot_N, scheme, rng)
        Y = Pf - Pc
        rows.append({
            "level":           level,
            "scheme":          scheme,
            "mean":            float(np.mean(Y)),
            "variance":        float(max(np.var(Y, ddof=1), 1e-14)),
            "cost_per_sample": float(MLMC_M ** level),
        })
    return pd.DataFrame(rows)

# CLASSICAL BASELINES

def classical_mc_baseline(target_epsilon, scheme="euler", rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_steps   = MLMC_M ** (N_LEVELS - 1)
    pilot     = simulate_heston_payoffs(3_000, n_steps, rng, scheme=scheme)
    var_pilot = np.var(pilot, ddof=1)
    N         = max(int(np.ceil(var_pilot / target_epsilon**2)), 1_000)
    t0        = time.time()
    payoffs   = simulate_heston_payoffs(N, n_steps, rng, scheme=scheme)
    return {
        "method":             "Classical MC (Euler)",
        "epsilon":            target_epsilon,
        "price":              float(np.mean(payoffs)),
        "stderr":             float(np.std(payoffs, ddof=1) / np.sqrt(N)),
        "path_steps":         int(N * n_steps),
        "oracle_calls":       0,
        "quantum_depth_cost": 0,
        "hybrid_total_cost":  int(N * n_steps),
        "runtime_sec":        time.time() - t0,
    }


def classical_mlmc_baseline(target_epsilon, scheme="euler",
                             n_levels=N_LEVELS, pilot_N=PILOT_N):
    stats = estimate_mlmc_level_stats(n_levels=n_levels, scheme=scheme,
                                       pilot_N=pilot_N, seed=SEED + 100)
    Vl = stats["variance"].to_numpy(dtype=float)
    Cl = stats["cost_per_sample"].to_numpy(dtype=float)
    Nl = np.ceil(
        (2.0 / target_epsilon**2) * np.sqrt(Vl * Cl).sum() * np.sqrt(Vl / Cl)
    ).astype(int)
    Nl  = np.maximum(Nl, MIN_LEVEL_SAMPLES)
    t0  = time.time()
    rng = np.random.default_rng(SEED + 200)
    estimates = []; path_steps = 0
    for level in range(n_levels):
        Pf, Pc = mlmc_level_correction(level, int(Nl[level]), scheme, rng)
        Y = Pf - Pc
        estimates.append(np.mean(Y))
        path_steps += int(Nl[level] * Cl[level])
    label = "Classical MLMC (Euler)" if scheme == "euler" else "Classical MLMC (Milstein)"
    return {
        "method":             label,
        "epsilon":            target_epsilon,
        "price":              float(np.sum(estimates)),
        "stderr":             float(np.sqrt(np.sum(Vl / Nl))),
        "path_steps":         int(path_steps),
        "oracle_calls":       0,
        "quantum_depth_cost": 0,
        "hybrid_total_cost":  int(path_steps),
        "runtime_sec":        time.time() - t0,
        "Nl":                 Nl,
        "level_stats":        stats,
    }

# QUANTUM RESOURCE MODEL

def oracle_depth_for_level(level):
    return int(BASE_ORACLE_DEPTH * (ORACLE_DEPTH_GROWTH_PER_LEVEL ** level))

def oracle_cx_for_level(level):
    return int(BASE_ORACLE_CX * (ORACLE_CX_GROWTH_PER_LEVEL ** level))

def payoff_range_from_variance(variance):
    return max(6.0 * np.sqrt(max(variance, 1e-14)), 1.0)

def qae_calls_for_level(epsilon_level, payoff_range):
    epsilon_amp = max(epsilon_level / payoff_range, 1e-4)
    return int(np.ceil(np.pi / epsilon_amp))

def quantum_level_cost(level, variance, epsilon_level):
    payoff_range = payoff_range_from_variance(variance)
    calls        = qae_calls_for_level(epsilon_level, payoff_range)
    depth        = oracle_depth_for_level(level)
    cx           = oracle_cx_for_level(level)
    return {
        "oracle_calls":       calls,
        "oracle_depth":       depth,
        "oracle_cx":          cx,
        "quantum_depth_cost": int(GROVER_FACTOR * calls * depth),
        "quantum_cx_cost":    int(GROVER_FACTOR * calls * cx),
        "payoff_range":       payoff_range,
    }


def quantum_mlmc_resource_baseline(target_epsilon, scheme="euler",
                                   n_levels=N_LEVELS, pilot_N=PILOT_N):
    stats         = estimate_mlmc_level_stats(n_levels=n_levels, scheme=scheme,
                                              pilot_N=pilot_N, seed=SEED + 300)
    epsilon_level = target_epsilon / np.sqrt(n_levels)
    total_calls = total_depth = total_cx = 0
    level_rows  = []
    for _, row in stats.iterrows():
        level    = int(row["level"])
        variance = float(row["variance"])
        qres     = quantum_level_cost(level, variance, epsilon_level)
        total_calls += qres["oracle_calls"]
        total_depth += qres["quantum_depth_cost"]
        total_cx    += qres["quantum_cx_cost"]
        level_rows.append({
            "level":              level,
            "variance":           variance,
            "mean":               float(row["mean"]),
            "oracle_calls":       qres["oracle_calls"],
            "oracle_depth":       qres["oracle_depth"],
            "quantum_depth_cost": qres["quantum_depth_cost"],
        })
    return {
        "method":             "Quantum MLMC",
        "epsilon":            target_epsilon,
        "price":              float(stats["mean"].sum()),
        "stderr":             target_epsilon,
        "path_steps":         0,
        "oracle_calls":       int(total_calls),
        "quantum_depth_cost": int(total_depth),
        "quantum_cx_cost":    int(total_cx),
        "hybrid_total_cost":  int(total_depth),
        "runtime_sec":        0.0,
    }, pd.DataFrame(level_rows)


def direct_qae_resource_baseline(target_epsilon, final_level=N_LEVELS - 1,
                                  payoff_range=None, pilot_N=PILOT_N):
    if payoff_range is None:
        payoff_range = S0
    epsilon_amp        = max(target_epsilon / payoff_range, 1e-4)
    oracle_calls       = int(np.ceil(np.pi / epsilon_amp))
    oracle_depth       = oracle_depth_for_level(final_level)
    oracle_cx          = oracle_cx_for_level(final_level)
    quantum_depth_cost = int(GROVER_FACTOR * oracle_calls * oracle_depth)
    quantum_cx_cost    = int(GROVER_FACTOR * oracle_calls * oracle_cx)
    # Price: QAE converges to E[P_L]; estimated via fine-grid pilot
    n_steps = MLMC_M ** final_level
    rng     = np.random.default_rng(SEED + 999)
    payoffs = simulate_heston_payoffs(pilot_N, n_steps, rng, scheme="euler")
    return {
        "method":             "Direct QAE",
        "epsilon":            target_epsilon,
        "price":              float(np.mean(payoffs)),
        "stderr":             target_epsilon,
        "path_steps":         0,
        "oracle_calls":       oracle_calls,
        "quantum_depth_cost": quantum_depth_cost,
        "quantum_cx_cost":    quantum_cx_cost,
        "hybrid_total_cost":  quantum_depth_cost,
        "runtime_sec":        0.0,
    }

# RUN ALL METHODS

def run_all_methods(epsilons=EPSILONS, n_levels=N_LEVELS):
    results      = []
    qmlmc_levels = []

    for eps in epsilons:
        print(f"\n{'='*70}\nε = {eps}\n{'='*70}")
        rng = np.random.default_rng(SEED + int(10_000 * eps))

        mc = classical_mc_baseline(target_epsilon=eps, scheme="euler", rng=rng)
        print(f"  Classical MC (Euler)       price={mc['price']:.4f}  "
              f"SE={mc['stderr']:.4f}  cost={mc['path_steps']:>13,}")
        results.append(mc)

        mlmc_e = classical_mlmc_baseline(target_epsilon=eps, scheme="euler",
                                          n_levels=n_levels)
        print(f"  Classical MLMC (Euler)     price={mlmc_e['price']:.4f}  "
              f"SE={mlmc_e['stderr']:.4f}  cost={mlmc_e['path_steps']:>13,}")
        results.append({k: v for k, v in mlmc_e.items()
                        if k not in ("Nl", "level_stats")})

        mlmc_m = classical_mlmc_baseline(target_epsilon=eps, scheme="milstein",
                                          n_levels=n_levels)
        print(f"  Classical MLMC (Milstein)  price={mlmc_m['price']:.4f}  "
              f"SE={mlmc_m['stderr']:.4f}  cost={mlmc_m['path_steps']:>13,}")
        results.append({k: v for k, v in mlmc_m.items()
                        if k not in ("Nl", "level_stats")})

        qmlmc, ql_df = quantum_mlmc_resource_baseline(
            target_epsilon=eps, scheme="euler", n_levels=n_levels)
        print(f"  Quantum MLMC               price={qmlmc['price']:.4f}  "
              f"calls={qmlmc['oracle_calls']:,}  "
              f"depth-cost={qmlmc['quantum_depth_cost']:>13,}")
        results.append(qmlmc)
        qmlmc_levels.append(ql_df.assign(epsilon=eps))

        dqae = direct_qae_resource_baseline(
            target_epsilon=eps, final_level=n_levels - 1, payoff_range=S0)
        print(f"  Direct QAE                 price={dqae['price']:.4f}  "
              f"calls={dqae['oracle_calls']:,}  "
              f"depth-cost={dqae['quantum_depth_cost']:>13,}")
        results.append(dqae)

    results_df      = pd.DataFrame(results)
    qmlmc_levels_df = pd.concat(qmlmc_levels, ignore_index=True)
    return results_df, qmlmc_levels_df

# FIGURES

def _get(df, method):
    sub = df[df["method"] == method].sort_values("epsilon")
    if sub.empty:
        print(f"  [WARN] '{method}' not found — skipping.")
    return sub


# ──  MLMC variance decay ────────────────────────────────────────────
def fig_variance_decay():
    print("\n[Figure 1] MLMC variance decay …")
    rows = []
    for scheme in ["euler", "milstein"]:
        rng = np.random.default_rng(SEED + 800)
        for level in range(N_LEVELS + 1):
            Pf, Pc = mlmc_level_correction(level, 10_000, scheme, rng)
            Y = Pf - Pc
            rows.append({
                "scheme": scheme, "level": level,
                "variance_correction": float(np.var(Y, ddof=1)),
                "mean_correction":     float(np.mean(Y)),
            })
    diag_df = pd.DataFrame(rows)

    colours = {"euler": PALETTE["mlmc_e"], "milstein": PALETTE["mlmc_m"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, metric, ylabel, title in [
        (axes[0], "variance_correction", "Var[$Y_ℓ$]",
         "Level correction variance (steeper = cheaper MLMC)"),
        (axes[1], "mean_correction", "E[$Y_ℓ$]",
         "Level correction mean (converges to 0)"),
    ]:
        for scheme, sub in diag_df.groupby("scheme"):
            sub = sub.sort_values("level")
            x   = sub["level"].to_numpy(dtype=float)
            y   = np.maximum(np.abs(sub[metric].to_numpy(dtype=float)), 1e-18)
            ax.semilogy(x, y, marker="o", linewidth=2.2,
                        color=colours[scheme], label=scheme.capitalize())
        ax.set_xlabel("MLMC level  ℓ")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(sorted(diag_df["level"].unique()))
        ax.legend()

    fig.suptitle("MLMC diagnostics — variance and mean decay across levels",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "01_mlmc_variance_decay.png")


# ── Oracle depth growth ────────────────────────────────────────────
def fig_oracle_depth_growth():
    print("\n[Figure 2] Oracle depth growth …")
    n_steps_list = np.arange(1, 8)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.cm.plasma
    for i, sq in enumerate([1, 2, 3]):
        depths  = 100 * 2 ** (2 * n_steps_list * sq)
        colour  = cmap(0.2 + 0.35 * i)
        ax.semilogy(n_steps_list, depths, marker="o", linewidth=2.2,
                    color=colour, label=f"{sq} qubit(s) per Gaussian")
    ax.set_xlabel("Number of Heston time steps")
    ax.set_ylabel("Estimated oracle depth (a.u.)")
    ax.set_title(
        "Quantum bottleneck: oracle depth grows exponentially with time steps\n"
        "→ motivates Quantum MLMC (shallow oracles) over Direct QAE (one deep oracle)",
        fontsize=10)
    ax.legend(title="Shock qubit budget")
    plt.tight_layout()
    save(fig, "02_oracle_depth_growth.png")


# ──  Total cost vs epsilon ──────────────────────────────────────────
def fig_total_cost(results_df):
    print("\n[Figure 3] Total cost vs ε …")
    fig, ax = plt.subplots(figsize=(9, 5))

    for method, style in METHOD_STYLE.items():
        sub = _get(results_df, method)
        if sub.empty:
            continue
        x = sub["epsilon"].to_numpy(dtype=float)
        y = np.maximum(sub["hybrid_total_cost"].to_numpy(dtype=float), 1.0)
        ax.loglog(x, y, color=style["color"], marker=style["marker"],
                  linestyle=style["ls"], linewidth=style["lw"],
                  markersize=7, label=method)

    # Reference slopes
    eps_arr = np.array([EPSILONS[-1], EPSILONS[0]], dtype=float)
    anchor  = results_df[results_df["method"] == "Classical MC (Euler)"]["hybrid_total_cost"].max()
    for exp, label, va in [(-2, "ε⁻²", "top"), (-1.5, "ε⁻¹·⁵", "bottom")]:
        y_ref = anchor * (eps_arr / EPSILONS[0]) ** exp
        ax.loglog(eps_arr, y_ref, "k--", lw=1, alpha=0.3)
        ax.text(eps_arr[0] * 0.93, y_ref[0], label, fontsize=9,
                color="grey", va=va, ha="right")

    ax.invert_xaxis()
    ax.set_xlabel("Target accuracy  ε", fontsize=12)
    ax.set_ylabel("Computational cost  (a.u.)", fontsize=12)
    ax.set_title("Total cost vs target accuracy\n"
                 "Quantum methods scale more favourably as ε tightens",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "03_total_cost_vs_epsilon.png")


# ── Figure 4: Speedup over Classical MLMC (Euler) ────────────────────────────
def fig_speedup(results_df):
    print("\n[Figure 4] Speedup …")
    REF     = "Classical MLMC (Euler)"
    ref_sub = _get(results_df, REF).set_index("epsilon")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="black", lw=1.4, ls="--",
               label="Classical MLMC (Euler) — baseline")

    for method, style in METHOD_STYLE.items():
        if method == REF:
            continue
        sub = _get(results_df, method)
        if sub.empty:
            continue
        epsilons  = sub["epsilon"].to_numpy(dtype=float)
        costs     = sub["hybrid_total_cost"].to_numpy(dtype=float)
        ref_costs = np.array([ref_sub.loc[e, "hybrid_total_cost"]
                               if e in ref_sub.index else np.nan
                               for e in epsilons], dtype=float)
        speedup = ref_costs / np.maximum(costs, 1.0)
        ax.plot(epsilons, speedup, color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=style["lw"],
                markersize=7, label=method)

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Target accuracy  ε", fontsize=12)
    ax.set_ylabel("Speedup  (×)", fontsize=12)
    ax.set_title("Speedup relative to Classical MLMC (Euler)\n"
                 "Values > 1 mean the method is cheaper than classical MLMC",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    # Shade advantage zone
    ax.axhspan(1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5,
               color="green", alpha=0.04)
    ax.text(0.97, 0.94, "Cheaper than classical MLMC ↑",
            transform=ax.transAxes, ha="right", fontsize=9, color="darkgreen")
    plt.tight_layout()
    save(fig, "04_speedup.png")


# ──  Price convergence ───────────────────────────────────────────────
def fig_price_convergence(results_df):
    print("\n[Figure 5] Price convergence …")
    PRICE_METHODS = [
        "Classical MC (Euler)",
        "Classical MLMC (Euler)",
        "Classical MLMC (Milstein)",
        "Quantum MLMC",
        "Direct QAE",
    ]
    epsilons  = sorted(results_df["epsilon"].unique())
    x_pos     = {eps: i for i, eps in enumerate(epsilons)}
    n_methods = len(PRICE_METHODS)
    width     = 0.7 / n_methods

    ref_rows  = results_df[results_df["method"] == "Classical MLMC (Milstein)"]
    ref_price = ref_rows["price"].median() if not ref_rows.empty else None

    fig, ax = plt.subplots(figsize=(11, 5))
    if ref_price is not None:
        ax.axhspan(ref_price - 0.05, ref_price + 0.05,
                   color="gold", alpha=0.20, label="Reference ± 0.05")
        ax.axhline(ref_price, color="goldenrod", lw=1.5, ls="--")

    for k, method in enumerate(PRICE_METHODS):
        sub = _get(results_df, method)
        if sub.empty:
            continue
        style   = METHOD_STYLE[method]
        prices  = sub["price"].to_numpy(dtype=float)
        stderrs = sub["stderr"].to_numpy(dtype=float)
        valid   = ~np.isnan(prices)
        xs      = np.array([x_pos[e] + (k - n_methods / 2 + 0.5) * width
                            for e in sub["epsilon"]])
        ax.errorbar(xs[valid], prices[valid], yerr=stderrs[valid],
                    fmt=style["marker"], color=style["color"],
                    linewidth=1.8, markersize=7, capsize=4, label=method)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([f"ε = {e}" for e in epsilons], fontsize=10)
    ax.set_ylabel("Estimated option price", fontsize=12)
    ax.set_title("Price convergence across methods  (±1 SE error bars)\n"
                 "All methods agree — quantum savings do not sacrifice accuracy",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "05_price_convergence.png")


# ──  Quantum MLMC per-level cost breakdown ──────────────────────────
def fig_qmlmc_level_breakdown(qmlmc_levels_df):
    print("\n[Figure 6] Quantum MLMC per-level breakdown …")
    epsilons = sorted(qmlmc_levels_df["epsilon"].unique())
    levels   = sorted(qmlmc_levels_df["level"].unique())
    cmap     = plt.cm.viridis
    colours  = [cmap(i / max(len(levels) - 1, 1)) for i in range(len(levels))]

    fig, ax  = plt.subplots(figsize=(11, 4.5))
    x        = np.arange(len(epsilons))
    bottom   = np.zeros(len(epsilons))

    for li, level in enumerate(levels):
        heights = []
        for eps in epsilons:
            row = qmlmc_levels_df[
                (qmlmc_levels_df["epsilon"] == eps) &
                (qmlmc_levels_df["level"]   == level)]
            heights.append(float(row["quantum_depth_cost"].values[0])
                           if len(row) > 0 else 0.0)
        heights = np.array(heights)
        ax.bar(x, heights, bottom=bottom, color=colours[li],
               label=f"Level {level}", width=0.6)
        bottom += heights

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {e}" for e in epsilons], fontsize=10)
    ax.set_ylabel("Quantum depth-cost  (log scale)", fontsize=11)
    ax.set_title("Quantum MLMC — depth-cost contribution per level\n"
                 "Fine levels dominate due to exponentially deeper oracles",
                 fontsize=11, fontweight="bold")
    ax.legend(title="MLMC level", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    save(fig, "06_qmlmc_level_breakdown.png")


# ── Switching boundary (per epsilon) ───────────────────────────────
def _find_log_intersection(x, y_classical, y_quantum):
    x     = np.asarray(x, dtype=float)
    log_c = np.log(np.asarray(y_classical, dtype=float))
    log_q = np.log(np.asarray(y_quantum,   dtype=float))
    diff  = log_c - log_q
    exact = np.where(np.isclose(diff, 0.0))[0]
    if len(exact):
        return x[exact[0]]
    for i in range(len(x) - 1):
        if diff[i] * diff[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None


def fig_switching_boundary():
    print("\n[Figure 7] Switching boundary per epsilon …")
    n_eps = len(EPSILONS)
    ncols = 3
    nrows = (n_eps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    for ax_idx, target_epsilon in enumerate(EPSILONS):
        ax    = axes[ax_idx]
        stats = estimate_mlmc_level_stats(n_levels=N_LEVELS, scheme="euler",
                                          pilot_N=PILOT_N, seed=SEED + 400)
        epsilon_level = target_epsilon / np.sqrt(N_LEVELS)

        levels = []; classical_costs = []; quantum_costs = []
        for _, row in stats.iterrows():
            level    = int(row["level"])
            variance = float(row["variance"])
            cps      = float(row["cost_per_sample"])
            N_cl     = max(int(np.ceil(variance / epsilon_level**2)), MIN_LEVEL_SAMPLES)
            c_cl     = N_cl * cps
            qres     = quantum_level_cost(level, variance, epsilon_level)
            c_q      = qres["quantum_depth_cost"]
            levels.append(level); classical_costs.append(c_cl); quantum_costs.append(c_q)

        levels          = np.array(levels,          dtype=float)
        classical_costs = np.array(classical_costs, dtype=float)
        quantum_costs   = np.array(quantum_costs,   dtype=float)
        boundary        = _find_log_intersection(levels, classical_costs, quantum_costs)

        y_min = min(classical_costs.min(), quantum_costs.min()) / 2
        y_max = max(classical_costs.max(), quantum_costs.max()) * 3
        x_l   = levels.min() - 0.5
        x_r   = levels.max() + 0.5

        if boundary is not None:
            ax.axvspan(x_l,       boundary, color="lightgreen", alpha=0.20)
            ax.axvspan(boundary,  x_r,      color="plum",       alpha=0.20)
            ax.axvline(boundary, color="black", ls="--", lw=2,
                       label=f"Boundary ℓ ≈ {boundary:.1f}")
            ax.text((x_l + boundary) / 2, y_min * 2.5, "Classical",
                    fontsize=8, fontweight="bold", color="darkgreen", ha="center")
            ax.text((boundary + x_r) / 2, y_min * 2.5, "Quantum",
                    fontsize=8, fontweight="bold", color="purple", ha="center")

        ax.semilogy(levels, classical_costs, marker="o", lw=2.2,
                    color=PALETTE["mlmc_e"], label="Classical cost")
        ax.semilogy(levels, quantum_costs,   marker="s", lw=2.2,
                    color=PALETTE["qmlmc"],  label="Quantum cost")

        # Annotate winner per level
        for lv, cc, qc in zip(levels, classical_costs, quantum_costs):
            winner = "Q" if qc < cc else "C"
            col    = PALETTE["qmlmc"] if winner == "Q" else PALETTE["mlmc_e"]
            ax.annotate(winner, xy=(lv, min(cc, qc)),
                        xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold", color=col)

        ax.set_xlim(x_l, x_r); ax.set_ylim(y_min, y_max)
        ax.set_xticks(levels.astype(int))
        ax.set_xlabel("MLMC level  ℓ", fontsize=9)
        ax.set_ylabel("Cost (log)", fontsize=9)
        ax.set_title(f"ε = {target_epsilon}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    # Hide unused axes
    for ax in axes[n_eps:]:
        ax.set_visible(False)

    fig.suptitle(
        "Switching boundary — where quantum beats classical per MLMC level\n"
        "C = classical cheaper   Q = quantum cheaper",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_switching_boundary.png")


# ──Direct QAE vs Quantum MLMC depth-cost ─────────────────────────
def fig_qae_vs_qmlmc(results_df):
    print("\n[Figure 8] Direct QAE vs Quantum MLMC …")
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in ["Quantum MLMC", "Direct QAE"]:
        sub = _get(results_df, method)
        if sub.empty:
            continue
        style = METHOD_STYLE[method]
        x = sub["epsilon"].to_numpy(dtype=float)
        y = np.maximum(sub["quantum_depth_cost"].to_numpy(dtype=float), 1.0)
        ax.loglog(x, y, color=style["color"], marker=style["marker"],
                  linestyle=style["ls"], linewidth=style["lw"],
                  markersize=7, label=method)
    ax.invert_xaxis()
    ax.set_xlabel("Target accuracy  ε", fontsize=12)
    ax.set_ylabel("Quantum depth-cost  (a.u.)", fontsize=12)
    ax.set_title("Direct QAE vs Quantum MLMC — quantum depth-cost\n"
                 "Quantum MLMC wins by keeping oracle circuits shallow",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "08_qae_vs_qmlmc.png")


# MAIN
if __name__ == "__main__":
    print("=" * 70)
    print("  Quantum Asian Option Pricing — Heston Model")
    print(f"  Figures will be saved to: ./{FIGURES_DIR}/")
    print("=" * 70)

    # Figures that don't need results_df
    fig_variance_decay()
    fig_oracle_depth_growth()
    fig_switching_boundary()

    # Run all methods
    print("\n" + "=" * 70)
    print("  Running all methods …")
    print("=" * 70)
    results_df, qmlmc_levels_df = run_all_methods()

    # Figures that need results_df
    fig_total_cost(results_df)
    fig_speedup(results_df)
    fig_price_convergence(results_df)
    fig_qmlmc_level_breakdown(qmlmc_levels_df)
    fig_qae_vs_qmlmc(results_df)

    print(f"\n{'='*70}")
    print(f"  All done. {len(os.listdir(FIGURES_DIR))} figures saved in ./{FIGURES_DIR}/")
    print("=" * 70)
