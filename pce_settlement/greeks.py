"""Black-Scholes option Greeks and Greek-laden settlement instances.

This module extends PCE settlement from the pure cash leg to *risk-aware*
settlement / Greek-constrained netting. Each transaction is a trade in an
option contract that carries a Greek vector g_t = (delta, gamma, vega). The
solver then picks the feasible subset that maximizes settled value *and* drives
the net portfolio Greeks toward neutrality (qubo.py folds the residual net-Greek
exposure in as a quadratic penalty, exactly like the solvency penalty).

Grounded in:
  - Portfolio compression (ISDA post-trade netting): reduce a book to fewer
    trades while keeping net Greeks near-neutral (arXiv:2402.17941).
  - Greek-neutral hedge selection via quadratic programming (MPRA "Hedging
    Greeks for a portfolio of options using Linear and Quadratic Programming").
  - Budget-constrained PCE (arXiv:2602.17479v2): the penalty-in-PCE machinery.

Dependencies: numpy + scipy only (scipy.stats.norm for the BS formulas).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .instance import Instance


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              kind: str = "call") -> tuple[float, float, float]:
    """Black-Scholes Greeks (delta, gamma, vega) for a single European option.

    Parameters
    ----------
    S : spot price of the underlying.
    K : strike price.
    T : time to expiry in years (> 0).
    r : risk-free rate (continuous).
    sigma : implied volatility (> 0).
    kind : 'call' or 'put'.

    Returns
    -------
    (delta, gamma, vega) for a long single contract.
      delta : d(price)/d(S).      call: N(d1);  put: N(d1) - 1.
      gamma : d2(price)/d(S)^2.   n(d1) / (S sigma sqrt(T)).  Same call/put.
      vega  : d(price)/d(sigma).  S n(d1) sqrt(T).  Same call/put.

    Vega convention: per 1.00 (100%) change in vol (NOT per 1%). Divide by 100
    for a per-1%-vol move if you prefer that scale; the QUBO is scale-invariant
    because default_lambdas normalizes by sum_t |g_t| per Greek.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, sigma must be positive")
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    pdf = norm.pdf(d1)
    if kind == "call":
        delta = norm.cdf(d1)
    elif kind == "put":
        delta = norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT
    return float(delta), float(gamma), float(vega)


def bs_greeks_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float,
                  sigma: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Vectorized BS Greeks. ``kind`` is an array of +1 (call) / -1 (put).

    Returns an (M, 3) array of (delta, gamma, vega), one row per option.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    kind = np.asarray(kind)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    pdf = norm.pdf(d1)
    # call delta = N(d1); put delta = N(d1) - 1. kind==1 -> call, ==-1 -> put.
    delta = norm.cdf(d1) - np.where(kind > 0, 0.0, 1.0)
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT
    return np.stack([delta, gamma, vega], axis=1)


def greek_gridlock_demo(amount: int = 100, delta_mag: float = 0.6,
                        gamma_mag: float = 0.02, vega_mag: float = 12.0
                        ) -> Instance:
    """Hero demo: a delta-laden settlement where cash-greedy leaves residual risk.

    Legible story in one picture. A hub party (P0) trades with three
    counterparties; the batch carries directional risk that does NOT net to zero
    if you settle everything:

        trade 0:  P0 -> P1,  amount 100,  delta +0.6   (long delta)
        trade 1:  P0 -> P2,  amount 100,  delta +0.6   (long delta)
        trade 2:  P3 -> P0,  amount 100,  delta -0.6   (short delta)
        trade 3:  P1 -> P0,  amount 100,  delta -0.6   (short delta)
        trade 4:  P2 -> P0,  amount 100,  delta -0.6   (short delta)

    Three short-delta legs but only two long-delta legs, so settling the whole
    batch (the cash-only optimum, value 500) leaves a residual net delta of -0.6.
    A risk-averse desk drops one short-delta leg to flatten delta to zero,
    settling 400 of 500 value -- the Greek/value trade-off. Gamma and vega
    follow the same per-trade signs at smaller magnitude.

    The Greek penalty must be weighted heavily enough for the desk to prefer the
    neutral set; use cfg.risk_aversion >= 5 (the demo default). Round,
    hand-checkable numbers for a clean before/after story.
    """
    N, M = 4, 5
    #            t0  t1  t2  t3  t4
    senders   = np.array([0, 0, 3, 1, 2], dtype=np.int64)
    receivers = np.array([1, 2, 0, 0, 0], dtype=np.int64)
    amounts   = np.full(M, amount, dtype=np.int64)
    weights   = amounts.astype(float)
    balances  = np.full(N, amount, dtype=np.int64)

    signs = np.array([+1.0, +1.0, -1.0, -1.0, -1.0])
    greeks = np.stack([
        signs * delta_mag,
        signs * gamma_mag,
        signs * vega_mag,
    ], axis=1)  # (M, 3)

    return Instance(N, M, balances, senders, receivers, amounts, weights,
                    greeks=greeks, name="greek_gridlock_hub")


def option_settlement_instance(N: int = 6, M: int = 14, tightness: float = 1.5,
                               seed: int = 0, r: float = 0.02,
                               spot_range: tuple[float, float] = (80.0, 120.0),
                               vol_range: tuple[float, float] = (0.15, 0.45),
                               expiry_range: tuple[float, float] = (0.1, 1.5),
                               contracts_range: tuple[int, int] = (1, 5),
                               ) -> Instance:
    """Random Greek-laden option book settled between ``N`` parties.

    Each transaction is a trade: a party (seller) delivers ``contracts`` of an
    option contract to another party (buyer) for an integer cash ``amount``.
    Spot/strike/expiry/vol are drawn at random; the per-contract Greeks come from
    Black-Scholes. The trade carries the Greeks signed by the *seller's* book:
    the seller is short the option (-g per contract), so the settlement decision
    "do we clear this trade" moves -g*contracts of net Greek for the batch.

    Cash sizing reuses the gridlock logic of ``instance.random_instance``:
    balances are a fraction (1/tightness) of each party's gross outflow so that
    settling everything is infeasible and selection matters.
    """
    rng = np.random.default_rng(seed)

    senders = rng.integers(0, N, size=M).astype(np.int64)
    receivers = senders.copy()
    while True:
        clash = receivers == senders
        if not clash.any():
            break
        receivers[clash] = rng.integers(0, N, size=int(clash.sum()))

    S = rng.uniform(*spot_range, size=M)
    K = rng.uniform(*spot_range, size=M)
    T = rng.uniform(*expiry_range, size=M)
    sigma = rng.uniform(*vol_range, size=M)
    kind = rng.choice([1, -1], size=M)  # +1 call, -1 put
    contracts = rng.integers(contracts_range[0], contracts_range[1] + 1, size=M)

    per_contract = bs_greeks_vec(S, K, T, r, sigma, kind)  # (M, 3)
    # Seller is short the option: settling the trade moves -g per contract for
    # the batch. Scale by number of contracts.
    greeks = -per_contract * contracts[:, None]

    # Cash amount: notional-like, proportional to |delta| * spot * contracts,
    # kept as a positive integer (clean binary slack encoding).
    raw_amount = np.abs(per_contract[:, 0]) * S * contracts
    amounts = np.maximum(np.round(raw_amount).astype(np.int64), 1)
    weights = amounts.astype(float)

    gross_out = np.zeros(N, dtype=np.int64)
    np.add.at(gross_out, senders, amounts)
    balances = np.floor(gross_out / max(tightness, 1e-9)).astype(np.int64)

    return Instance(N, M, balances, senders, receivers, amounts, weights,
                    greeks=greeks, name=f"option_book_N{N}_M{M}")
