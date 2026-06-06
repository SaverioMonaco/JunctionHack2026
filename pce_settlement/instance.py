"""Settlement instance generation (plan §1.1, Phase 1).

A settlement instance is a pool of pending transactions moving cash between
parties, each with an initial balance. Settling *all* transactions is often
infeasible (gridlock); the solver picks a feasible subset of maximal value.

Amounts and balances are kept as integers (minor units, e.g. cents) so the
binary slack encoding in qubo.py is clean (plan §6 "Balances must be integers").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    """A batch of pending settlement transactions.

    Attributes
    ----------
    N : number of parties (indexed 0..N-1).
    M : number of transactions (indexed 0..M-1).
    balances : (N,) int array, initial cash balance B_p >= 0 per party.
    senders : (M,) int array, sender party index s(t) for each transaction.
    receivers : (M,) int array, receiver party index d(t).
    amounts : (M,) int array, cash amount a_t > 0 moved by each transaction.
    weights : (M,) float array, value/priority weight w_t (defaults to amount).
    greeks : optional (M, G) float array of per-transaction option Greeks
        g_t = (delta, gamma, vega). None => no risk leg (pure cash settlement).
        Drives the Greek-neutrality penalty in qubo.build_qubo (see greeks.py).
    greek_names : labels for the G Greek columns.
    """

    N: int
    M: int
    balances: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    amounts: np.ndarray
    weights: np.ndarray
    greeks: np.ndarray | None = None
    greek_names: tuple[str, ...] = ("delta", "gamma", "vega")
    name: str = "instance"

    def __post_init__(self) -> None:
        self.balances = np.asarray(self.balances, dtype=np.int64)
        self.senders = np.asarray(self.senders, dtype=np.int64)
        self.receivers = np.asarray(self.receivers, dtype=np.int64)
        self.amounts = np.asarray(self.amounts, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=float)
        assert self.balances.shape == (self.N,)
        assert self.senders.shape == (self.M,)
        assert self.receivers.shape == (self.M,)
        assert self.amounts.shape == (self.M,)
        assert self.weights.shape == (self.M,)
        assert (self.amounts > 0).all(), "amounts must be positive"
        assert (self.balances >= 0).all(), "balances must be non-negative"
        if self.greeks is not None:
            self.greeks = np.asarray(self.greeks, dtype=float)
            assert self.greeks.shape == (self.M, len(self.greek_names)), (
                f"greeks must be (M={self.M}, G={len(self.greek_names)}), "
                f"got {self.greeks.shape}")

    def net_flow(self, x: np.ndarray) -> np.ndarray:
        """Net cash outflow per party for a decision vector x in {0,1}^M.

        Returns (N,) array: outflow - inflow for each party. Feasible iff
        net_flow(x) <= balances elementwise.
        """
        x = np.asarray(x).astype(np.int64)
        flow = np.zeros(self.N, dtype=np.int64)
        np.add.at(flow, self.senders, self.amounts * x)   # outflow
        np.subtract.at(flow, self.receivers, self.amounts * x)  # inflow
        return flow

    def value(self, x: np.ndarray) -> float:
        """Total settled value for decision vector x in {0,1}^M."""
        return float(self.weights @ np.asarray(x).astype(float))

    def net_greeks(self, x: np.ndarray) -> np.ndarray:
        """Net portfolio Greeks Sum_t g_t x_t for decisions x in {0,1}^M.

        Returns a (G,) array (zeros of shape (len(greek_names),) if no greeks).
        """
        G = len(self.greek_names)
        if self.greeks is None:
            return np.zeros(G, dtype=float)
        x = np.asarray(x).astype(float)[: self.M]
        return self.greeks.T @ x


def gridlock_cycle(k_parties: int = 3, amount: int = 100, seed: int = 0) -> Instance:
    """Canonical hero demo: A->B->C->...->A circular obligations (plan §1, §9).

    Each party owes ``amount`` to the next around the cycle and starts with zero
    cash. No single payment can settle alone (sender would go negative), but
    settling the WHOLE cycle nets every party to zero -> feasible. This is the
    textbook liquidity-saving-mechanism (LSM) example: the optimum settles all
    transactions, demonstrating the netting value.
    """
    N = k_parties
    M = k_parties
    balances = np.zeros(N, dtype=np.int64)          # nobody can pay alone
    senders = np.arange(N, dtype=np.int64)
    receivers = (np.arange(N, dtype=np.int64) + 1) % N
    amounts = np.full(M, amount, dtype=np.int64)
    weights = amounts.astype(float)
    return Instance(N, M, balances, senders, receivers, amounts, weights,
                    name=f"gridlock_cycle_{k_parties}")


def random_instance(N: int, M: int, tightness: float = 1.5,
                    seed: int = 0, amount_range: tuple[int, int] = (10, 100)) -> Instance:
    """Random gridlock-prone instance (plan Phase 1).

    ``tightness`` > 1 makes total obligations exceed available liquidity, so the
    solver must select a subset. Larger tightness => deeper gridlock. Balances
    are sized so that roughly total_obligations / tightness cash is available.
    """
    rng = np.random.default_rng(seed)
    lo, hi = amount_range
    senders = rng.integers(0, N, size=M).astype(np.int64)
    # receiver != sender
    receivers = senders.copy()
    while True:
        clash = receivers == senders
        if not clash.any():
            break
        receivers[clash] = rng.integers(0, N, size=int(clash.sum()))
    amounts = rng.integers(lo, hi + 1, size=M).astype(np.int64)
    weights = amounts.astype(float)

    # Total gross outflow per party at full settlement.
    gross_out = np.zeros(N, dtype=np.int64)
    np.add.at(gross_out, senders, amounts)
    # Give each party only a fraction of its gross obligation as starting cash.
    balances = np.floor(gross_out / max(tightness, 1e-9)).astype(np.int64)

    return Instance(N, M, balances, senders, receivers, amounts, weights,
                    name=f"random_N{N}_M{M}")
