r"""qff_pricing.classical.monte_carlo
=====================================

Plain classical Monte-Carlo prices using the *same* exact fast-forwarding
transitions as the quantum circuits.  These provide the ground-truth the quantum
estimates are checked against.

A classical MC estimate of a price to additive error ``eps`` needs
``O(1/eps^2)`` samples; the quantum QMCI estimator needs ``O(1/eps)`` -- the
quadratic speedup.  Here we just want an accurate reference, so we use many
samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qff_pricing.data import Dataset
from qff_pricing.models.cir import CIRModel
from qff_pricing.models.heston import HestonModel
from qff_pricing.quantum.payoff import asian_payoff, european_payoff


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int
    model: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{self.model} MC price = {self.price:.6f} "
                f"+/- {1.96 * self.std_error:.6f} (95% CI, {self.n_paths} paths)")


def _discount(ds: Dataset) -> float:
    return float(np.exp(-ds.market.risk_free_rate * ds.contract.maturity))


def price_cir_european(ds: Dataset, n_paths: int = 1_000_000,
                       seed: int = 0) -> MCResult:
    """Reference price for a European option on a CIR underlying."""
    if ds.cir is None:
        raise ValueError("Dataset has no CIR params.")
    rng = np.random.default_rng(seed)
    model = CIRModel(ds.cir, ds.contract.dt)
    paths = model.sample_paths(n_paths, ds.contract.n_steps, rng)
    terminal = paths[:, -1]
    payoff = european_payoff(terminal, ds.contract.strike, ds.contract.option_type)
    disc = _discount(ds)
    disc_payoff = disc * payoff
    return MCResult(
        price=float(disc_payoff.mean()),
        std_error=float(disc_payoff.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
        model="CIR",
    )


def price_heston_european(ds: Dataset, n_paths: int = 200_000, seed: int = 0,
                          exact_integral: bool = False) -> MCResult:
    """Reference price for a European option on a Heston asset price."""
    if ds.heston is None:
        raise ValueError("Dataset has no Heston params.")
    rng = np.random.default_rng(seed)
    model = HestonModel(ds.heston, ds.contract.dt)
    terminal = model.sample_paths(n_paths, ds.contract.n_steps, rng,
                                  exact_integral=exact_integral)["S"][:, -1]
    payoff = european_payoff(terminal, ds.contract.strike, ds.contract.option_type)
    disc = _discount(ds)
    disc_payoff = disc * payoff
    return MCResult(
        price=float(disc_payoff.mean()),
        std_error=float(disc_payoff.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
        model="Heston",
    )


# ---------------------------------------------------------------------------
# Path-dependent (Asian) prices -- these genuinely use the full T-step path
# ---------------------------------------------------------------------------


def price_cir_asian(ds: Dataset, n_paths: int = 1_000_000, seed: int = 0,
                    include_initial: bool = False) -> MCResult:
    """Reference price for an arithmetic-average (Asian) option on CIR.

    The payoff is ``max(mean_t V(t) - K, 0)`` over the ``T`` monitored points, so
    the whole fast-forwarded path matters (not just the terminal value).
    """
    if ds.cir is None:
        raise ValueError("Dataset has no CIR params.")
    rng = np.random.default_rng(seed)
    model = CIRModel(ds.cir, ds.contract.dt)
    paths = model.sample_paths(n_paths, ds.contract.n_steps, rng)
    payoff = asian_payoff(paths, ds.contract.strike, ds.contract.option_type,
                          include_initial=include_initial)
    disc_payoff = _discount(ds) * payoff
    return MCResult(
        price=float(disc_payoff.mean()),
        std_error=float(disc_payoff.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
        model="CIR-Asian",
    )


def price_heston_asian(ds: Dataset, n_paths: int = 200_000, seed: int = 0,
                       exact_integral: bool = False,
                       include_initial: bool = False) -> MCResult:
    """Reference price for an arithmetic-average (Asian) option on Heston.

    Uses the nested ``T``-step fast-forwarding path (variance + integrated
    variance + log-price increment), demonstrating that Heston is fast-forwarded
    for ``T != 1`` exactly as the paper's Section 6.2.1 scheme prescribes.
    """
    if ds.heston is None:
        raise ValueError("Dataset has no Heston params.")
    rng = np.random.default_rng(seed)
    model = HestonModel(ds.heston, ds.contract.dt)
    paths = model.sample_paths(n_paths, ds.contract.n_steps, rng,
                               exact_integral=exact_integral)["S"]
    payoff = asian_payoff(paths, ds.contract.strike, ds.contract.option_type,
                          include_initial=include_initial)
    disc_payoff = _discount(ds) * payoff
    return MCResult(
        price=float(disc_payoff.mean()),
        std_error=float(disc_payoff.std(ddof=1) / np.sqrt(n_paths)),
        n_paths=n_paths,
        model="Heston-Asian",
    )


__all__ = [
    "MCResult",
    "price_cir_european",
    "price_heston_european",
    "price_cir_asian",
    "price_heston_asian",
]
MARKER_LINE
