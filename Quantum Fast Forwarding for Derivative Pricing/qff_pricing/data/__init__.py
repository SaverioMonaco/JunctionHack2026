"""qff_pricing.data
====================

The **data layer**.  Everything the pricing engine needs is described by a
small set of plain dataclasses:

  * :class:`CIRParams`     -- parameters of a Cox--Ingersoll--Ross process.
  * :class:`HestonParams`  -- parameters of a (single-asset) Heston model.
  * :class:`ContractSpec`  -- the European option contract (strike, maturity ...).
  * :class:`MarketData`    -- the observable market state (spot, risk-free rate).
  * :class:`Dataset`       -- a bundle tying the above together.

The default :func:`make_synthetic_dataset` returns a **made-up** dataset so the
package runs out of the box.  To use *real* data, implement the
:class:`DataProvider` protocol in :mod:`qff_pricing.data.providers` (a worked
CSV example is included) and hand its output to the pricers -- nothing else in
the package needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CIRParams:
    """Cox--Ingersoll--Ross process  dV = kappa (theta - V) dt + sigma sqrt(V) dW.

    Attributes
    ----------
    kappa : float
        Mean-reversion rate.
    theta : float
        Long-run mean level.
    sigma : float
        Volatility ("vol-of-vol" when used inside Heston).
    v0 : float
        Deterministic initial value V(0).
    """

    kappa: float
    theta: float
    sigma: float
    v0: float

    # --- derived "Feller" quantities (see paper Sec. 6.1.1) ----------------
    @property
    def feller_eta(self) -> float:
        r"""eta = 4 theta kappa / sigma^2  (degrees of freedom parameter)."""
        return 4.0 * self.theta * self.kappa / (self.sigma ** 2)

    @property
    def feller_gap(self) -> float:
        r"""xi = eta/2 - 1.  Feller condition is xi > 0; the paper assumes eta >= 5."""
        return self.feller_eta / 2.0 - 1.0

    def validate(self) -> None:
        if min(self.kappa, self.theta, self.sigma) <= 0:
            raise ValueError("kappa, theta, sigma must all be positive.")
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative.")
        if self.feller_eta < 5.0:
            # The paper assumes eta >= 5 to avoid singularities in chi^2_{eta-1}.
            import warnings

            warnings.warn(
                f"eta = {self.feller_eta:.3f} < 5; the QFF loading analysis "
                "(Sec. 6.1.1) assumes eta >= 5. Results may be less accurate.",
                stacklevel=2,
            )


@dataclass(frozen=True)
class HestonParams:
    """Single-asset Heston model (price S coupled to CIR variance V).

    dS = mu S dt + sqrt(V) S dW^S
    dV = kappa (theta - V) dt + sigma sqrt(V) dW^V
    corr(dW^S, dW^V) = rho dt
    """

    mu: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    s0: float

    def cir(self) -> CIRParams:
        """The embedded variance process as a :class:`CIRParams`."""
        return CIRParams(kappa=self.kappa, theta=self.theta, sigma=self.sigma, v0=self.v0)

    def validate(self) -> None:
        self.cir().validate()
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")
        if self.s0 <= 0:
            raise ValueError("s0 must be positive.")


# ---------------------------------------------------------------------------
# Contract + market
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractSpec:
    """A European option contract.

    Attributes
    ----------
    option_type : str
        ``"call"`` -> payoff max(S_T - K, 0); ``"put"`` -> max(K - S_T, 0).
    strike : float
        Strike price K.
    maturity : float
        Time to maturity in years (calendar time = T * delta).
    n_steps : int
        Number of monitoring points T in the fast-forwarded path process.
        For a *European* option the payoff only depends on the terminal point,
        but the path still has to be propagated through T steps.
    """

    option_type: str = "call"
    strike: float = 100.0
    maturity: float = 1.0
    n_steps: int = 4

    @property
    def dt(self) -> float:
        """Time increment Delta between monitoring points."""
        return self.maturity / self.n_steps

    def validate(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if self.strike <= 0:
            raise ValueError("strike must be positive.")
        if self.maturity <= 0 or self.n_steps < 1:
            raise ValueError("maturity must be > 0 and n_steps >= 1.")


@dataclass(frozen=True)
class MarketData:
    """Observable market state."""

    spot: float = 100.0
    risk_free_rate: float = 0.03


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """Everything needed to price one European option.

    Exactly one of ``cir`` / ``heston`` is populated depending on the model.
    """

    market: MarketData
    contract: ContractSpec
    cir: Optional[CIRParams] = None
    heston: Optional[HestonParams] = None
    name: str = "unnamed"
    meta: dict = field(default_factory=dict)

    def validate(self) -> "Dataset":
        self.contract.validate()
        if self.cir is None and self.heston is None:
            raise ValueError("Dataset must contain either CIR or Heston params.")
        if self.cir is not None:
            self.cir.validate()
        if self.heston is not None:
            self.heston.validate()
        return self


# ---------------------------------------------------------------------------
# Synthetic ("made-up") datasets
# ---------------------------------------------------------------------------


def make_synthetic_dataset(model: str = "cir", seed: int = 0) -> Dataset:
    """Return a self-consistent, **made-up** dataset for quick experiments.

    Parameters
    ----------
    model : {"cir", "heston"}
        Which underlying to build the dataset for.
    seed : int
        Reproducibility seed (parameters are jittered slightly).

    Notes
    -----
    The parameters are deliberately chosen to satisfy the QFF analysis
    assumptions (Feller parameter ``eta >= 5`` for CIR; for Heston the
    truncation conditions of Lemma D.1 are *not* strictly enforced here -- this
    is synthetic test data, not a calibrated market fit).
    """
    rng = np.random.default_rng(seed)
    jitter = lambda x, p=0.05: float(x * (1.0 + p * (rng.random() - 0.5)))

    market = MarketData(spot=100.0, risk_free_rate=0.03)

    if model.lower() == "cir":
        # eta = 4*theta*kappa/sigma^2 = 4*0.04*2.0/0.10^2 = 32 >> 5  (well-behaved).
        cir = CIRParams(
            kappa=jitter(2.0),
            theta=jitter(0.04),
            sigma=jitter(0.10),
            v0=0.04,
        )
        contract = ContractSpec(
            option_type="call",
            strike=0.04,        # strike on the *variance* level
            maturity=1.0,
            n_steps=1,          # European: payoff sees only the terminal -> T=1
        )
        return Dataset(market=market, contract=contract, cir=cir,
                       name=f"synthetic-cir-{seed}").validate()

    if model.lower() == "heston":
        heston = HestonParams(
            mu=market.risk_free_rate,
            kappa=jitter(2.0),
            theta=jitter(0.04),
            sigma=jitter(0.10),
            rho=jitter(-0.5, 0.02),
            v0=0.04,
            s0=market.spot,
        )
        contract = ContractSpec(
            option_type="call",
            strike=100.0,
            maturity=1.0,
            n_steps=1,          # European: payoff sees only the terminal -> T=1
        )
        return Dataset(market=market, contract=contract, heston=heston,
                       name=f"synthetic-heston-{seed}").validate()

    raise ValueError("model must be 'cir' or 'heston'.")


__all__ = [
    "CIRParams",
    "HestonParams",
    "ContractSpec",
    "MarketData",
    "Dataset",
    "make_synthetic_dataset",
]
