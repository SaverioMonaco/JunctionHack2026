"""qff_pricing.data.providers
==============================

Plug points for **real** data.

The rest of the package only ever consumes a :class:`~qff_pricing.data.Dataset`.
To switch from the synthetic generator to real market/calibration data you only
need to produce a ``Dataset`` here -- the models, quantum circuits and scaling
analysis are completely agnostic to where the numbers came from.

Two patterns are shown:

  * :class:`DataProvider` -- a tiny Protocol describing the one method a
    provider must implement (``get_dataset``).
  * :func:`from_csv` -- a worked example that reads model parameters and a
    contract from a CSV file (e.g. exported from a calibration pipeline or a
    market-data vendor).
"""

from __future__ import annotations

import csv
from typing import Protocol, runtime_checkable

from qff_pricing.data import (
    CIRParams,
    ContractSpec,
    Dataset,
    HestonParams,
    MarketData,
)


@runtime_checkable
class DataProvider(Protocol):
    """Anything with a ``get_dataset`` method can feed the pricers."""

    def get_dataset(self) -> Dataset:  # pragma: no cover - interface only
        ...


def from_csv(path: str, model: str = "heston") -> Dataset:
    """Build a :class:`Dataset` from a flat ``key,value`` CSV file.

    Expected keys (Heston example)::

        model,heston
        name,my-real-data
        spot,100.0
        risk_free_rate,0.03
        mu,0.03
        kappa,2.0
        theta,0.04
        sigma,0.10
        rho,-0.5
        v0,0.04
        s0,100.0
        option_type,call
        strike,100.0
        maturity,1.0
        n_steps,4

    For a CIR dataset use ``model,cir`` and omit the price-process rows
    (``mu``, ``rho``, ``s0``); ``strike`` is then a strike on the variance.
    """
    kv: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            kv[row[0].strip()] = row[1].strip()

    model = kv.get("model", model).lower()
    market = MarketData(
        spot=float(kv.get("spot", 100.0)),
        risk_free_rate=float(kv.get("risk_free_rate", 0.03)),
    )
    contract = ContractSpec(
        option_type=kv.get("option_type", "call"),
        strike=float(kv.get("strike", 100.0)),
        maturity=float(kv.get("maturity", 1.0)),
        n_steps=int(kv.get("n_steps", 4)),
    )

    if model == "cir":
        cir = CIRParams(
            kappa=float(kv["kappa"]),
            theta=float(kv["theta"]),
            sigma=float(kv["sigma"]),
            v0=float(kv["v0"]),
        )
        return Dataset(market=market, contract=contract, cir=cir,
                       name=kv.get("name", "csv-cir")).validate()

    if model == "heston":
        heston = HestonParams(
            mu=float(kv.get("mu", market.risk_free_rate)),
            kappa=float(kv["kappa"]),
            theta=float(kv["theta"]),
            sigma=float(kv["sigma"]),
            rho=float(kv["rho"]),
            v0=float(kv["v0"]),
            s0=float(kv.get("s0", market.spot)),
        )
        return Dataset(market=market, contract=contract, heston=heston,
                       name=kv.get("name", "csv-heston")).validate()

    raise ValueError("CSV 'model' field must be 'cir' or 'heston'.")


__all__ = ["DataProvider", "from_csv"]
