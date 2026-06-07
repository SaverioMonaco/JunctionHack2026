r"""qff_pricing.quantum.pricer
==============================

End-to-end European-option pricers wiring together the full QFF / QMCI pipeline:

    dataset --> model (fast-forwarding) --> U_path --> U_f --> QAE --> price

Both pricers expose:

  * :meth:`price`     -- run the pipeline and return a :class:`PricingResult`.
  * :meth:`encoder`   -- the composed amplitude encoder ``A`` (a Qiskit circuit)
                          for inspection / resource counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit

from qff_pricing.data import Dataset
from qff_pricing.models.cir import CIRModel
from qff_pricing.models.heston import HestonModel
from qff_pricing.quantum.payoff import (
    european_payoff,
    payoff_oracle_exact,
    scaled_angles,
)
from qff_pricing.quantum.primitives import qsample_loader
from qff_pricing.quantum.qsvt_loader import build_qsvt_loader
from qff_pricing.quantum.qmci import (
    AmplitudeEstimate,
    build_amplitude_encoder,
    build_amplitude_encoder_qsvt,
    estimate_amplitude_exact,
    estimate_amplitude_exact_conditional,
    estimate_amplitude_mlae,
    estimate_amplitude_iqae,
)


@dataclass
class PricingResult:
    price: float                       # discounted fair price
    undiscounted: float                # E[payoff] under the (grid) measure
    amplitude: float                   # a = E[f]/M estimated by QAE
    payoff_scale: float                # M
    discount: float                    # exp(-r * maturity)
    n_qubits: int
    n_steps: int
    truncation: tuple[float, float]
    method: str
    estimate: AmplitudeEstimate
    model: str = ""
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.model} European {self.meta.get('option_type','?')} "
            f"price = {self.price:.6f}  (a={self.amplitude:.5f}, M={self.payoff_scale:.4f}, "
            f"n={self.n_qubits} qubits, T={self.n_steps} steps, method={self.method})"
        )


class _BasePricer:
    def __init__(self, dataset: Dataset, n_qubits: int = 4, n_sigma: float = 6.0,
                 loader: str = "stateprep", qsvt_degree: Optional[int] = None):
        dataset.validate()
        if loader not in ("stateprep", "qsvt"):
            raise ValueError("loader must be 'stateprep' or 'qsvt'.")
        self.dataset = dataset
        self.n_qubits = int(n_qubits)
        self.n_sigma = float(n_sigma)
        self.loader = loader
        self.qsvt_degree = qsvt_degree
        self._encoder: Optional[QuantumCircuit] = None
        self._objective: Optional[int] = None
        self._anc_qubits: list[int] = []
        self._qsvt_info = None
        self._scale: float = 1.0

    # subclasses build (centers, probs, truncation, underlying_label)
    def _grid(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _model_name(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def encoder(self) -> QuantumCircuit:
        """Build (and cache) the amplitude encoder circuit ``A = U_f . U_path``.

        With ``loader="stateprep"`` the qsample is loaded with Qiskit's
        ``StatePreparation`` (exact, ``Theta(N)`` gates).  With ``loader="qsvt"``
        it is loaded with the implemented polylog QET/QSVT loader (two ancillas,
        post-selected); the price estimator then conditions on the ancillas.
        """
        centers, probs, trunc = self._grid()
        c = self.dataset.contract
        payoff_vals = european_payoff(centers, c.strike, c.option_type)
        angles, scale = scaled_angles(payoff_vals)
        self._scale = scale
        self._trunc = trunc
        u_f = payoff_oracle_exact(self.n_qubits, angles)

        if self.loader == "stateprep":
            u_path = qsample_loader(probs)
            encoder, objective = build_amplitude_encoder(u_path, u_f)
            self._encoder = encoder
            self._objective = objective
            self._anc_qubits = []
        else:  # qsvt
            # financial terminal laws are skewed (non-central chi^2 / lognormal),
            # so default to a higher polynomial degree than the smooth-Gaussian
            # schedule; still linear in n (=> polylog in N).
            deg = self.qsvt_degree if self.qsvt_degree is not None else 4 * self.n_qubits + 4
            u_qsvt, info = build_qsvt_loader(probs, degree=deg)
            encoder, flag, ancs = build_amplitude_encoder_qsvt(
                u_qsvt, u_f, self.n_qubits)
            self._encoder = encoder
            self._objective = flag
            self._anc_qubits = ancs
            self._qsvt_info = info
        return self._encoder

    def price(self, method: str = "exact",
              epsilon_target: float = 0.01) -> PricingResult:
        """Run QMCI.  ``method`` is ``"exact"`` (statevector) or ``"iqae"``."""
        encoder = self.encoder()
        if self.loader == "qsvt":
            if method == "exact":
                # ancilla-conditioned statevector read-out (divides out F^2).
                est = estimate_amplitude_exact_conditional(
                    encoder, self._objective, self._anc_qubits)
            elif method in ("mlae", "iqae", "shots"):
                # FULL shot-based amplitude estimation through the loader, via a
                # Grover operator on the (flag=1 & ancillas=0) good subspace.
                est = estimate_amplitude_mlae(
                    encoder, self._objective, self._anc_qubits,
                    filling_sq=self._qsvt_info.filling_sq)
            else:
                raise ValueError("method must be 'exact' or 'mlae'/'iqae'.")
        elif method == "iqae":
            est = estimate_amplitude_iqae(encoder, self._objective,
                                          epsilon_target=epsilon_target)
        elif method == "exact":
            est = estimate_amplitude_exact(encoder, self._objective)
        else:
            raise ValueError("method must be 'exact' or 'iqae'.")

        undiscounted = self._scale * est.amplitude
        r = self.dataset.market.risk_free_rate
        discount = float(np.exp(-r * self.dataset.contract.maturity))
        price = discount * undiscounted
        return PricingResult(
            price=price,
            undiscounted=undiscounted,
            amplitude=est.amplitude,
            payoff_scale=self._scale,
            discount=discount,
            n_qubits=self.n_qubits,
            n_steps=self.dataset.contract.n_steps,
            truncation=self._trunc,
            method=est.method,
            estimate=est,
            model=self._model_name(),
            meta={"option_type": self.dataset.contract.option_type},
        )


class CIRPricer(_BasePricer):
    """Price a European option written on a CIR underlying (e.g. variance)."""

    def _model_name(self) -> str:
        return "CIR"

    def _grid(self):
        model = CIRModel(self.dataset.cir, self.dataset.contract.dt)
        return model.terminal_grid_distribution(
            self.n_qubits, self.dataset.contract.n_steps, n_sigma=self.n_sigma
        )


class HestonPricer(_BasePricer):
    """Price a European option on a Heston asset price S(T)."""

    def __init__(self, dataset: Dataset, n_qubits: int = 4, n_sigma: float = 6.0,
                 n_paths: int = 200_000, exact_integral: bool = False, seed: int = 1,
                 loader: str = "stateprep", qsvt_degree: Optional[int] = None):
        super().__init__(dataset, n_qubits=n_qubits, n_sigma=n_sigma,
                         loader=loader, qsvt_degree=qsvt_degree)
        self.n_paths = int(n_paths)
        self.exact_integral = bool(exact_integral)
        self.seed = int(seed)

    def _model_name(self) -> str:
        return "Heston"

    def _grid(self):
        model = HestonModel(self.dataset.heston, self.dataset.contract.dt)
        return model.terminal_grid_distribution(
            self.n_qubits,
            self.dataset.contract.n_steps,
            n_paths=self.n_paths,
            n_sigma=self.n_sigma,
            seed=self.seed,
            exact_integral=self.exact_integral,
        )


__all__ = ["CIRPricer", "HestonPricer", "PricingResult"]
