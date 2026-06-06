"""Configuration for the PCE settlement solver.

All hyperparameters live here. The penalty weight ``P`` is the #1 time sink
(see plan §1.3) -- it is exposed and swept by ``compare.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    # ---- Problem / encoding ------------------------------------------------
    k: int = 3                  # Pauli correlation order (k-body strings)
    P: float | None = None      # penalty weight; None => auto (10 * sum w_t)

    # ---- Greek-neutrality penalty (risk-aware settlement) -----------------
    # lambdas: per-Greek penalty weights (G,); None => qubo.default_lambdas when
    # the instance carries greeks, or no Greek term when it doesn't.
    # greek_targets: per-Greek target exposure (G,); None => zero (neutralize).
    lambdas: np.ndarray | None = None
    greek_targets: np.ndarray | None = None
    # risk_aversion scales the auto default_lambdas (only when lambdas is None).
    # 1.0 = neutrality weighted like the value objective; >1 = a risk-averse desk
    # that gives up settled value to flatten net Greeks. Ignored if lambdas set.
    risk_aversion: float = 1.0

    # ---- Backend -----------------------------------------------------------
    backend: str = "numpy"      # 'numpy' (COBYLA core) | 'jax' (PennyLane+Adam)

    # ---- Optimizer (COBYLA, numpy backend) --------------------------------
    n_restarts: int = 6         # random restarts; keep the best final loss
    maxiter: int = 2000         # COBYLA iterations per restart
    rhobeg: float = 0.5         # COBYLA initial step
    tol: float = 1e-6           # COBYLA tolerance

    # ---- Optimizer (Adam, jax backend; plan v2 §8) ------------------------
    lr: float = 0.05            # Adam learning rate
    n_steps: int = 300          # Adam steps per restart

    # ---- PCE loss ----------------------------------------------------------
    beta: float = 0.5           # regularizer weight (plan §1.5)

    # ---- Iterative-alpha PCE (arXiv:2602.17479v2 Algorithm 1) -------------
    # Fixed alpha can leave penalty-constrained variables un-binarized (stuck in
    # the continuous regime), so constraint satisfaction degrades with size.
    # Ramping alpha across rounds (warm-starting theta) forces binarization.
    # Default off => exact current behaviour. Turn on for dense Greek QUBOs.
    iterative_alpha: bool = False
    alpha_rounds: int = 4       # number of alpha-ramp rounds
    alpha_growth: float = 1.7   # alpha multiplier per round (alpha_0 = n^floor(k/2))

    # ---- Post-processing ---------------------------------------------------
    bit_swap: bool = True       # run one Ising local-search sweep after readout
    repair: bool = True         # classical feasibility repair before reporting

    # ---- Reproducibility ---------------------------------------------------
    seed: int = 0

    # ---- Hardware (v2, unused by the NumPy core) --------------------------
    provider: str = "sim"       # 'sim' | 'emerald' | 'garnet' | 'ionq' | ...
    n_shots: int = 1024
    topology: str = "linear"    # 'linear' (sim/IonQ) | 'square' (IQM)


DEFAULT = Config()
