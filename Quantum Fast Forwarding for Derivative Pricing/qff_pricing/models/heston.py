r"""qff_pricing.models.heston
=============================

The (single-asset) Heston stochastic-volatility model and its
**fast-forwarding scheme** (paper Section 6.2.1, Appendix D).

Heston SDE
----------
    dS(t) = mu S dt + sqrt(V(t)) S dW^S
    dV(t) = kappa (theta - V(t)) dt + sigma sqrt(V(t)) dW^V
    corr(dW^S, dW^V) = rho

Fast-forwarding (paper Sec. 6.2.1)
----------------------------------
The model is fast-forwardable *conditioned on being able to sample*
``I(t) = \int_t^{t+Delta} V(s) ds``.  The construction uses three ingredients:

  1. the **CIR variance** propagated exactly (see :mod:`qff_pricing.models.cir`);
  2. the **time-integral of the variance** ``I(t)``, whose conditional
     characteristic function (given the two variance endpoints) is known in
     closed form -- Broadie & Kaya (2006), paper Eq. (6, "int_cir_char_func").
     The quantum scheme loads the inverse-Fourier transform of this CF;
  3. an independent Gaussian ``Z_perp`` driving the price's idiosyncratic part.

The exact log-price increment (Broadie--Kaya / Andersen), which is the algebraic
content of the paper's ``U(t)`` increment Eq. (6.log-increment-heston) for a
single asset, is

    log S(t+Delta) = log S(t) + mu Delta - 0.5 I
                     + (rho/sigma) ( V(t+Delta) - V(t) - kappa theta Delta + kappa I )
                     + sqrt( (1 - rho^2) I ) * Z_perp ,   Z_perp ~ N(0,1).

This module provides both the *exact* integral sampler (Fourier inversion of the
CF -- faithful but slow) and a fast *trapezoidal* approximation used by default
when building Monte-Carlo histograms.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

from qff_pricing.data import HestonParams
from qff_pricing.models.cir import CIRModel


class HestonModel:
    """Fast-forwardable single-asset Heston model."""

    def __init__(self, params: HestonParams, dt: float):
        params.validate()
        self.params = params
        self.dt = float(dt)
        self.cir = CIRModel(params.cir(), dt)

    # ==================================================================
    # Conditional characteristic function of  I = int_t^{t+dt} V ds
    # given the variance endpoints  V_t, V_T  (Broadie & Kaya 2006).
    # This is the closed form the quantum scheme amplitude-encodes.
    # ==================================================================
    def integral_cir_char_func(self, a, v_t: float, v_T: float) -> complex:
        r"""Phi(a) = E[ exp(i a I) | V_t, V_{t+dt} ]  (complex).

        Uses the standard Broadie--Kaya expression with Bessel-I ratio.
        """
        kappa, sigma = self.params.kappa, self.params.sigma
        dt = self.dt
        d = self.cir.k.eta            # df = 4 theta kappa / sigma^2
        order = 0.5 * d - 1.0         # xi = eta/2 - 1  (Bessel order)

        gamma = np.sqrt(kappa ** 2 - 2.0 * sigma ** 2 * 1j * a)

        # Prefactor  gamma * sinh(kappa dt) / (kappa * sinh(gamma dt))  ... written
        # in the exp/sinh form of the paper, but using a numerically stable ratio.
        sinh_k = np.sinh(kappa * dt)
        sinh_g = np.sinh(gamma * dt)
        pref = gamma * sinh_k / (kappa * sinh_g)

        # exp term in (V_t + V_T)
        exp_term = np.exp(
            (v_t + v_T) / sigma ** 2
            * (kappa / np.tanh(kappa * dt) - gamma / np.tanh(gamma * dt))
        )

        # Bessel ratio
        arg_num = np.sqrt(v_t * v_T) / sigma ** 2 * 2.0 * gamma / sinh_g
        arg_den = np.sqrt(v_t * v_T) / sigma ** 2 * 2.0 * kappa / sinh_k
        # special.iv supports complex argument; guard tiny argument.
        bessel_ratio = special.iv(order, arg_num) / special.iv(order, arg_den)

        return pref * exp_term * bessel_ratio

    def _integral_cdf(self, x: float, v_t: float, v_T: float,
                      u_max: float = 200.0) -> float:
        """P(I <= x | V_t, V_T) via the Gil-Pelaez / Broadie-Kaya inversion."""
        if x <= 0:
            return 0.0

        def integrand(u):
            phi = self.integral_cir_char_func(u, v_t, v_T)
            return np.sin(u * x) / u * phi.real

        val, _ = integrate.quad(integrand, 1e-8, u_max, limit=200)
        return float(np.clip(2.0 / np.pi * val, 0.0, 1.0))

    def sample_integral_broadie_kaya(self, v_t: float, v_T: float,
                                     rng: np.random.Generator) -> float:
        """Exact draw of I = int V via numerical inversion of its CDF (slow)."""
        # crude location/scale for the search bracket
        mean_guess = 0.5 * (v_t + v_T) * self.dt
        hi = max(mean_guess * 10.0, mean_guess + 5.0 * mean_guess + 1e-6)
        target = rng.random()
        lo = 0.0
        # bisection on the monotone CDF
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self._integral_cdf(mid, v_t, v_T) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def sample_integral_trapezoid(v_t: np.ndarray, v_T: np.ndarray,
                                  dt: float) -> np.ndarray:
        """Fast approximation  I ~ dt (V_t + V_{t+dt}) / 2  (vectorised)."""
        return dt * 0.5 * (np.asarray(v_t) + np.asarray(v_T))

    # ==================================================================
    # Exact log-price increment (Broadie-Kaya / Andersen) -- Eq. (6.x)
    # ==================================================================
    def log_price_increment(self, v_t, v_T, integral_v, z_perp):
        p = self.params
        return (
            p.mu * self.dt
            - 0.5 * integral_v
            + (p.rho / p.sigma) * (v_T - v_t - p.kappa * p.theta * self.dt
                                   + p.kappa * integral_v)
            + np.sqrt(np.maximum((1.0 - p.rho ** 2) * integral_v, 0.0)) * z_perp
        )

    # ==================================================================
    # Path simulation
    # ==================================================================
    def sample_paths(self, n_paths: int, n_steps: int,
                     rng: np.random.Generator,
                     exact_integral: bool = False) -> dict:
        """Simulate Heston paths via exact V transitions + the log-S increment.

        Parameters
        ----------
        exact_integral : bool
            If True, sample ``int V`` with the (slow) Broadie--Kaya inversion;
            otherwise use the fast trapezoidal approximation (default).

        Returns
        -------
        dict with keys ``"S"`` (n_paths, n_steps+1) and ``"V"`` (same shape).
        """
        p = self.params
        V = np.full(n_paths, p.v0)
        logS = np.full(n_paths, np.log(p.s0))
        S_out = np.empty((n_paths, n_steps + 1))
        V_out = np.empty((n_paths, n_steps + 1))
        S_out[:, 0] = p.s0
        V_out[:, 0] = p.v0

        for t in range(1, n_steps + 1):
            V_next = self.cir.sample_step(V, rng)
            if exact_integral:
                integral_v = np.array([
                    self.sample_integral_broadie_kaya(float(a), float(b), rng)
                    for a, b in zip(V, V_next)
                ])
            else:
                integral_v = self.sample_integral_trapezoid(V, V_next, self.dt)
            z_perp = rng.standard_normal(n_paths)
            logS = logS + self.log_price_increment(V, V_next, integral_v, z_perp)
            V = V_next
            V_out[:, t] = V
            S_out[:, t] = np.exp(logS)

        return {"S": S_out, "V": V_out}

    # ==================================================================
    # Terminal grid distribution for the quantum loader (MC histogram)
    # ==================================================================
    def terminal_grid_distribution(
        self,
        n_qubits: int,
        n_steps: int,
        n_paths: int = 200_000,
        n_sigma: float = 6.0,
        seed: int = 1,
        exact_integral: bool = False,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """Discretise the terminal price law S(T) onto ``2**n_qubits`` cells.

        Heston has no closed-form terminal density, so we estimate it by Monte
        Carlo and bin it.  The resulting ``probs`` vector is what the quantum
        ``U_path`` amplitude-encodes (its square-root).  In the genuine quantum
        algorithm this marginal is produced *coherently* by the fast-forwarding
        circuit; here we approximate it classically so the demo is runnable.
        """
        N = 2 ** n_qubits
        rng = np.random.default_rng(seed)
        ST = self.sample_paths(n_paths, n_steps, rng,
                               exact_integral=exact_integral)["S"][:, -1]
        mean, std = float(ST.mean()), float(ST.std())
        if lo is None:
            lo = max(0.0, mean - n_sigma * std)
        if hi is None:
            hi = mean + n_sigma * std
        edges = np.linspace(lo, hi, N + 1)
        counts, _ = np.histogram(ST, bins=edges)
        probs = counts.astype(float)
        total = probs.sum()
        if total <= 0:
            raise RuntimeError("No samples landed in the truncation window.")
        probs /= total
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, probs, (float(lo), float(hi))

    def underlying_name(self) -> str:
        return "S"
