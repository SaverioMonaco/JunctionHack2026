r"""qff_pricing.models.cir
==========================

The Cox--Ingersoll--Ross (CIR) process and its **fast-forwarding scheme**
(paper Section 6.1.1, Appendix C).

CIR SDE
-------
    dV(t) = kappa (theta - V(t)) dt + sigma sqrt(V(t)) dW(t)

Exact one-step transition (Glasserman 2004; paper Sec. 6.1.1)
------------------------------------------------------------
Conditioned on V(t), the next value is a *scaled non-central chi-square*:

    V(t+Delta) | V(t) =_d  c ( Y + (sqrt(beta V(t)) + Z)^2 ),
        Y ~ chi^2_{eta-1},   Z ~ N(0, 1),

with

    beta  = 4 kappa e^{-kappa Delta} / (sigma^2 (1 - e^{-kappa Delta}))
    eta   = 4 theta kappa / sigma^2
    c     = sigma^2 (1 - e^{-kappa Delta}) / (4 kappa)
    gamma = sqrt(c beta) = e^{-kappa Delta / 2}

Equivalently  V(t+Delta) | V(t) = c * NCX2(df = eta, nc = beta V(t)),
where NCX2(df, nc) is a non-central chi-square.  This is the *primitive*
distribution pair (Gaussian Z, central chi-square Y) that the quantum scheme
loads, together with the deterministic transition function

    g((y, z), v) = c ( y + (sqrt(beta v) + z)^2 ).               (Eq. 6.1)

Because CIR is an affine process, the **terminal marginal** V(T*Delta) | V(0)
is itself a scaled non-central chi-square over the total horizon, which we use
to build an accurate quantum loader for European (terminal-only) payoffs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from qff_pricing.data import CIRParams


@dataclass
class CIRStepConstants:
    """The Delta-dependent constants of the CIR one-step transition."""

    beta: float
    eta: float
    c: float
    gamma: float

    @classmethod
    def from_params(cls, p: CIRParams, dt: float) -> "CIRStepConstants":
        kappa, sigma = p.kappa, p.sigma
        e = np.exp(-kappa * dt)
        beta = 4.0 * kappa * e / (sigma ** 2 * (1.0 - e))
        eta = 4.0 * p.theta * kappa / (sigma ** 2)
        c = sigma ** 2 * (1.0 - e) / (4.0 * kappa)
        gamma = np.exp(-kappa * dt / 2.0)
        return cls(beta=beta, eta=eta, c=c, gamma=gamma)


class CIRModel:
    """Fast-forwardable CIR process.

    Parameters
    ----------
    params : CIRParams
        Process parameters (kappa, theta, sigma, v0).
    dt : float
        Time increment Delta between monitoring points.
    """

    def __init__(self, params: CIRParams, dt: float):
        params.validate()
        self.params = params
        self.dt = float(dt)
        self.k = CIRStepConstants.from_params(params, dt)

    # ------------------------------------------------------------------
    # Transition function  g((y, z), v)   --  Eq. 6.1 in the paper
    # ------------------------------------------------------------------
    def transition(self, y: np.ndarray, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""g((y, z), v) = c ( y + (sqrt(beta v) + z)^2 ).

        ``y`` is a draw of the central chi^2_{eta-1} primitive, ``z`` a draw of
        the standard-Gaussian primitive, ``v`` the current process value.
        Works element-wise on NumPy arrays.
        """
        return self.k.c * (y + (np.sqrt(self.k.beta * v) + z) ** 2)

    # ------------------------------------------------------------------
    # Exact one-step sampler (used by the classical reference + Heston)
    # ------------------------------------------------------------------
    def sample_step(self, v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw V(t+Delta) | V(t)=v exactly via scaled non-central chi-square."""
        v = np.asarray(v, dtype=float)
        nc = self.k.beta * v
        return self.k.c * stats.ncx2.rvs(df=self.k.eta, nc=nc, random_state=rng)

    def sample_paths(self, n_paths: int, n_steps: int,
                     rng: np.random.Generator) -> np.ndarray:
        """Exact CIR paths.  Returns array of shape (n_paths, n_steps+1)."""
        v = np.full(n_paths, self.params.v0, dtype=float)
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = v
        for t in range(1, n_steps + 1):
            v = self.sample_step(v, rng)
            out[:, t] = v
        return out

    # ------------------------------------------------------------------
    # Terminal marginal V(T*Delta) | V(0)  -- exact scaled non-central chi^2
    # ------------------------------------------------------------------
    def terminal_constants(self, n_steps: int) -> CIRStepConstants:
        """Transition constants for the *total* horizon T*Delta (one big step)."""
        return CIRStepConstants.from_params(self.params, self.dt * n_steps)

    def terminal_law(self, n_steps: int):
        """Return a frozen ``scipy.stats`` distribution for V(T) | V(0).

        V(T) = c_T * NCX2(df=eta, nc=beta_T v0), realised as a *scaled* ncx2.
        """
        kT = self.terminal_constants(n_steps)
        nc = kT.beta * self.params.v0
        return stats.ncx2(df=kT.eta, nc=nc, scale=kT.c)

    def terminal_mean_std(self, n_steps: int) -> tuple[float, float]:
        d = self.terminal_law(n_steps)
        return float(d.mean()), float(d.std())

    # ------------------------------------------------------------------
    # Discretised grid distribution  --  input to the quantum loader U_path
    # ------------------------------------------------------------------
    def terminal_grid_distribution(
        self,
        n_qubits: int,
        n_steps: int,
        n_sigma: float = 6.0,
        lo: float | None = None,
        hi: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        r"""Discretise the terminal law onto ``N = 2**n_qubits`` grid cells.

        Returns ``(centers, probs, (lo, hi))`` where ``probs[k]`` is the
        (renormalised) probability mass in cell ``k`` -- this is exactly the
        quantity whose square-root is amplitude-encoded in the discrete qsample
        (paper Definition "Discrete Qsample").

        The truncation interval ``[lo, hi]`` defaults to ``mean +/- n_sigma std``
        clipped at 0 (CIR is non-negative).  The number of cells needed to keep
        the discretisation error below ``eps`` scales only **polynomially** in
        ``T`` and ``log(1/eps)`` -- see :mod:`qff_pricing.scaling`.
        """
        N = 2 ** n_qubits
        law = self.terminal_law(n_steps)
        mean, std = law.mean(), law.std()
        if lo is None:
            lo = max(0.0, mean - n_sigma * std)
        if hi is None:
            hi = mean + n_sigma * std
        edges = np.linspace(lo, hi, N + 1)
        cdf = law.cdf(edges)
        probs = np.diff(cdf)
        total = probs.sum()
        if total <= 0:
            raise RuntimeError("Degenerate terminal distribution; check params.")
        probs = probs / total
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, probs, (float(lo), float(hi))

    # convenience -------------------------------------------------------
    def underlying_name(self) -> str:
        return "V"  # the CIR European option is written on the variance level
