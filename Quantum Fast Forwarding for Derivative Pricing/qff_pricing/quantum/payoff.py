r"""qff_pricing.quantum.payoff
===============================

The payoff oracle ``U_f`` -- the second half of the QMCI amplitude encoder.

It acts as

    U_f |k> |0>  =  |k> ( sqrt(f~(x_k)) |0> + sqrt(1 - f~(x_k)) |1> ) ,

so that after ``U_path`` the probability of measuring the flag qubit in ``|0>``
(here we use ``|1>`` as the "good" subspace) equals ``sum_k p_k f~(x_k)``.  With
``f~ = f / M`` a rescaling into ``[0, 1]``, amplitude estimation of that
probability returns ``E[f] / M``.

Two constructions:

* :func:`payoff_oracle_exact` -- one multi-controlled ``RY`` per occupied grid
  cell, with the *exact* angle ``theta_k = 2 arcsin(sqrt(f~(x_k)))``.  ``O(N)``
  gates, unambiguous, used by the runnable pricer for correctness on small ``n``.

* :func:`affine_rotation` -- the ``O(n)`` "bank of singly-controlled rotations"
  that the paper's ``U_f`` uses (Sec. 3.1.3): it encodes an *affine* function of
  the integer index, which is exactly one linear piece of a piecewise-linear
  payoff.  Provided for the efficient construction / resource counting.

European payoffs supported: ``call`` -> max(x - K, 0), ``put`` -> max(K - x, 0).
Both are piecewise-linear and ``B``-Lipschitz with slope ``B = 1`` in ``x``,
matching the paper's assumptions.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# Classical payoff evaluation on the grid
# ---------------------------------------------------------------------------


def european_payoff(values: np.ndarray, strike: float,
                    option_type: str = "call") -> np.ndarray:
    """Vanilla European payoff evaluated at terminal underlying ``values``."""
    values = np.asarray(values, dtype=float)
    if option_type == "call":
        return np.maximum(values - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - values, 0.0)
    raise ValueError("option_type must be 'call' or 'put'.")


# ---------------------------------------------------------------------------
# Path-dependent payoffs (these are what make T > 1 meaningful)
# ---------------------------------------------------------------------------


def path_average(paths: np.ndarray, include_initial: bool = False) -> np.ndarray:
    r"""Arithmetic mean of the monitored values along each path.

    ``paths`` has shape ``(n_paths, n_steps + 1)`` (column 0 is the deterministic
    initial value).  By default we average over the ``T`` *monitored* points
    ``X(Delta), ..., X(T Delta)`` (columns ``1:``), the usual convention for a
    discretely-monitored arithmetic-average option.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    cols = paths if include_initial else paths[:, 1:]
    return cols.mean(axis=1)


def asian_payoff(paths: np.ndarray, strike: float, option_type: str = "call",
                 include_initial: bool = False) -> np.ndarray:
    r"""Arithmetic-average (Asian) option payoff over the whole path.

    payoff = max(mean(path) - K, 0)   (call)  /  max(K - mean(path), 0)  (put).

    Unlike :func:`european_payoff`, this depends on *every* monitored point, so
    it genuinely exercises the ``T``-step fast-forwarding construction rather than
    collapsing to the terminal marginal.
    """
    avg = path_average(paths, include_initial=include_initial)
    return european_payoff(avg, strike, option_type)


def scaled_angles(payoff_values: np.ndarray,
                  scale: float | None = None) -> tuple[np.ndarray, float]:
    r"""Return ``(angles, scale)`` with ``angles[k] = 2 arcsin(sqrt(f/scale))``.

    ``scale`` (= ``M``) defaults to the maximum payoff on the grid so that
    ``f~ in [0, 1]``.  ``E[f] = scale * P(flag = 1)``.
    """
    f = np.asarray(payoff_values, dtype=float)
    if scale is None:
        scale = float(f.max())
    if scale <= 0:
        # Degenerate: payoff is zero everywhere on the grid.
        return np.zeros_like(f), 1.0
    ftil = np.clip(f / scale, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(ftil)), scale


# ---------------------------------------------------------------------------
# Exact oracle (multi-controlled, O(N)) -- correctness-first
# ---------------------------------------------------------------------------


def payoff_oracle_exact(n_qubits: int, angles: np.ndarray,
                        label: str = "U_f") -> QuantumCircuit:
    r"""Diagonal payoff oracle on ``n_qubits`` index qubits + 1 flag qubit.

    Qubit ``n_qubits`` is the flag (the "objective" qubit).  Little-endian:
    qubit 0 is the least-significant index bit, matching ``qsample_loader``.
    """
    N = 2 ** n_qubits
    if len(angles) != N:
        raise ValueError("angles length must equal 2**n_qubits.")
    qc = QuantumCircuit(n_qubits + 1, name=label)
    flag = n_qubits
    controls = list(range(n_qubits))
    for k in range(N):
        theta = float(angles[k])
        if abs(theta) < 1e-12:
            continue
        # flip control qubits whose bit in k is 0, so the all-ones control fires.
        zero_bits = [i for i in range(n_qubits) if not (k >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        if n_qubits == 0:
            qc.ry(theta, flag)
        elif n_qubits == 1:
            qc.cry(theta, 0, flag)
        else:
            qc.mcry(theta, controls, flag)
        for i in zero_bits:
            qc.x(i)
    return qc


# ---------------------------------------------------------------------------
# Efficient affine rotation (O(n)) -- the paper's "bank of controlled R_Y"
# ---------------------------------------------------------------------------


def affine_rotation(n_qubits: int, slope: float, intercept: float,
                    label: str = "U_f_affine") -> QuantumCircuit:
    r"""Encode an affine angle ``theta(k) = intercept + slope * k`` on the flag.

    Implements ``RY(intercept)`` on the flag plus, for each index qubit ``i``, a
    controlled ``RY(slope * 2**i)``.  This is ``O(n)`` gates and reproduces one
    linear piece of a piecewise-linear payoff (Woerner-Egger / paper Sec. 3.1.3).
    The full piecewise-linear ``U_f`` is assembled by gating such pieces on a
    comparator against the kink index (see Qiskit's ``LinearAmplitudeFunction``).
    """
    qc = QuantumCircuit(n_qubits + 1, name=label)
    flag = n_qubits
    qc.ry(intercept, flag)
    for i in range(n_qubits):
        qc.cry(slope * (2 ** i), i, flag)
    return qc


__all__ = [
    "european_payoff",
    "path_average",
    "asian_payoff",
    "scaled_angles",
    "payoff_oracle_exact",
    "affine_rotation",
]
