"""JAX + PennyLane PCE training (plan v2 §6).

Stage 1 of the v2 pipeline: train PCE parameters on a local JAX statevector via
Adam (optax), backprop through default.qubit. The loss is identical to the
NumPy core (pce_solve.pce_loss) -- only the optimizer (Adam vs COBYLA) and the
simulator backend (PennyLane vs hand-rolled) differ, so parameters and readouts
are interchangeable. Pre-trained theta* are later loaded onto hardware
(hardware.py) for a single 3-basis forward pass.

This module is import-safe only where jax/pennylane are installed; the numpy
core does not import it.
"""
from __future__ import annotations

from functools import reduce

import jax
jax.config.update("jax_enable_x64", True)  # match the NumPy core's float64 precision
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

from . import pauli
from .config import Config
from .device import get_device
from .hea import build_hea
from .pce_solve import SolveResult, bit_swap
from .qubo import ising_energy

_PAULI_OP = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def pauli_pennylane_op(string, n: int):
    """Convert a PCE string (frozenset, letter) to a PennyLane observable."""
    subset, letter = string
    op = _PAULI_OP[letter]
    ops = [op(i) for i in sorted(subset)]
    return reduce(lambda a, b: a @ b, ops)


def make_circuit(dev, n: int, p: int, topology: str, strings):
    """Jit-able QNode returning all m Pauli expectations (plan v2 §6)."""
    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(theta):
        build_hea(theta, n, p, topology)
        return [qml.expval(pauli_pennylane_op(s, n)) for s in strings]
    return circuit


def _loss(theta, circuit, J, h, alpha, beta, nu):
    exps = jnp.stack(circuit(theta))
    t = jnp.tanh(alpha * exps)
    quad = t @ jnp.triu(J, k=1) @ t            # matches NumPy core (strict upper)
    field = h @ t
    reg = beta * nu * (jnp.mean(t ** 2) ** 2)
    return quad + field + reg


def expectations(theta, J, h, k, cfg, topology=None):
    """PennyLane expectations <Pi_i> for given theta (for cross-checking v1)."""
    m = J.shape[0]
    n = pauli.qubits_for(m, k)
    p = pauli.layers_for(m, n)
    topology = topology or cfg.topology
    strings = pauli.enumerate_pauli_strings(n, k, m)
    circuit = make_circuit(get_device("sim", n), n, p, topology, strings)
    return np.asarray(jnp.stack(circuit(jnp.asarray(theta))))


def solve(J: np.ndarray, h: np.ndarray, k: int, cfg: Config,
          const: float = 0.0, n_override: int | None = None,
          p_override: int | None = None) -> SolveResult:
    """Solve an Ising problem with PCE trained by Adam in JAX (plan v2 §6).

    Drop-in for pce_solve.solve: same SolveResult, same z in {-1,+1}^m, so the
    rest of the pipeline (repair, compare) is backend-agnostic.

    n_override / p_override let you deviate from the canonical sizing
    (n=qubits_for(m,k), p=floor(m/n)) -- needed for hardware, where canonical p
    is far too deep for big m. Set e.g. n=12, p=5 for a wide, shallow circuit.
    """
    m = J.shape[0]
    n = n_override or pauli.qubits_for(m, k)
    p = p_override or pauli.layers_for(m, n)
    if m > 3 * pauli.comb(n, k):
        raise ValueError(f"n={n},k={k} encodes <= {3*pauli.comb(n,k)} vars, need {m}")
    strings = pauli.enumerate_pauli_strings(n, k, m)
    alpha = pauli.alpha_for(n, k)
    nu = float(np.abs(np.triu(J, k=1)).sum() + np.abs(h).sum())

    Jj, hj = jnp.asarray(J), jnp.asarray(h)
    circuit = make_circuit(get_device("sim", n), n, p, cfg.topology, strings)

    @jax.jit
    def value_and_grad(theta):
        return jax.value_and_grad(_loss)(theta, circuit, Jj, hj, alpha, cfg.beta, nu)

    optimizer = optax.adam(cfg.lr)
    key = jax.random.PRNGKey(cfg.seed)

    best_theta, best_loss = None, np.inf
    history: list[float] = []
    for _ in range(cfg.n_restarts):
        key, subkey = jax.random.split(key)
        theta = jax.random.uniform(subkey, (p + 1, n), minval=0.0, maxval=2 * np.pi)
        opt_state = optimizer.init(theta)

        @jax.jit
        def step(theta, opt_state):
            loss, g = value_and_grad(theta)
            updates, opt_state = optimizer.update(g, opt_state)
            return optax.apply_updates(theta, updates), opt_state, loss

        last = np.inf
        for _ in range(cfg.n_steps):
            theta, opt_state, loss = step(theta, opt_state)
            last = float(loss)
            history.append(last)
        if last < best_loss:
            best_loss, best_theta = last, theta

    exps = np.asarray(jnp.stack(circuit(best_theta)))
    z_raw = np.sign(exps)
    z_raw[z_raw == 0] = 1.0
    z_raw = z_raw.astype(int)
    z = bit_swap(z_raw, J, h, const) if cfg.bit_swap else z_raw.copy()
    energy = ising_energy(z, J, h, const)

    return SolveResult(z=z, z_raw=z_raw, energy=energy, loss=float(best_loss),
                       n=n, p=p, m=m, k=k, loss_history=history,
                       theta=np.asarray(best_theta).reshape(p + 1, n))
