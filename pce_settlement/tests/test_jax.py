"""v2 backend tests (plan v2 §11): the JAX/PennyLane stack must agree with the
NumPy core, the Adam trainer must solve the hero gridlock, and shot-based
hardware decoding must converge to the analytic expectations. Skipped if
jax/pennylane are not installed."""
import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("jax")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import pennylane as qml  # noqa: E402

from pce_settlement import ansatz, baseline, pauli, pce_solve  # noqa: E402
from pce_settlement import pce_jax, hardware  # noqa: E402
from pce_settlement.config import Config  # noqa: E402
from pce_settlement.hea import build_hea  # noqa: E402
from pce_settlement.instance import gridlock_cycle  # noqa: E402


def test_pennylane_expectations_match_numpy_core():
    # Identical HEA + Pauli strings => identical expectations across backends.
    n, k = 5, 3
    m = 3 * pauli.comb(n, k)
    p = pauli.layers_for(m, n)
    strings = pauli.enumerate_pauli_strings(n, k, m)
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, (p + 1, n))

    exps_np = pauli.expectations(ansatz.run_hea(theta, n, p, "linear"), strings, n)

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="jax")
    def circ(th):
        build_hea(th, n, p, "linear")
        return [qml.expval(pce_jax.pauli_pennylane_op(s, n)) for s in strings]

    exps_pl = np.asarray(jnp.stack(circ(jnp.asarray(theta))))
    assert np.allclose(exps_np, exps_pl, atol=1e-9)


def test_jax_backend_solves_gridlock():
    # Adam-trained PCE on the hero gridlock should fully settle the cycle.
    inst = gridlock_cycle(3, amount=10)
    cfg = Config(backend="jax", seed=0, n_restarts=3, n_steps=300, lr=0.05)
    r = pce_solve.solve_settlement(inst, cfg)
    _, opt = baseline.brute_force(inst)
    assert r.feasible
    assert r.value >= 0.95 * opt


def test_hardware_decoding_converges_to_analytic():
    # Shot-based 3-basis decoding -> analytic expectations as shots grow.
    n, k = 4, 3
    m = 3 * pauli.comb(n, k)
    p = pauli.layers_for(m, n)
    strings = pauli.enumerate_pauli_strings(n, k, m)
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.uniform(0, 2 * np.pi, (p + 1, n)))

    exact = pauli.expectations(ansatz.run_hea(np.asarray(theta), n, p, "linear"),
                               strings, n)
    est = hardware.hardware_expectations(theta, "sim", n, p, "linear", strings,
                                         n_shots=20000)
    assert np.max(np.abs(exact - est)) < 0.05
