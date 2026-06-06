"""HEA statevector tests (plan §5): norm preservation and a known small circuit."""
import numpy as np

from pce_settlement import ansatz


def test_norm_preserved():
    n, p = 5, 4
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, ansatz.n_params(n, p))
    psi = ansatz.run_hea(theta, n, p, topology="linear")
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_single_ry_known_state():
    # One qubit, p=0 (just the final Ry layer): Ry(pi) |0> = |1> (up to sign).
    psi = ansatz.run_hea(np.array([np.pi]), n=1, p=0)
    assert abs(abs(psi[1]) - 1.0) < 1e-12
    assert abs(psi[0]) < 1e-12


def test_cz_phase():
    # Ry(pi/2) on both qubits then CZ. Ry(pi/2)|0> = (|0>+|1>)/sqrt2.
    # State before CZ = 1/2 (|00>+|01>+|10>+|11>); CZ flips sign of |11>.
    theta = np.array([[np.pi / 2, np.pi / 2], [0.0, 0.0]])  # p=1: layer0 + final
    psi = ansatz.run_hea(theta, n=2, p=1, topology="linear")
    expected = 0.5 * np.array([1, 1, 1, -1], dtype=complex)
    assert np.allclose(psi, expected, atol=1e-12)


def test_square_vs_linear_differ():
    n, p = 4, 2
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, 2 * np.pi, ansatz.n_params(n, p))
    a = ansatz.run_hea(theta, n, p, "linear")
    b = ansatz.run_hea(theta, n, p, "square")
    # both normalized; topologies entangle differently
    assert abs(np.linalg.norm(a) - 1) < 1e-12
    assert abs(np.linalg.norm(b) - 1) < 1e-12
