"""Pauli machinery tests (plan §5): 3-basis expectations vs direct-apply oracle,
Y-rotation sign on |+i>, and the qubit/layer sizing recap."""
import numpy as np

from pce_settlement import pauli


def _random_state(n, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return psi / np.linalg.norm(psi)


def test_three_basis_matches_oracle():
    n, k = 5, 3
    psi = _random_state(n, seed=1)
    strings = pauli.enumerate_pauli_strings(n, k, 3 * pauli.comb(n, k))
    exps = pauli.expectations(psi, strings, n)
    for i, s in enumerate(strings):
        assert abs(exps[i] - pauli.expectation_oracle(psi, s, n)) < 1e-10


def test_y_rotation_sign_on_plus_i():
    # |+i> = (|0> + i|1>)/sqrt(2) has <Y> = +1.
    psi = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    string = (frozenset({0}), "Y")
    exp = pauli.expectations(psi, [string], 1)[0]
    assert abs(exp - 1.0) < 1e-12


def test_sizing_recap():
    # (m, n, p) reference values for k=3 from the plan (§1.5 / portfolio §1).
    expected = {10: (4, 2), 20: (5, 4), 30: (5, 6), 50: (6, 8),
                100: (7, 14), 150: (8, 18), 200: (9, 22), 250: (9, 27)}
    for m, (n_exp, p_exp) in expected.items():
        n = pauli.qubits_for(m, 3)
        p = pauli.layers_for(m, n)
        assert n == n_exp, f"m={m}: n={n} != {n_exp}"
        assert p == p_exp, f"m={m}: p={p} != {p_exp}"


def test_alpha():
    assert pauli.alpha_for(7, 3) == 7.0   # n^floor(3/2) = n
    assert pauli.alpha_for(7, 2) == 7.0   # n^floor(2/2) = n
