"""Hardware deployment: 3-basis measurement + shot decoding (plan v2 §7).

Training is done; theta_star is fixed. Run exactly 3 circuits on the device
(one per Pauli basis X/Y/Z, since every PCE string is homogeneous), then decode
all m expectations from the shot bitstrings as Z-parity moments. This mirrors
the seminal PCE hardware workflow on IonQ/Quantinuum.

Set diff_method=None on the hardware QNode -- it's a forward pass only.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .device import get_device
from .hea import build_hea


def hardware_expectations(theta_star, provider: str, n: int, p: int, topology: str,
                          strings, n_shots: int):
    """Estimate all Pauli expectations from 3 basis-measurement circuits.

    Returns an (m,) array of <Pi_i>. Use provider='sv1' first (free-ish cloud
    sim, no queue) to confirm the decoding matches the JAX sim before burning
    QPU shots (plan v2 §10, §12).
    """
    dev = get_device(provider, n, shots=n_shots)

    def make_meas_circuit(basis):
        @qml.qnode(dev, diff_method=None)
        def circ():
            build_hea(theta_star, n, p, topology)
            for i in range(n):                 # rotate to measurement basis
                if basis == "X":
                    qml.Hadamard(wires=i)
                elif basis == "Y":
                    qml.adjoint(qml.S)(wires=i)
                    qml.Hadamard(wires=i)
                # Z: no rotation
            return qml.sample(wires=list(range(n)))
        return circ

    samples = {b: np.asarray(make_meas_circuit(b)()) for b in ("X", "Y", "Z")}
    return decode_expectations(samples, strings, n)


def decode_expectations(samples: dict, strings, n: int) -> np.ndarray:
    """<Pi> = mean over shots of (-1)^parity(bits on the string's qubits).

    samples: {'X': (n_shots, n) 0/1 array, 'Y': ..., 'Z': ...}.
    """
    exps = np.empty(len(strings), dtype=float)
    for i, (subset, letter) in enumerate(strings):
        bits = samples[letter]
        mask = np.array([1 if q in subset else 0 for q in range(n)])
        parities = np.mod((bits * mask).sum(axis=1), 2)
        exps[i] = float(np.mean((-1.0) ** parities))
    return exps
