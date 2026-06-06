"""Device abstraction (plan v2 §4) — the provider-lock-in prevention.

One function. Everything in the v2 stack imports from here. Switching hardware
is a one-string change in config.py. PennyLane is imported lazily so the rest of
the package stays importable without it.
"""
from __future__ import annotations

# ARNs for all Braket devices — swap here, nowhere else (plan v2 §4).
ARNS = {
    "emerald": "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",
    "garnet":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    "ionq":    "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
    "sv1":     "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "tn1":     "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
    "dm1":     "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
}

# Native square-lattice IQM machines use the 'square' HEA; IonQ is all-to-all so
# any pattern is native ('linear'). The local JAX sim uses 'linear'.
PROVIDER_TOPOLOGY = {
    "sim": "linear", "sv1": "linear", "tn1": "linear", "dm1": "linear",
    "emerald": "square", "garnet": "square", "ionq": "linear",
}


def get_device(provider: str, n_wires: int, shots: int | None = None):
    """Return a PennyLane device for the requested backend (plan v2 §4).

    provider: 'sim'  -> local JAX statevector (analytic, the training device)
              'emerald'|'garnet'|'ionq'  -> Braket QPUs (shots required)
              'sv1'|'tn1'|'dm1'          -> Braket cloud simulators
    """
    import pennylane as qml

    if provider == "sim":
        # Analytic when shots is None (training); shot-based when shots is set,
        # so hardware.py's 3-basis decoding can be validated locally with no QPU.
        return qml.device("default.qubit", wires=n_wires, shots=shots)

    if provider not in ARNS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {['sim', *ARNS]}")

    if shots is None:
        raise ValueError(f"Braket device '{provider}' requires shots (set cfg.n_shots).")

    return qml.device("braket.aws.qubit", device_arn=ARNS[provider],
                      wires=n_wires, shots=shots)
