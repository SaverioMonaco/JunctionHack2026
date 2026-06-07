"""qff_pricing.quantum
=======================

The Qiskit implementation of the QFF / QMCI pipeline:

    U_path  (qsample loader, :mod:`primitives`)
       |->  U_f  (payoff oracle, :mod:`payoff`)
             |->  amplitude estimation  (:mod:`qmci`)
                    |->  price  (:mod:`pricer`)
"""

from qff_pricing.quantum.pricer import CIRPricer, HestonPricer, PricingResult
from qff_pricing.quantum.qsvt_loader import (
    build_qsvt_loader,
    loader_fidelity,
)
from qff_pricing.quantum.fast_forward import (
    build_cir_ff_path,
    add_asian_payoff,
    price_cir_asian_quantum,
    FFCircuit,
    FFPrice,
)

__all__ = [
    "CIRPricer",
    "HestonPricer",
    "PricingResult",
    "build_qsvt_loader",
    "loader_fidelity",
    "build_cir_ff_path",
    "add_asian_payoff",
    "price_cir_asian_quantum",
    "FFCircuit",
    "FFPrice",
]
