"""qff_pricing
================

A Qiskit implementation of **Quantum Fast-Forwarding (QFF)** for European
derivative pricing over the Cox--Ingersoll--Ross (CIR) and Heston models.

The package follows the schemes in:

  * Section 6.1.1 "Quantum Fast-forwarding Scheme" (CIR), Appendix C, and
  * Section 6.2.1 "Quantum Fast-forwarding Scheme" (Heston), Appendix D

of the accompanying paper (``main.tex``).  The headline result is that the
state-preparation overhead of amplitude-encoding the discretised price is only
``poly(T, log(1/eps))`` -- i.e. *polylogarithmic* in the inverse error --
instead of the ``exp(T * n)`` cost of a naive amplitude encoding of the full
``T``-step path distribution.

Top-level entry points
----------------------
    from qff_pricing.data import make_synthetic_dataset
    from qff_pricing.quantum.pricer import CIRPricer, HestonPricer

See ``examples/demo.py`` and the README for an end-to-end walk-through.
"""

from qff_pricing.data import (
    CIRParams,
    HestonParams,
    ContractSpec,
    MarketData,
    Dataset,
    make_synthetic_dataset,
)

__all__ = [
    "CIRParams",
    "HestonParams",
    "ContractSpec",
    "MarketData",
    "Dataset",
    "make_synthetic_dataset",
]

__version__ = "0.1.0"
