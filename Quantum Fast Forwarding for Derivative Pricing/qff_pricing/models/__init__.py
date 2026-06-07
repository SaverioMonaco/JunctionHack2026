"""qff_pricing.models
======================

Classical descriptions of the fast-forwardable processes.  These objects are
the *single source of truth* for the dynamics; both the classical Monte-Carlo
reference and the quantum circuits are built from them.

  * :class:`~qff_pricing.models.cir.CIRModel`
  * :class:`~qff_pricing.models.heston.HestonModel`
"""

from qff_pricing.models.cir import CIRModel
from qff_pricing.models.heston import HestonModel

__all__ = ["CIRModel", "HestonModel"]
