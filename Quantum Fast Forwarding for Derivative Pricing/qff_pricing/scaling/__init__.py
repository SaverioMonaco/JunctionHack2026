"""qff_pricing.scaling
=======================

Resource / scaling analysis.

  * :mod:`resources` -- closed-form gate & qubit counts from the paper's
    theorems (CIR Thm "cir_qsample_resources" + Thm "sqrt_discretization";
    Heston Thm "heston_loading_resource" + Thm "hest_discretization"), the naive
    exponential amplitude-encoding baseline, and helpers to *measure* the actual
    Qiskit circuits.
  * :mod:`plots` -- matplotlib figures contrasting fast-forwarding (polynomial /
    polylogarithmic) with the naive exponential baseline.
"""

from qff_pricing.scaling import resources
from qff_pricing.scaling import advantage

__all__ = ["resources", "advantage"]
