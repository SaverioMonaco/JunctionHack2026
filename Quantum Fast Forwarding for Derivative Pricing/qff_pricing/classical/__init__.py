"""qff_pricing.classical
=========================

Classical fast-forwarding Monte-Carlo reference prices, used to validate the
quantum pipeline.  Same models, same exact transitions -- just sampled instead
of amplitude-encoded.
"""

from qff_pricing.classical.monte_carlo import (
    MCResult,
    price_cir_european,
    price_heston_european,
    price_cir_asian,
    price_heston_asian,
)

__all__ = [
    "MCResult",
    "price_cir_european",
    "price_heston_european",
    "price_cir_asian",
    "price_heston_asian",
]
