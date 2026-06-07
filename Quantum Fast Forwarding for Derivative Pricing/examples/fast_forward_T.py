r"""examples/fast_forward_T.py
=============================

Multi-step (``T != 1``) fast-forwarding for **path-dependent** (Asian) options,
plus the end-to-end quantum-advantage / barrier plots.

Run::

    python examples/fast_forward_T.py            # prices + validation
    python examples/fast_forward_T.py --plots    # also (re)generate the 3 figures
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qff_pricing.data import make_synthetic_dataset
from qff_pricing.classical import (price_cir_asian, price_cir_european,
                                   price_heston_asian, price_heston_european)
from qff_pricing.quantum.fast_forward import price_cir_asian_quantum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true", help="regenerate the 3 figures")
    ap.add_argument("--outdir", default="best_plots")
    args = ap.parse_args()

    print("=" * 72)
    print("Path-dependent (Asian) pricing over T-step fast-forwarded paths")
    print("=" * 72)

    # ---- CIR: European vs Asian over a genuine multi-step path ----
    ds = make_synthetic_dataset("cir", seed=0)
    ds.contract = replace(ds.contract, n_steps=12, strike=0.04)
    print("\nCIR (T = 12 monitoring points):")
    print("  ", price_cir_european(ds, n_paths=300_000))
    print("  ", price_cir_asian(ds, n_paths=300_000))

    # ---- Heston: nested fast-forwarding for T != 1 ----
    dh = make_synthetic_dataset("heston", seed=0)
    dh.contract = replace(dh.contract, n_steps=12, strike=100.0)
    print("\nHeston (T = 12, nested fast-forwarding: V + integral-V + log-S):")
    print("  ", price_heston_european(dh, n_paths=80_000))
    print("  ", price_heston_asian(dh, n_paths=80_000))

    # ---- the runnable quantum T-step circuit, validated exactly ----
    print("\nRunnable quantum fast-forwarding circuit (CIR Asian):")
    dq = make_synthetic_dataset("cir", seed=0)
    for T in (1, 2):
        dq.contract = replace(dq.contract, n_steps=T, strike=0.038)
        r = price_cir_asian_quantum(dq, n_inc=2, n_v=2, count_gates=True)
        print(f"  T={T}: {r.n_qubits} qubits, {r.transpiled_gates} gates  "
              f"quantum={r.price:.8f}  exact-grid={r.grid_price:.8f}  "
              f"|diff|={abs(r.price - r.grid_price):.1e}")

    if args.plots:
        from qff_pricing.scaling.advantage_plots import generate_all
        paths = generate_all(args.outdir, measured_T=(1, 2, 3, 4, 5))
        print("\nWrote figures:")
        for p in paths:
            print("  ", p)


if __name__ == "__main__":
    main()
