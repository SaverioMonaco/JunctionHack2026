r"""qff_pricing.quantum.fast_forward
====================================

A **runnable, multi-step** (``T != 1``) quantum fast-forwarding loader ``U_path``
for the CIR process, following the paper's scheme (Sec. 6.1.1 / Appendix C):

    U_path = U_jump_T . U_incr_T . ... . U_jump_1 . U_incr_1 .

Per step ``t`` we

  * **U_incr_t**: load the one-dimensional primitive increments
    ``Z_t ~ N(0,1)`` and ``Y_t ~ chi^2_{eta-1}`` as discrete qsamples onto fresh
    registers (``qsample_loader``), and
  * **U_jump_t**: apply the *coherent transition* ``g((y,z), v) = c(y+(sqrt(beta v)+z)^2)``
    (paper Eq. 6.1 / ``cir_recursion``) which writes the next variance value
    ``V_t`` into a fresh register, conditioned on ``V_{t-1}`` and the increments.

Unlike a European (``T=1``) pricer, this keeps **every** monitored ``V_1..V_T``
so a *path-dependent* payoff (here arithmetic-average / Asian) can be evaluated
coherently.  This is the construction that genuinely exercises the ``T``-step
fast-forwarding, rather than collapsing to the terminal marginal.

Honesty note (matches the package's style).  The jump ``g`` here is realised as
an **exact coherent lookup** over the discretised grid (a multiplexed
permutation), which is correct to machine precision but costs ``O(N_v N_inc^2)``
gates -- the pedagogical, *non-scalable* realisation of ``U_jump``.  The
*scalable* ``U_jump`` is the fixed-point Newton-sqrt arithmetic whose gate count
is modelled (not built) in :mod:`qff_pricing.scaling.advantage`.  What is real
and verified here is the **end-to-end T-step pipeline and its exact agreement
with the classical grid expectation** -- i.e. the fast-forwarding *path
construction* is wired correctly for ``T != 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy import stats

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qff_pricing.models.cir import CIRModel
from qff_pricing.quantum.primitives import qsample_loader
from qff_pricing.quantum.payoff import european_payoff


# --------------------------------------------------------------------------- #
#  Grids for the primitives and the variance register
# --------------------------------------------------------------------------- #
def _grid_pmf(dist, lo, hi, n_bits):
    N = 2 ** n_bits
    edges = np.linspace(lo, hi, N + 1)
    pmf = np.diff(dist.cdf(edges))
    pmf = np.clip(pmf, 0.0, None)
    total = pmf.sum()
    if total <= 0:
        raise ValueError("primitive puts no mass in truncation window")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pmf / total


def cir_increment_grids(model: CIRModel, n_inc: int, a: float = 4.0,
                        chi2_lo: float = 1e-3, chi2_sigma: float = 6.0):
    """Discrete grids (centers, probs) for the Z (Gaussian) and Y (chi^2) primitives."""
    eta = model.k.eta
    z_c, z_p = _grid_pmf(stats.norm(), -a, a, n_inc)
    chi = stats.chi2(eta - 1.0)
    y_hi = float(chi.mean() + chi2_sigma * chi.std())
    y_c, y_p = _grid_pmf(chi, chi2_lo, y_hi, n_inc)
    return (y_c, y_p), (z_c, z_p)


def cir_v_grid(model: CIRModel, n_v: int, n_steps: int, n_sigma: float = 5.0):
    """Variance-register grid centres covering the path's support over all steps."""
    law = model.terminal_law(n_steps)
    mean, std = float(law.mean()), float(law.std())
    lo = 0.0
    hi = mean + n_sigma * std
    edges = np.linspace(lo, hi, 2 ** n_v + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, (lo, hi)


def _nearest(centers: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(centers - value)))


# --------------------------------------------------------------------------- #
#  Circuit construction
# --------------------------------------------------------------------------- #
@dataclass
class FFCircuit:
    circuit: QuantumCircuit
    v_regs: list                       # list of QuantumRegister, one per step
    flag: object                       # the flag QuantumRegister
    v_centers: np.ndarray
    grids: dict = field(default_factory=dict)
    n_inc: int = 0
    n_v: int = 0
    T: int = 0


def _set_index(qc, controls, pattern_bits, targets, index):
    """Controlled write of integer ``index`` into ``targets`` when ``controls`` match ``pattern_bits``."""
    flips = [c for c, b in zip(controls, pattern_bits) if b == 0]
    for c in flips:
        qc.x(c)
    for j, tq in enumerate(targets):
        if (index >> j) & 1:
            qc.mcx(list(controls), tq)
    for c in flips:
        qc.x(c)


def build_cir_ff_path(model: CIRModel, T: int, n_inc: int = 2, n_v: int = 2,
                      a: float = 4.0, n_sigma: float = 5.0) -> FFCircuit:
    """Build the coherent ``T``-step CIR fast-forwarding loader ``U_path`` (no payoff)."""
    (y_c, y_p), (z_c, z_p) = cir_increment_grids(model, n_inc, a=a)
    v_centers, _ = cir_v_grid(model, n_v, T, n_sigma=n_sigma)
    N_inc = 2 ** n_inc

    Y = [QuantumRegister(n_inc, f"Y{t}") for t in range(1, T + 1)]
    Z = [QuantumRegister(n_inc, f"Z{t}") for t in range(1, T + 1)]
    V = [QuantumRegister(n_v, f"V{t}") for t in range(1, T + 1)]
    flag = QuantumRegister(1, "flag")
    qc = QuantumCircuit(*Y, *Z, *V, flag, name=f"U_path_T{T}")

    y_loader = qsample_loader(y_p, label="U_Y")
    z_loader = qsample_loader(z_p, label="U_Z")

    v0 = model.params.v0
    for t in range(T):
        # ---- U_incr_t : load the primitives onto fresh registers ----
        qc.compose(y_loader, qubits=list(Y[t]), inplace=True)
        qc.compose(z_loader, qubits=list(Z[t]), inplace=True)
        # ---- U_jump_t : coherent transition g -> writes V_t ----
        ybits = [(iy, [(iy >> j) & 1 for j in range(n_inc)]) for iy in range(N_inc)]
        zbits = [(iz, [(iz >> j) & 1 for j in range(n_inc)]) for iz in range(N_inc)]
        if t == 0:
            controls = list(Y[t]) + list(Z[t])
            for iy, yb in ybits:
                for iz, zb in zbits:
                    vnew = model.transition(y_c[iy], z_c[iz], v0)
                    iv = _nearest(v_centers, vnew)
                    _set_index(qc, controls, yb + zb, list(V[t]), iv)
        else:
            controls = list(V[t - 1]) + list(Y[t]) + list(Z[t])
            for ivp in range(2 ** n_v):
                vpb = [(ivp >> j) & 1 for j in range(n_v)]
                for iy, yb in ybits:
                    for iz, zb in zbits:
                        vnew = model.transition(y_c[iy], z_c[iz], v_centers[ivp])
                        iv = _nearest(v_centers, vnew)
                        _set_index(qc, controls, vpb + yb + zb, list(V[t]), iv)

    return FFCircuit(circuit=qc, v_regs=V, flag=flag, v_centers=v_centers,
                     grids={"y": (y_c, y_p), "z": (z_c, z_p)},
                     n_inc=n_inc, n_v=n_v, T=T)


def add_asian_payoff(ff: FFCircuit, strike: float, option_type: str = "call",
                     include_initial: bool = False, scale: float | None = None):
    """Append the Asian (arithmetic-average) payoff oracle, returning the scale ``M``."""
    qc = ff.circuit
    V = ff.v_regs
    v_centers = ff.v_centers
    T = ff.T
    n_v = ff.n_v
    flag_q = ff.flag[0]

    combos = list(product(range(2 ** n_v), repeat=T))
    payoffs = np.array([
        european_payoff(np.mean([v_centers[i] for i in combo]), strike, option_type)
        for combo in combos
    ])
    M = float(payoffs.max()) if scale is None else float(scale)
    if M <= 0:
        return 0.0

    controls = [q for reg in V for q in reg]
    for combo, f in zip(combos, payoffs):
        if f <= 0:
            continue
        theta = 2.0 * np.arcsin(np.sqrt(min(f / M, 1.0)))
        pattern = []
        for i in combo:
            pattern += [(i >> j) & 1 for j in range(n_v)]
        flips = [c for c, b in zip(controls, pattern) if b == 0]
        for c in flips:
            qc.x(c)
        qc.mcry(theta, list(controls), flag_q)
        for c in flips:
            qc.x(c)
    return M


# --------------------------------------------------------------------------- #
#  Read-out + exact classical reference on the SAME discrete grid
# --------------------------------------------------------------------------- #
def flag_probability(ff: FFCircuit) -> float:
    sv = Statevector(ff.circuit)
    idx = ff.circuit.find_bit(ff.flag[0]).index
    return float(sv.probabilities([idx])[1])


def classical_grid_asian(model: CIRModel, ff: FFCircuit, strike: float,
                         option_type: str = "call") -> float:
    """Exact E[payoff] over the SAME discretised increment grid + rounding.

    This is the ground truth the quantum statevector read-out must reproduce
    (to machine precision): both propagate the discrete increments through the
    identical rounded transition ``g`` and average.
    """
    (y_c, y_p) = ff.grids["y"]
    (z_c, z_p) = ff.grids["z"]
    v_centers = ff.v_centers
    T = ff.T
    N_inc = len(y_c)
    v0 = model.params.v0

    total = 0.0
    # enumerate increments for all T steps
    for combo in product(range(N_inc), repeat=2 * T):
        prob = 1.0
        v = v0
        vals = []
        ok = True
        for t in range(T):
            iy = combo[2 * t]
            iz = combo[2 * t + 1]
            prob *= y_p[iy] * z_p[iz]
            if prob == 0.0:
                ok = False
                break
            vnew = model.transition(y_c[iy], z_c[iz], v)
            iv = _nearest(v_centers, vnew)
            v = v_centers[iv]          # snap to grid, exactly like the circuit
            vals.append(v)
        if not ok:
            continue
        avg = float(np.mean(vals))
        total += prob * float(european_payoff(avg, strike, option_type))
    return total


# --------------------------------------------------------------------------- #
#  End-to-end convenience pricer
# --------------------------------------------------------------------------- #
@dataclass
class FFPrice:
    price: float
    grid_price: float
    undiscounted: float
    flag_prob: float
    scale: float
    n_qubits: int
    T: int
    transpiled_gates: int | None = None


def price_cir_asian_quantum(ds, n_inc: int = 2, n_v: int = 2, a: float = 4.0,
                            n_sigma: float = 5.0, count_gates: bool = False) -> FFPrice:
    """Price an Asian option on CIR via the runnable T-step quantum FF circuit."""
    if ds.cir is None:
        raise ValueError("dataset has no CIR params")
    model = CIRModel(ds.cir, ds.contract.dt)
    T = ds.contract.n_steps
    ff = build_cir_ff_path(model, T, n_inc=n_inc, n_v=n_v, a=a, n_sigma=n_sigma)
    M = add_asian_payoff(ff, ds.contract.strike, ds.contract.option_type)
    p1 = flag_probability(ff)
    grid = classical_grid_asian(model, ff, ds.contract.strike, ds.contract.option_type)
    disc = float(np.exp(-ds.market.risk_free_rate * ds.contract.maturity))
    undisc = M * p1
    gates = None
    if count_gates:
        from qiskit import transpile
        gates = transpile(ff.circuit, basis_gates=["cx", "u"],
                          optimization_level=0).size()
    return FFPrice(price=disc * undisc, grid_price=disc * grid, undiscounted=undisc,
                   flag_prob=p1, scale=M, n_qubits=ff.circuit.num_qubits, T=T,
                   transpiled_gates=gates)


__all__ = [
    "FFCircuit", "FFPrice", "cir_increment_grids", "cir_v_grid",
    "build_cir_ff_path", "add_asian_payoff", "flag_probability",
    "classical_grid_asian", "price_cir_asian_quantum",
]
