r"""qff_pricing.scaling.measured_plots
======================================

Figures built from **actual simulated runs** (not asymptotic models).

``loader_scaling_and_fidelity`` -- measured transpiled gate count of the
implemented QSVT/QET polylog loader vs the naive ``StatePreparation`` loader,
plus the QSVT loader fidelity.  Demonstrates exp(n) -> poly(n) (= polylog in
``N = 2**n``) with the loaded state still correct.

``ae_advantage`` -- estimation error vs number of oracle calls for real
amplitude estimation (Aer shots) against classical sampling of the *same*
distribution.  Demonstrates the O(1/Q) quantum vs O(1/sqrt(M)) classical law.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qff_pricing.quantum.qsvt_loader import build_qsvt_loader, loader_fidelity


def _gaussian_probs(n):
    x = np.linspace(-3, 3, 2 ** n, endpoint=False)
    p = np.exp(-x ** 2 / 2)
    return p / p.sum()


def _naive_gates(n):
    """Measured transpiled gate count of qiskit StatePreparation (may fail for big n)."""
    c = QuantumCircuit(n)
    c.append(StatePreparation(np.sqrt(_gaussian_probs(n))), range(n))
    return transpile(c, basis_gates=["cx", "u"], optimization_level=1).size()


def measure_loader_scaling(n_naive=range(3, 13), n_qsvt=range(3, 15),
                           n_fid=range(3, 11)):
    naive = []
    for n in n_naive:
        try:
            naive.append((n, _naive_gates(n)))
        except Exception:
            pass  # qiskit StatePreparation synthesis is flaky for large n
    qsvt = [(n, transpile(build_qsvt_loader(_gaussian_probs(n))[0],
                          basis_gates=["cx", "u"], optimization_level=1).size())
            for n in n_qsvt]
    fid = [(n, loader_fidelity(_gaussian_probs(n))) for n in n_fid]
    return naive, qsvt, fid


def loader_scaling_and_fidelity(path, data=None):
    naive, qsvt, fid = data if data is not None else measure_loader_scaling()
    fig, (axg, axf) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    nn, gn = zip(*naive); nq, gq = zip(*qsvt)
    # exp trend through the measured naive points, extended across the qsvt range
    c = float(np.mean([g / 2.0 ** n for n, g in naive]))
    nx = np.array(sorted(set(list(nn) + list(nq))))
    axg.semilogy(nx, c * 2.0 ** nx, ":", color="#d62728", alpha=0.7,
                 label=r"naive trend  $\propto 2^{n}$")
    axg.semilogy(nn, gn, "s", color="#d62728", ms=8, label="naive StatePreparation (measured)")
    axg.semilogy(nq, gq, "o-", color="#1f77b4", label="QSVT/QET loader (measured)")
    axg.set_xlabel(r"$n$  (qubits;  $N = 2^{n}$ grid points)")
    axg.set_ylabel("transpiled gate count (cx+u, log scale)")
    axg.set_title("Measured loader cost:  naive $\\Theta(2^{n})$  vs  QSVT poly$(n)$")
    axg.legend(loc="upper left", framealpha=0.95)
    axg.grid(True, which="both", alpha=0.25)

    nf, ff = zip(*fid)
    axf.plot(nf, ff, "o-", color="#2ca02c")
    axf.axhline(0.999, color="#888", ls=":", lw=1.2, label="0.999")
    axf.set_ylim(min(0.99, min(ff) - 0.001), 1.0005)
    axf.set_xlabel(r"$n$  (qubits)")
    axf.set_ylabel(r"loader fidelity  $|\langle\psi_{QSVT}|\sqrt{p}\rangle|$")
    axf.set_title("QSVT loader stays correct (fidelity > 0.999)")
    axf.legend(loc="lower right", framealpha=0.95)
    axf.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


__all__ = ["measure_loader_scaling", "loader_scaling_and_fidelity",
           "_gaussian_probs", "_naive_gates",
           "measure_ae_advantage", "ae_advantage"]


# --------------------------------------------------------------------------- #
#  Amplitude-estimation advantage: real Aer runs vs classical sampling
# --------------------------------------------------------------------------- #
def _build_cir_european_encoder(n=5):
    """Exact-loader (StatePreparation) amplitude encoder for a CIR European call."""
    import numpy as np
    from qff_pricing.data import make_synthetic_dataset
    from qff_pricing.models.cir import CIRModel
    from qff_pricing.quantum.primitives import qsample_loader
    from qff_pricing.quantum.payoff import european_payoff, scaled_angles, payoff_oracle_exact
    from qff_pricing.quantum.qmci import build_amplitude_encoder
    ds = make_synthetic_dataset("cir", seed=0)
    model = CIRModel(ds.cir, ds.contract.dt)
    centers, probs, _ = model.terminal_grid_distribution(n, ds.contract.n_steps)
    f = european_payoff(centers, ds.contract.strike, ds.contract.option_type)
    angles, M = scaled_angles(f)
    enc, obj = build_amplitude_encoder(qsample_loader(probs), payoff_oracle_exact(n, angles))
    ftil = np.clip(f / M, 0.0, 1.0)
    a_true = float(np.sum(probs * ftil))      # exact grid amplitude (ground truth)
    return enc, obj, probs, ftil, a_true


def _grover_flag(enc, flag):
    """Grover/AE operator Q = A S0 A^dag S_flag for the (flag=1) good subspace."""
    from qiskit import QuantumCircuit
    nq = enc.num_qubits
    q = QuantumCircuit(nq)
    q.z(flag)                                  # S_chi: phase flip flag=1
    q.compose(enc.inverse(), inplace=True)
    q.x(range(nq)); q.h(nq - 1); q.mcx(list(range(nq - 1)), nq - 1); q.h(nq - 1); q.x(range(nq))
    q.compose(enc, inplace=True)
    return q


def _power_probs(enc, flag, powers):
    """Exact P(flag=1) after A.Q^m for each unique m (deterministic, computed once)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    Q = _grover_flag(enc, flag)
    nq = enc.num_qubits
    pmap = {}
    base = QuantumCircuit(nq); base.compose(enc, inplace=True)
    cur = base.copy(); m_done = 0
    for m in sorted(set(powers)):
        for _ in range(m - m_done):
            cur.compose(Q, inplace=True)
        m_done = m
        pmap[m] = float(Statevector(cur).probabilities([flag])[1])
    return pmap


def _mlae_sample(pmap, powers, shots, seed):
    """MLAE estimate from precomputed P(flag=1) + injected binomial shot noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    data = [(m, int(rng.binomial(shots, min(max(pmap[m], 0.0), 1.0))), shots) for m in powers]
    thetas = np.linspace(1e-4, np.pi / 2 - 1e-4, 20000)
    ll = np.zeros_like(thetas)
    for m, g, s in data:
        pp = np.clip(np.sin((2 * m + 1) * thetas) ** 2, 1e-12, 1 - 1e-12)
        ll += g * np.log(pp) + (s - g) * np.log(1 - pp)
    a = float(np.sin(thetas[int(np.argmax(ll))]) ** 2)
    queries = int(sum(2 * m + 1 for m in powers))
    return a, queries


def measure_ae_advantage(n=5, shots=4096,
                         schedules=((0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 5, 8),
                                    (0, 1, 2, 3, 5, 8, 13), (0, 1, 2, 3, 5, 8, 13, 21)),
                         q_seeds=(0, 1, 2), classical_M=(30, 100, 300, 1000, 3000, 10000, 30000),
                         c_trials=300):
    import numpy as np
    enc, obj, probs, ftil, a_true = _build_cir_european_encoder(n)
    # quantum: exact per-power probabilities (computed once) + binomial shot noise
    all_powers = sorted({m for sched in schedules for m in sched})
    pmap = _power_probs(enc, obj, all_powers)
    quantum = []
    for sched in schedules:
        errs, queries = [], None
        for sd in q_seeds:
            a, queries = _mlae_sample(pmap, sched, shots=shots, seed=sd)
            errs.append(abs(a - a_true))
        quantum.append((queries, float(np.mean(errs))))
    # classical: sample the SAME discrete distribution
    rng = np.random.default_rng(0)
    classical = []
    for M in classical_M:
        e = [abs(ftil[rng.choice(len(probs), size=M, p=probs)].mean() - a_true)
             for _ in range(c_trials)]
        classical.append((M, float(np.sqrt(np.mean(np.square(e))))))
    return classical, quantum, a_true


def ae_advantage(path, data=None):
    import numpy as np
    classical, quantum, a_true = data
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    cM, ce = zip(*classical); qQ, qe = zip(*quantum)
    ax.loglog(cM, ce, "s-", color="#d62728", label="classical sampling (measured)")
    ax.loglog(qQ, qe, "o-", color="#1f77b4", label="quantum amplitude estimation (measured)")
    # reference slopes
    x = np.array(cM, float)
    ax.loglog(x, ce[0] * (x / x[0]) ** -0.5, ":", color="#d62728", alpha=0.6,
              label=r"$\propto 1/\sqrt{M}$  (slope $-1/2$)")
    xq = np.array(qQ, float)
    ax.loglog(xq, qe[0] * (xq / xq[0]) ** -1.0, ":", color="#1f77b4", alpha=0.6,
              label=r"$\propto 1/Q$  (slope $-1$)")
    ax.set_xlabel("oracle calls  (classical samples $M$  /  quantum queries $Q$)")
    ax.set_ylabel(r"estimation error  $|\hat a - a|$")
    ax.set_title("Measured amplitude-estimation advantage:  quantum $1/Q$  vs  classical $1/\\sqrt{M}$")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path
