# Implementation Plan — Pauli Correlation Encoding (PCE) for Financial Transaction Settlement

Hackathon build spec for Claude Code. Target: a qubit-efficient quantum (and quantum-inspired) solver for the **transaction settlement / netting** problem, pitched at OP Financial Group's settlement and intraday-liquidity operations.

PCE core comes from Sciorilli et al. (2025), "Towards large-scale quantum optimization solvers with few qubits". The settlement application follows the qubit-efficient lineage of Huber, Tan, Griffin & Angelakis (2024), "Exponential qubit reduction in optimization for financial transaction settlement" (EPJ Quantum Technology), which applied this style of encoding to real exchange data and handles **any QUBO with linear inequality constraints**.

Read §0 and §1 before writing code; the conventions and the constraint-to-QUBO mapping are where this goes wrong if rushed.

---

## 0. What this builds, in one paragraph

A pool of pending transactions moves cash (and optionally securities) between parties. Settling *all* of them is often infeasible because some party would go negative ("gridlock"). We pick the subset to settle that maximizes settled value subject to every party staying solvent. That is a **constrained QUBO**. We fold the constraints in as penalties, convert to an Ising problem, solve it with PCE on `n = O(M^{1/k})` qubits (vs. QAOA's `n = M`), read out the chosen subset, and repair any residual infeasibility classically. Single solve — no recursion (unlike the portfolio clustering pipeline). The headline result is the qubit-count gap plus matching solution quality against an exact classical baseline on simulable sizes.

**Why settlement is a *better* PCE showcase than MaxCut:** the settlement QUBO is **dense** (every pair of transactions sharing a party gets coupled by the constraint penalties). QAOA depth scales with the number of quadratic terms, so a dense QUBO makes QAOA circuits enormous. PCE's circuit depth is `floor(m/n)` **regardless of density** — the circuit never sees the problem structure. Lean on this in the pitch.

---

## 1. The settlement QUBO (get this exactly right)

### 1.1 Problem data
- Parties `p ∈ {1..N}` with initial cash balance `B_p ≥ 0` (extend to per-asset balances `B_{p,a}` for delivery-versus-payment).
- Transactions `t ∈ {1..M}`. Each `t` moves amount `a_t > 0` of cash from sender `s(t)` to receiver `d(t)`. Priority/value weight `w_t` (use `a_t`, or a business priority).
- Decision: `x_t ∈ {0,1}`, `x_t = 1` ⟺ transaction `t` is settled in this batch.

### 1.2 Objective and constraints
Maximize settled value: `max Σ_t w_t x_t`.

Solvency per party (net outflow cannot exceed available balance):
```
Σ_{t: s(t)=p} a_t x_t  −  Σ_{t: d(t)=p} a_t x_t  ≤  B_p      for every party p
```
Define `c_{p,t} = +a_t` if `p=s(t)`, `−a_t` if `p=d(t)`, else `0`. Constraint: `Σ_t c_{p,t} x_t ≤ B_p`.

### 1.3 Inequality → equality via slack, then penalty
Introduce slack `S_p ≥ 0` with `Σ_t c_{p,t} x_t + S_p = B_p`. Binary-encode `S_p = Σ_{l=0}^{L_p-1} 2^l y_{p,l}`, with `L_p = ceil(log2(B_p + 1))` bits. Penalty:
```
QUBO objective (minimize):
  H = − Σ_t w_t x_t  +  P · Σ_p ( Σ_t c_{p,t} x_t + S_p − B_p )²
```
`P` = penalty weight, set `P > max_t w_t / (min nonzero balance gap)` — practically, `P ≈ 10 · Σ_t w_t` to start, then tune down until feasible solutions appear without swamping the objective. **Tuning `P` is the #1 time sink; expose it and sweep it.**

Total binary variables `m = M + Σ_p L_p` (transactions + slack bits). All of them go into PCE, so the qubit count stays `O(m^{1/k})` even with slacks.

> **Optimization (do after a working baseline):** the Huber et al. paper substitutes *continuous* slack variables to avoid the slack-bit overhead entirely. For the hackathon, ship binary slack first; mention continuous slack as the resource-reduction upgrade.

### 1.4 QUBO → Ising (the conversion PCE needs)
Expanding `H` gives a quadratic form `Σ_{i≤j} Q_ij x_i x_j + const` over all `m` binary vars `x_i ∈ {0,1}`. Map to spins `z_i ∈ {−1,+1}` via `x_i = (1 + z_i)/2`. Collect terms into
```
H(z) = Σ_{i<j} J_ij z_i z_j  +  Σ_i h_i z_i  + const
```
`J_ij` (couplings) come from off-diagonal `Q`; `h_i` (fields) come from diagonal + linear pieces. Keep a clean `qubo_to_ising(Q, c) -> (J, h, const)` function and unit-test it (energies must match for random `x`).

### 1.5 Generalized PCE loss (with fields — this is the key extension)
The seminal PCE loss is MaxCut-only (no field term). General QUBO has fields, so extend it:
```
α = n^floor(k/2)                                   # = n for k=2,3
t_i = tanh(α · ⟨Π_i⟩)
L(θ) = Σ_{i<j} J_ij · t_i · t_j  +  Σ_i h_i · t_i  +  L_reg
L_reg = β · ν · [ (1/m) · Σ_i t_i² ]²,   β = 1/2
```
`ν` = scale estimate of the objective magnitude. For general QUBO use `ν = Σ_{i<j}|J_ij| + Σ_i|h_i|` (a safe O(1)-normalizing scale), or the Poljak–Turzík bound on the coupling graph if you want to mirror the paper. Minimizing `L` drives each `z_i = sgn(⟨Π_i⟩)` toward the configuration that lowers `H`. Readout: `z_i = sgn(⟨Π_i⟩)`, then `x_i = (1+z_i)/2`.

Everything else about PCE (Pauli-string enumeration, qubit sizing, HEA, expectation values, barren-plateau suppression) is identical to the portfolio plan — **reuse `pauli.py` and `ansatz.py` unchanged.**

Sizing recap (verified against the seminal paper's Table 2): `n` = smallest integer with `m ≤ 3·C(n,k)`; layers `p = floor(m/n)`; `α = n` for k∈{2,3}.

---

## 2. Tech stack

- Python 3.10+, `numpy`, `scipy` (COBYLA), `matplotlib`.
- Classical baseline solver: `PuLP` or OR-tools or `python-mip` (exact ILP for medium `M`) — this is your honest competitor.
- Quantum sim: custom NumPy statevector + the 3-basis expectation trick from the portfolio plan (`n` stays small, ≤ ~14 for demo sizes).
- Optional: `qiskit`/`pennylane` only if you want a QAOA gate-count comparison rendered from a real circuit.
- Do **not** reach for JAX/Adam unless COBYLA is too slow; get correctness first.

---

## 3. Project structure
```
pce_settlement/
  pauli.py        # REUSE from portfolio plan: string enumeration + 3-basis expectations
  ansatz.py       # REUSE: HEA (Ry + CZ), NumPy statevector
  instance.py     # generate / load settlement instances (gridlock scenarios)
  qubo.py         # build constrained QUBO, slack encoding, qubo_to_ising
  pce_solve.py    # generalized PCE loss, COBYLA wrapper, readout, bit-swap
  repair.py       # feasibility check + classical repair to a valid settled subset
  baseline.py     # brute force (small M) + ILP (PuLP/OR-tools) + greedy
  compare.py      # qubit-count vs QAOA, solution-quality vs ILP, runtime
  plots.py        # gridlock graph, settled subset, qubit-gap bar, value vs size
  config.py       # M, N, k, P, optimizer settings, seed
  main.py         # end-to-end demo
  tests/
```

---

## 4. Phase-by-phase

### Phase 1 — Instances (`instance.py`)
Generate batches where settling everything is infeasible, so selection/netting matters.
```python
def random_instance(N, M, tightness, seed) -> Instance:
    # parties with balances B_p; M transactions (sender, receiver, amount, weight)
    # 'tightness' controls how far total obligations exceed liquidity (gridlock depth)

def gridlock_cycle(k_parties) -> Instance:
    # canonical demo: A→B→C→A circular obligations. No party can pay alone,
    # but settling ALL simultaneously nets to feasible. Shows the netting value.
```
Ship the cyclic gridlock instance as the hero demo — it's the textbook liquidity-saving-mechanism (LSM) example and instantly legible to a banking audience.

### Phase 2 — QUBO build (`qubo.py`)
```python
def build_qubo(inst, P) -> (Q, c, var_index)   # §1.3; var_index maps x_t / slack bits
def qubo_to_ising(Q, c) -> (J, h, const)        # §1.4, unit-tested
def ising_energy(z, J, h, const) -> float       # for baselines / local search
```

### Phase 3 — PCE solve (`pce_solve.py`)
```python
def pce_loss(theta, J, h, strings, n, p, k, beta, nu) -> float   # §1.5
def solve(J, h, k, cfg) -> np.ndarray:           # returns z in {-1,+1}^m
    n = qubits_for(m, k); p = layers_for(m, n)
    strings = enumerate_pauli_strings(n, k, m)
    # COBYLA over theta (several random restarts, keep best L)
    # z = sign(expectations at optimum)
def bit_swap(z, J, h) -> np.ndarray              # one Θ(#terms) Ising local-search sweep
```

### Phase 4 — Feasibility repair (`repair.py`)
Penalty solutions can violate a balance constraint. Repair before reporting (a demo that "settles" an infeasible batch is worthless).
```python
def feasible(x, inst) -> bool
def repair(x, inst) -> x:
    # while infeasible: drop the lowest-(value/priority) transaction from the
    # most-violated party; recheck. Then a greedy re-add pass for any tx that
    # now fits, to recover value. Cheap and makes the output trustworthy.
```

### Phase 5 — Baselines (`baseline.py`)
```python
def brute_force(inst) -> (x_opt, value)          # M <= ~20
def ilp_optimal(inst) -> (x_opt, value)          # PuLP/OR-tools, exact, medium M
def greedy(inst) -> (x, value)                   # value-sorted feasible insertion
```

### Phase 6 — Comparison & headline (`compare.py`)
```python
def qubit_counts(m, k) -> dict     # PCE n vs QAOA n=m  (the money table)
def quality_vs_ilp(...)            # PCE settled value / ILP optimum (approx ratio)
def runtime(...)                   # wall-clock per solve
```
Headline numbers to print: for the demo `M`, PCE qubits vs QAOA qubits; PCE settled value vs ILP optimum (aim ≥ 0.95). Then the *projection table* (no run needed, pure arithmetic) for the pitch:

| qubits n (k=3) | variables addressable 3·C(n,3) |
|---|---|
| 12 | ~660 |
| 17 | ~2,040 |
| 50 | ~58,800 |
| 100 | ~485,100 |
| 200 | ~3,940,200 |

### Phase 7 — Plots (`plots.py`)
Gridlock graph (parties = nodes, transactions = directed weighted edges; settled = green, dropped = grey); a qubit-gap bar (PCE vs QAOA, log scale); settled-value vs problem-size; optional COBYLA loss curve.

---

## 5. Validation / tests
- `qubo_to_ising`: Ising energy == QUBO objective for 1000 random assignments.
- PCE recovers the brute-force optimum on small instances (M ≤ 14) within the bit-swap pass.
- Cyclic gridlock: solver + repair settles the full cycle (the netting win) when liquidity permits, and the max-value feasible subset when it doesn't.
- Penalty sanity: as `P` increases, fraction of infeasible raw readouts → 0.
- Reuse the portfolio plan's `pauli.py`/`ansatz.py` tests (3-basis expectations vs direct-apply oracle; Y-rotation sign on |+i⟩).

---

## 6. Gotchas
- **Spin convention.** Use `x = (1+z)/2`, `z ∈ {−1,+1}` consistently across `qubo_to_ising`, the loss, and readout. Mixing `(1+z)/2` and `(1−z)/2` silently flips your answer.
- **`m` includes slack bits.** Size `n`, `p`, `α`, `strings` on `m = M + Σ_p L_p`, not on `M`.
- **Dense QUBO is fine for PCE.** If your circuit grows with the number of `J_ij` terms, you've built QAOA. PCE depth = `floor(m/n)`, period.
- **Penalty weight `P` dominates behavior.** Too small → infeasible; too large → flat landscape / barren training. Sweep it; report the value used.
- **Always repair before reporting.** Raw penalty readout is not guaranteed feasible.
- **Balances must be integers** for clean binary slack encoding; scale amounts to integer minor units (cents) first.

---

## 7. Build order (hackathon-paced)
1. `instance.py` (cyclic gridlock + random) and `baseline.py` (brute force) — defines "truth."
2. `qubo.py` + ising unit test.
3. Drop in `pauli.py`/`ansatz.py` from the portfolio plan.
4. `pce_solve.py` + `repair.py`; match brute force on M ≤ 14.
5. `compare.py` qubit table + ILP quality check — this is the demo's spine.
6. `plots.py`, then `main.py` end-to-end on the hero gridlock instance.
7. If time: continuous-slack upgrade, multi-asset DvP, QAOA gate-count contrast, quantum-inspired classical mode (swap the circuit for a classical correlation sampler).

---

## 8. Pitch / narrative section (for the slides)

**The problem is real and valuable.** Settlement netting is NP-hard and grows combinatorially with payment count; national RTGS and securities-settlement systems clear huge batches in tight time windows. The Bank of Canada's 2022 quantum-annealing settlement pilot reported potential liquidity savings up to CAD 275M. Maps directly onto OP's TARGET2 / T2S / SEPA settlement and intraday-liquidity operations. The Bank of Finland is publicly tracking quantum readiness; optimisation is a named use case.

**The advantage we demonstrate today (true, measurable): space.** PCE encodes `m = O(n^k)` variables in `n` qubits. Our demo solves an `M`-transaction batch on `~O(M^{1/3})` qubits where QAOA needs `~M`. Show the side-by-side qubit table and matching solution quality vs an exact ILP solver on the sizes we can simulate. Plus a structural win unique to settlement: the QUBO is dense, which explodes QAOA depth, while PCE depth is problem-independent.

**The advantage that's coming (the scaling story).** Classical exact simulation of a quantum circuit dies near 50 qubits (~2⁵⁰ ≈ 10¹⁵ amplitudes). At that exact wall, PCE (k=3) already addresses ~59,000 binary variables; at 100 qubits, ~485,000; at 200, ~4M. So the moment a PCE circuit becomes classically unsimulable, it's addressing settlement batches that genuinely strain classical solvers inside the minutes-long settlement cycle. Hardware is on a timeline to meet this: Google Willow (105 qubits) and Quantinuum H2 (56) today; IBM targeting first large-scale fault-tolerant machine (Starling, 200 logical qubits) by 2029 and Quantinuum targeting fault-tolerance by 2030. As clean logical-qubit counts climb to a few hundred over 2029–2033, PCE's reach climbs from ~10⁴ to millions of transactions — the size of a national daily batch.

**Three caveats we state up front (credibility, not weakness).** (1) No proof yet that PCE beats the best classical solver on a hard instance at scale — it's a well-motivated conjecture. (2) The crossover needs fault-tolerant qubits (~2029–2030+), not today's noisy ones; today's value is as a quantum-inspired heuristic and as readiness. (3) Shallow circuits can sometimes be tensor-network-simulated past 50 qubits, so the advantage hinges on the trained circuit carrying enough entanglement — an open research question.

**One-line close.** "We're not beating your netting engine today. We're building the algorithm that will — it already matches classical quality at the sizes we can test, on a qubit budget ~1000× smaller than the obvious quantum approach — and it scales onto the fault-tolerant machines arriving at the end of the decade."
