# Implementation Plan — Pauli Correlation Encoding (PCE) for Portfolio Optimization

This is a build spec for Claude Code. It reproduces the pipeline in Soloviev & Krompiec (2025), "Large-scale Portfolio Optimization using Pauli Correlation Encoding", using the core PCE algorithm from Sciorilli et al. (2025), "Towards large-scale quantum optimization solvers with few qubits". Read this whole document before writing code; the math conventions matter.

---

## 0. The one architectural idea you must not get wrong

PCE does **not** solve the Markowitz mean-variance QUBO directly. The QUBO that PCE solves at every step is **weighted MaxCut on a market graph**. Returns enter the pipeline only at the final representative-selection step. The full chain is:

1. Build a **market graph** `G`: nodes = assets, edge weight `w_ij = 1 - |ρ_ij|` when `|ρ_ij| > λ` (Pearson correlation of returns), no edge otherwise.
2. **Recursively bipartition** `G` by solving weighted MaxCut with PCE. Because high correlation → small edge weight, MaxCut preferentially cuts the *weakly* correlated edges, so each resulting cluster contains *highly correlated* (redundant) assets.
3. From each cluster pick **one representative** = the asset with the highest historical mean return.
4. The portfolio is the set of representatives (diversified: one per redundancy cluster). Evaluate on a held-out test set via Sharpe ratio and cumulative return vs. a baseline.

PCE is the engine for step 2 only. Everything else is classical finance plumbing.

A direct Markowitz QUBO (`minimize xᵀΣx − q·μᵀx`) is also PCE-solvable since PCE handles any QUBO, but it is **not** what this plan builds — the graph-partition route is the one that scales to 250 assets and is benchmarked in the paper. Note it as a stretch variant only.

---

## 1. What PCE is (concise math reference)

Encode `m` binary variables into `n` qubits via Pauli-string expectation signs:

```
x_i = sgn(⟨Ψ| Π_i |Ψ⟩),   i = 1..m
```

Each `Π_i` is an order-`k` Pauli string: pick a `k`-subset `S ⊆ {0..n-1}` of qubits and one Pauli type `P ∈ {X,Y,Z}`; place `P` on every qubit in `S` and identity elsewhere. There are `3·C(n,k)` such strings; assign the first `m` of them (in a fixed, deterministic order) to the variables.

Sizing (verified against the paper's Table 2 — reproduce these exactly):

```
n = smallest integer with  m ≤ 3·C(n,k)
p = floor(m / n)                      # HEA layers
α = n^floor(k/2)                      # = n for both k=2 and k=3
```

Reference values (k=3): `(m,n,p)` = (10,4,2),(20,5,4),(30,5,6),(50,6,8),(100,7,14),(150,8,18),(200,9,22),(250,9,27). Two-qubit-gate-bearing circuit size scales as `3np+1` (k=2 gates: 25,61,91,148,298,430,586,715; k=3 gates: 25,61,91,145,295,433,595,730). The circuit is **problem-independent**: its depth depends only on `m,n`, never on the number of edges — this is PCE's headline advantage over QAOA.

**Loss (minimize):**

```
L(θ) = Σ_{(i,j)∈E} w_ij · tanh(α⟨Π_i⟩) · tanh(α⟨Π_j⟩)  +  L_reg
L_reg = β · ν · [ (1/m) · Σ_{i∈V} tanh(α⟨Π_i⟩)² ]²
```

with `β = 1/2`, and `ν` = Poljak–Turzík bound for weighted MaxCut: `ν = w(G)/2 + w(T_min)/4`, where `w(G)` is the total edge weight and `T_min` is a minimum spanning tree. Minimizing `L` drives connected nodes to opposite signs (i.e. cut), which maximizes the weighted cut.

**Why this is trainable at scale:** the loss variance is `Var(L) = (α⁴/d²)·Σ w_ij² + O(α⁶/d³)` with `d = 2ⁿ`. Since `n = O(m^{1/k})`, the barren-plateau decay in `m` is super-polynomially suppressed for `k>1`. You get this for free from the encoding; no special handling needed, but it justifies using a modest, fixed ansatz depth.

**Post-processing — local bit-swap search** (from the seminal paper, recommended): after reading `x = sgn(⟨Π_i⟩)`, sweep once over all variables; flip each, recompute the cut value `V(x) = Σ_{(i,j)∈E} w_ij(1 − x_i x_j)` with `x ∈ {−1,+1}`, keep the flip iff it increases `V`. Total cost `Θ(|E|)`. Cheap and meaningfully improves quality.

---

## 2. Tech stack

Keep it dependency-light. `n ≤ 14` means statevectors are tiny (`2¹⁴ = 16384`), so a custom NumPy simulator is both fastest for this structure and fully transparent.

- Python 3.10+
- `numpy`, `scipy` (COBYLA / SLSQP), `pandas`, `networkx` (graph ops, MST), `matplotlib`
- Optional upgrade path: `jax` for autodiff + Adam (the seminal paper uses Adam for larger instances); or `pennylane`/`qiskit-aer` if you'd rather not hand-roll the simulator. Do **not** start here — get the NumPy + COBYLA version correct first, it matches the finance paper directly (COBYLA, statevector).

Do **not** build dense `2ⁿ × 2ⁿ` Pauli matrices for `n > 10` (memory blows up). Apply Paulis to the statevector as sparse ops, or use the 3-basis trick below.

---

## 3. Project structure

```
pce_portfolio/
  data.py          # load S&P500, returns, train/test split, market graph
  pauli.py         # Pauli-string enumeration + fast expectation values
  ansatz.py        # Hardware-Efficient Ansatz (Ry + CZ), statevector sim
  pce_solver.py    # loss, optimizer wrapper, single bipartition, local search
  partition.py     # Algorithm 1: recursive bipartitioning
  portfolio.py     # representative selection, weights, Sharpe, backtest
  benchmarks.py    # classical EDA baseline (+ optional QAOA for m=10)
  validate.py      # reproduce Table 1 / Table 2 / Figure 1 numbers
  plots.py         # dendrogram, equity curves, Sharpe bars, runtime
  config.py        # all hyperparameters in one place
  main.py          # end-to-end run for m in {10,20,30,50,100,150,200,250}
  tests/
```

Put every tunable in `config.py`: `k`, `λ` (correlation threshold), `β`, `r_f`, optimizer settings (maxiter, tol), the `nsplits` schedule, the asset-count sweep, random seed.

---

## 4. Phase-by-phase build

### Phase 1 — Data & market graph (`data.py`)

Dataset: Kaggle `camnugent/sandp500`, file `all_stocks_5yr.csv` (≈505 tickers, 2013–2018, ~1259 trading days). Columns: `date, open, high, low, close, volume, Name`.

```python
def load_prices(path) -> pd.DataFrame:
    # pivot to wide: index=date (sorted), columns=Name, values=close
    # drop tickers with any missing values (or forward-fill, but be consistent)
    # return clean wide price frame with a deterministic column order

def daily_returns(prices) -> pd.DataFrame:        # prices.pct_change().dropna()

def train_test_split(returns, frac=0.8):          # CHRONOLOGICAL, not random
    # first 80% rows = train, last 20% = test

def select_first_m(returns, m):                   # first m columns (fixed order)

def market_graph(train_returns_m, lam) -> nx.Graph:
    # rho = train_returns_m.corr(method='pearson')   (m x m)
    # add edge (i,j) with weight 1 - |rho_ij|  iff |rho_ij| > lam
    # nodes carry the original asset index/ticker as an attribute
```

`λ` is not stated numerically in the paper. Calibrate it so graph density roughly matches Table 1 (m=10 → 39 edges / density 0.867; m=250 → 20948 edges / density 0.660). A single global `λ ≈ 0.2–0.3` lands in the right band; expose it and verify against Table 1 in `validate.py`.

`select_first_m` reproduces the paper's "first m columns" so larger-m runs nest smaller ones. Document that exact column ordering affects exact reproduction.

### Phase 2 — Pauli machinery (`pauli.py`)

```python
def enumerate_pauli_strings(n, k, m) -> list[tuple[frozenset[int], str]]:
    # all (S, P) for S in combinations(range(n), k), P in 'XYZ'
    # deterministic order; return first m. Assert 3*C(n,k) >= m.

def qubits_for(m, k) -> int:        # smallest n with 3*C(n,k) >= m
def layers_for(m, n) -> int:        # max(1, m // n)
```

Expectation values — implement the **3-basis trick** (fast and exact for statevector):

```python
def all_expectations(state, strings, n) -> np.ndarray:   # length m, real
    # probs in 3 bases:
    #   Z basis: pZ = |state|**2
    #   X basis: pX = |(H^⊗n) state|**2
    #   Y basis: pY = |((H·S†)^⊗n) state|**2   <-- VERIFY rotation with a unit test
    # for a string (S, P): ⟨Π⟩ = Σ_b (-1)^{parity(b & maskS)} · p_basis[b]
    # precompute parity-sign vectors per S once; this is the inner-loop hot path
```

Unit-test the Y rotation against a known state, e.g. `⟨Y⟩ = +1` for `|+i⟩ = (|0⟩+i|1⟩)/√2`. A simple, slower fallback for cross-checking: apply each Pauli string to the statevector directly (reshape to `[2]*n`, X = axis flip, Z = sign on the `1` index, Y = i·X·Z) and take `Re(⟨state|Π|state⟩)`. Keep the fallback as the test oracle.

### Phase 3 — Ansatz & simulator (`ansatz.py`)

Hardware-Efficient Ansatz exactly as the paper's Fig. 2: `p` layers, each = a layer of parameterized `R_y(θ)` on all `n` qubits followed by a CZ entangling pattern (linear/ladder; the paper says "linear entangling structure with CZ gates"). Parameter count ≈ `n·p` (add a final `R_y` layer if you want `n·(p+1)`; keep it configurable).

```python
def build_state(theta, n, p) -> np.ndarray:
    # start |0...0>, apply Ry layer, CZ ladder, repeat p times
    # pure NumPy statevector (apply 1- and 2-qubit gates by tensor reshaping)

def num_params(n, p) -> int
```

Validate gate count against `3np+1` and the Table 2 columns. The exact "+1"/factor depends on how you count the entangler; matching scaling is what matters, not the literal constant.

### Phase 4 — Single bipartition (`pce_solver.py`)

```python
def poljak_turzik_nu(G) -> float:
    # w(G)/2 + w(MST)/4   (networkx.minimum_spanning_tree on weights)

def loss(theta, G, strings, n, p, k, beta) -> float:
    state = build_state(theta, n, p)
    exp   = all_expectations(state, strings, n)        # ⟨Π_i⟩
    alpha = n ** (k // 2)
    t     = np.tanh(alpha * exp)
    main  = sum(w_ij * t[i] * t[j] for (i,j,w_ij) in edges_local(G))
    reg   = beta * poljak_turzik_nu(G) * (np.mean(t**2))**2
    return main + reg

def cut_value(x, G) -> float:        # Σ w_ij (1 - x_i x_j), x in {-1,+1}
def local_bit_swap(x, G) -> np.ndarray   # one Θ(|E|) sweep, keep improving flips

def bipartition(G, k, cfg) -> tuple[list, list]:
    m   = G.number_of_nodes()
    n   = qubits_for(m, k); p = layers_for(m, n)
    strings = enumerate_pauli_strings(n, k, m)
    theta0  = rng.uniform(0, 2π, num_params(n,p))
    res = scipy.optimize.minimize(loss, theta0, method='COBYLA',
                                  args=(G, strings, n, p, k, cfg.beta),
                                  options={'maxiter': cfg.maxiter})
    exp = all_expectations(build_state(res.x,n,p), strings, n)
    x   = np.where(exp > 0, 1, -1)
    x   = local_bit_swap(x, G)                 # optional but recommended
    S1  = [nodes[i] for i in range(m) if x[i] > 0]   # map local idx -> global id
    S2  = [nodes[i] for i in range(m) if x[i] <= 0]
    return S1, S2
```

Edges must be indexed **locally** (0..m-1) inside the loss but mapped back to **global** asset IDs in the returned sets. Keep a `local_index ↔ global_id` map per subgraph and never lose it — this is the most common bug source in the recursion.

Optimizer: COBYLA matches the paper. For larger `m` (more params), switch to SLSQP (finite-diff) or the JAX+Adam path. Use a few random restarts and keep the best `L`.

### Phase 5 — Recursive partition (`partition.py`)

Faithful to Algorithm 1 (FIFO queue, BFS order):

```python
def recursive_partition(G, nsplits, k, cfg) -> list[list]:
    queue = deque([list(G.nodes())])      # each item = list of global asset ids
    done  = []
    splits_done = 0
    while splits_done < nsplits and queue:
        nodes = queue.popleft()
        if len(nodes) > 1:
            S1, S2 = bipartition(G.subgraph(nodes), k, cfg)
            queue.append(S1); queue.append(S2)
            splits_done += 1
        else:
            done.append(nodes)            # singleton, can't split
    return done + list(queue)             # nsplits+1 clusters total
```

`nsplits` schedule from the paper: `nsplits = 2,4,6,9` for `m < 100`, and `nsplits = m/10 − 1` for `m ≥ 100`. Record the split sequence so you can draw the dendrogram (Fig. 6).

### Phase 6 — Portfolio & evaluation (`portfolio.py`)

```python
def representatives(clusters, train_returns) -> list:
    # for each cluster pick argmax_j mean(train_returns[asset_j])  (Eq. 1)

def equal_weight(assets) -> dict                     # 1/N each
def portfolio_returns(weights, returns) -> pd.Series # period returns
def sharpe(returns, r_f=0.0, periods=252) -> float   # (μ - r_f)/σ, optional ×√periods
def backtest(weights, test_returns, init=1000.0) -> pd.Series   # equity curve

# baseline = equal-weight ALL m assets (the "S&P 500" proxy in the paper)
```

Outputs to reproduce: Sharpe ratio bars, train and test (Fig. 8 — PCE should beat baseline on most sizes); `$1000` equity curves on the test set vs. baseline for each `m` (Fig. 7); the dendrogram for `m=50` (Fig. 6).

Defaults to state explicitly: `r_f = 0`, equal weighting, chronological 80/20 split, Sharpe on daily returns (annualize with `√252` only if you label it as such).

### Phase 7 — Benchmarks (`benchmarks.py`) — optional

- **EDA (classical heuristic):** an Estimation-of-Distribution Algorithm solving the same MaxCut as a drop-in for `bipartition`. Sample binary vectors from a per-variable Bernoulli model, score by `cut_value`, keep the top fraction, refit probabilities, iterate. Used for all `m ≥ 10` in the paper. Easiest external dep: `EDAspy`, or hand-roll a UMDA in ~40 lines.
- **QAOA (m=10 only):** one-hot encoding = PCE with `k=1`. Use `qiskit` or `pennylane`. The paper only runs it at `m=10` because gate count explodes (Table 2: QAOA p=2 needs ~42k gates at m=250 vs. PCE's ~730). Mostly a gate-count/runtime contrast (Fig. 9).

### Phase 8 — Validation (`validate.py`)

Assert against the papers so regressions surface immediately:

- `qubits_for` / `layers_for` reproduce the `(m,n,p)` table in §1 and Fig. 1.
- Gate counts track `3np+1` and Table 2 (both `k`).
- Market-graph edge counts / density approximate Table 1 for the calibrated `λ`.
- On a tiny hand-built weighted graph, PCE + local search recovers the known max cut.
- Loss-variance spot check: random-θ `Var(L)` shrinks with `n` consistent with the `α⁴/2^{2n}` leading term.

---

## 5. Testing strategy

- `pauli.py`: 3-basis expectations match the direct-apply oracle to ~1e-10 on random states; Y-rotation sign verified on `|+i⟩`.
- `ansatz.py`: `build_state` norm = 1; known small circuits (e.g. one Ry then CZ) match analytic states.
- `pce_solver.py`: on K3, K4, and a 6-node weighted graph, recovered cut ≥ 0.95 of optimum (brute-force optimum for n_nodes ≤ 16).
- `partition.py`: produces exactly `nsplits+1` clusters; clusters are disjoint and cover all nodes; global-id mapping is bijective.
- `portfolio.py`: Sharpe and equity-curve math checked against hand calculations on a 3-asset toy.

---

## 6. Gotchas (read before coding)

- **Sign conventions differ between the two papers.** Seminal uses `x ∈ {−1,+1}`; finance uses `x ∈ {0,1}` with `S1 = {i : x_i > 0}`. Pick `{−1,+1}` internally for MaxCut/`cut_value`, and derive the two subsets by sign. Be consistent.
- **`m` in the loss/regularization is the current subgraph size**, not the global asset count. Recompute `n, p, α, ν, strings` fresh on every bipartition call.
- **Local vs. global node indices.** Build the circuit/loss over local indices `0..m-1`; always map results back to global asset IDs before returning.
- **Time-series split is chronological.** Never shuffle — that leaks the future into training.
- **COBYLA scales poorly in parameter count.** At `m=250` (`~243` params) expect long runs (the paper quotes ~1 hour for `m≈200` on statevector). If it's too slow, move to JAX+Adam (autodiff) before blaming correctness.
- **Don't build dense Pauli matrices for `n>10`.** Use the 3-basis probabilities.
- **Number of edges never affects circuit depth** — if your circuit grows with `|E|`, you've implemented QAOA by accident, not PCE.

---

## 7. Suggested build order

1. `data.py` + `validate.py` market-graph check against Table 1.
2. `pauli.py` (with both expectation methods) + its tests — this is the correctness foundation.
3. `ansatz.py` + gate-count validation.
4. `pce_solver.py` single `bipartition`, test on toy graphs against brute-force.
5. `partition.py` recursion.
6. `portfolio.py` + `plots.py`; run `m=10` end-to-end and eyeball the equity curve vs. baseline.
7. Scale the `m`-sweep; reproduce Sharpe bars (Fig. 8).
8. Optional: EDA and QAOA benchmarks.

## 8. Stretch goals

- JAX/Adam optimizer with analytic gradients for `m ≥ 100`.
- Shot-based measurement (3 settings, since the Pauli sets are mutually commuting) instead of statevector, to mirror real hardware.
- Direct Markowitz QUBO variant solved by PCE (`minimize xᵀΣx − q·μᵀx` with a cardinality penalty) as an alternative architecture — compare against the graph-partition route.
- Smarter representative selection (e.g. signed-correlation / Sharpe-within-cluster instead of raw mean return), flagged as future work in the paper.
- Run-time profiling to reproduce Fig. 9.
