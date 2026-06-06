# PCE Settlement — Implementation Plan v2
## JAX simulation → Hardware demo (provider-agnostic)

Supersedes v1. QUBO formulation (§1) is unchanged — re-read it if you haven't. Everything else has been rewritten for the JAX + PennyLane + hardware-agnostic stack.

---

## 0. The three-stage pipeline (read this first)

```
STAGE 1 — Laptop JAX sim          STAGE 2 — Validate               STAGE 3 — Hardware demo
─────────────────────────          ─────────────────────────          ────────────────────────
default.qubit + JAX backprop  →   match ILP baseline on small M  →   load θ*, run 3-basis
Adam via optax, n=5..12           confirm topology-aware HEA            measurement on device
all development here              produces same solution as ideal        decode settlement subset
```

**Key decision baked in:** training happens entirely offline on the JAX simulator. Pre-trained parameters θ* are loaded onto hardware for a single forward pass (3 circuit evaluations, one per Pauli basis). This is exactly the workflow the seminal PCE paper used on IonQ/Quantinuum. Queue times and limited shot budgets make on-hardware training impossible at a hackathon. Never attempt gradient descent on real hardware during the event.

---

## 1. Hardware available via Qmill / AWS Braket

All accessed through `pennylane-braket` — one unified path, no qiskit bridge needed.

| Device | ARN (eu-north-1 unless noted) | Qubits | Topology | 2Q fidelity | Notes |
|---|---|---|---|---|---|
| **IQM Garnet** ⭐ reliable demo | `…/iqm/Garnet` | **20** | Square lattice | 99.51% | EU data residency, IQM native, 19h/day |
| **IQM Emerald** ⭐ headline demo | `…/iqm/Emerald` | **54** | Square lattice | 99.5% | Largest IQM on Braket, same native gates |
| **IonQ Forte Enterprise 1** ⭐ cleanest | `…/ionq/Forte-Enterprise-1` | **36** | **All-to-all** | ~99.9% | Zero topology issues; same family used in seminal PCE paper |
| SV1 (simulator) | `…/amazon/sv1` | up to 34 | — | — | Cloud statevector; use for n>20 pre-hardware checks |
| TN1 (simulator) | `…/amazon/tn1` | up to 50 | — | — | Tensor-network; useful for shallow low-entanglement circuits |
| dm1 (simulator) | `…/amazon/dm1` | up to 17 | — | — | Density matrix with noise; pre-flight hardware prediction |

Full ARNs:
```
arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet
arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald
arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1
```

**Demo strategy:**
- **Primary:** IQM Emerald (54Q, eu-north-1). Use n=12 qubits, p=5 layers (m=60 settlement vars on hardware; m=660 on JAX sim). Finnish company, EU-hosted, OP-relevant narrative, 55× compression headline.
- **Fallback:** IQM Garnet (20Q, same topology, same code). Identical HEA — swap one ARN string.
- **Bonus:** IonQ Forte Enterprise 1 if budget allows. All-to-all means the 'square' entangler works perfectly and you can use more qubits without topology penalty. The PCE seminal paper ran hardware experiments on IonQ — mention this connection.

Coming hardware (narrative, no code needed): IQM 150Q to VTT mid-2026, 300Q late 2027.

---

## 2. Tech stack

```
pennylane >= 0.38                  # circuit definition, device abstraction, qnode
amazon-braket-pennylane-plugin     # pennylane → AWS Braket (all hardware + cloud sims)
jax >= 0.4                         # JIT-compiled autodiff through statevector
optax                              # Adam — JAX-native optimizer
numpy, scipy, matplotlib, PuLP     # unchanged from v1
```

Install: `pip install pennylane amazon-braket-pennylane-plugin jax optax pulp`

The qiskit stack (qiskit-on-iqm, qiskit-ibm-runtime, pennylane-qiskit) is **not needed** — `pennylane-braket` handles all hardware paths via the Braket SDK. If you want IBM as a fallback you can add qiskit later, but don't start with it.

---

## 3. Project structure

```
pce_settlement/
  pauli.py          # Pauli string enumeration, 3-basis decoder (unchanged from portfolio plan)
  hea.py            # Topology-aware HEA circuit in PennyLane
  device.py         # get_device(provider, n, shots) → PennyLane device (THE abstraction layer)
  pce_jax.py        # JAX-native loss, Adam training loop, solution readout
  hardware.py       # 3-basis measurement circuits, shot-based expectation decoding
  qubo.py           # build_qubo, qubo_to_ising (unchanged from v1)
  instance.py       # settlement instance generation (unchanged from v1)
  repair.py         # feasibility repair (unchanged from v1)
  baseline.py       # ILP + brute force + greedy (unchanged from v1)
  compare.py        # qubit-gap table, quality vs ILP
  plots.py          # gridlock graph, equity, qubit gap bar
  config.py         # all hyperparams + target device selection
  main.py           # end-to-end: train on sim → optionally deploy on hardware
  tests/
```

---

## 4. Device abstraction (`device.py`) — the provider-lock-in prevention

One function. Everything else imports from here. Switching hardware = changing one string in `config.py`.

```python
import pennylane as qml

# ARNs for all devices — swap here, nowhere else in codebase
ARNS = {
    "emerald":  "arn:aws:braket:eu-north-1::device/qpu/iqm/Emerald",
    "garnet":   "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    "ionq":     "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1",
    "sv1":      "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "tn1":      "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
    "dm1":      "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
}

def get_device(provider: str, n_wires: int, shots: int | None = None):
    """
    provider: 'sim' | 'emerald' | 'garnet' | 'ionq' | 'sv1' | 'tn1' | 'dm1'
    shots: None → analytic (sim/JAX only); integer → required for all Braket devices.
    """
    if provider == "sim":
        # Local JAX statevector — the training device. No cost, exact, fast.
        return qml.device("default.qubit", wires=n_wires)

    if provider not in ARNS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(ARNS)}")

    return qml.device(
        "braket.aws.qubit",
        device_arn=ARNS[provider],
        wires=n_wires,
        shots=shots,
    )

# config.py controls everything:
#   PROVIDER = "sim"       → laptop JAX training
#   PROVIDER = "emerald"   → IQM Emerald hardware demo (primary)
#   PROVIDER = "garnet"    → IQM Garnet fallback
#   PROVIDER = "ionq"      → IonQ Forte (all-to-all bonus)
#   PROVIDER = "sv1"       → cloud statevector for n>20 pre-hardware checks
```

**Topology note:** IQM Garnet and Emerald are both square-lattice. The `'square'` topology in `hea.py` handles both. IonQ Forte is all-to-all — use `'linear'` (or any pattern; all CZ pairs are native). No qiskit, no transpilation, no NordIQuEst fork.

---

## 5. Topology-aware HEA (`hea.py`)

IQM Garnet and Emerald are both square-lattice — the same HEA pattern works for both, and it maps cleanly to their native CZ gates with no SWAP overhead. IonQ Forte is all-to-all — any CZ pattern is native.

```python
import pennylane as qml

def build_hea(theta, n: int, p: int, topology: str = 'square'):
    """
    p-layer HEA. theta shape: (p+1, n).
    topology: 'square' (IQM Garnet/Emerald) | 'linear' (sim / IonQ all-to-all)
    """
    def ry_layer(l):
        for i in range(n):
            qml.RY(theta[l, i], wires=i)

    def entangler():
        if topology == 'linear':
            for i in range(n - 1):
                qml.CZ(wires=[i, i + 1])
        elif topology == 'square':
            # Two-pass brick-layer — maps to square-lattice native edges.
            # Even pass: (0,1), (2,3), (4,5), ...
            # Odd pass:  (1,2), (3,4), (5,6), ...
            for i in range(0, n - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, n - 1, 2):
                qml.CZ(wires=[i, i + 1])

    for layer in range(p):
        ry_layer(layer)
        entangler()
    ry_layer(p)   # final Ry; total params = n*(p+1)
```

**IQM native gate alignment:** IQM Garnet/Emerald on Braket support `ry` and `cz` directly. The HEA uses exactly those — no decomposition, no overhead. PennyLane's Braket plugin transpiles to the device gate set automatically, but since you're already using the native gates it's a no-op pass.

**IonQ:** all CZ pairs are native regardless of qubit indices. Use `topology='linear'` or any pattern; the transpiler on IonQ won't insert SWAPs. This is the advantage of trapped-ion all-to-all connectivity and the reason the PCE paper chose IonQ for hardware.

Default `TOPOLOGY = 'square'` in config works for both IQM machines and the JAX sim.

---

## 6. JAX training (`pce_jax.py`)

```python
import jax, jax.numpy as jnp
import optax
import pennylane as qml
from hea import build_hea
from pauli import enumerate_pauli_strings, pauli_pennylane_op

def make_circuit(dev, n, p, topology, strings):
    """Returns a jit-able QNode that outputs all M Pauli expectations."""
    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(theta):
        build_hea(theta, n, p, topology)
        return [qml.expval(pauli_pennylane_op(s, n)) for s in strings]
    return circuit

@jax.jit
def loss_fn(theta, circuit_fn, J, h, alpha, beta, nu):
    exps = jnp.array(circuit_fn(theta))
    t = jnp.tanh(alpha * exps)
    # Quadratic (coupling) term — sparse dot product over nonzero J entries
    quad  = jnp.sum(J * jnp.outer(t, t)) / 2      # J symmetric, /2 for i<j
    field = jnp.dot(h, t)
    reg   = beta * nu * jnp.mean(t**2) ** 2
    return quad + field + reg

def train(J, h, n, p, topology, k, cfg, key=jax.random.PRNGKey(0)):
    """
    Returns (theta_star, loss_history).
    Runs cfg.n_restarts restarts of cfg.n_steps Adam steps, keeps best.
    """
    dev = get_device('sim', n)
    m = J.shape[0]
    alpha = float(n ** (k // 2))
    nu = float(jnp.sum(jnp.abs(J)) / 2 + jnp.sum(jnp.abs(h)))
    circuit_fn = make_circuit(dev, n, p, topology, enumerate_pauli_strings(n, k, m))
    optimizer = optax.adam(cfg.lr)

    best_theta, best_loss = None, jnp.inf
    for restart in range(cfg.n_restarts):
        key, subkey = jax.random.split(key)
        theta = jax.random.uniform(subkey, (p + 1, n), minval=0, maxval=2 * jnp.pi)
        opt_state = optimizer.init(theta)

        @jax.jit
        def step(theta, opt_state):
            loss, g = jax.value_and_grad(loss_fn)(theta, circuit_fn, J, h, alpha, cfg.beta, nu)
            updates, opt_state = optimizer.update(g, opt_state)
            return optax.apply_updates(theta, updates), opt_state, loss

        for _ in range(cfg.n_steps):
            theta, opt_state, loss = step(theta, opt_state)

        if loss < best_loss:
            best_loss, best_theta = loss, theta

    return best_theta, best_loss

def readout(theta_star, circuit_fn, alpha):
    exps = jnp.array(circuit_fn(theta_star))
    z = jnp.sign(exps)                       # z in {-1, +1}^m
    x = ((1 + z) / 2).astype(int)            # x in {0, 1}^m (settlement decisions)
    return x
```

`pauli_pennylane_op(s, n)` converts a `(frozenset[int], 'X'|'Y'|'Z')` PCE string to a PennyLane observable: `qml.PauliX(i) @ qml.PauliX(j)` for a 2-body XX string, etc.

---

## 7. Hardware deployment (`hardware.py`)

Training is done. `theta_star` is fixed. Run 3 circuits on hardware.

```python
def hardware_expectations(theta_star, provider, n, p, topology, strings, n_shots, k):
    """
    Run 3 basis-measurement circuits on real hardware.
    Returns estimated Pauli expectations for all M strings.
    """
    dev_hw = get_device(provider, n, shots=n_shots)

    def make_meas_circuit(basis):
        @qml.qnode(dev_hw, shots=n_shots)
        def circ():
            build_hea(theta_star, n, p, topology)
            # Rotate to measurement basis
            for i in range(n):
                if basis == 'X':  qml.Hadamard(wires=i)
                elif basis == 'Y': qml.adjoint(qml.S)(wires=i); qml.Hadamard(wires=i)
                # Z: no rotation
            return qml.sample(wires=list(range(n)))
        return circ

    samples = {b: make_meas_circuit(b)() for b in ('X', 'Y', 'Z')}
    return decode_expectations(samples, strings, n)   # parity-weighted averages

def decode_expectations(samples, strings, n):
    """
    For each string (S, P): estimate <Π> = mean over shots of (-1)^parity(bits at S).
    samples: dict{'X': (n_shots, n) bitarray, 'Y': ..., 'Z': ...}
    """
    exps = []
    for (qubits, pauli) in strings:
        bits = samples[pauli]                         # (n_shots, n)
        mask = jnp.array([1 if i in qubits else 0 for i in range(n)])
        parities = jnp.mod(jnp.sum(bits * mask, axis=1), 2)   # 0 or 1 per shot
        exps.append(jnp.mean((-1.0) ** parities))
    return jnp.array(exps)
```

**Shot budget guidance (IBM 10-min plan):** IBM open-plan jobs submit to a shared queue; the 10 min is circuit-execution time, not wall-clock. Budget 3 × 1000 shots for the hardware run. That's 3 circuits × ~30 seconds each = ~90 seconds execution, well within 10 min. IBM allows shots up to 4096 per circuit; use 1024–2048 for a balance of statistical accuracy and queue speed.

**Helmi shots:** Helmi access via LUMI/CSC is allocation-based. Typically a few hundred to a few thousand shots per session. 500 shots per basis setting is enough to estimate parity correlations.

---

## 8. Configuration (`config.py`)

```python
# ── Simulation (laptop, all development) ──────────────────────────
PROVIDER    = "sim"
N_QUBITS    = 5
K_ORDER     = 3
TOPOLOGY    = "square"    # same for sim and IQM hardware
N_PARTIES   = 3
SLACK_BITS  = 7

# ── Optimizer ─────────────────────────────────────────────────────
LR          = 0.01
N_STEPS     = 500
N_RESTARTS  = 3
BETA        = 0.5

# ── Hardware demo: change PROVIDER + N_QUBITS only ────────────────
# PROVIDER  = "emerald"   # IQM Emerald 54Q — primary demo
# PROVIDER  = "garnet"    # IQM Garnet 20Q  — fallback
# PROVIDER  = "ionq"      # IonQ Forte 36Q  — bonus (all-to-all)
# PROVIDER  = "sv1"       # cloud statevector for n>20 pre-flight
# N_QUBITS  = 12          # use 12 of the device's qubits
N_SHOTS     = 1024        # per basis setting (3 × N_SHOTS total shots)
```

---

## 9. Settlement instance sizing

| Target | n used | k | m_max (k=3) | Hardware run (p=5) | JAX sim | QAOA would need | Compression |
|---|---|---|---|---|---|---|---|
| JAX dev | 5 | 3 | 30 | — | 30 vars, brute-force checkable | 30Q | 6× |
| IQM Garnet (20Q) | 12 | 3 | 660 | m=60, p=5, ~180 gates | 660 vars | 660Q | **55×** |
| IQM Emerald (54Q) ⭐ | 12 | 3 | 660 | m=60, p=5, ~180 gates | 660 vars | 660Q | **55×** |
| IQM Emerald (54Q) max | 20 | 3 | 3,420 | m=100, p=5 | 3,420 vars | 3,420Q | **285×** |
| IonQ Forte (36Q) | 20 | 3 | 3,420 | m=100, p=5, all-to-all | 3,420 vars | 3,420Q | **285×** |

**Practical rule:** Hardware circuit always uses p=5 layers (180 gates at n=12, manageable at 99.5% 2Q fidelity). The full m_max simulation runs on JAX. The pitch shows both numbers: "620 transactions solved on 12 qubits on sim, verified on hardware at 60 transactions — same algorithm, same parameters."

**Hero instance:** 3-party circular gridlock (A→B→C→A). 3 transactions + 3-party slack = ~24 binary variables, n=5 on JAX. Show it first, then scale.

---

## 10. Build order (hackathon-paced)

**Hour 1–2:** `instance.py` (gridlock cycle + random), `qubo.py` + ising unit test, `baseline.py` brute-force. This defines ground truth before a circuit is written.

**Hour 3–4:** `pauli.py`, `hea.py` (sim/linear topology first), `pce_jax.py`. Run on n=5, the cyclic gridlock. Match brute-force. This is the moment the algorithm works.

**Hour 5–6:** `repair.py`, scale to n=12, compare qubit table vs ILP. All on sim.

**Hour 7–8:** `device.py` + `hardware.py` 3-basis circuits. Test that `hardware.py` gives same expectations as JAX sim (run it against `default.qubit` with shots before touching real hardware).

**Hour 9–10:** Set `PROVIDER = "sv1"` first (cloud statevector simulator, free-ish, no queue). Run the full hardware.py 3-basis workflow on SV1 to confirm the shot-based decoding matches the JAX sim. Then submit to IQM Emerald or Garnet. Jobs are typically fast (minutes, not hours, unlike IBM queues).

**Polish:** `plots.py`, qubit-gap bar, narrative slide with the projection table.

---

## 11. Tests (unchanged from v1, plus new ones)

- `qubo_to_ising`: energy matches QUBO objective on 1000 random assignments.
- `hea.py` (star topology): Ry + star-CZ on n=5 circuit preserves norm; no gate between non-adjacent qubits.
- `hardware.py` vs `default.qubit.jax`: shot-based expectations converge to exact expectations as shots → ∞ (test at 10k shots, allow ε=0.05 tolerance).
- PCE recovers ILP optimum on M≤14 (brute-force checkable) within local bit-swap pass.
- Feasibility: `repair(readout(theta_star))` is always feasible regardless of penalty weight.

---

## 12. Gotchas specific to this stack

**No topology headaches on IQM Braket.** Garnet and Emerald are square-lattice with tunable couplers — the brick-layer HEA already in `hea.py` maps to native edges. The Braket SDK/PennyLane transpiles automatically; you won't hit the Helmi star-topology problem.

**IonQ all-to-all means zero transpilation.** Every CZ between any two wires is native. If IQM has queue issues, IonQ Forte is the no-fuss backup.

**AWS credentials.** The Braket plugin reads `~/.aws/credentials` or env vars (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`). Set region to `eu-north-1` for IQM machines. Set up and test with `sv1` (free-tier cloud sim) before attempting a QPU call.

**Braket task costs.** IQM Emerald and Garnet: approximately $0.00145 per shot + $0.01 per task (check current Braket pricing). 3 settings × 1024 shots ≈ $0.04 + task fees — very cheap. Run a cost estimate with `device.aws_session.get_device_capabilities()` before submitting.

**JAX + PennyLane `diff_method`.** Use `"backprop"` for `default.qubit` (training). For hardware, set `diff_method=None` — you're doing a forward pass only, no gradients. The hardware QNode just returns samples, no AD needed.

**p=55 is too deep for hardware.** With n=12, m=660, p=floor(660/12)=55 — too noisy. For the hardware run, fix m=60, p=5. On JAX sim demonstrate m=660. Be explicit in the pitch: "we simulate 660-variable settlement on classical hardware and verify the circuit structure on the QPU at a smaller instance."

**SV1 for pre-flight.** Before burning QPU shots, run `hardware.py` with `PROVIDER="sv1"` and the same `theta_star`. SV1 is exact up to shot noise; if results match JAX sim within ε=0.05, the circuit is correct. Then switch to `"emerald"`.

**IQM availability window.** Both Garnet and Emerald are available 19 hours/day. Check the Braket console for the current maintenance window before planning the demo slot.

---

## 13. The pitch narrative (updated for Braket hardware)

- **Today — IQM Garnet (20Q, eu-north-1):** same code, use as fallback. n=12, 55× compression.
- **Today — IQM Emerald (54Q, eu-north-1) ⭐:** hardware demo at n=12, p=5 (60 transactions, 180 gates, ~99.5% 2Q fidelity). JAX sim at n=12, m=660 (620 transactions). QAOA would need 660 qubits. **55× compression headline.** Finnish company, EU data residency — pitch directly to OP.
- **Today — IonQ Forte (36Q, all-to-all):** same code, `TOPOLOGY='linear'`, n=20 → 3,420 variables. 285× compression. Zero topology complexity. The PCE seminal paper ran experiments on IonQ — mention this.
- **Mid-2026 — IQM 150Q to VTT:** PCE k=3 → 3·C(150,3) = 1.6M variables. Same code, zero changes.
- **2027 — IQM 300Q:** ~13M variables. National RTGS daily batch territory.
- **Crossover (~50 clean logical qubits, 2029–2030):** PCE circuit classically unsimulable, ~59k transactions, genuine quantum-advantage territory — and the settlement problem at that scale is genuinely hard classically within real-time windows.

