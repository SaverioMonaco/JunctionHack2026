"""Constrained settlement QUBO and QUBO->Ising conversion (plan §1, Phase 2).

QUBO representation
-------------------
We store a QUBO as a dense symmetric-free upper form ``Q`` (m x m) plus a scalar
``const``. The objective on a binary vector x in {0,1}^m is

    f(x) = sum_i Q[i, i] * x_i  +  sum_{i<j} Q[i, j] * x_i x_j  +  const

i.e. the diagonal holds linear coefficients and the strict upper triangle holds
the quadratic couplings (lower triangle is unused / zero).

Spin convention (plan §6): x_i = (1 + z_i) / 2, z_i in {-1, +1}. Used
consistently here, in the PCE loss, and at readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance


@dataclass
class VarIndex:
    """Maps QUBO variable indices to their meaning.

    Indices 0..M-1 are transaction decisions x_t. The remaining indices are
    slack bits, grouped by party. ``slack_bits[p]`` is the list of variable
    indices (low bit first) encoding party p's slack S_p = sum_l 2^l y_{p,l}.
    """

    M: int
    m: int                       # total binary variables = M + sum_p L_p
    slack_bits: list[list[int]]  # per-party slack variable indices

    def slack_value(self, x: np.ndarray, p: int) -> int:
        bits = self.slack_bits[p]
        return int(sum((1 << l) * int(x[idx]) for l, idx in enumerate(bits)))


def slack_bit_count(max_slack: int) -> int:
    """Bits to represent slack in [0, max_slack]: ceil(log2(max_slack + 1)).

    NOTE: the plan (§1.3) writes L_p = ceil(log2(B_p + 1)), which only suffices
    when every constraint coefficient is non-negative. In settlement a party
    both sends and receives, so the constraint Σ c_{p,t} x_t ≤ B_p has signed
    coefficients: when p is a net receiver, Σ c·x goes negative and the required
    slack S_p = B_p − Σ c·x exceeds B_p. The true upper bound is
    max_slack = B_p + (total amount p can receive). Size to that, or the QUBO
    cannot represent the feasible optimum.
    """
    if max_slack <= 0:
        return 0
    return int(np.ceil(np.log2(max_slack + 1)))


def max_slack_for_party(inst: Instance, p: int) -> int:
    """Largest slack S_p needed: B_p plus every amount party p could receive."""
    inflow = int(inst.amounts[inst.receivers == p].sum())
    return int(inst.balances[p]) + inflow


def default_penalty(inst: Instance) -> float:
    """Starting penalty weight P ~ 10 * sum_t w_t (plan §1.3)."""
    return 10.0 * float(inst.weights.sum())


def default_lambdas(inst: Instance) -> np.ndarray:
    """Per-Greek penalty weights lambda_g for the neutrality term.

    Greeks live on a different numeric scale than cash, so we normalize each
    dimension by its total magnitude: lambda_g = kappa / (sum_t |g_t|)^2 with
    kappa ~ sum_t w_t (the value-objective scale). This makes each Greek penalty
    comparable to the value objective at full settlement, the same role the
    data-driven beta(c) heuristic plays in arXiv:2602.17479v2 (Eq. 19).

    Returns a (G,) array. Raises if the instance carries no Greeks.
    """
    if inst.greeks is None:
        raise ValueError("instance has no greeks; default_lambdas needs them")
    kappa = float(inst.weights.sum())
    scale = np.maximum(np.abs(inst.greeks).sum(axis=0), 1e-9)  # (G,)
    return kappa / (scale ** 2)


def _add_square_penalty(Q: np.ndarray, const: float, a: np.ndarray,
                        b: float, weight: float) -> float:
    """Accumulate ``weight * (a . v - b)^2`` into (Q, const) for v in {0,1}^m.

    Exact binary expansion (v_j^2 = v_j):
        (a.v - b)^2 = sum_j a_j^2 v_j + 2 sum_{j<k} a_j a_k v_j v_k
                      - 2 b sum_j a_j v_j + b^2
    The linear pieces land on the diagonal of Q, the quadratic pieces on the
    strict upper triangle, the b^2 on the constant. Returns the updated const.

    Shared by the per-party solvency penalty and the per-Greek neutrality
    penalty -- they are the same quadratic form, only the coefficient vector,
    target and weight differ.
    """
    nz = np.nonzero(a)[0]
    for j in nz:
        Q[j, j] += weight * (a[j] * a[j] - 2.0 * b * a[j])
    for ia in range(len(nz)):
        for ib in range(ia + 1, len(nz)):
            j, kk = nz[ia], nz[ib]
            Q[j, kk] += weight * 2.0 * a[j] * a[kk]   # j < kk preserved
    return const + weight * (b * b)


def build_qubo(inst: Instance, P: float | None = None,
               lambdas: np.ndarray | None = None,
               greek_targets: np.ndarray | None = None
               ) -> tuple[np.ndarray, float, VarIndex]:
    """Build the constrained settlement QUBO (plan §1.2-1.3), optionally with
    a Greek-neutrality penalty.

        H = - sum_t w_t x_t
            + P * sum_p ( sum_t c_{p,t} x_t + S_p - B_p )^2          (solvency)
            + sum_g lambda_g * ( sum_t g_{t} x_t - target_g )^2      (Greeks)

    with c_{p,t} = +a_t if p == s(t), -a_t if p == d(t), else 0, and slack
    S_p binary-encoded with L_p = ceil(log2(B_p + 1)) bits.

    The Greek term is added only when ``inst.greeks`` is set and ``lambdas`` is
    given. It adds NO new variables: the coefficient vector is the Greek column
    on the M transaction vars and zero on every slack bit (an equality-target
    soft penalty needs no slack). ``greek_targets`` defaults to zero (neutralize
    net Greeks). See greeks.py / default_lambdas.

    Returns (Q, const, var_index).
    """
    if P is None:
        P = default_penalty(inst)

    M, N = inst.M, inst.N

    # Variable layout: transactions first, then slack bits per party.
    slack_bits: list[list[int]] = []
    m = M
    for p in range(N):
        L_p = slack_bit_count(max_slack_for_party(inst, p))
        slack_bits.append(list(range(m, m + L_p)))
        m += L_p
    var_index = VarIndex(M=M, m=m, slack_bits=slack_bits)

    Q = np.zeros((m, m), dtype=float)
    const = 0.0

    # ---- Objective: maximize value => minimize -sum_t w_t x_t -------------
    for t in range(M):
        Q[t, t] += -inst.weights[t]

    # ---- Solvency penalty: P * sum_p (A_p . v - B_p)^2 --------------------
    # A_p is the linear coefficient vector over all m binary vars for party p.
    for p in range(N):
        B_p = int(inst.balances[p])
        A = np.zeros(m, dtype=float)
        # transaction contributions c_{p,t}
        send_mask = inst.senders == p
        recv_mask = inst.receivers == p
        A[:M] += np.where(send_mask, inst.amounts, 0.0)
        A[:M] -= np.where(recv_mask, inst.amounts, 0.0)
        # slack bit contributions 2^l
        for l, idx in enumerate(var_index.slack_bits[p]):
            A[idx] += float(1 << l)
        const = _add_square_penalty(Q, const, A, float(B_p), P)

    # ---- Greek-neutrality penalty: sum_g lambda_g (A_g . v - target_g)^2 --
    if inst.greeks is not None and lambdas is not None:
        G = inst.greeks.shape[1]
        lambdas = np.asarray(lambdas, dtype=float)
        targets = (np.zeros(G) if greek_targets is None
                   else np.asarray(greek_targets, dtype=float))
        for g in range(G):
            A = np.zeros(m, dtype=float)
            A[:M] = inst.greeks[:, g]   # zero on slack bits
            const = _add_square_penalty(Q, const, A, float(targets[g]),
                                        float(lambdas[g]))

    return Q, const, var_index


def embed_solution(inst: Instance, x: np.ndarray, var_index: VarIndex) -> np.ndarray:
    """Full m-bit QUBO vector for transaction decisions x, with slack bits set.

    Slack S_p = B_p - net_flow_p. Requires x to be feasible so each S_p is
    non-negative; our slack sizing (max_slack_for_party) guarantees it is
    representable. Useful for verification and warm-starting.
    """
    x = np.asarray(x).astype(int)
    full = np.zeros(var_index.m, dtype=int)
    full[: inst.M] = x
    slack = inst.balances - inst.net_flow(x)
    for p in range(inst.N):
        s = int(slack[p])
        assert s >= 0, f"infeasible x: party {p} slack {s} < 0"
        for l, idx in enumerate(var_index.slack_bits[p]):
            full[idx] = (s >> l) & 1
    return full


def qubo_to_ising(Q: np.ndarray, const: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO to Ising via x_i = (1 + z_i) / 2 (plan §1.4).

    Returns (J, h, ising_const) with energy
        H(z) = sum_{i<j} J[i, j] z_i z_j + sum_i h_i z_i + ising_const
    J is strict-upper-triangular (lower triangle zero).
    """
    m = Q.shape[0]
    diag = np.diag(Q).copy()
    # Symmetric quadratic coupling matrix from the strict upper triangle.
    Qu = np.triu(Q, k=1)
    Qsym = Qu + Qu.T  # Qsym[i, j] = quadratic coeff between i and j (i != j)

    J = Qu / 4.0
    h = diag / 2.0 + Qsym.sum(axis=1) / 4.0
    ising_const = const + diag.sum() / 2.0 + Qu.sum() / 4.0
    return J, h, ising_const


def qubo_energy(x: np.ndarray, Q: np.ndarray, const: float = 0.0) -> float:
    """Evaluate the QUBO objective f(x) for x in {0,1}^m."""
    x = np.asarray(x, dtype=float)
    lin = np.diag(Q) @ x
    quad = x @ np.triu(Q, k=1) @ x
    return float(lin + quad + const)


def ising_energy(z: np.ndarray, J: np.ndarray, h: np.ndarray, const: float = 0.0) -> float:
    """Evaluate Ising energy H(z) for z in {-1,+1}^m (plan §1.4)."""
    z = np.asarray(z, dtype=float)
    quad = z @ np.triu(J, k=1) @ z
    return float(quad + h @ z + const)
