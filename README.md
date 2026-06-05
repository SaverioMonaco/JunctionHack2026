# Quantum Machine Learning for Finance — Hackathon Challenges

## Datasets

| Source | Description | How to access |
|--------|-------------|---------------|
| **Yahoo Finance** | Historical market data (prices, volumes, etc.) | `pip install yfinance` → [pypi.org/project/yfinance](https://pypi.org/project/yfinance/) · [docs](https://ranaroussi.github.io/yfinance/) |
| **Qiskit Finance** | Simulated financial data (random portfolios, option scenarios) | Part of [qiskit-finance](https://github.com/qiskit-community/qiskit-finance); includes `RandomDataProvider` and `YahooDataProvider` |
| **Normal distribution** | Synthetic baseline | `numpy.random.normal` or `scipy.stats.norm` |

> **Note:** Qiskit Finance tutorials are illustrative only. A good solution should not rely on Qiskit Finance alone.

---

## Getting Started

- Yahoo Finance data: `pip install yfinance`
- Qiskit Finance tutorials serve to illustrate concepts — a competitive solution goes beyond them.
- Data can also be sampled from a normal distribution for synthetic experiments.

---

## Challenge Option 1 — Portfolio Optimization

Use a quantum approach to tackle portfolio optimization.

**Suggested approaches:**
- **QAOA** — Quantum Approximate Optimization Algorithm
- **VQE** — Variational Quantum Eigensolver
- **Grover adaptive search** — for constrained polynomial binary optimization

**References:**
- Brandhofer et al. (2023): [*Benchmarking the performance of portfolio optimization with QAOA*](https://doi.org/10.1007/s11128-022-03766-5), Quantum Information Processing, Volume 22, article 25
- Buonaiuto et al. (2023): [*Best practices for portfolio optimization by quantum computing, experimented on real quantum devices*](https://doi.org/10.1038/s41598-023-45392-w), Scientific Reports 13, 19434
- Gilliam et al. (2021): [*Grover adaptive search for constrained polynomial binary optimization*](https://doi.org/10.22331/q-2021-04-08-428), Quantum 5, 428

---

## Challenge Option 2 — Option Pricing

Apply quantum methods to financial derivative pricing.

**Suggested approaches:**
- **Quantum Amplitude Estimation (QAE)** — to simulate scenarios
- **Quantum PDE solvers** — for the ambitious ones
- **Quantum Generative Adversarial Networks (qGAN)** — to simulate densities

**References:**
- Stamatopoulos et al. (2020): [*Option Pricing using Quantum Computers*](https://doi.org/10.22331/q-2020-07-06-291), Quantum 4, 291
- Wang and Kan (2024): [*Option pricing under stochastic volatility on a quantum computer*](https://doi.org/10.22331/q-2024-10-23-1504), Quantum 8, 1504
- Herman et al. (2026): [*Quantum Speedups for Derivative Pricing Beyond Black-Scholes*](https://arxiv.org/abs/2602.03725), arXiv:2602.03725
- Zoufal et al. (2019): [*Quantum Generative Adversarial Networks for learning and loading random distributions*](https://doi.org/10.1038/s41534-019-0223-2), npj Quantum Information 5, 103

---

## Challenge Option 3 — QML for Finance (Hybrid Approach)

Select a **classical ML method** and replicate its results using a **quantum method**.

**Classical approaches to consider:**
- Neural Network
- PCA
- Autoencoder
- etc.

**References:**
- Stamatopoulos et al. (2020): [*Option Pricing using Quantum Computers*](https://doi.org/10.22331/q-2020-07-06-291), Quantum 4, 291
- Wang and Kan (2024): [*Option pricing under stochastic volatility on a quantum computer*](https://doi.org/10.22331/q-2024-10-23-1504), Quantum 8, 1504
- Herman et al. (2026): [*Quantum Speedups for Derivative Pricing Beyond Black-Scholes*](https://arxiv.org/abs/2602.03725), arXiv:2602.03725
- Zoufal et al. (2019): [*Quantum Generative Adversarial Networks for learning and loading random distributions*](https://doi.org/10.1038/s41534-019-0223-2), npj Quantum Information 5, 103

---

*© OP Pohjola / OP-Luottamuksellinen*
