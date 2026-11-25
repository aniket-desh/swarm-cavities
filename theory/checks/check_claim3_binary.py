"""Check of claim3.md eqs. (3)-(4) and (15): equal-time two-site recursion for nonlinear stochastic agents at gamma = 0.

Agents: s_i = +-1 with P(+1) = (1 + tanh(beta h_i(t) + hpriv_i)) / 2, J zero-mean, variance g^2/N, gamma = 0.
Theory (nested single-site sampling; frozen part of the field shared by replicas, thermal part redrawn):
    mu_i(t+1) = E_y tanh(beta (eta_f(t) + g sqrt(1 - q(t)) y) + hpriv),   Cov(eta_f(t), eta_f(t')) = g^2 q(t, t')
    q(t, t') = <mu_i(t) mu_i(t')>,   v_i(t) = 1 - mu_i(t)^2,   r_i(t) = beta E_y [1 - tanh^2(.)]
    psi(t+1) = g^4 <r_i(t)^2>^2 ( <v_i(t)^2> + psi(t) ),   PR(t)/N = <v>^2 / (<v^2> + psi)
Simulation: N agents, R replicas with the same J and hpriv; U-statistic estimates of Tr Sigma and Tr Sigma^2.

Run: uv run python theory/checks/check_claim3_binary.py
"""
import numpy as np
from toy import sample_J
from pr_estimators import trace_moments_ustat

beta, g, sig, T = 1.2, 1.0, 0.5, 8
N, R, n_dis, M = 1000, 600, 4, 200_000
rng = np.random.default_rng(0)
y, wy = np.polynomial.hermite_e.hermegauss(40); wy = wy / wy.sum()

# theory
hp = sig * rng.standard_normal(M)
mu = np.zeros((T + 1, M)); mu[0] = np.tanh(hp)
eta = np.zeros((T, M)); zf = rng.standard_normal((T, M))
psi = np.zeros(T + 1); pr_th = np.zeros(T + 1)
for t in range(T):
    q = mu[: t + 1] @ mu[: t + 1].T / M                          # q(t, t') for t, t' <= t
    K = g**2 * q
    w = np.linalg.pinv(K[:t, :t], hermitian=True) @ K[:t, t] if t else np.zeros(0)
    eta[t] = w @ eta[:t] + np.sqrt(max(K[t, t] - K[:t, t] @ w, 0.0)) * zf[t]
    th = np.tanh(beta * (eta[t][:, None] + g * np.sqrt(1 - q[t, t]) * y[None, :]) + hp[:, None])
    v = 1 - mu[t] ** 2
    r = beta * ((1 - th**2) @ wy)
    pr_th[t] = v.mean() ** 2 / ((v**2).mean() + psi[t])
    psi[t + 1] = g**4 * (r**2).mean() ** 2 * ((v**2).mean() + psi[t])
    mu[t + 1] = th @ wy
v = 1 - mu[T] ** 2
pr_th[T] = v.mean() ** 2 / ((v**2).mean() + psi[T])

# simulation
pr_sim, psi_sim = [], []
for d in range(n_dis):
    J = sample_J(rng, np.zeros(N, int), [[0.0]], [[g]], [[0.0]])
    hpi = sig * rng.standard_normal(N)
    S = np.where(rng.random((R, N)) < (1 + np.tanh(hpi)) / 2, 1.0, -1.0)
    rows_pr, rows_psi = [], []
    for t in range(T + 1):
        t1sq, t2 = trace_moments_ustat(S)
        vi = S.var(0, ddof=1)
        # <v_i^2> estimated without bias for +-1 data is awkward; the plug-in bias is O(1/R) and negligible here
        rows_pr.append(t1sq / t2 / N); rows_psi.append(t2 / N - np.mean(vi**2))
        if t < T:
            S = np.where(rng.random((R, N)) < (1 + np.tanh(beta * S @ J.T + hpi)) / 2, 1.0, -1.0)
    pr_sim.append(rows_pr); psi_sim.append(rows_psi)
pr_sim, psi_sim = np.array(pr_sim), np.array(psi_sim)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
print(f"beta={beta} g={g} sigma_priv={sig} gamma=0;  simulation N={N}, R={R} replicas, {n_dis} disorder samples")
print("t            ", np.arange(T + 1))
print("psi   theory ", psi)
print("psi   sim    ", psi_sim.mean(0)); print("      sem    ", psi_sim.std(0, ddof=1) / np.sqrt(n_dis))
print("PR/N  theory ", pr_th)
print("PR/N  sim    ", pr_sim.mean(0)); print("      sem    ", pr_sim.std(0, ddof=1) / np.sqrt(n_dis))
print("max relative error of PR/N: %.4f" % np.max(np.abs(pr_sim.mean(0) / pr_th - 1)))
