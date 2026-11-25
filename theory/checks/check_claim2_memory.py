"""Check of claim2.md eq. (4): own-history memory splits the static and the period-2 thresholds.

Agent s(t+1) = tanh(beta h(t) + kappa s(t)) near s = 0, so R0(omega) = beta / (e^{i omega} - kappa) and the linearized map
is kappa + beta J. Predicted: static mode grows iff g (1+gamma) beta / (1-kappa) > 1, the alternating mode iff
g (1+gamma) beta / (1+kappa) > 1. The script runs the nonlinear map from a small random state and reports the growth
factor per round and the lag-1 correlation of the growing pattern (+1: static, -1: alternating).

Run: uv run python theory/checks/check_claim2_memory.py
"""
import numpy as np
from toy import sample_J

N, g, gamma = 1500, 1.0, 0.5
rng = np.random.default_rng(0)
J = sample_J(rng, np.zeros(N, int), [[0.0]], [[g]], [[gamma]])
print(f"N={N} g={g} gamma={gamma}")
print(" kappa  beta | g(1+gamma)beta/(1-kappa)  g(1+gamma)beta/(1+kappa) | growth per round: sim  predicted | lag-1 correlation")
for kappa, beta in [(0.3, 0.43), (0.3, 0.50), (0.3, 0.80), (-0.3, 0.43), (-0.3, 0.50), (0.0, 0.62), (0.0, 0.70)]:
    s = 1e-9 * rng.standard_normal(N); logs = []
    for t in range(400):
        s_new = np.tanh(beta * J @ s + kappa * s)
        logs.append(np.log(np.linalg.norm(s_new) / np.linalg.norm(s)))
        c1 = s_new @ s / (np.linalg.norm(s_new) * np.linalg.norm(s))
        s = s_new * (1e-9 / np.linalg.norm(s_new))          # keep the state in the linear regime
    stat, alt = g * (1 + gamma) * beta / (1 - kappa), g * (1 + gamma) * beta / (1 + kappa)
    pred = max(abs(kappa + beta * g * (1 + gamma)), abs(kappa - beta * g * (1 + gamma)))
    print(f" {kappa:5.2f} {beta:5.2f} |          {stat:.3f}                    {alt:.3f}          |"
          f"       {np.exp(np.mean(logs[200:])):.3f}  {pred:.3f}     |   {c1:+.2f}")
