"""Check of claim4.md section 5: benign-plus-zealot transient from the dynamic two-population DMFT (claim1.md eqs. (4)-(8)).

Benign agents s(t+1) = tanh(beta h(t) + hpriv) start at s(0) = tanh(hpriv); zealots are frozen at -1.
Compares m_B(t), t <= T, from the single-site process (population 1 frozen) with direct simulation at N = 3000,
and reports the zealot fraction needed to flip the benign mean within T = 10 rounds against the static threshold.

Run: uv run python theory/checks/check_claim4_transient.py
"""
import numpy as np
from toy import make_labels, sample_J, simulate, dmft_mc

beta, J0, mu, sig, g, gamma, T, N, n_dis, M = 1.5, 1.0, 0.3, 0.2, 0.3, 0.5, 10, 3000, 6, 200_000
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"beta={beta} J0={J0} mu={mu} sigma_priv={sig} g={g} gamma={gamma}; simulation N={N}, {n_dis} samples; static rho_up = 0.219")
print("  rho  | m_B(t) at t = 2, 4, 6, 8, 10: DMFT / simulation                               | max |diff|")
for rho in [0.10, 0.20, 0.24, 0.28, 0.32]:
    pops = [1 - rho, rho]
    o = dmft_mc(np.random.default_rng(0), pops, [[J0, J0], [J0, J0]], [[g, g], [g, g]], [[gamma, gamma], [gamma, gamma]],
                [mu, -20.0], [sig, 0.0], [beta, beta], [0.0, 0.0], 0.0, T, M, frozen_pops=(1,))
    labels = make_labels(N, pops); ms = []
    for d in range(n_dis):
        rng = np.random.default_rng(10 + d)
        J = sample_J(rng, labels, [[J0, J0], [J0, J0]], [[g, g], [g, g]], [[gamma, gamma], [gamma, gamma]])
        hp = np.where(labels == 0, mu + sig * rng.standard_normal(N), -20.0)
        s = simulate(rng, J, labels, hp, [beta, beta], [0.0, 0.0], 0.0, T, frozen=labels == 1)
        ms.append(s[:, labels == 0].mean(1))
    ms = np.mean(ms, 0)
    print(f" {rho:.2f}  | {o['m'][0, [2, 4, 6, 8, 10]]} / {ms[[2, 4, 6, 8, 10]]} | {np.abs(o['m'][0] - ms).max():.4f}")
