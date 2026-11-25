"""How PR/N and the replica variance behave as the linear swarm approaches the instability (claim3.md section 4).

x(t+1) = a J x(t) + xi(t), stationary covariance from Sigma <- A Sigma A^T + 1 iterated to convergence, N = 600.
The boundary is a_c = 1 / (g (1 + |gamma|)). For gamma = 0 the closed forms are vbar = 1/(1-x), PR/N = 1 - x^2,
x = (a g)^2 (eq. (14)).

Run: uv run python theory/checks/check_claim3_critical.py
"""
import numpy as np
from toy import sample_J

N, g = 600, 1.0
rng = np.random.default_rng(0)
print(f"N={N}, one sample per cell")
print(" gamma  a/a_c |  vbar    PR/N   | gamma=0 closed forms: vbar   PR/N")
for gamma in [0.0, 0.6, 1.0]:
    a_c = 1.0 / (g * (1 + abs(gamma)))
    J = sample_J(rng, np.zeros(N, int), [[0.0]], [[g]], [[gamma]])
    for frac in [0.5, 0.7, 0.8, 0.9, 0.95]:
        A = frac * a_c * J
        S = np.eye(N); prev = 0.0
        for t in range(3000):
            S = A @ S @ A.T + np.eye(N)
            tr = np.trace(S) / N
            if abs(tr - prev) < 1e-6 * tr:
                break
            prev = tr
        if t == 2999:
            print(f" {gamma:4.1f}  {frac:5.2f} | not stationary: this finite-N sample is already above the boundary")
            continue
        pr = np.trace(S) ** 2 / np.sum(S * S) / N
        x = (frac * a_c * g) ** 2
        extra = f"{1/(1-x):6.3f}  {1-x**2:.4f}" if gamma == 0 else ""
        print(f" {gamma:4.1f}  {frac:5.2f} | {tr:6.3f}  {pr:.4f} | {extra}")
