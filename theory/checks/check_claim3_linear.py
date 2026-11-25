"""Check of claim3.md eqs. (6)-(13) and the closed form eq. (14) on the linear swarm x(t+1) = a J x(t) + sigma xi(t).

For a sampled J the replica covariance Sigma(t) = sum_{k<t} A^k Sigma_xi A^kT obeys Sigma(t+1) = A Sigma(t) A^T + Sigma_xi
exactly (A = diag(a_i) J), so Tr Sigma and Tr Sigma^2 are computed without sampling noise and compared with the
two-site cavity prediction.

Run: uv run python theory/checks/check_claim3_linear.py
"""
import numpy as np
from toy import sample_J
from two_site import four_point, pr_over_n, linear_single_site

np.set_printoptions(precision=4, suppress=True, linewidth=160)
N, n_dis = 1000, 4
labels = np.zeros(N, int)
rng = np.random.default_rng(0)


def exact(a_i, sig_i, g, gamma, n):
    out = []
    for d in range(n_dis):
        A = a_i[:, None] * sample_J(rng, labels, [[0.0]], [[g]], [[gamma]])
        S = np.zeros((N, N)); rows = []
        for t in range(n):
            tr, tr2, dg2 = np.trace(S) / N, np.sum(S * S) / N, np.sum(np.diag(S) ** 2) / N
            rows.append([tr, dg2, tr2 - dg2, tr**2 / tr2 if tr2 > 0 else np.nan])
            S = A @ S @ A.T + np.diag(sig_i**2)
        out.append(rows)
    return np.mean(out, 0), np.std(out, 0, ddof=1)       # columns: vbar, <v_i^2>, psi, PR/N


cases = [
    ("homogeneous, gamma=0  ", [0.8], [1.0], 1.0, 0.0, 12),
    ("homogeneous, gamma=0.6", [0.55], [1.0], 1.0, 0.6, 12),
    ("homogeneous, gamma=-0.6", [0.55], [1.0], 1.0, -0.6, 12),
    ("homogeneous, gamma=1  ", [0.45], [1.0], 1.0, 1.0, 12),
    ("two site types, gamma=0.5", [0.3, 0.8], [1.0, 0.5], 1.0, 0.5, 12),
]
for name, a, sig, g, gamma, n in cases:
    a, sig = np.array(a), np.array(sig)
    R, v, w = linear_single_site(a, g, gamma, sig, n)
    fp = four_point(R, v, w, g, gamma)
    pr, psi = pr_over_n(v, w, fp["SS"])
    types = np.arange(N) % len(a)
    ex, ex_sd = exact(a[types], sig[types], g, gamma, n)
    vd = np.einsum("ktt->kt", v)
    print(f"\n{name}: a={a} sigma={sig} g={g}   (N={N}, {n_dis} samples; cavity iterations {fp['iterations']})")
    ts = [1, 2, 3, 5, 8, 11]
    print("  t               ", ts)
    print("  vbar   exact    ", ex[ts, 0]); print("         cavity   ", (w @ vd)[ts])
    print("  psi    exact    ", ex[ts, 2]); print("         cavity   ", psi[ts])
    print("  PR/N   exact    ", ex[ts, 3]); print("         cavity   ", pr[ts])
    print("  max relative error over t>=2:  psi %.4f   PR/N %.4f    (sample-to-sample sd of exact psi at t=11: %.4f, relative %.4f)" %
          (np.nanmax(np.abs(psi[2:] / ex[2:, 2] - 1)), np.nanmax(np.abs(pr[2:] / ex[2:, 3] - 1)), ex_sd[11, 2], ex_sd[11, 2] / ex[11, 2]))

# frozen agents: a fraction 1 - rho_b of the agents never moves (x = 0); only the benign block of J matters
rho_b, a_, gam_, n = 0.6, 0.75, 0.5, 12
R, v, w = linear_single_site([a_], 1.0, gam_, [1.0], n, rho_b=rho_b)
fp = four_point(R, v, w, 1.0, gam_, rho_b=rho_b); pr, psi = pr_over_n(v, w, fp["SS"])
Nb = int(rho_b * 1500); vals = []
for d in range(3):
    A = a_ * sample_J(rng, np.zeros(1500, int), [[0.0]], [[1.0]], [[gam_]])[:Nb, :Nb]
    S = np.zeros((Nb, Nb))
    for t in range(n - 1):
        S = A @ S @ A.T + np.eye(Nb)
    vals.append([np.trace(S) / Nb, np.trace(S) ** 2 / np.sum(S * S) / Nb])
vals = np.mean(vals, 0)
print(f"\nfrozen fraction {1-rho_b:.1f}, a={a_}, gamma={gam_}, t=11:  vbar exact {vals[0]:.4f} cavity {(w @ np.einsum('ktt->kt', v))[11]:.4f};"
      f"  PR/N_b exact {vals[1]:.4f} cavity {pr[11]:.4f}")

# stationary closed form for gamma = 0, homogeneous: PR/N = 1 - (a g)^4   (eq. (14))
a, g = 0.8, 1.0
A = a * sample_J(rng, labels, [[0.0]], [[g]], [[0.0]])
S = np.zeros((N, N))
for t in range(200):
    S = A @ S @ A.T + np.eye(N)
print(f"\nstationary gamma=0: PR/N exact {np.trace(S)**2 / np.sum(S*S) / N:.4f}   1-(ag)^4 = {1 - (a*g)**4:.4f}"
      f"   vbar exact {np.trace(S)/N:.4f}  1/(1-(ag)^2) = {1/(1-(a*g)**2):.4f}")
