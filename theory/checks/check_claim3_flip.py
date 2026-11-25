"""Check of claim3.md section 5: does PR fall before the benign mean flips at a zealot-driven saddle-node?

Stochastic agents: benign s_i = +-1 with P(+1) = (1 + tanh(beta h_i + hpriv_i)) / 2, zealots frozen at -1,
J_ij = J0/N + (g/sqrt(N)) z_ij. For each zealot fraction rho, 2000 replicas share couplings, private fields and
zealot set and differ in sampling noise. At round T the script reports, over benign agents only:
    PR / N_B              (U-statistic estimator)            and the mean-mode prediction eq. (16) (stationary, g = 0)
    N_B Var(m_B)          variance of the benign mean across replicas, and its prediction <v> / (1 - lam^2)
    flipped               fraction of replicas with m_B(T) < 0
The same two statistics are also computed over the replicas that have not flipped (m_B(T) > 0).
Averages over 8 draws of (J, hpriv).

Run: uv run python theory/checks/check_claim3_flip.py
"""
import numpy as np
from toy import sample_J
from pr_estimators import pr_ustat

beta, J0, mu, sig = 1.5, 1.0, 0.3, 0.2
R, n_dis = 2000, 8
e, we = np.polynomial.hermite_e.hermegauss(48); we = we / we.sum()


def mean_field(rho):
    """g = 0 fixed point on the benign-majority branch: m, lam = d m(t+1)/d m(t), <v>, <v^2>, kappa^2."""
    m = 1.0
    for _ in range(5000):
        th = np.tanh(beta * J0 * ((1 - rho) * m - rho) + mu + sig * e)
        m = we @ th
    r = beta * (1 - th**2); v = 1 - th**2
    return m, J0 * (1 - rho) * (we @ r), we @ v, we @ v**2, (J0 * (1 - rho)) ** 2 * (we @ r**2)


for N, g, gamma in [(32, 0.0, 0.0), (128, 0.0, 0.0), (32, 0.3, 0.5)]:
    print(f"\nN={N} g={g} gamma={gamma} beta={beta} J0={J0} mu={mu} sigma_priv={sig}; {R} replicas x {n_dis} disorder draws")
    print("  rho | lam   | PR/N_B all replicas:     | unflipped only | N_B Var(m_B) all:      | unflipped only | flipped fraction")
    print("      |       | eq.(16)  T=10   T=40     |  T=10   T=40   |   pred   T=10   T=40   |  T=10   T=40   |  T=10   T=40")
    for nz in sorted({int(round(rho * N)) for rho in [0.0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]}):
        nb = N - nz
        m, lam, v1, v2, k2 = mean_field(nz / N)
        on_branch = m > 0
        Lam1 = k2 * v1 / (1 - lam**2) if on_branch and lam < 1 else np.nan
        pr_pred = v1**2 / (v2 + Lam1**2 / nb)
        rows = []
        for d in range(n_dis):
            rng = np.random.default_rng(100 * d + N)
            J = sample_J(rng, np.zeros(N, int), [[J0]], [[g]], [[gamma]])
            hp = mu + sig * rng.standard_normal(N)
            S = np.where(rng.random((R, N)) < (1 + np.tanh(hp)) / 2, 1.0, -1.0)
            S[:, :nz] = -1.0
            row = []
            for t in range(1, 41):
                P = (1 + np.tanh(beta * S @ J.T + hp)) / 2
                S = np.where(rng.random((R, N)) < P, 1.0, -1.0); S[:, :nz] = -1.0
                if t in (10, 40):
                    mB = S[:, nz:].mean(1)
                    keep = mB > 0          # replicas still on the benign-majority side
                    pr_keep = pr_ustat(S[keep][:, nz:]) / nb if keep.sum() >= 100 else np.nan
                    row += [pr_ustat(S[:, nz:]) / nb, nb * mB.var(), (mB < 0).mean(), pr_keep, nb * mB[keep].var() if keep.sum() >= 100 else np.nan]
            rows.append(row)
        r = np.nanmean(rows, 0)
        print(f" {nz/N:.3f}| {lam:.3f} |   {pr_pred:.3f}   {r[0]:.3f}  {r[5]:.3f} |   {r[3]:.3f}  {r[8]:.3f} |"
              f"   {v1/(1-lam**2) if on_branch and lam < 1 else np.nan:6.2f} {r[1]:6.2f} {r[6]:6.2f} |  {r[4]:6.2f} {r[9]:6.2f} |  {r[2]:.3f}  {r[7]:.3f}")
