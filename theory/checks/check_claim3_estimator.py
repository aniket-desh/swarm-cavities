"""Check of claim3.md section 6: finite-replica bias of the PR estimator, its correction, and the replica count at N = 32.

Draws R replicas of an N = 32 state with a known covariance (bulk + one common mode, set to give a target PR),
Gaussian or binary (+-1 from a thresholded Gaussian), and reports mean and sd of three estimators over 400 trials.

Run: uv run python theory/checks/check_claim3_estimator.py
"""
import numpy as np
from pr_estimators import pr_naive, pr_wishart, pr_ustat

N, trials = 32, 400
rng = np.random.default_rng(0)


def make_cov(pr_target):
    """Sigma = diag(v) + c u u^T with heterogeneous v; c tuned by bisection to hit the target PR."""
    v = rng.uniform(0.5, 1.0, N); u = rng.uniform(0.5, 1.0, N)
    pr = lambda c: np.trace(S := np.diag(v) + c * np.outer(u, u)) ** 2 / np.sum(S * S)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if pr(mid) > pr_target else (lo, mid)
    return np.diag(v) + lo * np.outer(u, u)


print(f"N={N}, {trials} trials per cell; entries are mean (sd) of the estimate")
for kind in ["gaussian", "binary"]:
    for pr_target in [4.0, 12.0, 24.0]:
        S = make_cov(pr_target)
        L = np.linalg.cholesky(S)
        draw = lambda R: rng.standard_normal((R, N)) @ L.T
        if kind == "binary":
            thr = rng.normal(0, 0.5, N)
            g_draw = draw
            draw = lambda R: np.where(g_draw(R) > thr, 1.0, -1.0)
            big = draw(400_000); Sb = np.cov(big.T); pr_true = np.trace(Sb) ** 2 / np.sum(Sb * Sb)
        else:
            pr_true = np.trace(S) ** 2 / np.sum(S * S)
        print(f"\n{kind}, true PR = {pr_true:.2f}")
        print("    R |  naive          n*PR/(n+PR) | Gaussian-corrected | U-statistic")
        for R in [16, 32, 64, 128, 256, 512]:
            est = np.array([[f(X) for f in (pr_naive, pr_wishart, pr_ustat)] for X in (draw(R) for _ in range(trials))])
            mu, sd = est.mean(0), est.std(0)
            n = R - 1
            print(f"  {R:3d} | {mu[0]:6.2f} ({sd[0]:5.2f})   {n*pr_true/(n+pr_true):6.2f}    | {mu[1]:6.2f} ({sd[1]:5.2f})     | {mu[2]:6.2f} ({sd[2]:5.2f})")
