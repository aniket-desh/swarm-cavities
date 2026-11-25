"""Reference numbers quoted in theory/solver_spec.md. Each block is an independent calculation that
theory/dmft_solver.py must reproduce.

Run: uv run python theory/checks/reference_values.py
"""
import numpy as np
from math import comb
from toy import dmft_mc
from two_site import linear_single_site, four_point, pr_over_n
from static_cavity import fixed_point, saddle_nodes

np.set_printoptions(precision=5, suppress=True, linewidth=160)

# ---- A1. gamma = 0, memoryless mock Ising agent: exact Gaussian integration --------------------------------
# s(t+1) = tanh(beta (Jbar m(t) + eta(t)) + hpriv), s(0) = tanh(hpriv), <eta(t) eta(t')> = g^2 C(t,t')
beta, g, Jbar, mu, sig, T = 1.2, 1.0, 0.5, 0.3, 0.5, 6
x, w = np.polynomial.hermite_e.hermegauss(60); w = w / w.sum()
hp = mu + sig * x
m = np.zeros(T + 1); C = np.zeros((T + 1, T + 1))
m[0] = w @ np.tanh(hp); C[0, 0] = w @ np.tanh(hp) ** 2


def state(t, z):
    """s(t) as a function of the standard-normal deviate z of eta(t-1) and of hpriv (grid [z, hp])."""
    return np.tanh(beta * (Jbar * m[t - 1] + g * np.sqrt(C[t - 1, t - 1]) * z[:, None]) + hp[None, :])


for t in range(T):
    S = state(t + 1, x)                                   # [z, hp]
    m[t + 1] = w @ S @ w
    C[t + 1, t + 1] = w @ S**2 @ w
    C[t + 1, 0] = C[0, t + 1] = (w @ S) @ (w * np.tanh(hp))
    for tp in range(t):                                   # C(t+1, tp+1): eta(t), eta(tp) correlated with coefficient c
        c = C[t, tp] / np.sqrt(C[t, t] * C[tp, tp])
        Sp = state(tp + 1, x)
        # z_tp = c z_t + sqrt(1-c^2) u : integrate over (z_t, u, hp)
        acc = 0.0
        for iu, u in enumerate(x):
            Sp_u = np.tanh(beta * (Jbar * m[tp] + g * np.sqrt(C[tp, tp]) * (c * x[:, None] + np.sqrt(1 - c * c) * u)) + hp[None, :])
            acc += w[iu] * (w @ (S * Sp_u) @ w)
        C[t + 1, tp + 1] = C[tp + 1, t + 1] = acc
print("A1. gamma=0 memoryless, beta=1.2 g=1 Jbar=0.5 mu=0.3 sigma=0.5 (exact Gaussian integration)")
print("  t         ", np.arange(T + 1))
print("  m(t)      ", m)
print("  C(t,t)    ", np.diag(C))
print("  C(t,t-1)  ", np.diag(C, 1))
print("  C(t,0)    ", C[:, 0])
o = dmft_mc(np.random.default_rng(3), [1.0], [[Jbar]], [[g]], [[0.0]], [mu], [sig], [beta], [0.0], 0.0, T, 400_000)
print("  Monte Carlo (M=4e5) max |diff|:  m %.5f   C %.5f" % (np.abs(o["m"][0] - m).max(), np.abs(o["C"][0] - C).max()))

# ---- A2. linear map, closed forms -----------------------------------------------------------------------------
a, gam, n = 0.55, 0.6, 12
R, v, wt = linear_single_site([a], 1.0, gam, [1.0], n)
cat = [comb(2 * k, k) // (k + 1) for k in range(5)]
print("\nA2. linear x(t+1) = a J x + xi, a=0.55 g=1 gamma=0.6")
print("  R(t0+lag, t0), lag 1,3,5,7: recursion ", R[0, [1, 3, 5, 7], 0], "  closed form a^(2k+1) gamma^k Cat_k ",
      np.array([a ** (2 * k + 1) * gam**k * cat[k] for k in range(4)]))
print("  R at even lags (must vanish):", R[0, [2, 4, 6], 0])
print("  v(t), t=1..5:", np.diag(v[0])[1:6], "  v(3) closed form 1 + x + x^2 (1 + gamma^2) =", 1 + a**2 + a**4 * (1 + gam**2))
fp = four_point(R, v, wt, 1.0, gam); pr, psi = pr_over_n(v, wt, fp["SS"])
print("  PR(t)/N, t=1..5 and t=11:", pr[1:6], pr[11])
xg = 0.45**2
print("  gamma=1, a=0.45: stationary variance (1 - sqrt(1-4x))/(2x) = %.5f;  gamma=0, a=0.8: 1/(1-x) = %.5f, PR/N = 1-x^2 = %.5f"
      % ((1 - np.sqrt(1 - 4 * xg)) / (2 * xg), 1 / (1 - 0.64), 1 - 0.64**2))

# ---- A3. Novikov (regression) response against the tangent response -------------------------------------------
o = dmft_mc(np.random.default_rng(4), [1.0], [[0.3]], [[1.0]], [[0.5]], [0.2], [0.5], [1.0], [0.3], 0.5, 8, 400_000)
K = 1.0 * o["C"][0][:8, :8]                              # <eta eta> = g^2 C
eta, s = o["eta"][0], o["s"][0]
Rn = np.zeros((9, 8))
for t in range(1, 9):
    Rn[t, :t] = np.linalg.solve(K[:t, :t], eta[:t] @ s[t] / s.shape[1])
print("\nA3. response estimators, one population with memory (beta=1 g=1 gamma=0.5 Jbar=0.3 mu=0.2 sigma=0.5 kappa=0.3 lam=0.5)")
print("  tangent  R(8, t'), t'=3..7:", o["R"][0][8, 3:8])
print("  Novikov  R(8, t'), t'=3..7:", Rn[8, 3:8])
print("  max |diff| over all (t,t'): %.4f   condition number of the noise covariance: %.0f" % (np.abs(Rn - o["R"][0][:, :8]).max(), np.linalg.cond(K)))

# white-probe estimator: inject theta(t) ~ N(0, eps^2) into the field, R(t,t') = <s(t) theta(t')> / eps^2
for eps, Mp in [(0.3, 400_000), (0.3, 2000), (0.3, 500)]:
    errs, bias = [], None
    for seed in range(1 if Mp > 10_000 else 20):
        op = dmft_mc(np.random.default_rng(50 + seed), [1.0], [[0.3]], [[1.0]], [[0.5]], [0.2], [0.5], [1.0], [0.3], 0.5, 8, Mp, probe_eps=eps)
        Rp = np.einsum("tm,um->tu", op["s"][0], op["theta"][0]) / Mp / eps**2 * np.tri(9, 8, -1)
        errs.append(Rp - o["R"][0][:, :8])
    errs = np.array(errs)
    print("  white probe eps=%.1f M=%d:  R(8,7) = %.4f (unprobed tangent %.4f);  rms error per entry %.4f;  predicted sqrt(<s^2>)/(eps sqrt(M)) = %.4f"
          % (eps, Mp, (errs[0] + o["R"][0][:, :8])[8, 7], o["R"][0][8, 7], np.sqrt((errs[:, 1:, :] ** 2).mean()), np.sqrt(o["C"][0][8, 8]) / (eps * np.sqrt(Mp))))

# ---- A4. static solver: small-g expansion and zealot thresholds ------------------------------------------------
b, mu4, s4, g4, ga4 = 1.0, 0.5, 0.3, 0.2, 0.5
fp = fixed_point(b, g4, ga4, mu4, s4, m_init=0.5)
e, we = np.polynomial.hermite_e.hermegauss(60); we = we / we.sum()
ph = np.tanh(mu4 + s4 * e); d1 = 1 - ph**2
m0, q0 = we @ ph, we @ ph**2
m1 = m0 + g4**2 * b**2 * (0.5 * q0 * (we @ (-2 * ph * d1)) + ga4 * (we @ d1) * (we @ (ph * d1)))
print("\nA4. static fixed point, beta=1 g=0.2 gamma=0.5 mu=0.5 sigma=0.3:  m = %.5f  q = %.5f  chi = %.5f   (g=0: m = %.5f; second order in g: m = %.5f)"
      % (fp["m"], fp["q"], fp["chi"], m0, m1))
# homogeneous Curie-Weiss with zealots, closed form: beta (1-rho)(1-m^2) = 1 and atanh(m) = beta((1-rho) m - rho) + mu
bz, muz = 1.5, 0.3
f = lambda r: np.arctanh(np.sqrt(1 - 1 / (bz * (1 - r)))) - (bz * ((1 - r) * np.sqrt(1 - 1 / (bz * (1 - r))) - r) + muz)
lo, hi = 0.0, 1 - 1 / bz - 1e-9
for _ in range(80):
    mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) * f(lo) > 0 else (lo, mid)
sn = saddle_nodes(bz, 0.0, 0.0, muz, 0.0, 1.0)
print("  zealots, g=0, sigma=0, beta=1.5 mu=0.3: closed-form rho_up = %.5f (m_c = %.5f);  solver rho_up = %.5f, rho_down = %.5f"
      % (lo, np.sqrt(1 - 1 / (bz * (1 - lo))), sn[0], sn[2]))
