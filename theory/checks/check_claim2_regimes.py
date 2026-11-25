"""What happens above the instability as gamma -> 1 (claim2.md section 4), and what T = 10, N <= 128 can resolve (section 5).

Part A (large N, deterministic, no private fields): s(t+1) = tanh(beta J s(t)), g = 1, beta = 2.
    For each gamma: lag displacements A1 = <(s(t+1)-s(t))^2>, A2 = <(s(t+2)-s(t))^2>, the largest Lyapunov exponent,
    and the even-lag correlation time tau (first even lag with c(tau)/c(0) < 1/e).
Part B (N = 64, T = 10, noisy agents with private fields): s(t+1) = tanh(beta h(t) + hpriv + sig_xi xi).
    Replica observables defined in claim2.md section 5, as a function of gamma, below and above the instability.

Part C (deterministic, no fields): number of rounds until an exact 2-cycle is reached, against N and gamma.

Run: uv run python theory/checks/check_claim2_regimes.py [A|B|C]
"""
import sys
import numpy as np
from toy import sample_J

part = sys.argv[1] if len(sys.argv) > 1 else "AB"
np.set_printoptions(precision=3, suppress=True, linewidth=160)

if "A" in part:
    N, beta, burn, rec = 2000, 2.0, 3000, 5000
    labels = np.zeros(N, int)
    print(f"Part A: N={N} beta={beta} g=1, burn-in {burn}, record {rec} rounds, 2 disorder samples per gamma")
    print(" gamma | q      A1/q    A2/q   | Lyapunov | tau_even (rounds) | c(tau)/c(0) at tau = 2, 10, 100, 1000")
    for gamma in [-1.0, -0.5, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rows = []
        for d in range(2):
            rng = np.random.default_rng(100 * d + 7)
            J = beta * sample_J(rng, labels, [[0.0]], [[1.0]], [[gamma]])
            s = rng.uniform(-1, 1, N)
            v = rng.standard_normal(N); v /= np.linalg.norm(v)
            for t in range(burn):
                s = np.tanh(J @ s)
            S = np.empty((rec, N)); lyap = 0.0
            for t in range(rec):
                S[t] = s
                s_new = np.tanh(J @ s)
                v = (1 - s_new**2) * (J @ v)          # tangent map: delta s(t+1) = D(t) J delta s(t)
                nv = np.linalg.norm(v); lyap += np.log(nv); v /= nv
                s = s_new
            q = np.mean(S**2)
            A1 = np.mean((S[1:] - S[:-1]) ** 2); A2 = np.mean((S[2:] - S[:-2]) ** 2)
            lags = np.arange(0, 3001, 2)
            c = np.array([np.mean(S[: rec - L] * S[L:]) for L in lags]) / q
            below = np.nonzero(c < np.exp(-1))[0]
            tau = lags[below[0]] if len(below) else np.inf
            rows.append([q, A1 / q, A2 / q, lyap / rec, tau, c[1], c[5], c[50], c[500]])
        r = np.mean(rows, 0)
        print(f" {gamma:5.2f} | {r[0]:.3f}  {r[1]:.4f}  {r[2]:.4f} | {r[3]:+.4f} | {[x[4] for x in rows]} | {r[5:]}")

if "B" in part:
    N, T, sig, sig_xi, nJ = 64, 10, 0.5, 0.3, 400
    labels = np.zeros(N, int)

    def run(rng, J, hp, s0, beta):
        s = np.empty((T + 1, N)); s[0] = s0
        for t in range(T):
            s[t + 1] = np.tanh(beta * J @ s[t] + hp + sig_xi * rng.standard_normal(N))
        return s

    print(f"\nPart B: N={N} T={T} g=1 sigma_priv={sig} sigma_xi={sig_xi}, {nJ} graphs per cell; mean (SEM over graphs)")
    print(" beta gamma | dq = q_same - q_diff | amplification D_ab/D_floor | A1      A2      A2/A1 | frozen index 1 - A2/D_ab")
    for beta in [0.4, 2.0]:
        for gamma in [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0]:
            rows = []
            for d in range(nJ):
                rng = np.random.default_rng(d)
                hp = sig * rng.standard_normal(N)
                s0 = np.tanh(hp)
                J = sample_J(rng, labels, [[0.0]], [[1.0]], [[gamma]])
                J2 = sample_J(rng, labels, [[0.0]], [[1.0]], [[gamma]])
                a, b, c = run(rng, J, hp, s0, beta), run(rng, J, hp, s0, beta), run(rng, J2, hp, s0, beta)
                # one-step noise floor: redraw the last round of replica a from the same s_a(T-1)
                a_re = np.tanh(beta * J @ a[T - 1] + hp + sig_xi * rng.standard_normal(N))
                q_same, q_diff = np.mean(a[T] * b[T]), np.mean(a[T] * c[T])
                D_ab, D_floor = np.mean((a[T] - b[T]) ** 2), np.mean((a[T] - a_re) ** 2)
                A1, A2 = np.mean((a[T] - a[T - 1]) ** 2), np.mean((a[T] - a[T - 2]) ** 2)
                rows.append([q_same - q_diff, D_ab, D_floor, A1, A2])
            r = np.array(rows); mu = r.mean(0); se = r.std(0, ddof=1) / np.sqrt(nJ)
            amp = mu[1] / mu[2]
            print(f" {beta:4.1f} {gamma:5.2f} | {mu[0]:+.3f} ({se[0]:.3f}) | {amp:6.2f} | {mu[3]:.3f}  {mu[4]:.3f}  {mu[4]/mu[3]:.2f} | {1 - mu[4]/mu[1]:+.2f}")

if "C" in part:
    beta = 2.0
    print(f"\nPart C: rounds until max_i |s_i(t+2) - s_i(t)| < 1e-6 (median over samples; inf = not reached), beta={beta}, g=1")
    print(" gamma | " + "  ".join(f"N={N:5d}" for N in [125, 250, 500, 1000, 2000]))
    for gamma in [0.7, 0.8, 0.9, 0.95, 1.0]:
        meds = []
        for N in [125, 250, 500, 1000, 2000]:
            t_max, n_s = (20000, 8) if N <= 1000 else (8000, 4)
            times = []
            for d in range(n_s):
                rng = np.random.default_rng(1000 * d + N)
                J = beta * sample_J(rng, np.zeros(N, int), [[0.0]], [[1.0]], [[gamma]])
                s = rng.uniform(-1, 1, N); t_freeze = np.inf
                for t in range(0, t_max, 10):
                    for _ in range(10):
                        s = np.tanh(J @ s)
                    if np.max(np.abs(np.tanh(J @ np.tanh(J @ s)) - s)) < 1e-6:
                        t_freeze = t + 10
                        break
                times.append(t_freeze)
            meds.append(np.median(times))
        print(f" {gamma:5.2f} | " + "  ".join(f"{m:7.0f}" for m in meds))
