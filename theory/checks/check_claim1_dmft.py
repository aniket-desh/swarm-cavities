"""Check of claim1.md eqs. (4)-(8): two-population DMFT with memory against direct simulation.

Compares four versions of the single-site process with a direct simulation at N = 4000:
    correct                 full correlation <s s> in the noise, reaction term sum_{t'<t} R(t,t') s(t')
    spec noise              connected correlation <ds ds> in the noise (as written in docs/spec.md)
    spec time index         reaction term sum_{t'<t} R(t,t'+1) s(t') (as written in docs/spec.md)
    no reaction             gamma set to zero in the single-site process
Also measures R(t, t') in the direct simulation by a random-sign field kick at one round.

Run: uv run python theory/checks/check_claim1_dmft.py [--stochastic]
"""
import sys
import numpy as np
from toy import make_labels, sample_J, simulate, dmft_mc, empirical_moments

stochastic = "--stochastic" in sys.argv
rho = [0.6, 0.4]
Jbar = [[0.5, -0.3], [0.8, 0.2]]
g = [[1.0, 0.6], [0.8, 1.2]]
gamma = [[0.6, 0.5], [0.5, -0.4]]
mu, sig = [0.3, -0.2], [0.5, 0.3]
beta, kappa, lam = [1.2, 0.9], [0.4, 0.0], 0.5
T, N, n_dis, M = 10, 4000, 40, 400_000
P = 2

rng = np.random.default_rng(0)
labels = make_labels(N, rho)
ms, Cs, Rs = [], [], []
t_kick, eps = 3, 1e-4
for d in range(n_dis):
    J = sample_J(rng, labels, Jbar, g, gamma)
    hpriv = np.asarray(mu)[labels] + np.asarray(sig)[labels] * rng.standard_normal(N)
    s = simulate(np.random.default_rng(1000 + d), J, labels, hpriv, beta, kappa, lam, T, stochastic)
    mm, CC = empirical_moments(s, labels, P)
    ms.append(mm); Cs.append(CC)
    if not stochastic:
        # diagonal response: kick h_i(t_kick) by eps r_i, r_i = +-1; R(t, t_kick) = <r_i ds_i(t)> / eps
        r = rng.choice([-1.0, 1.0], N)
        theta = np.zeros((T, N)); theta[t_kick] = eps * r
        s2 = simulate(np.random.default_rng(1000 + d), J, labels, hpriv, beta, kappa, lam, T, False, theta=theta)
        Rs.append(np.stack([((s2 - s) * r)[:, labels == a].mean(1) / eps for a in range(P)]))
ms, Cs = np.array(ms), np.array(Cs)
m_sim, m_sem = ms.mean(0), ms.std(0, ddof=1) / np.sqrt(n_dis)
C_sim, C_sem = Cs.mean(0), Cs.std(0, ddof=1) / np.sqrt(n_dis)

variants = {
    "correct": dict(),
    "spec noise (connected)": dict(cov="connected"),
    "spec time index": dict(echo_shift=1),
    "no reaction": dict(reaction=False),
}
out = {}
for name, kw in variants.items():
    out[name] = dmft_mc(np.random.default_rng(1), rho, Jbar, g, gamma, mu, sig, beta, kappa, lam, T, M,
                        stochastic=stochastic, **kw)

np.set_printoptions(precision=4, suppress=True, linewidth=150)
print(f"stochastic={stochastic}  N={N}  disorder samples={n_dis}  DMFT trajectories M={M}  T={T}")
print("simulation SEM (max over t and population): m", m_sem.max().round(4),
      " C(t,t)", np.diagonal(C_sem, axis1=1, axis2=2).max().round(4),
      " C(t,t-2)", np.diagonal(C_sem, offset=2, axis1=1, axis2=2).max().round(4))
tt = np.arange(T + 1)
for a in range(P):
    print(f"\npopulation {a}:  t =", tt[[2, 4, 6, 8, 10]])
    print("  sim   m(t)      ", m_sim[a, [2, 4, 6, 8, 10]])
    for name in variants:
        print(f"  {name:24s} m", out[name]["m"][a, [2, 4, 6, 8, 10]])
    print("  sim   C(t,t)    ", np.diag(C_sim[a])[[2, 4, 6, 8, 10]])
    for name in variants:
        print(f"  {name:24s} C", np.diag(out[name]["C"][a])[[2, 4, 6, 8, 10]])
print("\nmax abs error over t >= 1 and both populations   [m(t), C(t,t), C(t,t-2)]")
for name in variants:
    o = out[name]
    em = np.abs(o["m"] - m_sim)[:, 1:].max()
    ed = np.abs(np.diagonal(o["C"] - C_sim, axis1=1, axis2=2))[:, 1:].max()
    e2 = np.abs(np.diagonal(o["C"] - C_sim, offset=2, axis1=1, axis2=2)).max()
    print(f"  {name:24s} {em:.4f}  {ed:.4f}  {e2:.4f}")
if not stochastic:
    R_sim = np.mean(Rs, 0)
    print(f"\nresponse to a kick at t'={t_kick}: R(t, t') for t = t'+1 .. t'+5")
    for a in range(P):
        print(f"  pop {a} sim  ", R_sim[a, t_kick + 1 : t_kick + 6])
        print(f"  pop {a} DMFT ", out["correct"]["R"][a, t_kick + 1 : t_kick + 6, t_kick])
