"""Check of claim2.md eqs. (5)-(8): instability boundary of the synchronous fixed point with quenched private fields.

Model: s(t+1) = tanh(beta J s(t) + hpriv), J zero-mean with variance g^2/N and pair correlation gamma, hpriv ~ N(0, sig^2).
For each (gamma, beta) the script
  1. finds the fixed point of one N-agent sample by damped iteration,
  2. computes the eigenvalues of D J, D = diag(beta (1 - s*^2)), and reports the spectral radius and where on the
     circle the largest eigenvalue sits,
  3. compares with the cavity prediction for the spectral edge and with Lam = g^2 <chi_i^2> (stable iff Lam < 1),
  4. prints the two formulas that do not work: g (1+gamma) <d> (spec / literature.md) and its square-mean variant.

Run: uv run python theory/checks/check_claim2_boundary.py
"""
import numpy as np
from toy import sample_J
from static_cavity import fixed_point, spectral_radius, stability_param

N, g, sig, n_dis = 1200, 1.0, 0.7, 3
rng = np.random.default_rng(0)
labels = np.zeros(N, int)


def cavity_radius(beta, gamma):
    fp = fixed_point(beta, g, gamma, sig=sig)
    return spectral_radius(fp["d"], fp["W"], g, gamma), fp


def Lam_of(fp, gamma):
    """Eq. (8). gamma >= 0: static (and period-2) mode, dressed with the fixed point's own chi.
    gamma < 0: period-4 mode, dressed susceptibility at omega = pi/2 (solved separately, same d_i)."""
    return fp["Lam"] if gamma >= 0 else stability_param(fp["d"], fp["W"], g, gamma)


def beta_c(gamma, lo=0.3, hi=4.0):
    """beta at which the cavity stability parameter Lam (eq. 8) reaches 1."""
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fp = fixed_point(mid, g, gamma, sig=sig)
        stable = fp["converged"] and Lam_of(fp, gamma) < 1
        lo, hi = (mid, hi) if stable else (lo, mid)
    return hi


def sim_radius(beta, gamma):
    rads, angs, qs = [], [], []
    for d in range(n_dis):
        J = sample_J(rng, labels, [[0.0]], [[g]], [[gamma]])
        hp = sig * rng.standard_normal(N)
        s = np.tanh(hp)
        for it in range(20000):          # damped iteration: also converges past a flip (-1) or +-i instability
            s_new = np.tanh(beta * J @ s + hp)
            if np.max(np.abs(s_new - s)) < 1e-12:
                break
            s = 0.7 * s + 0.3 * s_new
        if np.max(np.abs(np.tanh(beta * J @ s + hp) - s)) > 1e-8:
            continue
        ev = np.linalg.eigvals((beta * (1 - s**2))[:, None] * J)
        top = ev[np.argmax(np.abs(ev))]
        rads.append(np.abs(top)); angs.append(abs(np.degrees(np.angle(top)))); qs.append(np.mean(s**2))
    return rads, angs, qs


print(f"N={N}  g={g}  sigma_priv={sig}  disorder samples={n_dis}")
print(" gamma  beta | radius: sim  cavity | |angle| of top eigenvalue (deg) | Lam (eq.8) | g(1+|gamma|)<d>  g(1+|gamma|)sqrt<d^2> | q: sim  cavity")
for gamma in [-0.6, 0.0, 0.5, 1.0]:
    bc = beta_c(gamma)
    for beta in [0.5, 0.7, 0.85 * bc, 0.95 * bc]:
        rad_th, fp = cavity_radius(beta, gamma)
        Lam = Lam_of(fp, gamma)
        rads, angs, qs = sim_radius(beta, gamma)
        naive1 = g * (1 + abs(gamma)) * fp["d_mean"]
        naive2 = g * (1 + abs(gamma)) * np.sqrt(fp["d2_mean"])
        print(f" {gamma:5.2f} {beta:5.3f} |       {np.mean(rads):.3f}  {rad_th:.3f} | {np.round(angs, 0)} | {Lam:.3f} | "
              f"{naive1:.3f}  {naive2:.3f} | {np.mean(qs):.4f} {fp['q']:.4f}  ({len(rads)}/{n_dis} converged)")
    fp = cavity_radius(bc, gamma)[1]
    print(f"   -> cavity boundary beta_c = {bc:.4f};  at beta_c: g(1+|gamma|)<d> = {g*(1+abs(gamma))*fp['d_mean']:.3f}, "
          f"g(1+|gamma|)sqrt<d^2> = {g*(1+abs(gamma))*np.sqrt(fp['d2_mean']):.3f}  (both would be 1 if those formulas were exact)")
    if gamma < 0:
        # the fixed point survives the +-i instability, so the simulated radius can be followed through 1
        b1, b2 = 0.95 * bc, 1.05 * bc
        r1, r2 = np.mean(sim_radius(b1, gamma)[0]), np.mean(sim_radius(b2, gamma)[0])
        print(f"   -> simulated radius crosses 1 at beta = {b1 + (1 - r1) * (b2 - b1) / (r2 - r1):.4f}")
