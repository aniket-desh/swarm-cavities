"""Static cavity (fixed-point) equations for the deterministic mock agent, claim2.md eqs. (5)-(6), claim4.md eqs. (2)-(4).

One population of updating agents, optionally a fraction rho_z of zealots frozen at s = -1:
    s = tanh( beta * [ J0 ((1 - rho_z) m - rho_z) + g sqrt(Q) z + Gam chi s ] + hpriv ),   z ~ N(0,1), hpriv ~ N(mu, sig^2)
    m = <s>,  q = <s^2>,  Q = (1 - rho_z) q + rho_z,  Gam = (1 - rho_z) gamma g^2
    d = beta (1 - s^2)                     bare one-step gain of the agent
    chi_i = d / (1 - d Gam chi)            dressed static susceptibility of the agent in the swarm
    chi = <chi_i>
Stability parameter (claim2.md eq. (8)):  Lam = (1 - rho_z) g^2 <chi_i^2>;  the fixed point is stable iff Lam < 1.
"""
import numpy as np


def _nodes(n_z=64, n_h=48):
    """Gauss-Hermite nodes/weights for z ~ N(0,1) and for the private-field deviate."""
    z, wz = np.polynomial.hermite_e.hermegauss(n_z)
    e, we = np.polynomial.hermite_e.hermegauss(n_h)
    return z, wz / wz.sum(), e, we / we.sum()


def solve_site(a, k, s_init, iters=200):
    """Solve s = tanh(a + k s) by Newton from s_init (elementwise); selects the branch continuous with s_init."""
    s = s_init.copy()
    for _ in range(iters):
        f = np.tanh(a + k * s) - s
        df = k * (1 - np.tanh(a + k * s) ** 2) - 1
        s_new = np.clip(s - f / np.where(np.abs(df) < 1e-9, -1e-9, df), -1, 1)
        if np.max(np.abs(s_new - s)) < 1e-13:
            s = s_new
            break
        s = s_new
    return s


def fixed_point(beta, g, gamma, mu=0.0, sig=0.0, J0=0.0, rho_z=0.0, m_init=0.0, m_clamp=None,
                n_z=64, n_h=48, iters=4000, damp=0.5, tol=1e-11):
    """Solve the static cavity equations by damped iteration from m_init (which selects the branch).

    If m_clamp is given, the mean entering the field is held at m_clamp and only (q, chi) are solved; the returned
    m_out = <s> then gives the residual m_out - m_clamp of the mean equation, used to find all branches.
    """
    z, wz, e, we = _nodes(n_z, n_h if sig > 0 else 1)
    Z, E = np.meshgrid(z, e, indexing="ij")
    W = np.outer(wz, we)
    hp = mu + sig * E
    m = m_init if m_clamp is None else m_clamp
    q, chi = max(m * m, 0.3), 0.0
    s = np.full_like(Z, m)
    Gam = (1 - rho_z) * gamma * g**2
    for it in range(iters):
        Q = (1 - rho_z) * q + rho_z
        a = beta * (J0 * ((1 - rho_z) * m - rho_z) + g * np.sqrt(Q) * Z) + hp
        s = solve_site(a, beta * Gam * chi, s)
        d = beta * (1 - s**2)
        chi_i = d / (1 - d * Gam * chi)
        m_new, q_new, chi_new = (W * s).sum(), (W * s**2).sum(), (W * chi_i).sum()
        err = abs(q_new - q) + abs(chi_new - chi) + (abs(m_new - m) if m_clamp is None else 0.0)
        if m_clamp is None:
            m = (1 - damp) * m + damp * m_new
        q = (1 - damp) * q + damp * q_new
        chi = (1 - damp) * chi + damp * chi_new
        if err < tol:
            break
    Lam = (1 - rho_z) * g**2 * (W * chi_i**2).sum()
    return dict(m=m, m_out=m_new, q=q, chi=chi, Lam=Lam, d=d, W=W, s=s, converged=err < tol,
                d_mean=(W * d).sum(), d2_mean=(W * d**2).sum())


def branches(beta, g, gamma, mu, sig, J0, rho_z, n_grid=161):
    """All solutions m of the mean equation at zealot fraction rho_z: roots of <s>(m) - m, refined by bisection."""
    F = lambda m: fixed_point(beta, g, gamma, mu, sig, J0, rho_z, m_clamp=m, tol=1e-10)["m_out"] - m
    ms = np.linspace(-0.999, 0.999, n_grid)
    Fs = np.array([F(m) for m in ms])
    roots = []
    for i in np.nonzero(np.sign(Fs[:-1]) != np.sign(Fs[1:]))[0]:
        lo, hi = ms[i], ms[i + 1]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.sign(F(mid)) == np.sign(Fs[i]) else (lo, mid)
        roots.append(0.5 * (lo + hi))
    return roots


def saddle_nodes(beta, g, gamma, mu, sig, J0, n_grid=61, tol=2e-4):
    """Forward and backward critical zealot fractions (claim4.md eq. (5)).

    rho_up:   largest rho_z at which the benign-majority branch (largest root) still exists, raising rho_z.
    rho_down: smallest rho_z at which the captured branch (smallest root) still exists, lowering rho_z;
              0 if the captured state survives the removal of every zealot.
    Returns (rho_up, m at rho_up, rho_down, m at rho_down), or None if there is no bistable window.
    """
    top = lambda r: max(branches(beta, g, gamma, mu, sig, J0, r, n_grid))
    bot = lambda r: min(branches(beta, g, gamma, mu, sig, J0, r, n_grid))
    n3 = lambda r: len(branches(beta, g, gamma, mu, sig, J0, r, n_grid)) >= 3
    rs = np.linspace(0.0, 0.4, 21)
    bist = [r for r in rs if n3(r)]
    if not bist:
        return None
    lo, hi = max(bist), max(bist) + rs[1]          # forward saddle-node between the last bistable grid point and the next
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if n3(mid) else (lo, mid)
    rho_up, m_up = lo, top(lo)
    if min(bist) == 0.0:
        return rho_up, m_up, 0.0, bot(0.0)
    lo, hi = min(bist) - rs[1], min(bist)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); lo, hi = (lo, mid) if n3(mid) else (mid, hi)
    return rho_up, m_up, hi, bot(hi)


def _dressed(d, W, lam, ag, rho_b):
    """Physical root k of k = rho_b <d / (lam - ag k d)> (the smallest non-negative one, continuous with lam -> infinity).

    Returns (k, per-site d / (lam - ag k d)), or (None, None) if no root exists below the pole, which means lam lies
    inside the spectrum.
    """
    if ag == 0.0:
        return rho_b * (W * d).sum() / lam, d / lam
    f = lambda k: rho_b * (W * d / (lam - ag * k * d)).sum() - k
    k_pole = lam / (ag * d.max())
    ks = np.linspace(0.0, k_pole, 4001)[:-1]
    fs = np.array([f(k) for k in ks])
    neg = np.nonzero(fs < 0)[0]
    if len(neg) == 0:
        return None, None
    lo, hi = ks[neg[0] - 1], ks[neg[0]]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    k = 0.5 * (lo + hi)
    return k, d / (lam - ag * k * d)


def stability_param(d, W, g, gamma, rho_b=1.0):
    """Lam = rho_b g^2 <|chi_i(omega*)|^2> at the most unstable frequency (claim2.md eq. (8)); inf if already unstable.

    omega* = 0 and pi for gamma > 0 (static and period-2 modes), +-pi/2 for gamma < 0 (period-4 mode). In both cases
    |chi_i(omega*)| = d_i / (1 - |gamma| g^2 k d_i) with k = rho_b <d_i / (1 - |gamma| g^2 k d_i)>.
    """
    k, gi = _dressed(d, W, 1.0, abs(gamma) * g**2, rho_b)
    return np.inf if k is None else rho_b * g**2 * (W * gi**2).sum()


def spectral_radius(d, W, g, gamma, rho_b=1.0):
    """Edge of the spectrum of D J for J elliptic (variance g^2/N, pair correlation gamma), D = diag(d) (claim2.md eq. (7)).

    On the real axis (gamma >= 0) or the imaginary axis (gamma < 0, which maps onto the same equations with |gamma|):
        k = rho_b <d / (lam - |gamma| g^2 k d)>,   edge:  rho_b g^2 <(d / (lam - |gamma| g^2 k d))^2> = 1.
    """
    ag = abs(gamma) * g**2

    def outside(lam):
        k, gi = _dressed(d, W, lam, ag, rho_b)
        return k is not None and rho_b * g**2 * (W * gi**2).sum() < 1.0

    lo, hi = 1e-9, 4.0 * g * d.max() + 1e-9
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if outside(mid) else (mid, hi)
    return hi
