"""Check of claim4.md: zealot saddle-node, sign of the gamma effect, hysteresis width.

Model: fraction rho of zealots frozen at s = -1; benign agents s(t+1) = tanh(beta h(t) + hpriv), hpriv ~ N(mu, sig^2),
J_ij = J0/N + (g/sqrt(N)) z_ij with Corr(z_ij, z_ji) = gamma.
  1. Static cavity (eqs. (2)-(4)): forward threshold rho_up, backward threshold rho_down, width, for a (g, gamma) grid,
     and the first-order formula eq. (7) for the shift of rho_up.
  2. Direct simulation, adiabatic sweep of rho up and then down at N = 3000.

Run: uv run python theory/checks/check_claim4_zealots.py [g values, default 0 0.3 0.45]
"""
import sys
import numpy as np
from toy import sample_J
from static_cavity import saddle_nodes, fixed_point, _nodes

beta, J0, mu, sig = 1.5, 1.0, 0.3, 0.2
N, T_step, d_rho = 3000, 150, 0.005


def first_order_shift(g, gamma):
    """eq. (7): rho_up(g, gamma) - rho_up(0) = g^2 (1 - rho_c) df / (1 + m_c), evaluated at the g = 0 saddle-node."""
    rho_c, m_c, _, _ = saddle_nodes(beta, 0.0, 0.0, mu, sig, J0)
    e, we = _nodes(64, 48)[2:]
    x = beta * J0 * ((1 - rho_c) * m_c - rho_c) + mu + sig * e
    phi = np.tanh(x); d1 = 1 - phi**2; d2 = -2 * phi * d1
    Q = (1 - rho_c) * (we @ phi**2) + rho_c
    df = beta**2 * (0.5 * Q * (we @ d2) + (1 - rho_c) * gamma * (we @ d1) * (we @ (phi * d1)))
    return rho_c, g**2 * (1 - rho_c) * df / (1 + m_c)


def sweep(g, gamma, seed):
    """Adiabatic sweep: raise rho in steps of d_rho until the benign mean turns negative, then lower it again."""
    rng = np.random.default_rng(seed)
    J = sample_J(rng, np.zeros(N, int), [[J0]], [[g]], [[gamma]])
    hp = mu + sig * rng.standard_normal(N)
    order = rng.permutation(N)
    s = np.tanh(hp)

    def relax(s, n_z):
        zeal = order[:n_z]
        s = s.copy(); s[zeal] = -1.0
        for t in range(T_step):
            s = np.tanh(beta * J @ s + hp); s[zeal] = -1.0
        ben = order[n_z:]
        return s, s[ben].mean()

    rho_up = rho_down = None
    rho = 0.0
    while rho < 0.6:
        s, mB = relax(s, int(round(rho * N)))
        if mB < 0:
            rho_up = rho
            break
        rho += d_rho
    while rho >= 0:
        s, mB = relax(s, int(round(rho * N)))
        if mB > 0:
            rho_down = rho
            break
        rho -= d_rho
    return rho_up, (rho_down if rho_down is not None else 0.0)


print(f"beta={beta} J0={J0} mu={mu} sigma_priv={sig};  simulation N={N}, {T_step} rounds per step of {d_rho} in rho, 2 samples")
print("   g  gamma | cavity: rho_up  m_c    rho_down  width  Lam at rho_up | first order rho_up | simulation: rho_up  rho_down  width")
for g in ([float(x) for x in sys.argv[1:]] or [0.0, 0.3, 0.45]):
    for gamma in ([0.0] if g == 0 else [-0.5, 0.0, 0.5, 1.0]):
        sn = saddle_nodes(beta, g, gamma, mu, sig, J0)
        rc0, shift = first_order_shift(g, gamma)
        sims = np.array([sweep(g, gamma, seed) for seed in (1, 2)]).mean(0)
        if sn is None:
            print(f" {g:4.2f} {gamma:5.2f} |        no bistable window: the benign mean changes sign continuously          |"
                  f"      {rc0 + shift:.4f}        |      {sims[0]-d_rho/2:.4f}  {sims[1]+d_rho/2:.4f}   {sims[0]-sims[1]-d_rho:.4f}")
            continue
        fp = fixed_point(beta, g, gamma, mu, sig, J0, sn[0], m_init=sn[1])
        # the simulated thresholds are the first step at which the state has left the branch, so subtract half a step
        print(f" {g:4.2f} {gamma:5.2f} |        {sn[0]:.4f}  {sn[1]:+.3f}  {sn[2]:.4f}   {sn[0]-sn[2]:.4f}   {fp['Lam']:.3f}       |"
              f"      {rc0 + shift:.4f}        |      {sims[0]-d_rho/2:.4f}  {sims[1]+d_rho/2:.4f}   {sims[0]-sims[1]-d_rho:.4f}")
