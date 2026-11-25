"""Check design_lib.dmft against theory/dmft_solver.forward_pass where the two must agree: no memory (kappa = 0),
and the Poisson-degree noise covariance g^2 C (fixed_indegree=False), with the same effective (Jbar, g, gamma).

The private fields differ slightly (design_lib uses the realized evidence fields of envs.evidence_game, the solver
a Gaussian with the same mean and s.d.), and the solver draws round-0 noise per copy, so agreement is expected at
the 0.01 level, not to Monte Carlo precision.

Run: uv run python experiments/04_swarm_design/check_dmft.py
"""
import numpy as np

import design_lib as dl
from theory.dmft_solver import Blocks, IsingAgent, forward_pass

np.set_printoptions(precision=4, suppress=True, linewidth=150)
T, K, p_neg = 8, 4, 0.5
ap, am = dl.GAINS[("inverted", "large")]
hp = dl.make_game(20000, seed=0, truth=1).h_priv
for gamma in (1.0, -1.0):
    e = dl.effective(ap, am, K, p_neg, gamma)
    mine = dl.dmft(K, gamma, p_neg, ap, am, T=T, kappa=0.0, fixed_indegree=False)
    ref = forward_pass(Blocks([1.0], e["Jbar"], e["g"], e["gamma_eff"]),
                       [IsingAgent(beta=1.0, w_priv=dl.A_P, bias=dl.C0, out_noise=dl.SIG_XI)],
                       [hp.mean()], [hp.std()], T, 20000, np.random.default_rng(1), copies=4)
    print(f"gamma_eff={e['gamma_eff']:+.2f}  Jbar={e['Jbar']:.2f}  g={e['g']:.3f}")
    print("  m(t)    design_lib", mine["m"][1:]); print("          solver    ", ref["m"][0][1:])
    print("  C(t,t)  design_lib", np.diag(mine["C"])[1:]); print("          solver    ", np.diag(ref["C"][0])[1:])
    print("  R(T,T-lag), lag 1..4  design_lib", mine["R"][T, T - 1::-1][:4], " solver", ref["R"][0][T, T - 1::-1][:4])
    print("  max |diff|: m %.4f  C(t,t) %.4f  R %.4f" % (np.abs(mine["m"] - ref["m"][0])[1:].max(),
          np.abs(np.diag(mine["C"]) - np.diag(ref["C"][0]))[1:].max(), np.abs(mine["R"] - ref["R"][0]).max()))
