"""Refrozen predictions for the revised kick estimators (experiments/07_kick/DESIGN.md revision 3, reviewer finding on
rule 2): F, F_open, F_open,matched and F - F_open,matched with the open-agent |s| >= 0.1 filter and the alignment split,
on surrogate v2 (theory/quiet_regime.md section 1) under two laws: linear response with noise sigma0 = 0.045 and
one-shot gains, and the anchoring law with target gains. Written by the design reviewer as a zero-cost check, extended
here to three independent batches of 16 graphs per cell (Monte Carlo s.e. = s.d. over batches / sqrt 3) and pooled
values (unweighted mean over the four |gamma| = 1 cells). Output: refrozen_predictions.json.
Run: uv run python experiments/07_kick/refreeze_predictions.py
"""
import json
import sys
from pathlib import Path
ROOT = Path("/workspace/swarm-cavity")
sys.path.insert(0, str(ROOT / "theory" / "checks"))
import numpy as np
import check_kick_echo as ck
sv = ck.sv
fit = sv.load_fit()

def cell(K, gamma, gains, law, n_graphs, rng):
    recs = []
    for g in range(n_graphs):
        J = ck.fixed_indegree_signed_couplings(ck.N, K, gamma, ck.P_NEG, rng)
        hp = ck.make_game(ck.N, mu=0.5, sigma=1.0, k=3, seed=int(rng.integers(2**31))).h_priv
        S = sv.simulate_v2(J, hp, gains, gains, law, ck.T, rng)
        for rep in range(2):
            for t0 in (6, 8):
                for i in range(ck.N):
                    if abs(S[rep, t0, i]) < 0.1: continue
                    tw = sv.kick_twin(S[rep], J, hp, gains, law, i, t0, rng, second_order=True, gains0=gains)
                    readers = list(tw["ds1"])
                    tw.update(i=i, J_ji={j: int(J[j, i]) for j in readers}, s0=S[rep, t0], s1=S[rep, t0 + 1],
                              d1=1 - S[rep, t0 + 1] ** 2, d2=1 - S[rep, t0 + 2] ** 2,
                              recip=[(j, int(J[i, j])) for j in readers if J[i, j] != 0],
                              open=[(k, [(j, int(J[k, j])) for j in readers if J[k, j] != 0]) for k in tw["ds2"]
                                    if k not in readers and abs(S[rep, t0, k]) >= 0.1])
                    recs.append(tw)
    return recs

def stats(recs):
    c = {}
    for ty in (1, -1):
        v = np.array([(r["ds1"][j], r["D"] * r["d1"][j]) for r in recs for j in r["ds1"] if r["J_ji"][j] == ty])
        c[ty] = v[:, 0] @ v[:, 1] / (v[:, 1] @ v[:, 1])
    y = np.array([r["echo"] for r in recs])
    p = np.array([r["d2"][r["i"]] * sum(c[ty] * r["ds1"][j] for j, ty in r["recip"]) for r in recs])
    al = np.array([p_ * r["s1"][r["i"]] > 0 for p_, r in zip(p, recs)])
    op = np.array([(r["ds2"][k], r["d2"][k] * sum(c[ty] * r["ds1"][j] for j, ty in snd), r["s1"][k])
                   for r in recs for k, snd in r["open"]]).reshape(-1, 3)
    oal = op[:, 1] * op[:, 2] > 0
    sl = lambda a, b: a @ b / (b @ b) if len(b) and b @ b > 0 else np.nan
    F, Fo = sl(y, p), sl(op[:, 0], op[:, 1])
    fc = np.sum(p[al] ** 2) / np.sum(p ** 2)
    Fom = sl(op[oal, 0], op[oal, 1]) if fc > 0.5 else sl(op[~oal, 0], op[~oal, 1])
    return F, Fo, Fom, fc


if __name__ == "__main__":
    out = {}
    for name in ("linear+noise", "anchoring"):
        law = (dict(rho0=0.0, rho_inf=0.0, u0=1.0, sig0=0.045, sig1=0.0, pi0=0.0, mu_pi=1.0) if name != "anchoring"
               else sv.law_swarm(fit, 2))
        for w in ("inverted", "label"):
            g = fit["gains_transfer"][f"{w}|K2"]["one_shot" if name != "anchoring" else "target"]
            for gamma in (-1.0, 1.0):
                batches = [stats(cell(2, gamma, g, law, 16, np.random.default_rng([11, b, int(gamma) + 1, len(w), len(name)])))
                           for b in range(3)]
                a = np.array(batches)            # (3, 4): F, F_open, F_open_matched, frac_confirming
                m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(3)
                diff = a[:, 0] - a[:, 2]
                key = f"{name}|{w}|g{gamma:+.0f}"
                out[key] = {"F": m[0], "F_se": se[0], "F_open": m[1], "F_open_se": se[1], "F_open_matched": m[2],
                            "F_open_matched_se": se[2], "F_minus_open_matched": diff.mean(),
                            "F_minus_open_matched_se": diff.std(ddof=1) / np.sqrt(3), "closed_frac_confirming": m[3]}
                o = out[key]
                print(f"{key:28s} F {o['F']:.3f}({o['F_se']:.3f}) F_open {o['F_open']:.3f}({o['F_open_se']:.3f}) "
                      f"matched {o['F_open_matched']:.3f} F-matched {o['F_minus_open_matched']:+.3f}"
                      f"({o['F_minus_open_matched_se']:.3f}) frac_conf {o['closed_frac_confirming']:.2f}", flush=True)
        cellsk = [k for k in out if k.startswith(name + "|")]
        pool = {s: float(np.mean([out[k][s] for k in cellsk])) for s in ("F", "F_open", "F_open_matched", "F_minus_open_matched")}
        out[f"{name}|pooled"] = pool
        print(f"{name} pooled (unweighted mean of the four cells):", {k: round(v, 3) for k, v in pool.items()}, flush=True)
    Path(__file__).with_name("refrozen_predictions.json").write_text(json.dumps(out, indent=1, default=float))
