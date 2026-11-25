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
                              # slope factors from the state before each response (D_LAG) or, as in stage 1's
                              # frozen numbers, from the same round's base draw
                              d1=1 - S[rep, t0 + (0 if D_LAG else 1)] ** 2, d2=1 - S[rep, t0 + (1 if D_LAG else 2)] ** 2,
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
    ratio = sl(op[oal, 0], op[oal, 1]) / sl(op[~oal, 0], op[~oal, 1])      # open confirm / open disconfirm
    return F, Fo, Fom, fc, ratio


D_LAG = False
LAWS = ("linear+noise", "anchoring")
SIG0 = {"linear+noise": 0.045}
CELLS = [("inverted", 2, -1.0), ("inverted", 2, 1.0), ("label", 2, -1.0), ("label", 2, 1.0)]
OUTFILE = "refrozen_predictions.json"   # stage-1 numbers in DESIGN.md came from an earlier seed form (int(gamma) + 1)
if "--stage2" in sys.argv:      # stage 2 cells (experiments/07_kick/DESIGN_stage2.md)
    # revision 2 after the stage-2 design review: lagged slope factors, two noise levels for the linear law (0.045 as in
    # stage 1, 0.063 about Haiku's measured K=2 noise), and the open-path confirm/disconfirm ratio
    CELLS = [("inverted", 2, -0.5), ("inverted", 2, 0.5), ("inverted", 4, -1.0), ("inverted", 4, 1.0)]
    OUTFILE = "refrozen_predictions_stage2.json"
    D_LAG = True
    LAWS = ("linear+noise", "linear+noise2", "anchoring")
    SIG0 = {"linear+noise": 0.045, "linear+noise2": 0.063}

if __name__ == "__main__":
    out = {}
    for name in LAWS:
        for w, K, gamma in CELLS:
            law = (dict(rho0=0.0, rho_inf=0.0, u0=1.0, sig0=SIG0[name], sig1=0.0, pi0=0.0, mu_pi=1.0)
                   if name != "anchoring" else sv.law_swarm(fit, K))
            g = fit["gains_transfer"][f"{w}|K{K}"]["one_shot" if name != "anchoring" else "target"]
            batches = [stats(cell(K, gamma, g, law, 16, np.random.default_rng([11, b, int(2 * gamma) + 2, len(w), len(name), K])))
                       for b in range(3)]
            a = np.array(batches)            # (3, 5): F, F_open, F_open_matched, frac_confirming, ratio
            m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(3)
            diff = a[:, 0] - a[:, 2]
            key = f"{name}|{w}|K{K}|g{gamma:+.1f}"
            out[key] = {"F": m[0], "F_se": se[0], "F_open": m[1], "F_open_se": se[1], "F_open_matched": m[2],
                        "F_open_matched_se": se[2], "F_minus_open_matched": diff.mean(),
                        "F_minus_open_matched_se": diff.std(ddof=1) / np.sqrt(3), "closed_frac_confirming": m[3],
                        "open_confirm_over_disconfirm": m[4], "open_confirm_over_disconfirm_se": se[4]}
            o = out[key]
            print(f"{key:28s} F {o['F']:.3f}({o['F_se']:.3f}) F_open {o['F_open']:.3f}({o['F_open_se']:.3f}) "
                  f"matched {o['F_open_matched']:.3f} F-matched {o['F_minus_open_matched']:+.3f}"
                  f"({o['F_minus_open_matched_se']:.3f}) frac_conf {o['closed_frac_confirming']:.2f} "
                  f"ratio {o['open_confirm_over_disconfirm']:.3f}({o['open_confirm_over_disconfirm_se']:.3f})", flush=True)
        for grp, sel in (("pooled", lambda k: True), ("pooled_K2", lambda k: "|K2|" in k), ("pooled_K4", lambda k: "|K4|" in k)):
            cellsk = [k for k in out if k.startswith(name + "|") and "|pooled" not in k and sel(k)]
            if not cellsk:
                continue
            pool = {s: float(np.mean([out[k][s] for k in cellsk])) for s in
                    ("F", "F_open", "F_open_matched", "F_minus_open_matched", "open_confirm_over_disconfirm")}
            out[f"{name}|{grp}"] = pool
            print(f"{name} {grp} (unweighted mean of {len(cellsk)} cells):", {k: round(v, 3) for k, v in pool.items()}, flush=True)
    Path(__file__).with_name(OUTFILE).write_text(json.dumps(out, indent=1, default=float))
