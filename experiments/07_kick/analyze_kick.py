"""Analysis of kick twins (experiments/07_kick/kick.py) with the estimators of theory/quiet_regime.md section 3 and
the controls of experiments/07_kick/DESIGN.md.

Per twin with kicked agent i (s = 2p - 1; Delta = twin - base; base states read from the base run):
    kick             D_i = -2 s_i(t0); twins with |s_i(t0)| < 0.1 are dropped (section 3.7)
    first order      r(type) = sum Delta s_j(t0+1) D_i / sum D_i^2 over readers j, type = J_ji                    (K1)
    echo             E = sum Delta s_i(t0+2) D_i / sum D_i^2                                                    (K2)
    factorization    c(type) = sum Delta s_j d_j D_i / sum d_j^2 D_i^2, d_j = 1 - s_j(t0+1)^2, leave-one-out over
                     twins; pred_i = d_i sum_{j reciprocated} c(J_ij) Delta s_j(t0+1), d_i = 1 - s_i(t0+2)^2;
                     F = sum Delta s_i(t0+2) pred_i / sum pred_i^2                                              (K5)
    open paths       the same regression for agents k != i that read a reader of i, are not readers, and have
                     |s_k(t0)| >= 0.1  ->  F_open
    alignment        F and F_open split by sign(pred * s(t0+1)) of the receiver: "confirming" (the returning
                     change points the way the receiver already leans) or "disconfirming"; r_A split by whether
                     reader and sender lean the same way at t0
    kick-size ratio  r_A over the smallest third of |D| divided by r_A over the largest third (terciles per cell)
    sham             sham twins (no message changed; readers redrawn on identical prompts) give the same slopes,
                     which must be zero within noise
    null ratio       (--null, base runs only) the kick-size ratio that saturation alone produces on the realized
                     base states: Delta s_j = tanh(atanh s_j(t0+1) + a_+ D_i / K) - s_j(t0+1) with the one-shot
                     transfer gain a_+, no anchoring and no noise

Uncertainties: bootstrap over graphs (replicas of a graph share its graph, evidence and round 0, so they are not
independent); with fewer than five graphs (the pilot) over twins, which understates the s.e. Pooled values over the
|gamma| = 1 cells are inverse-variance weighted.
Usage: uv run python experiments/07_kick/analyze_kick.py <kick_tag> [--methods]
       uv run python experiments/07_kick/analyze_kick.py <base_tag> --null
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
from cells import graph_for  # noqa: E402
from engine.ledger import read_jsonl  # noqa: E402

S_MIN = 0.1
TYPES = (+1, -1)
ONE_SHOT_A_PLUS = {("inverted", 2): 0.80, ("inverted", 4): 0.66, ("label", 2): 0.54, ("label", 4): 0.40}  # transfer check
_ROWS: dict = {}


def base_state(base_exp: str, rep: int, t: int) -> np.ndarray:
    if base_exp not in _ROWS:
        _ROWS[base_exp] = {(r["replica"], r["round_t"]): np.array(r["p_carried"], float)
                           for r in read_jsonl(ROOT / "results" / base_exp / "swarm_rounds.jsonl")}
    return 2 * _ROWS[base_exp][(rep, t)] - 1


def twin_sums(tw: dict, J: np.ndarray) -> dict:
    """Every per-twin sum the estimators need; cell estimates are ratios of (weighted) totals of these."""
    i = tw["kicked"][0]
    p_old, p_new = tw["kick"][str(i)]
    D = 2 * (p_new - p_old)
    s0 = base_state(tw["base_experiment_id"], tw["rep"], tw["t0"])
    base = 2 * np.array(tw["base_p"]) - 1                    # (2, N): rounds t0+1, t0+2
    ds = 2 * (np.array(tw["twin_p"]) - np.array(tw["base_p"]))
    d1, d2 = 1 - base[0] ** 2, 1 - base[1] ** 2
    readers = np.flatnonzero(J[:, i])
    o = {"i": i, "D": D, "s0": s0[i], "y": ds[1, i], "self_t1": ds[0, i], "graph": tw["graph"],
         "sent": np.array(tw["sent"]), "missing": np.array(tw["missing"]), "readers": readers}
    for ty in TYPES:
        js = [j for j in readers if J[j, i] == ty]
        o[f"r_num{ty}"] = sum(ds[0, j] for j in js) * D
        o[f"r_den{ty}"] = len(js) * D * D
        o[f"c_num{ty}"] = sum(ds[0, j] * d1[j] for j in js) * D
        o[f"c_den{ty}"] = sum(d1[j] ** 2 for j in js) * D * D
        o[f"X{ty}"] = d2[i] * sum(ds[0, j] for j in readers if J[i, j] == ty)
    o["s_i_t1"] = base[0, i]
    # agree readers split by whether reader and sender lean the same way at t0
    for al in ("same", "opp"):
        js = [j for j in readers if J[j, i] == +1 and ((s0[j] * s0[i] > 0) == (al == "same"))]
        o[f"rA_num_{al}"] = sum(ds[0, j] for j in js) * D
        o[f"rA_den_{al}"] = len(js) * D * D
    # open two-step paths, with the receiver's lean at t0+1 for the alignment split
    rset = set(readers.tolist())
    rows = []
    for k in np.flatnonzero((J[:, readers] != 0).any(1)):
        if k == i or k in rset or abs(s0[k]) < S_MIN:
            continue
        A = d2[k] * sum(ds[0, j] for j in readers if J[k, j] == +1)
        B = d2[k] * sum(ds[0, j] for j in readers if J[k, j] == -1)
        rows.append((A, B, ds[1, k], base[0, k]))
    o["open"] = np.array(rows).reshape(-1, 4)
    return o


def _slope_ratio(num, den):
    return num / den if den > 0 else np.nan


def estimates(Q: list[dict], w: np.ndarray, cuts) -> dict:
    """Cell estimates from per-twin sums with bootstrap weights w (counts per twin)."""
    col = lambda k: np.array([q[k] for q in Q])
    tot = lambda k: float(np.sum(w * col(k)))
    out = {}
    for ty, nm in ((+1, "agree"), (-1, "disagree")):
        out[f"r_{nm}"] = _slope_ratio(tot(f"r_num{ty}"), tot(f"r_den{ty}"))
    D, y = col("D"), col("y")
    out["echo"] = _slope_ratio(float(np.sum(w * y * D)), float(np.sum(w * D * D)))
    cn = {ty: col(f"c_num{ty}") for ty in TYPES}
    cd = {ty: col(f"c_den{ty}") for ty in TYPES}
    C = {ty: (np.sum(w * cn[ty]) - cn[ty]) / np.maximum(np.sum(w * cd[ty]) - cd[ty], 1e-12) for ty in TYPES}  # leave-one-out
    pred = C[+1] * col("X1") + C[-1] * col("X-1")
    conf = pred * col("s_i_t1") > 0
    for nm, m in (("F", np.ones(len(Q), bool)), ("F_confirm", conf), ("F_disconfirm", ~conf)):
        den = float(np.sum((w * pred * pred)[m]))
        out[nm] = float(np.sum((w * y * pred)[m])) / den if den > 0 else np.nan
    num = {"all": 0.0, "confirm": 0.0, "disconfirm": 0.0}
    den = {"all": 0.0, "confirm": 0.0, "disconfirm": 0.0}
    for k, q in enumerate(Q):
        if w[k] == 0 or not len(q["open"]):
            continue
        A, B, Y, S = q["open"].T
        pk = C[+1][k] * A + C[-1][k] * B
        cf = pk * S > 0
        for nm, m in (("all", np.ones(len(pk), bool)), ("confirm", cf), ("disconfirm", ~cf)):
            num[nm] += w[k] * float(np.sum((Y * pk)[m]))
            den[nm] += w[k] * float(np.sum((pk * pk)[m]))
    out["F_open"] = _slope_ratio(num["all"], den["all"])
    out["F_open_confirm"] = _slope_ratio(num["confirm"], den["confirm"])
    out["F_open_disconfirm"] = _slope_ratio(num["disconfirm"], den["disconfirm"])
    # composition test: compare F with the open-path slope of the alignment the closed paths mostly have
    # (theta-weighted by pred^2: at gamma=+1 the returning change opposes the receiver's lean, at inverted gamma=-1 it
    # agrees with it)
    frac_conf = float(np.sum((w * pred * pred)[conf]) / max(np.sum(w * pred * pred), 1e-12))
    out["closed_frac_confirming"] = frac_conf
    out["F_open_matched"] = out["F_open_confirm"] if frac_conf > 0.5 else out["F_open_disconfirm"]
    out["F_minus_open_matched"] = out["F"] - out["F_open_matched"]
    aD = np.abs(D)
    small, large = aD <= cuts[0], aD >= cuts[1]
    rs = _slope_ratio(float(np.sum((w * col("r_num1"))[small])), float(np.sum((w * col("r_den1"))[small])))
    rl = _slope_ratio(float(np.sum((w * col("r_num1"))[large])), float(np.sum((w * col("r_den1"))[large])))
    out.update(rA_small=rs, rA_large=rl, rA_ratio=rs / rl if rl else np.nan)
    out["rA_same_lean"] = _slope_ratio(tot("rA_num_same"), tot("rA_den_same"))
    out["rA_opp_lean"] = _slope_ratio(tot("rA_num_opp"), tot("rA_den_opp"))
    return out


def bootstrap(Q, n_boot=1000):
    """Point estimates and bootstrap s.e. (clusters: graphs; twins if fewer than five graphs)."""
    aD = np.abs([q["D"] for q in Q])
    cuts = (np.quantile(aD, 1 / 3), np.quantile(aD, 2 / 3))
    est = estimates(Q, np.ones(len(Q)), cuts)
    graphs = sorted({q["graph"] for q in Q})
    rng = np.random.default_rng(0)
    if len(graphs) >= 5:
        method = f"bootstrap over {len(graphs)} graphs"
        gid = np.array([graphs.index(q["graph"]) for q in Q])
        draws = (np.bincount(rng.integers(len(graphs), size=len(graphs)), minlength=len(graphs))[gid] for _ in range(n_boot))
    else:
        method = f"bootstrap over {len(Q)} twins ({len(graphs)} graph(s); understates the s.e.)"
        draws = (np.bincount(rng.integers(len(Q), size=len(Q)), minlength=len(Q)) for _ in range(n_boot))
    boots = [estimates(Q, w.astype(float), cuts) for w in draws]
    se = {f"{k}_se": float(np.nanstd([b[k] for b in boots], ddof=1)) for k in est}
    return {**est, **se, "se_method": method, "D_terciles": list(map(float, cuts))}


def analyze_cell(twins: list[tuple[dict, np.ndarray]], methods: bool) -> dict:
    Q_all = [twin_sums(tw, J) | {"sham": bool(tw.get("sham"))} for tw, J in twins]
    tw0 = twins[0][0]
    out = {k: tw0[k] for k in ("wording", "K", "p_neg", "gamma", "N")}
    real = [q for q in Q_all if not q["sham"]]
    out.update(n_twins=len(real), n_sham=len(Q_all) - len(real), calls_sent=int(sum(q["sent"].sum() for q in Q_all)),
               missing=int(sum(q["missing"].sum() for q in Q_all)),
               kicked_self_changed_t1=int(sum(q["self_t1"] != 0 for q in Q_all)),                       # must be 0
               non_readers_sent_t1=int(sum(np.delete(q["sent"][0], q["readers"]).sum() for q in Q_all)),  # must be 0
               graphs=sorted({q["graph"] for q in Q_all}))
    if methods:
        return out
    Q = [q for q in real if abs(q["s0"]) >= S_MIN]
    out.update(n_used=len(Q), D2_mean=float(np.mean([q["D"] ** 2 for q in Q])),
               open_per_twin=float(np.mean([len(q["open"]) for q in Q])),
               frac_echo_exactly_zero=float(np.mean([q["y"] == 0 for q in Q])), **bootstrap(Q))
    S = [q for q in Q_all if q["sham"] and abs(q["s0"]) >= S_MIN]
    if len(S) >= 10:
        b = bootstrap(S)
        out.update({f"sham_{k}": b[k] for k in ("r_agree", "r_agree_se", "r_disagree", "r_disagree_se", "echo",
                                                 "echo_se", "se_method")}, n_sham_used=len(S))
    return out


def null_ratio_from_base(base_tag: str, t0s=(6, 8), reps=(0, 1)) -> list[dict]:
    """Kick-size ratio from saturation alone on the realized base states, from the base runs only (no twin exists or
    is read): every eligible kick (|s_i(t0)| >= 0.1) of every base run of the |gamma| = 1 and gamma = 0 cells, and
    each agree reader's tanh response with the one-shot transfer gain, Delta s_j = tanh(atanh s_j(t0+1) + a_+ D_i/K)
    - s_j(t0+1), no anchoring and no noise. Same tercile bins and slope estimator as the measured ratio."""
    from cells import cell_id
    man = json.loads((ROOT / "results" / base_tag / "manifest.json").read_text())
    by = defaultdict(list)
    for c in man["cells"]:
        J = graph_for(c)
        a, K = ONE_SHOT_A_PLUS[(c["wording"], c["K"])], c["K"]
        exp = f"{base_tag}-{cell_id(c)}"
        for rep in reps:
            for t0 in t0s:
                s0, s1 = base_state(exp, rep, t0), base_state(exp, rep, t0 + 1)
                for i in np.flatnonzero(np.abs(s0) >= S_MIN):
                    D = -2 * s0[i]
                    for j in np.flatnonzero(J[:, i] == +1):
                        x = np.arctanh(np.clip(s1[j], -0.98, 0.98))
                        by[(c["wording"], K, c["gamma"])].append((abs(D), (np.tanh(x + a * D / K) - s1[j]) * D, D * D))
    out = []
    for (wording, K, gamma), rows in sorted(by.items()):
        v = np.array(rows)
        cuts = np.quantile(v[:, 0], [1 / 3, 2 / 3])
        sm, lg = v[:, 0] <= cuts[0], v[:, 0] >= cuts[1]
        out.append({"wording": wording, "K": K, "gamma": gamma, "n_agree_reader_terms": len(v),
                    "null_rA_ratio": float((v[sm, 1].sum() / v[sm, 2].sum()) / (v[lg, 1].sum() / v[lg, 2].sum()))})
    return out


def load(tag: str) -> dict:
    cells = defaultdict(list)
    Js = {}
    for p in sorted((ROOT / "results").glob(f"{tag}-*/twins.jsonl")):
        for tw in read_jsonl(p):
            key = (tw["wording"], tw["K"], tw["p_neg"], tw["gamma"], tw["N"])
            gk = key + (tw["graph"],)
            if gk not in Js:
                Js[gk] = graph_for(dict(wording=tw["wording"], K=tw["K"], p_neg=tw["p_neg"], gamma=tw["gamma"],
                                        N=tw["N"], graph=tw["graph"], T=10, mode="sync"))
            cells[key].append((tw, Js[gk]))
    return cells


def pooled(rows: list[dict], key: str) -> tuple[float, float]:
    """Inverse-variance pooled estimate over the |gamma| = 1 cells."""
    v = [(r[key], r[key + "_se"]) for r in rows if abs(r["gamma"]) == 1 and np.isfinite(r.get(key, np.nan))
         and r.get(key + "_se", 0) > 0]
    if not v:
        return np.nan, np.nan
    x, s = np.array(v).T
    wt = 1 / s**2
    return float(np.sum(wt * x) / np.sum(wt)), float(1 / np.sqrt(np.sum(wt)))


if __name__ == "__main__":
    tag = sys.argv[1]
    mode = "methods" if "--methods" in sys.argv else "null" if "--null" in sys.argv else "full"
    if mode == "null":                      # here <tag> is the BASE tag; reads base runs only
        out = null_ratio_from_base(tag, t0s=(5,) if tag.startswith("pilot") else (6, 8),
                                   reps=(0,) if tag.startswith("pilot") else (0, 1))
        for r in out:
            print(r)
        (ROOT / "results" / f"{tag}-null-ratio.json").write_text(json.dumps(out, indent=1))
        sys.exit()
    cells = load(tag)
    rows = []
    for key, twins in sorted(cells.items()):
        r = analyze_cell(twins, mode == "methods")
        rows.append(r)
        line = (f"{r['wording'][:3]} K={r['K']} g={r['gamma']:+.1f} | twins {r['n_twins']} sham {r['n_sham']} sent "
                f"{r['calls_sent']} missing {r['missing']} | self-changed@t0+1 {r['kicked_self_changed_t1']} "
                f"non-readers-sent@t0+1 {r['non_readers_sent_t1']}")
        if mode == "full":
            f = lambda k: f"{r[k]:+.4f}({r[k + '_se']:.4f})"
            line += (f"\n    used {r['n_used']} | rA {f('r_agree')} rD {f('r_disagree')} | E {f('echo')} | F {f('F')} "
                     f"[conf {f('F_confirm')} disc {f('F_disconfirm')}] | F_open {f('F_open')} [conf {f('F_open_confirm')} "
                     f"disc {f('F_open_disconfirm')}] F-F_open,matched {f('F_minus_open_matched')} | rA small/large {f('rA_ratio')} | rA same/opp lean "
                     f"{f('rA_same_lean')}/{f('rA_opp_lean')} | zero-echo {r['frac_echo_exactly_zero']:.2f} | {r['se_method']}")
            if "sham_echo" in r:
                line += (f"\n    sham ({r['n_sham_used']}): rA {r['sham_r_agree']:+.4f}({r['sham_r_agree_se']:.4f}) "
                         f"rD {r['sham_r_disagree']:+.4f}({r['sham_r_disagree_se']:.4f}) E {r['sham_echo']:+.4f}"
                         f"({r['sham_echo_se']:.4f})")
        print(line, flush=True)
    if mode == "full":
        pool = {k: pooled(rows, k) for k in ("F", "F_open", "F_open_matched", "F_minus_open_matched", "F_confirm",
                                              "F_open_confirm", "F_disconfirm", "F_open_disconfirm", "rA_ratio")}
        print("pooled over |gamma|=1 cells:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in pool.items()})
        (ROOT / "results" / f"{tag}-summary.json").write_text(json.dumps({"cells": rows, "pooled": pool}, indent=1,
                                                                          default=float))
