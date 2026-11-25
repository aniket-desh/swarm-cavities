"""Analysis of kick twins (experiments/07_kick/kick.py; predictions in theory/quiet_regime.md).

Per twin with kicked agent i (states s = 2p - 1, differences twin - base):
    kick size        Delta_i = s_i^kicked(t0) - s_i(t0) = -2 s_i(t0) for a mirror kick (p -> 1 - p);
                     zero for an agent at p = 0.50 (mirroring cannot move it) and small near it
    first order      r(type) = sum Delta s_j(t0+1) Delta_i / sum Delta_i^2 over readers j of the edge type J_ji
                     (+1: j reads i as a source it tends to agree with, -1: tends to disagree with)
    echo             E = sum Delta s_i(t0+2) Delta_i / sum Delta_i^2
    factorization    pred_i = sum_{j reads i and i reads j} rbar(J_ij) Delta s_j(t0+1),  with rbar(type) the
                     first-order slope of that edge type from the OTHER twins of the cell (leave-one-out, so a
                     twin's own noise is not in its prediction); E - E^pred = sum (Delta s_i(t0+2) - pred_i)
                     Delta_i / sum Delta_i^2 has mean zero if responses compose linearly, as the cavity picture
                     assumes.
Slopes through the origin weight each twin by its kick size, so near-zero kicks add nothing rather than blowing up a
ratio. Uncertainties: bootstrap over twins (the independent units; a twin's readers share its kick).
Usage: uv run python experiments/07_kick/analyze_kick.py <kick_tag> [--methods]
  --methods prints only cost, missing and replay-correctness numbers (used before the predictions are frozen).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
from cells import graph_for  # noqa: E402
from engine.ledger import read_jsonl  # noqa: E402


def twin_quantities(tw: dict, J: np.ndarray) -> dict:
    i = tw["kicked"][0]
    p_old, p_new = tw["kick"][str(i)]
    D = 2 * (p_new - p_old)
    ds = 2 * (np.array(tw["twin_p"]) - np.array(tw["base_p"]))          # (2 rounds, N)
    readers = np.flatnonzero(J[:, i])
    recip = [j for j in readers if J[i, j] != 0]
    return {"i": i, "D": D, "e2": ds[1, i], "ds1": ds[0], "readers": readers,
            "dsr": {int(j): ds[0, j] for j in readers}, "type_ji": {int(j): int(J[j, i]) for j in readers},
            "recip": [(int(j), int(J[i, j])) for j in recip], "self_t1": ds[0, i],
            "sent": np.array(tw["sent"]), "missing": np.array(tw["missing"])}


def analyze_cell(path: Path, methods: bool, n_boot: int = 2000) -> dict:
    tws = list(read_jsonl(path))
    c = {k: tws[0][k] for k in ("wording", "K", "p_neg", "gamma", "N", "graph")}
    c.update(T=10, mode="sync")
    J = graph_for(c)
    q = [twin_quantities(t, J) for t in tws]
    out = {**c, "n_twins": len(q), "calls_sent": int(sum(x["sent"].sum() for x in q)),
           "missing": int(sum(x["missing"].sum() for x in q)),
           "kicked_self_changed_t1": int(sum(x["self_t1"] != 0 for x in q)),
           "zero_kicks": int(sum(x["D"] == 0 for x in q)),        # must be 0
           "unaffected_sent_t1": int(sum(np.delete(x["sent"][0], x["readers"]).sum() for x in q))}  # non-readers sent at t0+1: must be 0 (readers can be skipped when a kick at p = 0.50 leaves the message unchanged)
    if methods:
        return out
    slope = lambda y, d: float(np.sum(y * d) / np.sum(d * d)) if np.sum(d * d) > 0 else 0.0
    # first-order responses by edge type: (twin index, Delta s_j, Delta_i)
    r_by = {+1: [], -1: []}
    for k, x in enumerate(q):
        for j, dsj in x["dsr"].items():
            r_by[x["type_ji"][j]].append((k, dsj, x["D"]))
    rng = np.random.default_rng(0)
    for ty, name in ((+1, "agree"), (-1, "disagree")):
        v = np.array(r_by[ty]).reshape(-1, 3)
        out[f"r_{name}_n"] = int(len(v))
        out[f"r_{name}"] = slope(v[:, 1], v[:, 2]) if len(v) else None
        if len(v) > 1:          # bootstrap over twins, keeping each twin's readers together
            ks = np.unique(v[:, 0]); bs = []
            for _ in range(n_boot):
                pick = rng.choice(ks, len(ks))
                vv = np.concatenate([v[v[:, 0] == k] for k in pick])
                bs.append(slope(vv[:, 1], vv[:, 2]))
            out[f"r_{name}_se"] = float(np.std(bs, ddof=1))
        else:
            out[f"r_{name}_se"] = None
    D = np.array([x["D"] for x in q]); E2 = np.array([x["e2"] for x in q])
    P = np.zeros(len(q))
    for k, x in enumerate(q):
        rbar = {}
        for ty in (+1, -1):
            v = np.array([(dsj, d) for kk, dsj, d in r_by[ty] if kk != k]).reshape(-1, 2)
            rbar[ty] = slope(v[:, 0], v[:, 1]) if len(v) else 0.0
        P[k] = sum(rbar[ty_ij] * x["ds1"][j] for j, ty_ij in x["recip"])
    idx = rng.integers(len(q), size=(n_boot, len(q)))
    bslope = lambda y: np.array([slope(y[b], D[b]) for b in idx])
    out.update(n_used=int(np.sum(D != 0)), kick_abs_mean=float(np.abs(D).mean()),
               echo=slope(E2, D), echo_se=float(bslope(E2).std(ddof=1)),
               echo_pred=slope(P, D), echo_pred_se=float(bslope(P).std(ddof=1)),
               echo_minus_pred=slope(E2 - P, D), echo_minus_pred_se=float(bslope(E2 - P).std(ddof=1)),
               echo_resid_sd=float(np.std(E2 - slope(E2, D) * D, ddof=1)),
               frac_echo_exactly_zero=float(np.mean(E2 == 0)))
    return out


if __name__ == "__main__":
    tag, methods = sys.argv[1], "--methods" in sys.argv
    rows = []
    for p in sorted((ROOT / "results").glob(f"{tag}-*/twins.jsonl")):
        r = analyze_cell(p, methods)
        rows.append(r)
        line = (f"{r['wording'][:3]} K={r['K']} g={r['gamma']:+.1f} | twins {r['n_twins']} sent {r['calls_sent']} "
                f"missing {r['missing']} zero-kicks {r['zero_kicks']} | self-changed@t0+1 {r['kicked_self_changed_t1']} "
                f"unaffected-sent@t0+1 {r['unaffected_sent_t1']}")
        if not methods:
            line += (f" | r+ {r['r_agree']:.3f}({r['r_agree_se']:.3f}) r- {r['r_disagree']:.3f}({r['r_disagree_se']:.3f}) "
                     f"| echo {r['echo']:+.4f}({r['echo_se']:.4f}) resid-sd {r['echo_resid_sd']:.3f} pred {r['echo_pred']:+.4f}"
                     f"({r['echo_pred_se']:.4f}) diff {r['echo_minus_pred']:+.4f}({r['echo_minus_pred_se']:.4f}) "
                     f"zero {r['frac_echo_exactly_zero']:.2f}")
        print(line, flush=True)
    if not methods:
        (ROOT / "results" / f"{tag}-summary.json").write_text(json.dumps(rows, indent=1))
