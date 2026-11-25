"""Observables of real swarm runs, in the same definitions as the predictions (experiments/04_swarm_design).

A run's state S (2 replicas, T+1 rounds, N agents) is rebuilt from results/<experiment_id>/swarm_rounds.jsonl,
using p_carried: the state the dynamics actually propagated, which equals the stated belief when the reply
parsed and the agent's previous message when it went missing (envs/swarm.py). States are truth-aligned,
s = truth * (2p - 1), so m > 0 means the swarm leans toward the truth. thermal_observables and overlap_excess
are imported from design_lib, so the formulas are identical to the predictor's.

Usage:  uv run python experiments/05_swarm/analyze.py <tag> [<tag> ...]
Writes results/<tag>/observables.json (one record per run) and prints a table.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "04_swarm_design"))
from design_lib import overlap_excess, thermal_observables  # noqa: E402


def load_run(exp_dir: Path) -> tuple[np.ndarray, dict]:
    meta = json.loads((exp_dir / "swarm_meta.json").read_text())
    rows = [json.loads(line) for line in open(exp_dir / "swarm_rounds.jsonl")]
    R, T = meta["config"]["replicas"], meta["config"]["T"]
    N = len(meta["h_priv"])
    S = np.full((R, T + 1, N), np.nan)
    missing = np.zeros((R, T + 1, N), bool)
    for r in rows:
        S[r["replica"], r["round_t"]] = np.array(r["p_carried"], float)
        missing[r["replica"], r["round_t"]] = r["missing"]
    if np.isnan(S).any():
        raise ValueError(f"{exp_dir.name}: incomplete run")
    truth = meta["game"]["truth"]
    return truth * (2 * S - 1), {"truth": truth, "missing": int(missing.sum()), "n": int(missing.size)}


def analyze(tag: str) -> list[dict]:
    man = json.loads((ROOT / "results" / tag / "manifest.json").read_text())
    from cells import cell_id   # noqa: E402  (same folder)
    recs, runs_by_game = [], defaultdict(list)
    for c in man["cells"]:
        d = ROOT / "results" / f"{tag}-{cell_id(c)}"
        if not (d / "swarm_rounds.jsonl").exists():
            continue
        try:
            S, info = load_run(d)
        except ValueError as e:
            print(f"skip: {e}")
            continue
        obs = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in thermal_observables(S).items()}
        rec = {**{k: c[k] for k in ("wording", "K", "p_neg", "gamma", "N", "T", "mode", "graph")}, **info, **obs}
        recs.append(rec)
        # runs that share an evidence draw and every cell parameter but the graph -> overlap excess dq
        runs_by_game[(c["wording"], c["K"], c["p_neg"], c["gamma"], c["N"], c["mode"], c.get("game", c["graph"]))].append(
            (len(recs) - 1, S))
    for group in runs_by_game.values():
        if len(group) > 1:
            for (i, _), dq in zip(group, overlap_excess([S for _, S in group])):
                recs[i]["dq"] = dq.tolist()
    (ROOT / "results" / tag / "observables.json").write_text(json.dumps(recs))
    return recs


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    for tag in sys.argv[1:]:
        for r in analyze(tag):
            print(f"{r['wording'][:3]} K={r['K']} g={r['gamma']:+.1f} {r['mode'][:2]} G{r['graph']} | missing "
                  f"{r['missing']}/{r['n']} | m(T) {r['m'][-1]:.3f} D(T) {r['D'][-1]:.4f} c1 {r['c1']:.3f} "
                  f"c2 {r['c2']:+.3f} P_delta {r['P_delta']:.2f} P_raw {r['P_raw']:.2f}")
