"""Kick-and-echo runs on finished Haiku swarm runs (envs/kick.py; design in theory/quiet_regime.md).

For each base run (a cell of experiments/05_swarm/cells.py), every requested agent of replica `rep` is kicked at
round t0 in its own twin (one kick per twin, so no twin can contaminate another), and rounds t0+1, t0+2 are rerun.
Twin records go to results/<base experiment_id>-kick/twins.jsonl, one line per twin, appended as each finishes; a
restart skips twins already written, and the engine's checkpoints and replay cache make interrupted twins free to
redo. Unaffected agents replay the base run's replies from the cache (identical, free).

Run from the repo root inside tmux:
  uv run python experiments/07_kick/kick.py <base_tag> <kick_tag> --t0 6 8 --rep 0 1 --min-abs-s 0.1 --cap USD
      [--cells SUBSTR ...]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from cells import H, REPLICAS, cell_id, game_for, graph_for  # noqa: E402
from engine import BudgetGuard  # noqa: E402
from engine.agent import AgentConfig  # noqa: E402
from engine.ledger import append_jsonl, read_jsonl  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.kick import base_rows, kick_twin  # noqa: E402
from envs.swarm import LLMBackend, Swarm, SwarmConfig  # noqa: E402

LEDGER = ROOT / "budget" / "ledger.jsonl"


async def kick_cell(c: dict, base_tag: str, kick_tag: str, t0s, reps, agents, guard, sem, min_abs_s: float,
                    sham_n: int = 0) -> int:
    base_exp = f"{base_tag}-{cell_id(c)}"
    base_dir = ROOT / "results" / base_exp
    out_dir = ROOT / "results" / f"{kick_tag}-{cell_id(c)}"
    out = out_dir / "twins.jsonl"
    done = {(r["rep"], r["t0"], tuple(r["kicked"]), bool(r.get("sham"))) for r in read_jsonl(out)}
    J, game = graph_for(c), game_for(c)
    runner = Runner(f"{kick_tag}-{cell_id(c)}", guard, cfg=AgentConfig(), concurrency=16, ledger_path=LEDGER,
                    checkpoint_replays=False)
    cfg = SwarmConfig(experiment_id=base_exp, T=c["T"], H=H, replicas=REPLICAS, mode=c["mode"], seed=c["graph"])
    sw = Swarm(J, game, cfg, LLMBackend(runner, disagree=c["wording"]))
    # sham twins: the readers redraw on identical prompts through a Runner with its own replay namespace
    sham_runner = Runner(f"{kick_tag}-{cell_id(c)}-sham", guard, cfg=AgentConfig(), concurrency=16, ledger_path=LEDGER,
                         replay_salt=f"sham|{kick_tag}", checkpoint_replays=False)
    sham_backend = LLMBackend(sham_runner, disagree=c["wording"])
    rows = base_rows(base_dir)
    # agents with |s_i(t0)| < min_abs_s are skipped: a mirror kick of size 2|s_i| adds almost nothing (and nothing at
    # p = 0.50), theory/quiet_regime.md section 3.7. The selection uses only the state at t0, never anything later.
    eligible = [(rep, t0, i) for rep in reps for t0 in t0s for i in agents
                if abs(2 * rows[(rep, t0)]["p_carried"][i] - 1) >= min_abs_s]
    # sham jobs: sham_n of the eligible jobs, evenly spaced (fixed by the base state only)
    sham_jobs = [eligible[k] for k in np.linspace(0, len(eligible) - 1, sham_n).round().astype(int)] if sham_n else []
    todo = [(rep, t0, i, False) for rep, t0, i in eligible if (rep, t0, (i,), False) not in done]
    todo += [(rep, t0, i, True) for rep, t0, i in sham_jobs if (rep, t0, (i,), True) not in done]

    async def one(job):
        rep, t0, i, sham = job
        async with sem:
            r = await kick_twin(sw, rows, rep, t0, [i], n_rounds=2, sham_backend=sham_backend if sham else None)
            append_jsonl(out, {"base_experiment_id": base_exp, **{k: c[k] for k in
                               ("wording", "K", "p_neg", "gamma", "N", "graph")},
                               "truth": game.truth, "rep": r["rep"], "t0": r["t0"], "kicked": r["kicked"],
                               "sham": r["sham"],
                               "kick": {str(k): v for k, v in r["kick"].items()},
                               "base_p": r["base_p"].tolist(), "twin_p": r["twin_p"].tolist(),
                               "missing": r["missing"].tolist(), "sent": r["sent"].tolist()})
            return int(r["sent"].sum())

    sent = await asyncio.gather(*(one(j) for j in todo))
    print(f"[{out_dir.name}] {len(todo)} twins, {sum(sent)} calls sent; spent so far ${guard.experiment_spent():.3f}",
          flush=True)
    return sum(sent)


async def main_async(a):
    man = json.loads((ROOT / "results" / a.base_tag / "manifest.json").read_text())
    cs = [c for c in man["cells"] if c["mode"] == "sync" and (not a.cells or any(s in cell_id(c) for s in a.cells))]
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=a.cap, experiment_id=f"{a.kick_tag}-*", ledger_path=LEDGER)
    # paid calls per twin ~ readers (K) at t0+1, plus readers again and their readers (K + K^2) at t0+2
    n_calls = (sum(c["N"] * (2 * c["K"] + c["K"] ** 2) for c in cs) * len(a.t0) * len(a.rep)
               if a.agents == "all" else 0)
    guard.project(n_calls, in_tokens=400, cached_tokens=1000, out_tokens=700, model="claude-haiku-5-5", batch=False)
    sem = asyncio.Semaphore(a.parallel)
    t = time.time()
    for c in cs:   # every base run must be complete before any twin is built from it
        rows = base_rows(ROOT / "results" / f"{a.base_tag}-{cell_id(c)}")
        if any((r, c["T"]) not in rows for r in range(REPLICAS)):
            sys.exit(f"base run {cell_id(c)} is incomplete; rerun the base command until every cell is done")
    def agents_for(c):
        if a.agents == "all":
            return range(c["N"])
        if a.agents == "first-eligible":       # one-twin replay check: the first agent that passes the |s| filter
            s0 = 2 * np.array(base_rows(ROOT / "results" / f"{a.base_tag}-{cell_id(c)}")[(a.rep[0], a.t0[0])]["p_carried"]) - 1
            return [int(np.flatnonzero(np.abs(s0) >= a.min_abs_s)[0])]
        return [int(x) for x in a.agents.split(",")]
    tot = sum(await asyncio.gather(*(kick_cell(c, a.base_tag, a.kick_tag, a.t0, a.rep, agents_for(c), guard, sem,
                                               a.min_abs_s, a.sham) for c in cs)))
    print(f"done: {tot} calls sent in {time.time() - t:.0f}s; spent ${guard.experiment_spent():.3f} on {a.kick_tag}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_tag")
    ap.add_argument("kick_tag")
    ap.add_argument("--t0", type=int, nargs="+", required=True)
    ap.add_argument("--rep", type=int, nargs="+", default=[0])
    ap.add_argument("--min-abs-s", type=float, default=0.0)
    ap.add_argument("--sham", type=int, default=0, help="sham twins per base run")
    ap.add_argument("--agents", default="all")
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--parallel", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))
