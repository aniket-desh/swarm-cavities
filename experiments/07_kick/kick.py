"""Kick-and-echo runs on finished Haiku swarm runs (envs/kick.py; design in theory/quiet_regime.md).

For each base run (a cell of experiments/05_swarm/cells.py), every requested agent of replica `rep` is kicked at
round t0 in its own twin (one kick per twin, so no twin can contaminate another), and rounds t0+1, t0+2 are rerun.
Twin records go to results/<base experiment_id>-kick/twins.jsonl, one line per twin, appended as each finishes; a
restart skips twins already written, and the engine's checkpoints and replay cache make interrupted twins free to
redo. Unaffected agents replay the base run's replies from the cache (identical, free).

Run from the repo root inside tmux:
  uv run python experiments/07_kick/kick.py <base_tag> <kick_tag> --t0 5 --rep 0 --cap USD [--cells SUBSTR ...]
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


async def kick_cell(c: dict, base_tag: str, kick_tag: str, t0: int, rep: int, agents, guard, sem) -> int:
    base_exp = f"{base_tag}-{cell_id(c)}"
    base_dir = ROOT / "results" / base_exp
    out_dir = ROOT / "results" / f"{kick_tag}-{cell_id(c)}"
    out = out_dir / "twins.jsonl"
    done = {(r["rep"], r["t0"], tuple(r["kicked"])) for r in read_jsonl(out)}
    J, game = graph_for(c), game_for(c)
    runner = Runner(f"{kick_tag}-{cell_id(c)}", guard, cfg=AgentConfig(), concurrency=16, ledger_path=LEDGER)
    cfg = SwarmConfig(experiment_id=base_exp, T=c["T"], H=H, replicas=REPLICAS, mode=c["mode"], seed=c["graph"])
    sw = Swarm(J, game, cfg, LLMBackend(runner, disagree=c["wording"]))
    rows = base_rows(base_dir)
    todo = [i for i in agents if (rep, t0, (i,)) not in done]

    async def one(i):
        async with sem:
            r = await kick_twin(sw, rows, rep, t0, [i], n_rounds=2)
            append_jsonl(out, {"base_experiment_id": base_exp, **{k: c[k] for k in
                               ("wording", "K", "p_neg", "gamma", "N", "graph")},
                               "truth": game.truth, "rep": r["rep"], "t0": r["t0"], "kicked": r["kicked"],
                               "kick": {str(k): v for k, v in r["kick"].items()},
                               "base_p": r["base_p"].tolist(), "twin_p": r["twin_p"].tolist(),
                               "missing": r["missing"].tolist(), "sent": r["sent"].tolist()})
            return int(r["sent"].sum())

    sent = await asyncio.gather(*(one(i) for i in todo))
    print(f"[{out_dir.name}] {len(todo)} twins, {sum(sent)} calls sent; spent so far ${guard.experiment_spent():.3f}",
          flush=True)
    return sum(sent)


async def main_async(a):
    man = json.loads((ROOT / "results" / a.base_tag / "manifest.json").read_text())
    cs = [c for c in man["cells"] if c["mode"] == "sync" and (not a.cells or any(s in cell_id(c) for s in a.cells))]
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=a.cap, experiment_id=f"{a.kick_tag}-*", ledger_path=LEDGER)
    # paid calls per twin ~ readers (K) at t0+1, plus readers again and their readers (K + K^2) at t0+2
    n_calls = sum(c["N"] * (2 * c["K"] + c["K"] ** 2) for c in cs) if a.agents == "all" else 0
    guard.project(n_calls, in_tokens=400, cached_tokens=1000, out_tokens=700, model="claude-haiku-5-5", batch=False)
    sem = asyncio.Semaphore(a.parallel)
    t = time.time()
    tot = 0
    for c in cs:
        agents = range(c["N"]) if a.agents == "all" else [int(x) for x in a.agents.split(",")]
        tot += await kick_cell(c, a.base_tag, a.kick_tag, a.t0, a.rep, agents, guard, sem)
    print(f"done: {tot} calls sent in {time.time() - t:.0f}s; spent ${guard.experiment_spent():.3f} on {a.kick_tag}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_tag")
    ap.add_argument("kick_tag")
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--agents", default="all")
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--parallel", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))
