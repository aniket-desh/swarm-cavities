"""Run sets of Haiku 5.5 swarm cells: the pilot (message pool + pipeline check) and the swarm tests.

A cell is (wording, K, p_neg, gamma, N, T, mode, graph). Each cell is one envs.swarm run: a fixed in-degree
signed graph (envs/graphs.fixed_indegree_signed_couplings, exactly round(|gamma| K) reciprocated ties per agent,
concordant for gamma >= 0 and discordant for gamma < 0), one evidence draw (envs/evidence_game.make_game with
mu=0.5, sigma=1, k=3, the values experiments/04_swarm_design assumed), two replicas sharing graph, evidence and
round-0 answers, own last answer shown (H=1).

Pairing. The graph depends on (K, p_neg, gamma, N, graph) and the game on (N, graph), not on mode or wording,
so the asynchronous control and the two wordings run on the same graph and evidence as the synchronous cell
they are compared with. The coin's truth is drawn from the game seed (both signs occur, so a HEADS bias of the
model shows up instead of hiding in m); analyses use truth-aligned states s * truth.

Every agent call is checkpointed by engine.Runner, so a crashed or interrupted set resumes without re-spending.
One BudgetGuard covers the whole set through the prefix cap "<tag>-*" (engine/ledger.total_api_spend).

Run from the repo root inside tmux:  uv run python experiments/05_swarm/cells.py <cellset> <tag> --cap USD
"""

import argparse
import asyncio
import itertools
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from engine import BudgetGuard  # noqa: E402
from engine.agent import AgentConfig  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.evidence_game import make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings, spectral_log  # noqa: E402
from envs.swarm import LLMBackend, Swarm, SwarmConfig  # noqa: E402

LEDGER = ROOT / "budget" / "ledger.jsonl"
BATCH = False   # set by --batch: synchronous rounds through Message Batches (50% off, ~6 min per round)
MU, SIGMA, K_CLUES, H, REPLICAS = 0.5, 1.0, 3, 1, 2


def cell_id(c: dict) -> str:
    return (f"{c['wording']}-K{c['K']}-pn{c['p_neg']}-g{c['gamma']:+.1f}-N{c['N']}-T{c['T']}-{c['mode']}"
            f"-G{c['graph']}")


def graph_for(c: dict) -> np.ndarray:
    key = zlib.crc32(f"K{c['K']}|pn{c['p_neg']}|g{c['gamma']}|N{c['N']}".encode())
    return fixed_indegree_signed_couplings(c["N"], c["K"], c["gamma"], c["p_neg"],
                                           np.random.default_rng([key, c["graph"]]))


def game_for(c: dict):
    return make_game(c["N"], mu=MU, sigma=SIGMA, k=K_CLUES, seed=10_000 + c["graph"])


def cells(name: str) -> list[dict]:
    base = dict(N=64, T=10, mode="sync")
    if name == "pilot":
        # message pool for the transfer check and a pipeline check at K=2 and N=64; excluded from every
        # confirmatory analysis (graphs 900+ are reserved for it)
        return [dict(base, wording="inverted", K=K, p_neg=0.5, gamma=g, graph=900)
                for K, g in itertools.product((2, 4), (-1.0, 0.0, 1.0))]
    if name == "kick1":
        # base runs for stage 1 of the kick-and-echo test (experiments/07_kick/DESIGN.md): both wordings, K=2,
        # gamma in {-1, 0, +1}, 8 graphs each (graphs 0-7; confirmatory)
        return [dict(base, wording=w, K=2, p_neg=0.5, gamma=g, graph=G)
                for w, g, G in itertools.product(("inverted", "label"), (-1.0, 0.0, 1.0), range(8))]
    if name == "kick2":
        # base runs for stage 2 of the kick-and-echo test (experiments/07_kick/DESIGN_stage2.md): inverted wording;
        # K=2 at gamma = +-0.5 (additivity against stage 1's gamma = +-1) and K=4 at gamma in {-1, 0, +1}; graphs 0-7
        return ([dict(base, wording="inverted", K=2, p_neg=0.5, gamma=g, graph=G) for g, G in
                 itertools.product((-0.5, 0.5), range(8))] +
                [dict(base, wording="inverted", K=4, p_neg=0.5, gamma=g, graph=G) for g, G in
                 itertools.product((-1.0, 0.0, 1.0), range(8))])
    raise ValueError(f"unknown cell set {name}")


async def run_cell(c: dict, tag: str, guard: BudgetGuard, sem: asyncio.Semaphore) -> dict:
    exp = f"{tag}-{cell_id(c)}"
    async with sem:
        t0 = time.time()
        runner = Runner(exp, guard, cfg=AgentConfig(), concurrency=16, ledger_path=LEDGER)
        cfg = SwarmConfig(experiment_id=exp, T=c["T"], H=H, replicas=REPLICAS, mode=c["mode"], seed=c["graph"])
        sw = Swarm(graph_for(c), game_for(c), cfg, LLMBackend(runner, batch=BATCH, poll_s=30, disagree=c["wording"]))
        out = await sw.run()
        n_miss = int(out["missing"].sum())
        print(f"[{exp}] done in {time.time() - t0:.0f}s, missing {n_miss}/{out['missing'].size}", flush=True)
        return {"experiment_id": exp, **c, "missing": n_miss, "n_calls": int(out["missing"].size)}


async def main_async(name: str, tag: str, cap: float, max_runs: int):
    cs = cells(name)
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=cap, experiment_id=f"{tag}-*", ledger_path=LEDGER)
    n_calls = sum(c["N"] * (1 + c["T"] * REPLICAS) for c in cs)
    # measured per call so far: ~1370 prompt tokens (mostly cache reads) and 600-700 output tokens incl. thinking
    usd = guard.project(n_calls, in_tokens=400, cached_tokens=1000, out_tokens=700, model="claude-haiku-5-5",
                        batch=BATCH)
    if usd > cap:
        sys.exit(f"projection ${usd:.2f} exceeds the cap ${cap:.2f}")
    man = ROOT / "results" / tag
    man.mkdir(parents=True, exist_ok=True)
    for c in cs:   # log every graph's realized statistics before spending anything
        log = spectral_log(graph_for(c), eigs=False)
        c["graph_log"] = {k: v for k, v in log.items() if isinstance(v, (int, float))}
        c["truth"] = game_for(c).truth
    (man / "manifest.json").write_text(json.dumps({"cellset": name, "tag": tag, "cap": cap, "cells": cs,
                                                    "projection_usd": usd}, indent=1, default=float))
    sem = asyncio.Semaphore(max_runs)
    res = await asyncio.gather(*(run_cell(c, tag, guard, sem) for c in cs), return_exceptions=True)
    bad = [(cell_id(c), r) for c, r in zip(cs, res) if isinstance(r, BaseException)]
    for cid, e in bad:
        print(f"FAILED {cid}: {e!r}", flush=True)
    print(f"spent ${guard.experiment_spent():.4f} on {tag}; {len(cs) - len(bad)}/{len(cs)} cells done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cellset")
    ap.add_argument("tag")
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    BATCH = a.batch
    asyncio.run(main_async(a.cellset, a.tag, a.cap, a.max_runs))
