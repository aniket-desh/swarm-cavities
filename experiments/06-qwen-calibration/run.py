"""Qwen3.5-9B calibration with real Qwen swarm messages (EXPERIMENT.md in this folder). GPU side only: every step
writes raw records under results/; fits, bootstraps and figures are in analyze.py and plot.py (coordinator, CPU).

  pilot       the Haiku pilot's cell set (experiments/05_swarm/cells.py "pilot": inverted wording, p_neg=0.5,
              K in {2, 4}, gamma in {-1, 0, +1}, N=64, T=10, two replicas, graph 900) on the vLLM backend, with
              the same graphs, evidence and seeds. Its messages are the pool for the next two steps.
  transfer    the paired real-vs-template design of experiments/05_swarm/transfer.py (functions imported, file
              unchanged) with the Qwen pilot as the message pool, for wording in {inverted, label} and K in {2, 4, 8}.
              K=8 has no pilot cell, so its inboxes draw from the K=4 pool (POOL_K).
  conformity  cavity.probes.conformity_design (uniform all-agree inboxes at fixed hbar), K in {4, 8}, both wordings,
              two text arms: "template" (probes.peer_text) and "real" (each slot's text replaced by a Qwen pilot
              message that stated the nearest probability on the same side, mirrored as needed; the displayed
              probability, hence hbar, is unchanged). Own last answer 0.70 HEADS in both arms, and also 0.30 TAILS in
              the real arm, which measures the weight kappa of the own last answer at fixed inbox.

Run on the worker:  uv run --no-sync python experiments/06-qwen-calibration/run.py pilot|transfer|conformity
"""

import argparse
import asyncio
import dataclasses
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))

import numpy as np  # noqa: E402

import cavity.probes as P  # noqa: E402
import cells as haiku_cells  # noqa: E402  (experiments/05_swarm/cells.py: cell set, graphs, games)
import transfer as haiku_transfer  # noqa: E402  (experiments/05_swarm/transfer.py: mirror, base_probes, context)
import envs.vllm_backend as vb  # noqa: E402
from envs.prompts import InboxMsg  # noqa: E402
from envs.swarm import Swarm, SwarmConfig  # noqa: E402

PILOT_TAG = "qwen-pilot-20261125"
TRANSFER_TAG = "qwen-transfer-20261125"
CONF_TAG = "qwen-conf-20261125"
OUT = ROOT / "results" / "qwen-calib"
POOL_K = {2: 2, 4: 4, 8: 4}
TRANSFER_KS, WORDINGS, ARMS = (2, 4, 8), ("inverted", "label"), ("real", "template")
N_BASE = 600
CONF_KS = (2, 4, 8)
CONF_HBARS = (-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
PROGRESS = os.environ.get("ORCH_PROGRESS")


def progress(msg: str):
    print(msg, flush=True)
    if PROGRESS:
        Path(PROGRESS).write_text(msg + "\n")


# ---------------------------------------------------------------- pilot

def pilot(llm):
    cs = haiku_cells.cells("pilot")
    man = ROOT / "results" / PILOT_TAG
    man.mkdir(parents=True, exist_ok=True)
    for c in cs:
        c["truth"] = haiku_cells.game_for(c).truth
    # same manifest layout as cells.py, so experiments/05_swarm/analyze.py reads this tag unchanged
    (man / "manifest.json").write_text(json.dumps({"cellset": "pilot", "tag": PILOT_TAG, "backend": vb.MODEL,
                                                    "cells": cs}, indent=1, default=float))
    for k, c in enumerate(cs):
        exp = f"{PILOT_TAG}-{haiku_cells.cell_id(c)}"
        t0 = time.time()
        be = vb.VLLMBackend(llm, exp, disagree=c["wording"])
        cfg = SwarmConfig(experiment_id=exp, T=c["T"], H=haiku_cells.H, replicas=haiku_cells.REPLICAS,
                          mode=c["mode"], seed=c["graph"])
        out = asyncio.run(Swarm(haiku_cells.graph_for(c), haiku_cells.game_for(c), cfg, be).run())
        progress(f"pilot cell {k + 1}/{len(cs)} {exp}: {time.time() - t0:.0f}s, missing "
                 f"{int(out['missing'].sum())}/{out['missing'].size}")


# ---------------------------------------------------------------- message pool

def qwen_pool(K: int) -> list[tuple[float, str]]:
    """(displayed probability, public message) of every parsed Qwen pilot reply at rounds >= 1 and in-degree
    POOL_K[K]; same selection as transfer.message_pool, from the vLLM checkpoints."""
    pool = []
    for f in sorted(glob.glob(str(ROOT / "results" / f"{PILOT_TAG}-*-K{POOL_K[K]}-*" / "vllm_calls.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["round_t"] >= 1 and r["status"] == "ok" and r["public_message"].strip():
                pool.append((round(float(r["calibrated_belief_prob"]), 2), r["public_message"]))
    if not pool:
        raise RuntimeError(f"no Qwen pilot messages for K={K}; run the pilot first")
    return pool


# ---------------------------------------------------------------- transfer

def transfer(llm):
    haiku_transfer.message_pool = qwen_pool          # base_probes draws its peer messages from the Qwen pilot
    OUT.mkdir(parents=True, exist_ok=True)
    for wording in WORDINGS:
        be = vb.VLLMBackend(llm, f"{TRANSFER_TAG}-{wording}", disagree=wording)
        recs = []
        for K in TRANSFER_KS:
            t0 = time.time()
            bps = haiku_transfer.base_probes(K, N_BASE)
            ctxs = [haiku_transfer.context(bp, arm, 50_000 + 1000 * K + bp["b"]) for bp in bps for arm in ARMS]
            res = be.respond_sync(0, 0, ctxs, 1)
            for ctx, r in zip(ctxs, res):
                bp = bps[int(ctx.agent_id.split("|b")[1].split("|")[0])]
                rec = r.record
                recs.append({"wording": wording, "K": K, "b": bp["b"], "arm": ctx.agent_id.rsplit("|", 1)[1],
                             "f_A": sum(2 * m.prob - 1 for m in ctx.agree) / K,
                             "f_D": sum(2 * m.prob - 1 for m in ctx.disagree) / K,
                             "n_agree": len(ctx.agree), "h_priv": ctx.h_priv, "u": 2 * bp["hist"][0][0] - 1,
                             "status": r.status, "reason": r.reason,
                             "s": 2 * r.prob - 1 if r.status == "ok" else None,
                             "s_sampled": 2 * rec["p_sampled"] - 1 if rec.get("p_sampled") is not None else None,
                             "valid_mass": rec.get("valid_mass"), "message": r.message})
            progress(f"transfer {wording} K={K}: {len(ctxs)} calls in {time.time() - t0:.0f}s")
        (OUT / f"transfer_{wording}.json").write_text(json.dumps(recs))


# ---------------------------------------------------------------- conformity

def real_texts(msgs, pool: list[tuple[float, str]], rng: np.random.Generator) -> tuple[InboxMsg, ...]:
    """Replace each message's text by a pilot message that stated the nearest probability on the same side
    (pool mirrored so both sides are covered); the displayed probability is kept."""
    both = pool + [haiku_transfer.mirror(p, t) for p, t in pool]
    probs = np.array([p for p, _ in both])
    out = []
    for m in msgs:
        d = np.abs(probs - m.prob)
        cand = np.flatnonzero(d <= d.min() + 1e-9)
        out.append(InboxMsg(m.prob, both[int(rng.choice(cand))][1]))
    return tuple(out)


def conformity(llm):
    OUT.mkdir(parents=True, exist_ok=True)
    for wording in WORDINGS:
        be = vb.VLLMBackend(llm, f"{CONF_TAG}-{wording}", disagree=wording)
        recs = []
        for K in CONF_KS:
            t0 = time.time()
            pool = qwen_pool(K)
            # own last answer 0.70 HEADS (+1) in both arms; 0.30 TAILS (-1) too in the real arm, so the weight of
            # the own last answer (kappa) is measured at fixed inbox
            probes = P.conformity_design(K=K, hbars=CONF_HBARS, ev_levels=(0.4, 1.4), hist_states=(+1, -1), n=20)
            arms = []
            for arm in ("template", "real"):
                for pr in probes:
                    if arm == "template" and pr.design["hist"] != +1:
                        continue
                    ctx = pr.ctx
                    if arm == "real":
                        rng = np.random.default_rng([K, P._key(pr.design["hbar"]), pr.design["sample"],
                                                     P._key(pr.design["ev_level"])])
                        ctx = dataclasses.replace(ctx, agree=real_texts(ctx.agree, pool, rng))
                    ctx = dataclasses.replace(ctx, agent_id=f"K{K}|{arm}|{ctx.agent_id}")
                    arms.append(P.Probe(ctx, {**pr.design, "arm": arm, "wording": wording}))
            res = be.respond_sync(0, 0, [p.ctx for p in arms], 1)
            for rec, r in zip(P._records(arms, res), res):
                rec["s_sampled"] = (2 * r.record["p_sampled"] - 1) if r.record.get("p_sampled") is not None else None
                rec["valid_mass"] = r.record.get("valid_mass")
                rec["p"] = None if not np.isfinite(rec["p"]) else rec["p"]
                rec["s"] = None if rec["p"] is None else rec["s"]
                recs.append(rec)
            progress(f"conformity {wording} K={K}: {len(arms)} calls in {time.time() - t0:.0f}s")
        (OUT / f"conformity_{wording}.json").write_text(json.dumps(recs, default=float))


STEPS = {"pilot": pilot, "transfer": transfer, "conformity": conformity}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(STEPS))
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--prefix-caching", type=int, default=1)
    a = ap.parse_args()
    llm = vb.make_llm(enable_prefix_caching=bool(a.prefix_caching), max_num_seqs=a.max_num_seqs)
    for s in a.steps:
        STEPS[s](llm)
