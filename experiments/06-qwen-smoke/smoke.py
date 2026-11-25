"""GPU smoke test and timing of envs/vllm_backend.py (Qwen3.5-9B on vLLM). See EXPERIMENT.md in this folder.

Steps (each writes into results/<tag>-*/ and appends one record to $EXP_DIR/summary.json):
  env      vLLM version, GPU, and whether the engine starts with the requested options
  tok      tokenizer facts the digit-path belief relies on: "0".."9" and "." single tokens, the generation prompt
           ends in the empty think block, and the sampled prefix ends in a lone " " token
  probes   16 single-agent calls (round 0, and round 1 with agree / disagree inboxes), printed in full
  swarm    envs/swarm.py at N=8, K=2, T=4, two replicas, sync and async (inverted wording, as the pilot)
  timing   one synchronous run at N in (256, 1024), K=4, T=2, one replica: seconds per round from vllm_timings

Run on the worker:  uv run --no-sync python experiments/06-qwen-smoke/smoke.py <step>... --tag qwen-smoke
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from envs.evidence_game import evidence_for_field, make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402
from envs.prompts import AgentContext, InboxMsg  # noqa: E402
from envs.swarm import Swarm, SwarmConfig  # noqa: E402
import envs.vllm_backend as vb  # noqa: E402

EXP_DIR = Path(os.environ.get("EXP_DIR", Path(__file__).parent))
SUMMARY = EXP_DIR / "summary.json"


def record(step: str, out: dict):
    s = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    s[step] = out
    SUMMARY.write_text(json.dumps(s, indent=1, default=float))
    print(f"== {step}: {json.dumps(out, default=float)[:2000]}", flush=True)


def step_env(llm, args) -> dict:
    import torch
    import vllm

    return {"vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "gpu_mem_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "prefix_caching": args.prefix_caching, "model": vb.MODEL}


def step_tok(llm, args) -> dict:
    tok = llm.get_tokenizer()
    ids = {s: tok.encode(s, add_special_tokens=False) for s in [str(d) for d in range(10)] + ["."]}
    ev = evidence_for_field(0.4, 3, np.random.default_rng(0))
    ctx = AgentContext(agent_id=0, session=0, round_t=0, h_priv=ev.h_priv, evidence=ev.text())
    prompt = vb.render_prompt(tok, ctx, 1)
    canon = tok.encode('{"rationale": "a", "public_message": "b", "calibrated_belief_prob": 0.62}',
                       add_special_tokens=False)
    out = {"single_token": {s: len(v) == 1 for s, v in ids.items()},
           "prompt_tail": prompt[-60:], "ends_with_empty_think": prompt.endswith("<think>\n\n</think>\n\n"),
           "canonical_number_tokens": [tok.decode([t]) for t in canon[-8:]],
           "n_prompt_tokens_round0": len(tok.encode(prompt, add_special_tokens=False))}
    assert all(out["single_token"].values()), out
    assert out["ends_with_empty_think"], out["prompt_tail"]
    return out


def probe_contexts() -> list[AgentContext]:
    rng = np.random.default_rng(1)
    ctxs = []
    for i in range(16):
        ev = evidence_for_field(rng.choice([-1, 1]) * 0.5 + rng.standard_normal(), 3, rng)
        if i < 4:
            ctxs.append(AgentContext(agent_id=f"p{i}", session=i, round_t=0, h_priv=ev.h_priv, evidence=ev.text()))
            continue
        p0 = round(float((1 + np.tanh(0.42 * ev.h_priv)) / 2), 2)
        msgs = [InboxMsg(p, t) for p, t in [(0.82, "Two of my three sensors say HEADS, the strongest at 80%."),
                                            (0.31, "My sensors lean TAILS, though one says HEADS."),
                                            (0.55, "Split reports, slight lean HEADS."),
                                            (0.12, "All my sensors report TAILS; I'm fairly confident.")]]
        rng.shuffle(msgs)
        ctxs.append(AgentContext(agent_id=f"p{i}", session=i, round_t=1, h_priv=ev.h_priv, evidence=ev.text(),
                                 history=((p0, "From my own sensors I lean one way."),),
                                 agree=tuple(msgs[:2]), disagree=tuple(msgs[2:])))
    return ctxs


def step_probes(llm, args) -> dict:
    out = {}
    for wording in ("label", "inverted"):
        be = vb.VLLMBackend(llm, f"{args.tag}-probes-{wording}", disagree=wording)
        res = be.respond_sync(0, 0, probe_contexts(), 1)
        for c, r in zip(probe_contexts(), res):
            rec = r.record
            print(f"[{wording} {c.agent_id} t={c.round_t} h={c.h_priv:+.2f}] {rec['status']} "
                  f"p={rec.get('p_logprob_raw')} vm={rec.get('valid_mass')} sampled={rec.get('p_sampled')}\n"
                  f"   rationale: {rec.get('rationale')}\n   message: {rec.get('public_message')}", flush=True)
        # the last tokens of the sampled prefixes: the number should start after a lone " " token
        out[wording] = {**vb.summarize_calls(be.ckpt),
                        "prefix_tails": sorted({"|".join(r.record["prefix_tail"]) for r in res
                                                if r.record["status"] == "ok"})}
    return out


def step_swarm(llm, args) -> dict:
    out = {}
    N, K, T = 8, 2, 4
    J = fixed_indegree_signed_couplings(N, K, 0.0, 0.5, np.random.default_rng(8))
    game = make_game(N, mu=0.5, sigma=1.0, k=3, seed=10_008)
    for mode in ("sync", "async"):
        exp = f"{args.tag}-swarm-{mode}"
        be = vb.VLLMBackend(llm, exp, disagree="inverted")
        t0 = time.time()
        o = asyncio.run(Swarm(J, game, SwarmConfig(experiment_id=exp, T=T, H=1, replicas=2, mode=mode, seed=8),
                              be).run())
        S = game.truth * o["s"]
        out[mode] = {"wall_s": time.time() - t0, "swarm_missing": int(o["missing"].sum()), "swarm_n": int(o["missing"].size),
                     "m_truth_aligned_by_round": np.nanmean(S, axis=(0, 2)).round(3).tolist(),
                     "replica_rms_diff_by_round": np.sqrt(np.nanmean((o["p"][0] - o["p"][1]) ** 2, axis=1)).round(3).tolist(),
                     **vb.summarize_calls(be.ckpt)}
    return out


def step_timing(llm, args) -> dict:
    out = {}
    for N in args.timing_N:
        exp = f"{args.tag}-timing-N{N}"
        J = fixed_indegree_signed_couplings(N, 4, 0.0, 0.5, np.random.default_rng(N))
        game = make_game(N, mu=0.5, sigma=1.0, k=3, seed=20_000 + N)
        be = vb.VLLMBackend(llm, exp, disagree="inverted")
        t0 = time.time()
        o = asyncio.run(Swarm(J, game, SwarmConfig(experiment_id=exp, T=2, H=1, replicas=1, seed=N), be).run())
        tm = [json.loads(line) for line in open(be.dir / "vllm_timings.jsonl")]
        out[f"N{N}"] = {"wall_s": time.time() - t0, "swarm_missing": int(o["missing"].sum()), "swarm_n": int(o["missing"].size),
                        "rounds": [{k: r[k] for k in ("round_t", "n_calls", "sample_s", "stage1_s", "stage2_s",
                                                      "n_stage2", "total_s", "prompt_tokens", "gen_tokens")}
                                   for r in tm],
                        **vb.summarize_calls(be.ckpt)}
    return out


STEPS = {"env": step_env, "tok": step_tok, "probes": step_probes, "swarm": step_swarm, "timing": step_timing}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(STEPS))
    ap.add_argument("--tag", default="qwen-smoke")
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--timing-N", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    llm = vb.make_llm(enable_prefix_caching=bool(a.prefix_caching), max_num_seqs=a.max_num_seqs)
    record("engine_start", {"load_s": time.time() - t0, "prefix_caching": a.prefix_caching,
                            "max_num_seqs": a.max_num_seqs})
    for s in a.steps:
        t1 = time.time()
        r = STEPS[s](llm, a)
        r["step_s"] = time.time() - t1
        record(s, r)
