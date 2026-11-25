"""De-biased sign-rule test (EXPERIMENT.md in this folder; predictions frozen in theory/analyst_signrule.md and
theory/checks/analyst_signrule_predictions.json). GPU side only; analysis in analyze.py, figures in plot.py.

Design (the note's sections 1, 6, 7): cell 08b P0 = wording "max", PromptVariant() default, news nu_nom = 0.5, note
a0_nom = 0.3; calls drawn by the experimenter from tanh(x - chat), chat = -0.6366..., with a stored uniform per
(replica, round, analyst); the model still writes its note from its own prompt after the sampled call.
N = 256, K = 16, p_neg = 1/2, T = 20 (21 rounds), R = 2 replicas (lockstep, same session tag, own call uniforms).
Runs: mutual (gamma = +1) g = 0, 1; one-way (gamma = 0) g = 0..4; antagonistic (gamma = -1) g = 0..4; each run its
own graph, notes, news and uniforms.

Kick twins (envs/analyst.kick_slice) on settled rounds t0 in {8, 12}, both replicas, one kick per analyst per slice:
  mutual 100 kicks + 25 null twins over 8 slices; one-way 20 kicks over 20 slices (protocol check: E = 0 exactly);
  antagonistic 400 kicks + 25 null twins over 20 slices. Analysts per slice drawn without replacement (seeded).

Asynchronous control (theory/analyst_signrule.md Amendment 1, added by amendment of EXPERIMENT.md): block-random
schedule, B = 16 blocks of 16, permutation per (graph, round) from the stored seed (8080, world, g, t), shared by
the replicas; 4 mutual and 4 one-way runs on new graphs g = 10..13, 2 replicas, everything else as above.

Steps (run on the worker, one engine per command):
  smoke                       decision tokens, one rendered prompt, and a timing run (N = 256, R = 2, T = 1, mutual;
                              its own tag, not analysed)
  base RUN [RUN ...]          base runs, RUN = world:g, in the order given
  kicks RUN [RUN ...]         the twins of those runs' slices
  asmoke                      timing of the block schedule: N = 256, R = 2, T = 2 (rounds 1 and 2 asynchronous)
  abase RUN [RUN ...]         asynchronous runs, RUN = world:g with g in 10..13
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import envs.analyst as A  # noqa: E402
import envs.vllm_backend as vb  # noqa: E402
from engine.ledger import append_jsonl, read_jsonl  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings, spectral_log  # noqa: E402

PRED = json.loads((ROOT / "theory" / "checks" / "analyst_signrule_predictions.json").read_text())
CHAT = float(PRED["design"]["chat"])                       # -0.6366486620518087, frozen
TAG = "analyst-signrule-20261125"
RESULTS = ROOT / "results"
N, K, T, R, P_NEG = 256, 16, 20, 2, 0.5
WORDING, VARIANT, NU_NOM, A0_NOM = "max", A.PromptVariant(), 0.5, 0.3
WORLDS = {"mutual": 1.0, "oneway": 0.0, "antagonistic": -1.0}
RUNS = [("mutual", 0), ("mutual", 1)] + [("oneway", g) for g in range(5)] + [("antagonistic", g) for g in range(5)]
ASYNC_RUNS = [("mutual", g) for g in range(10, 14)] + [("oneway", g) for g in range(10, 14)]
BLOCKS = 16
T0S = (8, 12)
# twins per slice, slices ordered (g, rep, t0) lexicographically
KICKS = {"mutual": [13, 13, 13, 13, 12, 12, 12, 12], "oneway": [1] * 20, "antagonistic": [20] * 20}
NULLS = {"mutual": [4, 3, 3, 3, 3, 3, 3, 3], "oneway": [0] * 20, "antagonistic": [2] * 5 + [1] * 15}
LOGPROBS = 100
PROGRESS = os.environ.get("ORCH_PROGRESS")


def progress(msg: str):
    print(msg, flush=True)
    if PROGRESS:
        Path(PROGRESS).write_text(msg + "\n")


def run_id(world: str, g: int) -> str:
    return f"{TAG}-{world}-g{g}"


def graph(world: str, g: int) -> np.ndarray:
    k = list(WORLDS).index(world)
    return fixed_indegree_signed_couplings(N, K, WORLDS[world], P_NEG, np.random.default_rng([9090, k, g]))


def swarm(world: str, g: int, backend, T_=T, n=N, tag=None, blocks=1) -> A.AnalystSwarm:
    k = list(WORLDS).index(world)
    J = graph(world, g) if n == N else fixed_indegree_signed_couplings(n, K, WORLDS[world], P_NEG,
                                                                          np.random.default_rng([9091, k, g]))
    game = A.make_game(n, T_, R=R, seed=9000 + 100 * k + g, nu_nom=NU_NOM, a0_nom=A0_NOM)
    cfg = A.AnalystSwarmConfig(tag or run_id(world, g), T=T_, R=R, seed=9000 + 100 * k + g, K=K, chat=CHAT,
                               session=100 * k + g, results_root=RESULTS, blocks=blocks,
                               order_seed=int(np.random.SeedSequence([8080, k, g]).generate_state(1)[0]))
    return A.AnalystSwarm(J, game, cfg, backend)


def backend(llm, name: str):
    return A.AnalystVLLM(llm, WORDING, name, K=K, logprobs=LOGPROBS, variant=VARIANT)


def slices(world: str, g: int):
    """(slice index within the world, rep, t0, kicked, nulls) for run g, as pre-registered."""
    out = []
    idx = 0
    gs = sorted({gg for w, gg in RUNS if w == world})
    for gg in gs:
        for rep in range(R):
            for t0 in T0S:
                if gg == g:
                    nk, nn = KICKS[world][idx], NULLS[world][idx]
                    pick = np.random.default_rng([7070, list(WORLDS).index(world), gg, rep, t0]).choice(N, nk + nn,
                                                                                                    replace=False)
                    out.append((idx, rep, t0, sorted(map(int, pick[:nk])), sorted(map(int, pick[nk:]))))
                idx += 1
    return out


# ---------------------------------------------------------------- steps

def step_smoke(llm, label: str):
    import torch
    import vllm
    print("vllm", vllm.__version__, "torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)
    tag = f"{TAG}-smoke-{label}"
    sw = swarm("mutual", 99, backend(llm, tag), T_=1, tag=tag)
    ids = sw.backend.decision_ids(sw.context(0, 0, 0, None, None))
    print("BUY/SELL ids", sw.backend.buy_id, sw.backend.sell_id, "round-0 prompt tokens", len(ids), flush=True)
    t0 = time.time()
    out = sw.run()
    print(f"smoke: 2 lockstep rounds x {R} replicas x {N} analysts in {time.time() - t0:.0f} s; missing "
          f"{int(out['missing'].sum())}/{out['missing'].size}", flush=True)
    for t in read_jsonl(RESULTS / tag / "analyst_timings.jsonl"):
        print("timing", t, flush=True)
    recs = list(read_jsonl(RESULTS / tag / "analyst_calls.jsonl"))
    for r in recs[::64]:
        print(f"t={r['round_t']} rep={r['replica']} i={r['agent_id']} x {r['x']:+.2f} p_sample {r['p_sample']:.3f} "
              f"call {r['call']:+d} [{r['note_status']}] {r['note']}", flush=True)
    print(sw.backend.tok.decode(sw.backend.decision_ids(sw.context(0, 1, 5, np.array([r['call'] for r in recs[:N]]),
                                                                      [r['note'] for r in recs[:N]]))), flush=True)


def step_base(llm, runs):
    for k, (w, g) in enumerate(runs):
        t0 = time.time()
        sw = swarm(w, g, backend(llm, run_id(w, g)))
        man = sw.dir / "graph.json"
        sw.dir.mkdir(parents=True, exist_ok=True)
        if not man.exists():
            man.write_text(json.dumps({k_: v for k_, v in spectral_log(sw.J, eigs=False).items() if k_ != "eigvals"},
                                      default=float))
        out = sw.run()
        s = np.tanh(np.nan_to_num(out["x"]) - CHAT)
        progress(f"base {k + 1}/{len(runs)} {w}:{g}: {time.time() - t0:.0f} s, missing {int(out['missing'].sum())}/"
                 f"{out['missing'].size}, mean sampling belief rounds 5-20 {s[5:].mean():+.3f}, q {(s[5:] ** 2).mean():.3f}")


def async_id(world: str, g: int) -> str:
    return f"{TAG}-async-{world}-g{g}"


def step_asmoke(llm, label: str):
    tag = f"{TAG}-asmoke-{label}"
    sw = swarm("mutual", 98, backend(llm, tag), T_=2, tag=tag, blocks=BLOCKS)
    t0 = time.time()
    out = sw.run()
    print(f"asmoke: rounds 0 (sync), 1, 2 (B = {BLOCKS}) x {R} replicas in {time.time() - t0:.0f} s; missing "
          f"{int(out['missing'].sum())}/{out['missing'].size}", flush=True)
    for t in read_jsonl(RESULTS / tag / "analyst_timings.jsonl"):
        print("timing", t, flush=True)


def step_abase(llm, runs):
    for k, (w, g) in enumerate(runs):
        t0 = time.time()
        sw = swarm(w, g, backend(llm, async_id(w, g)), tag=async_id(w, g), blocks=BLOCKS)
        sw.dir.mkdir(parents=True, exist_ok=True)
        (sw.dir / "graph.json").write_text(json.dumps({k_: v for k_, v in spectral_log(sw.J, eigs=False).items()
                                                       if k_ != "eigvals"}, default=float))
        out = sw.run()
        s = np.tanh(np.nan_to_num(out["x"]) - CHAT)
        progress(f"abase {k + 1}/{len(runs)} {w}:{g}: {time.time() - t0:.0f} s, missing {int(out['missing'].sum())}/"
                 f"{out['missing'].size}, mean sampling belief rounds 5-20 {s[5:].mean():+.3f}")


def step_kicks(llm, runs):
    for k, (w, g) in enumerate(runs):
        sw = swarm(w, g, backend(llm, run_id(w, g)))
        path = RESULTS / TAG / f"kicks_{w}_g{g}.jsonl"          # one file per run: the two workers never collide
        done = {r["slice"] for r in read_jsonl(path)}
        t0 = time.time()
        n = 0
        for idx, rep, t0_, kicked, nulls in slices(w, g):
            if idx in done:
                continue
            recs = A.kick_slice(sw, sw.backend, rep, t0_, kicked, nulls, tag=f"kick-{w}-g{g}")
            # one line per slice (a single append), so a stop never leaves a partial slice behind
            append_jsonl(path, {"world": w, "g": g, "slice": idx, "rep": rep, "t0": t0_, "twins": recs})
            n += len(recs)
        progress(f"kicks {k + 1}/{len(runs)} {w}:{g}: {n} twins in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["smoke", "base", "kicks", "asmoke", "abase"])
    ap.add_argument("runs", nargs="*", help="world:g")
    ap.add_argument("--label", default="W", help="worker label for the smoke tag")
    a = ap.parse_args()
    runs = [(r.split(":")[0], int(r.split(":")[1])) for r in a.runs]
    assert all(r in (ASYNC_RUNS if a.step == "abase" else RUNS) for r in runs), runs
    llm = vb.make_llm(max_num_seqs=256, max_logprobs=LOGPROBS)
    t = time.time()
    {"smoke": lambda: step_smoke(llm, a.label), "base": lambda: step_base(llm, runs), "kicks": lambda: step_kicks(llm, runs),
     "asmoke": lambda: step_asmoke(llm, a.label), "abase": lambda: step_abase(llm, runs)}[a.step]()
    print(f"== step {a.step} done in {time.time() - t:.0f} s", flush=True)
