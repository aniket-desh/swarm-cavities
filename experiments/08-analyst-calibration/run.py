"""Analyst scenario, step 2 (EXPERIMENT.md in this folder). GPU side only: every step writes raw records under
results/; fits, bootstraps and figures are in analyze.py and plot.py (coordinator, CPU).

  tok      versions, BUY/SELL single tokens after the decision prefix, one rendered prompt and its token count
  smoke    6 single-agent decisions per wording (round 0 and round 4 with a 16-colleague template inbox) printed,
           and a 3-round swarm at N=64, K=16 (mild wording, one-way world) whose notes are printed
  pilot    the three trust worlds (gamma = +1, 0, -1 at p_neg = 1/2, K = 16), N = 256, T = 10, one replica, mild
           wording, one graph each; same notes, news and call uniforms in the three worlds. Its notes are the pool for
           the probes. Excluded from all confirmatory analysis.
  probes   N_BASE base probes; each is sent with every wording in WORDINGS and two text arms (real pilot notes /
           template notes at the same calls), plus N_DUP repeats of mild-real probes in a different batch order.

Probe design (one base probe b, rng seeded by (PROBE_SEED, b)):
  - 16 colleague slots; each is a faded colleague with probability 1/2 (the graph's p_neg), so n_allies ~ Bin(16, 1/2);
  - group consensus m_A, m_D ~ U(-1, 1); each colleague's call is BUY with probability (1 + m) / 2, so the normalized
    fields f_A = sum_allies sigma / 16 and f_D = sum_faded sigma / 16 cover their whole range with every composition;
  - real arm: each slot's note is a pilot note written after the same call, drawn from the tercile of senders whose
    belief in that call was lowest ("hesitant") or highest ("confident"), one tercile per group chosen with
    probability 1/2. The sender's belief s_j = 2 P_buy - 1 is stored, so the regression can ask whether readers
    respond to the call or to the confidence in the note (leak). Template arm: the same calls with
    envs.analyst.template_note texts, which carry no confidence (a placebo for the leak regressor);
  - own previous call sigma_own = +-1 with probability 1/2, with a pilot note written after that call (same in both arms);
  - research note h0 ~ N(0, 1) at a0_nom = 0.3; news xi ~ N(0, 1) at nu_nom in {0.5, 1.0} with probability 1/2;
  - round t uniform in 1..9, session tag 90000 + b (the arms and wordings of one base probe share it).

Run on the worker:  uv run --no-sync python experiments/08-analyst-calibration/run.py tok|smoke|pilot|probes
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import envs.analyst as A  # noqa: E402
import envs.vllm_backend as vb  # noqa: E402
from engine.ledger import append_jsonl, read_jsonl  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings, spectral_log  # noqa: E402

PILOT_TAG = "analyst-pilot-20261125"
PROBE_TAG = "analyst-probes-20261125"
RESULTS = ROOT / "results"
OUT = RESULTS / PROBE_TAG
WORLDS = {"mutual": +1.0, "oneway": 0.0, "antagonistic": -1.0}
N_PILOT, K, T_PILOT, GAME_SEED, PILOT_WORDING = 256, 16, 10, 800, "mild"
WORDINGS = ("mild", "moderate", "strong", "max")
ARMS = ("real", "template")
N_BASE, N_DUP, PROBE_SEED = 1000, 100, 8008
NU_NOMS, A0_NOM = (0.5, 1.0), 0.3
BATCH = 512
LOGPROBS = 100          # top-k logprobs of the decision token: censoring of x needs P(other side) < ~1e-8
PROGRESS = os.environ.get("ORCH_PROGRESS")


def progress(msg: str):
    print(msg, flush=True)
    if PROGRESS:
        Path(PROGRESS).write_text(msg + "\n")


def make_llm():
    return vb.make_llm(max_num_seqs=256, max_logprobs=LOGPROBS)


def pilot_graph(world: str) -> np.ndarray:
    k = list(WORLDS).index(world)
    return fixed_indegree_signed_couplings(N_PILOT, K, WORLDS[world], 0.5, np.random.default_rng([GAME_SEED, k]))


# ---------------------------------------------------------------- tok, smoke

def step_tok(llm):
    import torch
    import vllm
    print("vllm", vllm.__version__, "torch", torch.__version__, "gpu", torch.cuda.get_device_name(0), flush=True)
    be = A.AnalystVLLM(llm, "strong", "tok", logprobs=LOGPROBS)
    g = A.make_game(1, 1, seed=0)
    ctx = A.AnalystContext(0, 1, 3, g.notes[0].text(), "UP", 68, own=("BUY", "Leaning up."),
                           allies=(("BUY", "Indicators up."),) * 8, faded=(("SELL", "Down."),) * 8)
    ids = be.decision_ids(ctx)
    print("BUY id", be.buy_id, "SELL id", be.sell_id, "note prefix", [be.tok.decode([i]) for i in be.note_prefix_ids["BUY"]])
    print("prompt tokens", len(ids), "tail", [be.tok.decode([i]) for i in ids[-8:]])
    print(be.tok.decode(ids), flush=True)


def _template_inbox(rng, m_A, m_D, n_A=8):
    calls = lambda m, n: [("BUY" if rng.random() < (1 + m) / 2 else "SELL") for _ in range(n)]
    return (tuple((c, A.template_note(c, rng)) for c in calls(m_A, n_A)),
            tuple((c, A.template_note(c, rng)) for c in calls(m_D, 16 - n_A)))


def step_smoke(llm):
    rng = np.random.default_rng(1)
    g = A.make_game(6, 1, seed=1)
    for w in WORDINGS:
        be = A.AnalystVLLM(llm, w, f"smoke-{w}", logprobs=LOGPROBS)
        ctxs = []
        for i in range(6):
            if i < 2:
                ctxs.append(A.AnalystContext(i, 7, 0, g.notes[i].text(), g.news_side[0, i], int(g.news_pct[0, i])))
            else:
                al, fd = _template_inbox(rng, [1, -1, 0.5, -0.5][i - 2], [-1, 1, 0.5, 0][i - 2])
                ctxs.append(A.AnalystContext(i, 7, 3, g.notes[i].text(), g.news_side[0, i], int(g.news_pct[0, i]),
                                             own=("BUY", "Leaning up."), allies=al, faded=fd))
        bel = be.beliefs(ctxs)
        notes = be.notes([c for c, b in zip(ctxs, bel) if b["ok"]],
                         ["BUY" if b["p_buy"] >= 0.5 else "SELL" for b in bel if b["ok"]], key=(0, 0))
        it = iter(notes)
        for c, b in zip(ctxs, bel):
            nA = sum(k == "BUY" for k, _ in c.allies) - sum(k == "SELL" for k, _ in c.allies)
            nD = sum(k == "BUY" for k, _ in c.faded) - sum(k == "SELL" for k, _ in c.faded)
            n = next(it) if b["ok"] else {"note": None}
            print(f"[{w}] t={c.round_t} news {c.news_side}{c.news_pct} netA {nA:+d} netD {nD:+d}: P(BUY) "
                  f"{b['p_buy']:.4f} x {b['x']:+.2f} valid {b['valid_mass']:.4f} cens {b['censored']} "
                  f"other {b['top_other']} | {n['note']}", flush=True)
    # a small swarm: N=64, K=16, one-way, mild, T=2
    J = fixed_indegree_signed_couplings(64, 16, 0.0, 0.5, np.random.default_rng(5))
    cfg = A.AnalystSwarmConfig("analyst-smoke-20261125", T=2, K=16, results_root=RESULTS)
    out = A.AnalystSwarm(J, A.make_game(64, 2, seed=5), cfg, A.AnalystVLLM(llm, "mild", cfg.experiment_id, logprobs=LOGPROBS)).run()
    recs = list(read_jsonl(RESULTS / cfg.experiment_id / "analyst_calls.jsonl"))
    for r in recs[::4]:
        print(f"t={r['round_t']} i={r['agent_id']} P {r['p_buy_raw']:.3f} call {r['call']:+d} [{r['note_status']}] {r['note']}")
    print("missing", int(out["missing"].sum()), "of", out["missing"].size, flush=True)
    for t in read_jsonl(RESULTS / cfg.experiment_id / "analyst_timings.jsonl"):
        print("timing", t, flush=True)


# ---------------------------------------------------------------- pilot

def step_pilot(llm):
    man = RESULTS / PILOT_TAG
    man.mkdir(parents=True, exist_ok=True)
    (man / "manifest.json").write_text(json.dumps(
        {"N": N_PILOT, "K": K, "T": T_PILOT, "game_seed": GAME_SEED, "wording": PILOT_WORDING, "p_neg": 0.5,
         "worlds": WORLDS, "nu_nom": 0.5, "a0_nom": A0_NOM, "model": vb.MODEL,
         "graphs": {w: {k: v for k, v in spectral_log(pilot_graph(w), eigs=False).items() if k != "eigvals"}
                    for w in WORLDS}}, indent=1, default=float))
    game = A.make_game(N_PILOT, T_PILOT, R=1, seed=GAME_SEED, nu_nom=0.5, a0_nom=A0_NOM)
    for k, w in enumerate(WORLDS):
        t0 = time.time()
        exp = f"{PILOT_TAG}-{w}"
        cfg = A.AnalystSwarmConfig(exp, T=T_PILOT, R=1, seed=GAME_SEED, K=K, results_root=RESULTS)
        out = A.AnalystSwarm(pilot_graph(w), game, cfg, A.AnalystVLLM(llm, PILOT_WORDING, exp, K=K, logprobs=LOGPROBS)).run()
        progress(f"pilot world {k + 1}/3 {w}: {time.time() - t0:.0f}s, missing {int(out['missing'].sum())}/"
                 f"{out['missing'].size}, mean s by round {np.round(np.nanmean(out['s'], (1, 2)), 2).tolist()}")


# ---------------------------------------------------------------- probes

def note_pool() -> dict:
    """Pilot notes (rounds >= 1, decision ok, note complete) by call, split into terciles of the sender's belief in
    its own call, sigma * s: 'hes' (lowest third) and 'conf' (highest third)."""
    rows = []
    for w in WORLDS:
        for r in read_jsonl(RESULTS / f"{PILOT_TAG}-{w}" / "analyst_calls.jsonl"):
            if r["round_t"] >= 1 and r["status"] == "ok" and r["note_status"] == "ok":
                rows.append((r["call"], r["note"], 2 * r["p_buy"] - 1, w, r["round_t"], r["agent_id"]))
    if not rows:
        raise RuntimeError("no pilot notes; run the pilot first")
    pool = {}
    for c in (1, -1):
        rc = sorted([r for r in rows if r[0] == c], key=lambda r: c * r[2])
        n3 = max(len(rc) // 3, 1)
        pool[c] = {"hes": rc[:n3], "conf": rc[-n3:], "all": rc}
    return pool


def pool_stats(pool: dict) -> dict:
    """sigma * s of the senders per call and tercile: the contrast that identifies the leak fraction (EXPERIMENT.md
    fixed rule: leak untestable if the hesitant-confident gap in mean sigma * s is below 0.4)."""
    out = {}
    for c, d in pool.items():
        for k in ("hes", "conf", "all"):
            v = np.array([c * r[2] for r in d[k]])
            out[f"{A.CALLS[c]}|{k}"] = {"n": len(v), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
        out[f"{A.CALLS[c]}|gap"] = out[f"{A.CALLS[c]}|conf"]["mean"] - out[f"{A.CALLS[c]}|hes"]["mean"]
    return out


def base_probes(pool: dict, n: int) -> list[dict]:
    out = []
    for b in range(n):
        rng = np.random.default_rng([PROBE_SEED, b])
        faded = rng.random(K) < 0.5
        m_A, m_D = rng.uniform(-1, 1, 2)
        calls = np.where(rng.random(K) < (1 + np.where(faded, m_D, m_A)) / 2, 1, -1)
        strat = {"A": str(rng.choice(["hes", "conf"])), "D": str(rng.choice(["hes", "conf"]))}
        real = []
        for j in range(K):
            src = pool[int(calls[j])][strat["D" if faded[j] else "A"]]
            r = src[rng.integers(len(src))]
            real.append((r[1], r[2]))                                  # (note, sender belief s)
        trng = np.random.default_rng([PROBE_SEED, b, 7])
        tmpl = [A.template_note(A.CALLS[int(c)], trng) for c in calls]
        own = int(rng.choice([-1, 1]))
        own_src = pool[own]["all"]
        own_note = own_src[rng.integers(len(own_src))][1]
        h0 = rng.standard_normal()
        note = A.research_note(h0, A0_NOM, rng)
        xi, nu_nom = rng.standard_normal(), float(rng.choice(NU_NOMS))
        side, pct, xi_real = A.news_item(xi, nu_nom)
        out.append(dict(b=b, faded=faded.tolist(), calls=calls.tolist(), m_A=m_A, m_D=m_D, strat=strat, real=real,
                        tmpl=tmpl, own=own, own_note=own_note, note=note, xi=xi, xi_real=xi_real, nu_nom=nu_nom,
                        side=side, pct=pct, t=int(rng.integers(1, 10)), order=rng.permutation(K).tolist()))
    return out


def probe_context(bp: dict, arm: str) -> A.AnalystContext:
    allies, faded = [], []
    for j in bp["order"]:
        c = A.CALLS[bp["calls"][j]]
        text = bp["real"][j][0] if arm == "real" else bp["tmpl"][j]
        (faded if bp["faded"][j] else allies).append((c, text))
    return A.AnalystContext(f"b{bp['b']}|{arm}", 90000 + bp["b"], bp["t"], bp["note"].text(), bp["side"], bp["pct"],
                            own=(A.CALLS[bp["own"]], bp["own_note"]), allies=tuple(allies), faded=tuple(faded))


def design_row(bp: dict) -> dict:
    fz, c = np.array(bp["faded"]), np.array(bp["calls"])
    s_send = np.array([s for _, s in bp["real"]])
    return dict(b=bp["b"], n_allies=int((~fz).sum()), fA=float(c[~fz].sum() / K), fD=float(c[fz].sum() / K),
                sA=float(s_send[~fz].sum() / K), sD=float(s_send[fz].sum() / K),       # sender-belief fields
                strat_A=bp["strat"]["A"], strat_D=bp["strat"]["D"], m_A=bp["m_A"], m_D=bp["m_D"],
                own=bp["own"], h0=bp["note"].h0, h0_real=bp["note"].h0_real, xi=bp["xi"], xi_real=bp["xi_real"],
                nu_nom=bp["nu_nom"], round_t=bp["t"])


def step_probes(llm, n_base: int = N_BASE):
    pool = note_pool()
    OUT.mkdir(parents=True, exist_ok=True)
    ps = pool_stats(pool)
    print(json.dumps(ps, indent=1), flush=True)
    (OUT / "pool_stats.json").write_text(json.dumps(ps, indent=1))
    bps = base_probes(pool, n_base)
    (OUT / "design.json").write_text(json.dumps([design_row(bp) for bp in bps]))
    jobs = [(w, arm) for w in WORDINGS for arm in ARMS] + [("mild", "dup")]
    for k, (w, arm) in enumerate(jobs):
        path = OUT / f"{w}_{arm}.jsonl"
        done = {r["b"] for r in read_jsonl(path)}
        todo = [bp for bp in (bps[:N_DUP][::-1] if arm == "dup" else bps) if bp["b"] not in done]
        be = A.AnalystVLLM(llm, w, f"{PROBE_TAG}-{w}", K=K, logprobs=LOGPROBS)
        t0 = time.time()
        for s in range(0, len(todo), BATCH):
            chunk = todo[s:s + BATCH]
            ctxs = [probe_context(bp, "real" if arm == "dup" else arm) for bp in chunk]
            for bp, b in zip(chunk, be.beliefs(ctxs)):
                append_jsonl(path, {"b": bp["b"], "wording": w, "arm": arm, "status": "ok" if b["ok"] else "missing",
                                    **{kk: b[kk] for kk in ("p_buy", "x", "valid_mass", "censored", "lp_buy",
                                                            "lp_sell", "lp_min", "top_other", "n_prompt_tokens")}})
        progress(f"probes {k + 1}/{len(jobs)} {w}/{arm}: {len(todo)} calls in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["tok", "smoke", "pilot", "probes"])
    ap.add_argument("--n-base", type=int, default=N_BASE, help="base probes (EXPERIMENT.md fixed-rule contingency: 600)")
    args = ap.parse_args()
    llm = make_llm()
    for st in args.steps:
        t0 = time.time()
        if st == "probes":
            step_probes(llm, args.n_base)
        else:
            {"tok": step_tok, "smoke": step_smoke, "pilot": step_pilot}[st](llm)
        print(f"== step {st} done in {time.time() - t0:.0f}s", flush=True)
