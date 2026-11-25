"""Prompt check of envs/prompts.py on claude-haiku-5-5: 20 realtime calls over a few inbox compositions,
capped at $0.05 by BudgetGuard(experiment_cap_usd=0.05). Costs go to budget/ledger.jsonl under
experiment_id prompt-check-<timestamp>; checkpoints and the replay cache under results/<experiment_id>/.

Calls (one agent, evidence level h_priv ~ 0.4, last own answer 0.70, K = 8 peers, cavity/probes.py inboxes):
  2  round 0 (no inbox, no history)
  10 uniform agree-group inboxes at hbar = -0.6, -0.3, 0, 0.3, 0.6 (2 samples each)
  4  disagree-only inboxes at x = -0.6, +0.6 (2 samples each)
  4  the four non-uniform compositions of claim1.md section 4.1 at hbar = +0.2
Run from the repo root: python experiments/01_prompt_check/prompt_check.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

import cavity.probes as P  # noqa: E402
from engine import BudgetGuard, read_jsonl  # noqa: E402
from engine.agent import AgentConfig, build_params  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.prompts import AgentContext, make_request  # noqa: E402
from envs.swarm import LLMBackend  # noqa: E402

LEDGER = ROOT / "budget" / "ledger.jsonl"
EXP = "prompt-check-" + time.strftime("%Y%m%d-%H%M%S")
H = 1


def probes() -> list[P.Probe]:
    ev = P._evidence(0.4, 3, 0)
    out = []
    for smp in range(2):  # round 0: evidence only
        ctx = AgentContext(agent_id=f"r0|{smp}", session=50_000 + smp, round_t=0, h_priv=ev.h_priv, evidence=ev.text())
        out.append(P.Probe(ctx, {"set": "round0", "h_priv": ev.h_priv, "K": 8, "u": 0.0, "f_A": 0.0, "f_D": 0.0,
                                 "hbar_realized": 0.0, "sample": smp}))
    out += P.conformity_design(hbars=(-0.6, -0.3, 0.0, 0.3, 0.6), n=2)
    out += [p for p in P.gains_design(xs=(-0.6, 0.6), n=2) if p.design["arm"] == "disagree"]
    out += [p for p in P.sufficiency_design(levels=(2,), ev_levels=(0.4,), hist_states=(1,), n=1)
            if p.design["comp"] in ("polarized", "few_strong", "route_A", "route_B")]
    assert len(out) == 20
    return out


async def main(out_dir: Path):
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=0.05, experiment_id=EXP, ledger_path=LEDGER)
    runner = Runner(EXP, guard, cfg=AgentConfig(), concurrency=4, ledger_path=LEDGER)
    prs = probes()
    # measured prompt size of the longest request (count_tokens is free), then the projection
    longest = max(prs, key=lambda p: len(make_request(p.ctx, H).dynamic))
    params = build_params(make_request(longest.ctx, H), runner.cfg)
    n_tok = (await runner.client.messages.count_tokens(model=params["model"], system=params["system"],
                                                       messages=params["messages"],
                                                       output_config=params["output_config"])).input_tokens
    print("longest prompt, count_tokens:", n_tok, "| chars:", sum(len(x["text"]) for x in params["system"])
          + len(params["messages"][0]["content"]))
    usd = guard.project(len(prs), in_tokens=n_tok, cached_tokens=0, out_tokens=600, model=params["model"], batch=False)
    P.project_cost(guard, [p.ctx for p in prs], H)
    if usd > 0.05:
        raise SystemExit(f"projection ${usd:.4f} exceeds the $0.05 cap")
    res = await LLMBackend(runner).respond(0, 0, [p.ctx for p in prs], H)
    rec = P._records(prs, res)
    for r, x in zip(rec, res):
        r["attempts"] = x.record["attempts"]
    led = [r for r in read_jsonl(LEDGER) if r["experiment_id"] == EXP]
    tok = {k: [r[k] for r in led] for k in ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")}
    refusals = sum(r["reason"] == "refusal" for r in rec)
    parse_fail = sum(1 for r in rec for a in r["attempts"] if a["reason"] not in (None, "refusal"))
    conf = [r for r in rec if r["set"] == "conformity"]
    hb = np.array([r["hbar_realized"] for r in conf])
    pp = np.array([r["p"] for r in conf])
    ok = np.isfinite(pp)
    slope = float(np.polyfit(hb[ok], pp[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    dis = {x: [r["p"] for r in rec if r["set"] == "gains" and r["x"] == x] for x in (-0.6, 0.6)}
    summary = {
        "experiment_id": EXP, "n_calls_ledger": len(led), "total_cost_usd": sum(r["cost_usd"] for r in led),
        "refusals": refusals, "missing": sum(r["status"] != "ok" for r in rec), "parse_failures_incl_retried": parse_fail,
        "tokens_mean": {k: float(np.mean(v)) for k, v in tok.items()}, "tokens_max": {k: int(np.max(v)) for k, v in tok.items()},
        "round0_p": [r["p"] for r in rec if r["set"] == "round0"], "h_priv": prs[0].ctx.h_priv,
        "uniform_p_by_hbar": {f"{h:+.2f}": [float(p) for p, hh in zip(pp, hb) if abs(hh - h) < 1e-9] for h in sorted(set(hb))},
        "uniform_slope_dp_dhbar": slope,
        "disagree_only_p": {str(k): v for k, v in dis.items()},
        "compositions_hbar_0.2": {r["comp"]: r["p"] for r in rec if r["set"] == "sufficiency"},
        "messages": [(r["id"], r["p"], r["message"]) for r in rec],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{EXP}.json").write_text(json.dumps({"summary": summary, "records": rec}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "messages"}, indent=1, default=str))
    for m in summary["messages"]:
        print(m)


if __name__ == "__main__":
    asyncio.run(main(Path(__file__).resolve().parent))
