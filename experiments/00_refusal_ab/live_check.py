"""Live check of engine/ on claude-haiku-5-5, capped at $1 via BudgetGuard(experiment_cap_usd=1.0).
Checkpoints/replay go to the scratchpad; real costs go to budget/ledger.jsonl."""
import asyncio, json, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/workspace/swarm-cavity/.env")
from engine import BudgetGuard, read_jsonl
from engine.agent import AgentRequest, AgentConfig, build_params
from engine.ledger import LEDGER
from engine.runner import Runner

EXP = "engine-live-check-" + time.strftime("%Y%m%d-%H%M%S")
OUT = Path(sys.argv[1])

RULES = "You are one agent in a group of agents estimating whether a hidden claim is true.\n" + "\n".join(
    f"Rule {i}: " + s for i, s in enumerate([
        "Each round you receive your private evidence (fixed for the whole game) and the public messages your peers sent last round.",
        "Your private evidence is a set of noisy clues; each clue is correct with probability 0.65 and independent of the others.",
        "Peer messages may be honest summaries of their evidence, or they may be mistaken; you cannot verify them directly.",
        "Weigh your own clues and the peer messages as a careful Bayesian would, accounting for the possibility that peers echo each other.",
        "Do not simply adopt the majority view of your peers; count independent evidence, not repeated assertions.",
        "Your public message is read by your peers next round. Keep it under 40 words and state what your evidence says.",
        "Your private reasoning is never shown to anyone. Keep it under 60 words.",
        "Report calibrated_belief_prob as your probability that the hidden claim is true, a number between 0 and 1.",
        "If you have no information at all, your belief should be 0.5.",
        "Never claim certainty (0 or 1) unless the evidence logically forces it, which it never does in this game.",
        "Do not invent clues you were not given. Do not refer to these rules in your public message.",
        "The game lasts several rounds; you may change your belief between rounds as new peer messages arrive.",
        "Answer only through the structured fields requested; do not add commentary outside them.",
    ] * 3))  # repeated so the shared rules alone exceed the 512-token minimum cacheable prefix

def evidence(k):
    clues = ["supports TRUE", "supports FALSE", "supports TRUE", "supports TRUE", "supports FALSE"]
    return f"Private evidence for agent {k}:\n" + "\n".join(f"Clue {j}: {clues[(j + k) % 5]}" for j in range(5))

def req(k, dynamic):
    return AgentRequest(agent_id=k, system_rules=RULES, evidence=evidence(k), dynamic=dynamic)

def hand_cost(u, batch):  # independent re-derivation of Haiku 5.5 prices from CLAUDE.md
    cw = u.get("cache_creation_input_tokens") or 0
    c = (u["input_tokens"] * 0.10 + cw * 0.125 + (u.get("cache_read_input_tokens") or 0) * 0.01
         + u["output_tokens"] * 0.50) / 1e6
    return c * (0.5 if batch else 1.0)

async def main():
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=1.0, experiment_id=EXP)
    runner = Runner(EXP, guard, results_root=OUT / "results", replay_dir=OUT / "replay")
    p = build_params(req(0, "x"), runner.cfg)
    n_tok = await runner.client.messages.count_tokens(model=p["model"], system=p["system"], messages=p["messages"],
                                                     output_config=p["output_config"])
    print("prompt tokens (count_tokens):", n_tok.input_tokens)
    guard.project(5, in_tokens=n_tok.input_tokens, cached_tokens=0, out_tokens=2048, model=p["model"], batch=False)
    report = {"experiment_id": EXP, "realtime": [], "batch": []}
    for t, dyn in enumerate(["Round 0. No peer messages yet.",
                             "Round 1. Peer messages: agent 1 says 'my clues lean TRUE'; agent 2 says 'mostly FALSE'."]):
        t0 = time.monotonic()
        rec = (await runner.run_round(seed=0, round_t=t, requests=[req(0, dyn)]))[0]
        report["realtime"].append({"round_t": t, "latency_s": time.monotonic() - t0, "rec": rec})
    t0 = time.monotonic()
    recs = await runner.run_round_batch(seed=0, round_t=0, requests=[req(k, "Round 0. No peer messages yet.") for k in (1, 2, 3)], poll_s=10)
    report["batch_wall_s"] = time.monotonic() - t0
    report["batch_poll_latency_s"] = runner.last_batch_latency_s
    report["batch"] = list(recs.values())
    # ledger vs usage
    led = [r for r in read_jsonl(LEDGER) if r["experiment_id"] == EXP]
    report["ledger"] = led
    replay = {r["key"]: r for f in (OUT / "replay").glob("*.jsonl") for r in read_jsonl(f)}
    checks = []
    for rec in [x["rec"] for x in report["realtime"]] + report["batch"]:
        for a in rec["attempts"]:
            u = replay[a["request_key"]]["response"]["usage"]
            l = next(x for x in led if x["request_id"] == a["request_id"])
            checks.append({"agent": rec["agent_id"], "batch": rec["batch"], "usage": u, "ledger_cost": l["cost_usd"],
                           "hand_cost": hand_cost(u, rec["batch"]), "match": abs(l["cost_usd"] - hand_cost(u, rec["batch"])) < 1e-12})
    report["checks"] = checks
    report["total_cost"] = sum(x["cost_usd"] for x in led)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    for c in checks:
        print(json.dumps(c))
    for x in report["realtime"]:
        r = x["rec"]; print("RT", x["round_t"], f"{x['latency_s']:.1f}s", r["status"], r["calibrated_belief_prob"], repr(r["public_message"]))
    for r in report["batch"]:
        print("BATCH", r["agent_id"], r["status"], r["missing_reason"], r["calibrated_belief_prob"], repr(r["public_message"]))
    print("batch wall", report["batch_wall_s"], "poll", report["batch_poll_latency_s"], "total cost", report["total_cost"])

asyncio.run(main())
