"""A/B/C test of reasoning_extraction refusals on claude-haiku-5-5 (realtime, 12 calls per arm).
A: current schema (private_reasoning, 'never shown to peers') + rules with 'private reasoning is never shown'.
B: field renamed to 'rationale', no 'private'/'never shown' wording anywhere.
C: no reasoning field at all (public_message + calibrated_belief_prob).
Costs go to budget/ledger.jsonl under experiment_id refusal-ab-*."""
import asyncio, copy, json, sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/swarm-cavity")
from dotenv import load_dotenv
load_dotenv("/workspace/swarm-cavity/.env")
import anthropic
from engine.agent import AgentConfig, build_params
from engine.pricing import cost_usd, usage_tokens
from engine.ledger import record_call

src = Path(__file__).with_name("live_check.py").read_text().split("def hand_cost")[0]
ns = {}; exec(src.replace("from engine import BudgetGuard, read_jsonl", "").replace("from engine.runner import Runner", ""), ns)
RULES_A = ns["RULES"]
RULES_B = RULES_A.replace("Your private reasoning is never shown to anyone. Keep it under 60 words.",
                          "Your rationale is one or two sentences on why you hold your belief.")
RULES_C = RULES_A.replace("Your private reasoning is never shown to anyone. Keep it under 60 words.",
                          "Think it through, then report only the fields requested.")
EXP = "refusal-ab-" + time.strftime("%Y%m%d-%H%M%S")
PEERS = ["agent 1 says 'my clues lean TRUE'; agent 2 says 'mostly FALSE'.",
         "agent 3 says 'three of five clues support TRUE'.", "agent 4 says 'I lean FALSE, weakly'.",
         "No peer messages yet."]


def params(arm, k):
    p = build_params(ns["req"](k, f"Round {k % 4}. Peer messages: {PEERS[k % 4]}"), AgentConfig())
    p = copy.deepcopy(p)
    sch = p["output_config"]["format"]["schema"]
    if arm == "A":
        return p
    p["system"][0]["text"] = RULES_B if arm == "B" else RULES_C
    sch["properties"].pop("private_reasoning")
    if arm == "B":
        sch["properties"] = {"rationale": {"type": "string", "description": "One or two sentences on why you hold your belief."},
                             **sch["properties"]}
    sch["required"] = list(sch["properties"])
    return p


async def one(client, arm, k):
    r = await client.messages.create(**params(arm, k))
    tok = usage_tokens(r.usage); c = cost_usd(r.model, r.usage, False)
    record_call(EXP, r.model, False, tok, c, r._request_id)
    belief = None
    if r.stop_reason != "refusal":
        txt = "".join(b.text for b in r.content if b.type == "text")
        try: belief = json.loads(txt).get("calibrated_belief_prob")
        except Exception: belief = "bad-json"
    cat = getattr(r.stop_details, "category", None) if r.stop_reason == "refusal" else None
    return arm, k, r.stop_reason, cat, belief, c


async def main():
    client = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(8)
    async def guarded(a, k):
        async with sem: return await one(client, a, k)
    res = await asyncio.gather(*[guarded(a, k) for a in "ABC" for k in range(12)])
    tot = 0
    for a in "ABC":
        rows = [r for r in res if r[0] == a]
        ref = sum(r[2] == "refusal" for r in rows)
        print(f"arm {a}: refusals {ref}/12  beliefs {[r[4] for r in rows if r[2] != 'refusal']}  cats {set(r[3] for r in rows if r[3])}")
        tot += sum(r[5] for r in rows)
    print(f"total cost ${tot:.4f}  experiment_id {EXP}")

asyncio.run(main())
