"""Test 1b: do the channel gains measured with template messages transfer to real swarm inboxes?

experiments/03_calibration measured a_+ and a_- with K=8 inboxes of short templated messages ("My sensors favor
HEADS."). Real swarm messages restate sensor reports ("My sensors (74% HEADS, 58% HEADS, 60% TAILS) ...") that a
reader can treat as independent evidence, and the swarm designs that discriminate the theory run at K=2 and 4.
This probe set measures the two-channel response in the swarm's own conditions:

    E[s] = tanh(a_+ f_A - a_- f_D + a_p h_priv + c),   f_A = sum_agree s_j / K,  f_D = sum_disagree s_j / K   (1)

(experiments/03_calibration/RESULT.md answer 2; s = 2p - 1, displayed probabilities). Design, per K in {2, 4} and
wording in {inverted, label}, n base probes:

  - inbox slots: each of the K slots is a disagree tie with probability 1/2 (p_neg = 0.5, as in the swarm cells);
  - peer messages: drawn from the pilot swarm's real messages at the same K (rounds >= 1, replies that parsed),
    each mirrored with probability 1/2 (HEADS <-> TAILS in the text, p -> 1 - p) so that the peer field is
    uncorrelated with the probe's own evidence and both signs occur;
  - own evidence: h ~ N(+-0.5, 1) with a random sign, rendered by evidence_for_field (k=3), like make_game;
  - own last answer (H=1): what the agent would have said in round 0, p0 = (1 + tanh(0.42 h_priv - 0.016)) / 2
    (the surrogate's round-0 map, experiments/04_swarm_design/design_lib.py) with a one-line templated message.

Each base probe is sent twice, identical except for the peer texts: arm "real" (pilot text) and arm "template"
(cavity.probes.peer_text for the same displayed probability). The paired difference a(real) - a(template) is the
transfer error of the calibration; the K=2 against K=4 comparison says whether the agent averages its inbox
(gains per unit f equal across K) or adds it up (gains per unit f growing like K).

Run from the repo root inside tmux:  uv run python experiments/05_swarm/transfer.py run|analyze
"""

import asyncio
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from cavity.probes import _fit_tanh, peer_text  # noqa: E402
from engine import BudgetGuard  # noqa: E402
from engine.agent import AgentConfig  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.evidence_game import evidence_for_field  # noqa: E402
from envs.prompts import AgentContext, InboxMsg  # noqa: E402
from envs.swarm import LLMBackend  # noqa: E402

TAG = "transfer-20261125"
POOL_TAG = "pilot-20261125"
OUT = ROOT / "results" / "transfer"
LEDGER = ROOT / "budget" / "ledger.jsonl"
N_BASE = 600
KS, WORDINGS, ARMS = (2, 4), ("inverted", "label"), ("real", "template")
A_P, C0 = 0.42, -0.016


def mirror(p: float, text: str) -> tuple[float, str]:
    """The same message about the opposite side: HEADS <-> TAILS in the text, p -> 1 - p."""
    swap = {"HEADS": "TAILS", "TAILS": "HEADS", "heads": "tails", "tails": "heads", "Heads": "Tails", "Tails": "Heads"}
    return round(1 - p, 2), re.sub(r"\b(HEADS|TAILS|heads|tails|Heads|Tails)\b", lambda m: swap[m.group(0)], text)


def message_pool(K: int) -> list[tuple[float, str]]:
    pool = []
    for f in sorted(glob.glob(str(ROOT / "results" / f"{POOL_TAG}-*-K{K}-*" / "checkpoints.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r["round_t"] >= 1 and r["status"] == "ok" and r["public_message"].strip():
                pool.append((round(float(r["calibrated_belief_prob"]), 2), r["public_message"]))
    if not pool:
        raise RuntimeError(f"no pilot messages for K={K}; run the pilot first")
    return pool


def base_probes(K: int, n: int, seed: int = 0) -> list[dict]:
    pool = message_pool(K)
    out = []
    for b in range(n):
        rng = np.random.default_rng([seed, K, b])
        neg = rng.random(K) < 0.5
        msgs = []
        for _ in range(K):
            p, t = pool[rng.integers(len(pool))]
            msgs.append(mirror(p, t) if rng.random() < 0.5 else (p, t))
        h = rng.choice([-1, 1]) * 0.5 + rng.standard_normal()
        ev = evidence_for_field(h, 3, rng)
        p0 = round((1 + np.tanh(A_P * ev.h_priv + C0)) / 2, 2)
        side = "HEADS" if p0 >= 0.5 else "TAILS"
        hist = ((p0, f"From my own sensors I lean {side}, about {round(100 * max(p0, 1 - p0))}%."),)
        out.append({"K": K, "b": b, "neg": neg.tolist(), "msgs": msgs, "evidence": ev, "hist": hist,
                    "tpl_seed": [seed, K, b, 7]})
    return out


def context(bp: dict, arm: str, session: int) -> AgentContext:
    trng = np.random.default_rng(bp["tpl_seed"])
    agree, disagree = [], []
    for (p, t), is_neg in zip(bp["msgs"], bp["neg"]):
        m = InboxMsg(p, t if arm == "real" else peer_text(p, trng))
        (disagree if is_neg else agree).append(m)
    return AgentContext(agent_id=f"K{bp['K']}|b{bp['b']}|{arm}", session=session, round_t=1,
                        h_priv=bp["evidence"].h_priv, evidence=bp["evidence"].text(), history=bp["hist"],
                        agree=tuple(agree), disagree=tuple(disagree))


async def run():
    OUT.mkdir(parents=True, exist_ok=True)
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=3.0, experiment_id=f"{TAG}-*", ledger_path=LEDGER)
    n_calls = len(KS) * len(WORDINGS) * N_BASE * len(ARMS)
    guard.project(n_calls, in_tokens=500, cached_tokens=1000, out_tokens=700, model="claude-haiku-5-5", batch=False)
    for wording in WORDINGS:
        exp = f"{TAG}-{wording}"
        runner = Runner(exp, guard, cfg=AgentConfig(), concurrency=24, ledger_path=LEDGER)
        backend = LLMBackend(runner, disagree=wording)
        recs = []
        for K in KS:
            bps = base_probes(K, N_BASE)
            # the two arms of a base probe get the same session tag, so they differ only in the peer texts
            ctxs = [context(bp, arm, 50_000 + 1000 * K + bp["b"]) for bp in bps for arm in ARMS]
            res = await backend.respond(0, 0, ctxs, 1)
            for ctx, r in zip(ctxs, res):
                bp = bps[int(ctx.agent_id.split("|b")[1].split("|")[0])]
                fA = sum(2 * m.prob - 1 for m in ctx.agree) / K
                fD = sum(2 * m.prob - 1 for m in ctx.disagree) / K
                recs.append({"wording": wording, "K": K, "b": bp["b"], "arm": ctx.agent_id.rsplit("|", 1)[1],
                             "f_A": fA, "f_D": fD, "n_agree": len(ctx.agree), "h_priv": ctx.h_priv,
                             "u": 2 * bp["hist"][0][0] - 1, "status": r.status, "reason": r.reason,
                             "s": 2 * r.prob - 1 if r.status == "ok" else None, "message": r.message})
            print(f"[{wording} K={K}] done; spent ${guard.experiment_spent():.3f}", flush=True)
        (OUT / f"{wording}.json").write_text(json.dumps(recs))


def fit(rs: list[dict]) -> np.ndarray:
    """Eq. (1) by least squares; returns (a_+, a_-, a_p)."""
    X = np.array([[r["f_A"], -r["f_D"], r["h_priv"]] for r in rs])
    y = np.array([r["s"] for r in rs])
    w, _, _ = _fit_tanh(X, y, np.zeros(len(rs)), [1.0, 0.5, 0.4])
    return w


def analyze(n_boot: int = 1000):
    summary = {}
    for wording in WORDINGS:
        recs = json.loads((OUT / f"{wording}.json").read_text())
        for K in KS:
            rk = [r for r in recs if r["K"] == K]
            ok_b = {b for b in {r["b"] for r in rk}
                    if all(r["s"] is not None for r in rk if r["b"] == b)}   # pairs with both arms answered
            by = {(r["b"], r["arm"]): r for r in rk if r["b"] in ok_b}
            bs = np.array(sorted(ok_b))
            est = {arm: fit([by[(b, arm)] for b in bs]) for arm in ARMS}
            rng = np.random.default_rng(0)
            boot = {arm: [] for arm in ARMS}
            for _ in range(n_boot):          # paired bootstrap over base probes
                pick = rng.choice(bs, len(bs))
                for arm in ARMS:
                    boot[arm].append(fit([by[(b, arm)] for b in pick]))
            boot = {a: np.array(v) for a, v in boot.items()}
            diff = boot["real"] - boot["template"]
            key = f"{wording}|K{K}"
            summary[key] = {
                "n_pairs": int(len(bs)), "n_missing": int(sum(r["s"] is None for r in rk)),
                **{f"{arm}_{nm}": float(est[arm][i]) for arm in ARMS for i, nm in enumerate(("a_plus", "a_minus", "a_p"))},
                **{f"{arm}_{nm}_se": float(boot[arm][:, i].std(ddof=1)) for arm in ARMS
                   for i, nm in enumerate(("a_plus", "a_minus", "a_p"))},
                **{f"diff_{nm}": float(est["real"][i] - est["template"][i]) for i, nm in enumerate(("a_plus", "a_minus", "a_p"))},
                **{f"diff_{nm}_se": float(diff[:, i].std(ddof=1)) for i, nm in enumerate(("a_plus", "a_minus", "a_p"))},
                "real_field_sd": float(np.std([by[(b, 'real')]["f_A"] - by[(b, 'real')]["f_D"] for b in bs])),
            }
            s = summary[key]
            print(f"{key}: n={s['n_pairs']} | real a+ {s['real_a_plus']:.2f}({s['real_a_plus_se']:.2f}) "
                  f"a- {s['real_a_minus']:.2f}({s['real_a_minus_se']:.2f}) ap {s['real_a_p']:.2f} | template a+ "
                  f"{s['template_a_plus']:.2f}({s['template_a_plus_se']:.2f}) a- {s['template_a_minus']:.2f}"
                  f"({s['template_a_minus_se']:.2f}) ap {s['template_a_p']:.2f} | diff a+ {s['diff_a_plus']:+.2f}"
                  f"({s['diff_a_plus_se']:.2f}) a- {s['diff_a_minus']:+.2f}({s['diff_a_minus_se']:.2f})", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"run": lambda: asyncio.run(run()), "analyze": analyze}[sys.argv[1]]()
