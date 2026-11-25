"""Test 1 of theory/README.md: single-agent calibration of claude-haiku-5-5 (the agent of envs/prompts.py, H=1).

Stages (each its own experiment_id, checkpointed and replay-cached by engine.Runner, so a re-run resumes free):
  suff       claim1.md section 4.1 sufficiency design: 5 field levels x 6 compositions x 2 evidence levels
             x 2 own-history states x 25 samples = 3000 calls  -> eta^2_comp
  conf       conformity curve: 9 hbar in [-0.8, 0.8] x 2 evidence levels x 2 history states x 20 = 720 calls
             -> beta_resp, and with fit_surrogate the response surface F(h_priv, u, f_A, f_D)
  gains      agree-only vs disagree-only inboxes, label wording: 5 x 2 x 20 = 200 calls -> a_+, a_-
  gains_inv  the same with the "inverted" disagree wording (envs/prompts.DISAGREE_STYLES) = 200 calls
  white      white-probe kernel R(t, t'): M=200 trajectories x 9 rounds = 1800 calls (solver_spec.md eq. (S1))

Realtime API (batch would halve a ~$2 cost but adds 5-30 min latency per stage). Caps: $10 for the whole
script via BudgetGuard(experiment_cap_usd=...) per stage, plus the global $1,900 cap.

Run from the repo root inside tmux:  python experiments/03_calibration/run.py [stage ...]
Outputs: results/calibration/<stage>.json (records), then analyze.py.
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
from engine import BudgetGuard  # noqa: E402
from engine.agent import AgentConfig  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.swarm import LLMBackend  # noqa: E402

TAG = "calib-20261125"          # fixed, so re-runs resume from checkpoints instead of re-spending
OUT = ROOT / "results" / "calibration"
LEDGER = ROOT / "budget" / "ledger.jsonl"
H = 1
CAPS = {"suff": 4.0, "conf": 1.0, "gains": 0.4, "gains_inv": 0.4, "white": 2.0}


def designs():
    return {
        "suff": (P.sufficiency_design(n=25), "label"),
        "conf": (P.conformity_design(ev_levels=(0.4, 1.4), hist_states=(+1, -1), n=20), "label"),
        "gains": (P.gains_design(n=20), "label"),
        "gains_inv": (P.gains_design(n=20, seed=1), "inverted"),
    }


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


async def run_stage(stage: str):
    exp = f"{TAG}-{stage}"
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=CAPS[stage], experiment_id=exp, ledger_path=LEDGER)
    runner = Runner(exp, guard, cfg=AgentConfig(), concurrency=16, ledger_path=LEDGER)
    t0 = time.time()
    if stage == "white":
        out = await P.run_white_probe(LLMBackend(runner), guard, M=200, T=8, K=8, eps=0.15, bg_sd=0.3, H=H)
        payload = {"experiment_id": exp, **out}
    else:
        probes, style = designs()[stage]
        recs = await P.run_probe_set(probes, LLMBackend(runner, disagree=style), guard, H=H)
        payload = {"experiment_id": exp, "disagree_style": style, "records": recs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stage}.json").write_text(json.dumps(payload, default=_json_default))
    print(f"[{stage}] done in {time.time() - t0:.0f}s; spent ${guard.experiment_spent():.4f}", flush=True)


def main():
    stages = sys.argv[1:] or ["gains", "gains_inv", "conf", "suff", "white"]
    for st in stages:
        asyncio.run(run_stage(st))


if __name__ == "__main__":
    main()
