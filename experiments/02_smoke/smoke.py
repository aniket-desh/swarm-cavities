"""Gate (b): smoke test of the full swarm pipeline on claude-haiku-5-5 at N=8, T=4 (CLAUDE.md guardrail 1).

Graph: fixed in-degree K=3 directed signed graph (envs/graphs.fixed_indegree_signed_couplings), gamma=0.5
(round(0.5*3)=2 reciprocated ties per agent), 25% negative ("disagree") ties. Evidence: make_game(mu=0.5,
sigma=1, k=3). Runs two synchronous replicas and one asynchronous replica, realtime API, H=1.
Checks: no refusals or parse failures, beliefs in [0,1], ledger cost matches the checkpoints, and a resume
re-run costs $0 (every round is checkpointed). Cap: $5 via BudgetGuard(experiment_cap_usd=5).

Run from the repo root:  python experiments/02_smoke/smoke.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from engine import BudgetGuard, read_jsonl  # noqa: E402
from engine.runner import Runner  # noqa: E402
from envs.evidence_game import make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings, spectral_log  # noqa: E402
from envs.swarm import LLMBackend, SwarmConfig, run_swarm  # noqa: E402

N, T, K, GAMMA, P_NEG = 8, 4, 3, 0.5, 0.25
EXP = sys.argv[1] if len(sys.argv) > 1 else "smoke-" + time.strftime("%Y%m%d-%H%M%S")
LEDGER = ROOT / "budget" / "ledger.jsonl"


def main():
    rng = np.random.default_rng(0)
    J = fixed_indegree_signed_couplings(N, K, GAMMA, P_NEG, rng)
    game = make_game(N, mu=0.5, sigma=1.0, k=3, seed=0)
    log = spectral_log(J, eigs=False)
    print("graph:", {k: log[k] for k in log if not isinstance(log[k], (list, dict))})
    print("truth:", game.truth, " h_priv:", np.round(game.h_priv, 2))

    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=5.0, experiment_id=EXP, ledger_path=LEDGER)
    n_calls = N * (T + 1) * 3
    guard.project(n_calls, in_tokens=400, cached_tokens=800, out_tokens=650, model="claude-haiku-5-5", batch=False)

    out = {}
    for mode, reps in (("sync", 2), ("async", 1)):
        exp = f"{EXP}-{mode}"
        runner = Runner(exp, guard, ledger_path=LEDGER)
        cfg = SwarmConfig(experiment_id=exp, T=T, H=1, replicas=reps, mode=mode, seed=0)
        res = run_swarm(J, game, cfg, LLMBackend(runner))
        out[mode] = res
        print(f"{mode}: p (replica, round, agent)\n", np.round(res["p"], 2))
        print(f"{mode}: missing {int(res['missing'].sum())} of {res['missing'].size}")

    led = [r for r in read_jsonl(LEDGER) if str(r["experiment_id"]).startswith(EXP)]
    spent = sum(r["cost_usd"] for r in led)
    print(f"ledger: {len(led)} calls, ${spent:.4f}")

    # resume check: identical re-run must make no new calls
    runner = Runner(f"{EXP}-sync", guard, ledger_path=LEDGER)
    run_swarm(J, game, SwarmConfig(experiment_id=f"{EXP}-sync", T=T, H=1, replicas=2, mode="sync", seed=0),
              LLMBackend(runner))
    led2 = [r for r in read_jsonl(LEDGER) if str(r["experiment_id"]).startswith(EXP)]
    print(f"resume re-run added {len(led2) - len(led)} ledger lines (must be 0)")

    summary = {
        "experiment_id": EXP, "N": N, "T": T, "K": K, "gamma": GAMMA, "p_neg": P_NEG,
        "calls": len(led), "cost_usd": spent, "resume_new_calls": len(led2) - len(led),
        "missing": {m: int(out[m]["missing"].sum()) for m in out},
        "m_sync": np.nanmean(out["sync"]["s"], axis=2).round(3).tolist(),
        "m_async": np.nanmean(out["async"]["s"], axis=2).round(3).tolist(),
        "truth": game.truth,
    }
    Path(__file__).with_name("summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
