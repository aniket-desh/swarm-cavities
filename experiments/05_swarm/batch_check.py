"""Check the Message Batches path of the swarm (engine.Runner.run_round_batch via LLMBackend(batch=True)) on a tiny
swarm before using it for sweeps: replies parse, ledger lines are marked batch and priced at 50%, a re-run replays
for free, and the per-round latency (which sets how long a T-round batch swarm takes).

N=8, K=2, inverted wording, T=2, two replicas (8 + 2*2*8 = 40 calls, about $0.01).
Run: uv run python experiments/05_swarm/batch_check.py
"""

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
from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402
from envs.swarm import LLMBackend, SwarmConfig, run_swarm  # noqa: E402

EXP = "batchcheck-20261125"
LEDGER = ROOT / "budget" / "ledger.jsonl"

if __name__ == "__main__":
    J = fixed_indegree_signed_couplings(8, 2, 0.0, 0.5, np.random.default_rng(0))
    game = make_game(8, seed=1)
    guard = BudgetGuard(cap_usd=1900, experiment_cap_usd=0.5, experiment_id=EXP, ledger_path=LEDGER)
    runner = Runner(EXP, guard, ledger_path=LEDGER)
    cfg = SwarmConfig(EXP, T=2, H=1, replicas=2)
    t0 = time.time()
    out = run_swarm(J, game, cfg, LLMBackend(runner, batch=True, poll_s=10, disagree="inverted"))
    wall = time.time() - t0
    led = [r for r in read_jsonl(LEDGER) if r["experiment_id"] == EXP]
    print(f"wall {wall:.0f}s for 3 sequential rounds ({wall / 3:.0f}s per round); missing {int(out['missing'].sum())}"
          f"/{out['missing'].size}; ledger {len(led)} lines, all batch={all(r['batch'] for r in led)}, "
          f"${sum(r['cost_usd'] for r in led):.4f}")
    n_before = len(led)
    run_swarm(J, game, cfg, LLMBackend(Runner(EXP, guard, ledger_path=LEDGER), batch=True, poll_s=10, disagree="inverted"))
    print("re-run added", len([r for r in read_jsonl(LEDGER) if r["experiment_id"] == EXP]) - n_before, "ledger lines (must be 0)")
