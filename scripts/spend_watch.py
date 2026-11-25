"""Watch cumulative project spend and log each new $100 threshold crossed.

Total spend = actual Anthropic API cost from experiment code (sum of `cost_usd` over budget/ledger.jsonl)
            + this project's Runpod pod spend (`orch spend --json`, project "swarm-cavity").

For pods we take max(billed, estimated): Runpod's billing records lag by hours, so `billed` reads $0 for
recent pods while `estimated` (hourly rate x time) is already accurate to a few percent.

Each time the total passes a new multiple of STEP, one line is appended to budget/alerts.log. The last
threshold alerted is kept in budget/alerts_state.json, so a restart never re-alerts. A coordinator session
tails alerts.log and turns each line into a phone notification.

Run:  uv run python scripts/spend_watch.py            # loop every 5 min (run inside tmux)
      uv run python scripts/spend_watch.py --once     # one check, print totals
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so `engine` imports under plain `python` as well as `uv run`
from engine.budget import pod_spend  # noqa: E402  (shared with engine.BudgetGuard)
from engine.ledger import total_api_spend  # noqa: E402

LEDGER = ROOT / "budget" / "ledger.jsonl"
ALERTS = ROOT / "budget" / "alerts.log"
STATE = ROOT / "budget" / "alerts_state.json"
PROJECT = "swarm-cavity"
CAP = 2000.0
STEP = 100.0


def api_spend() -> float:
    """Sum cost_usd over the ledger; skip blank or partially written lines."""
    return total_api_spend(LEDGER)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_threshold": 0.0, "last_pods": 0.0}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)  # atomic, so a crash mid-write can't corrupt the state


def check(state: dict) -> tuple[float, float, float]:
    api = api_spend()
    pods = pod_spend()
    if pods is None:  # orch unreachable: reuse the last known pod figure rather than under-count
        pods = state.get("last_pods", 0.0)
    state["last_pods"] = pods
    total = api + pods
    threshold = STEP * int(total // STEP)
    if threshold > state["last_threshold"]:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = (f"{now} swarm-cavity spend passed ${threshold:.0f} "
               f"(API ${api:.0f}, pods ${pods:.0f}) of ${CAP:.0f}")
        with ALERTS.open("a") as f:
            f.write(msg + "\n")
            f.flush()
        state["last_threshold"] = threshold
    save_state(state)
    return api, pods, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=300.0)
    args = ap.parse_args()
    ALERTS.touch()
    state = load_state()
    while True:
        api, pods, total = check(state)
        print(f"{datetime.now():%H:%M:%S} api=${api:.2f} pods=${pods:.2f} total=${total:.2f} "
              f"alerted_through=${state['last_threshold']:.0f}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
