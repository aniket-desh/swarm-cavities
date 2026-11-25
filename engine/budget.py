"""Hard dollar cap on experiment spend: Anthropic API (ledger) + this project's Runpod pods.

    spent = sum(cost_usd over budget/ledger.jsonl) + pod_spend("swarm-cavity")

BudgetGuard.check(projected) raises BudgetExceeded when spent + projected would pass the cap. pod_spend() is
shared with scripts/spend_watch.py so both count spend the same way.
"""

import json
import subprocess
import time
from pathlib import Path

from engine.ledger import LEDGER, ROOT, total_api_spend
from engine.pricing import cost_from_tokens

PROJECT = "swarm-cavity"
ALERTS_STATE = ROOT / "budget" / "alerts_state.json"  # written by spend_watch.py; read here as a fallback only


class BudgetExceeded(RuntimeError):
    pass


def pod_spend() -> float | None:
    """This project's pod spend since it started; None if orch could not be read.

    Takes max(billed, estimated): Runpod billing records lag by hours, so `billed` reads $0 for recent pods
    while `estimated` (hourly rate x time) is already accurate to a few percent.
    """
    try:
        out = subprocess.run(
            ["orch", "spend", "--json", "--days", "365"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        data = json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    for p in data.get("projects", []):
        if p.get("project") == PROJECT:
            return max(p.get("billed") or 0.0, p.get("estimated") or 0.0)
    return 0.0  # no pods tagged for this project yet


class BudgetGuard:
    """Refuses new spend once ledger + pod spend reaches `cap_usd` (and, optionally, per-experiment ledger
    spend reaches `experiment_cap_usd`).

    The ledger is re-read at most every `refresh_s` seconds; in between, costs this process reports through
    add() are added on top, so the guard never lags behind its own spending. Pod spend is cached for
    `pod_refresh_s` because `orch spend` takes a couple of seconds.
    """

    def __init__(self, cap_usd: float = 1900.0, experiment_cap_usd: float | None = None,
                 experiment_id: str | None = None, ledger_path: Path = LEDGER,
                 pod_spend_fn=pod_spend, refresh_s: float = 30.0, pod_refresh_s: float = 300.0):
        if experiment_cap_usd is not None and experiment_id is None:
            raise ValueError("experiment_cap_usd needs experiment_id")
        self.cap_usd = cap_usd
        self.experiment_cap_usd = experiment_cap_usd
        self.experiment_id = experiment_id
        self.ledger_path = Path(ledger_path)
        self.pod_spend_fn = pod_spend_fn
        self.refresh_s = refresh_s
        self.pod_refresh_s = pod_refresh_s
        self._ledger_t = -1e18
        self._ledger_total = 0.0
        self._ledger_exp = 0.0
        self._local = 0.0  # costs added since the last ledger read
        self._pod_t = -1e18
        self._pods: float | None = None

    def _pods_now(self) -> float:
        if time.monotonic() - self._pod_t > self.pod_refresh_s:
            pods = self.pod_spend_fn()
            if pods is not None:
                self._pods, self._pod_t = pods, time.monotonic()
        if self._pods is None:  # orch unreachable and nothing cached: use spend_watch's last reading
            try:
                return float(json.loads(ALERTS_STATE.read_text()).get("last_pods", 0.0))
            except (OSError, json.JSONDecodeError, ValueError):
                return 0.0
        return self._pods

    def _refresh_ledger(self, force: bool = False) -> None:
        if force or time.monotonic() - self._ledger_t > self.refresh_s:
            self._ledger_total = total_api_spend(self.ledger_path)
            if self.experiment_id is not None:
                self._ledger_exp = total_api_spend(self.ledger_path, self.experiment_id)
            self._local = 0.0
            self._ledger_t = time.monotonic()

    def add(self, cost: float) -> None:
        """Report a cost this process has just written to the ledger."""
        self._local += cost

    def api_spent(self) -> float:
        self._refresh_ledger()
        return self._ledger_total + self._local

    def experiment_spent(self) -> float:
        self._refresh_ledger()
        return self._ledger_exp + self._local

    def spent(self) -> float:
        return self.api_spent() + self._pods_now()

    def check(self, projected_usd: float = 0.0) -> None:
        total = self.spent()
        if total + projected_usd > self.cap_usd:
            raise BudgetExceeded(f"spent ${total:.2f} + projected ${projected_usd:.2f} > cap ${self.cap_usd:.2f}")
        if self.experiment_cap_usd is not None:
            exp = self.experiment_spent()
            if exp + projected_usd > self.experiment_cap_usd:
                raise BudgetExceeded(f"experiment {self.experiment_id}: spent ${exp:.2f} + projected "
                                     f"${projected_usd:.2f} > cap ${self.experiment_cap_usd:.2f}")

    def project(self, n_calls: int, in_tokens: int, cached_tokens: int, out_tokens: int, model: str,
                batch: bool, cache_write_tokens: int = 0) -> float:
        """Projected cost of n_calls, each with `in_tokens` uncached input, `cached_tokens` cache reads,
        `cache_write_tokens` 5-min cache writes and `out_tokens` output (thinking included). Prints the
        projection; any sweep over $50 must call this before launch (CLAUDE.md guardrail 1c)."""
        per_call = cost_from_tokens(model, batch, in_tokens, out_tokens, cached_tokens, cache_write_tokens)
        usd = n_calls * per_call
        print(f"[budget] projection: {n_calls} calls x (in {in_tokens} + cache_read {cached_tokens} + "
              f"cache_write {cache_write_tokens}, out {out_tokens}) on {model} batch={batch} -> ${usd:.2f} "
              f"(${per_call:.5f}/call); spent ${self.spent():.2f} of cap ${self.cap_usd:.2f}", flush=True)
        return usd
