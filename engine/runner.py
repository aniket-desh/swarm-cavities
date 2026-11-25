"""Run one round of N agent calls, realtime (async, bounded concurrency) or as one Message Batch.

Per agent and round the order of side effects is fixed so a crash at any point never re-spends:

    budget check -> API call -> ledger line (money spent) -> replay cache (response stored) -> checkpoint line

* Checkpoints: results/<experiment_id>/checkpoints.jsonl, one line per (experiment_id, seed, round_t, agent_id),
  written when the agent's record is final (ok or missing). Keys already present are skipped on restart.
* Replay cache: every response is stored under sha256(exact request params, attempt). Re-sending identical
  params returns the stored response at zero cost and writes no ledger line. Transcripts therefore replay
  exactly, and a cavity perturbation (one peer message changed) only pays for the requests whose params
  actually changed. The transport (batch or realtime) is not part of the key.
* Retry: a malformed / truncated reply is retried once (attempt 1 has its own replay key), then recorded as
  missing. Refusals are recorded as missing without retry. Transport errors (after the SDK's own retries, or
  errored/expired batch entries after the retry batch) are not checkpointed, so a restart retries them; the
  round raises RoundError after the other agents are saved.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path

from engine.agent import AgentConfig, AgentRequest, build_params, parse_belief
from engine.budget import BudgetExceeded, BudgetGuard
from engine.ledger import LEDGER, ROOT, append_jsonl, read_jsonl, record_call
from engine.pricing import cost_from_tokens, cost_usd, usage_tokens

MAX_ATTEMPTS = 2  # first try + one retry on malformed output


class RoundError(RuntimeError):
    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__(f"{len(errors)} agent call(s) failed: " +
                         "; ".join(f"{k}: {e!r}" for k, e in list(errors.items())[:5]))


def request_key(params: dict, attempt: int) -> str:
    blob = json.dumps({"params": params, "attempt": attempt}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _to_dict(msg) -> dict:
    return msg if isinstance(msg, dict) else msg.to_dict()


class ReplayCache:
    """Append-only store of request -> response, sharded by the first two hex chars of the key."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._shards: dict[str, dict] = {}

    def _load(self, shard: str) -> dict:
        self._shards[shard] = {r["key"]: r for r in read_jsonl(self.root / f"{shard}.jsonl")}
        return self._shards[shard]

    def get(self, key: str) -> dict | None:
        shard = key[:2]
        hit = self._shards.get(shard, {}).get(key)
        if hit is None:  # another process may have written it since we loaded the shard
            hit = self._load(shard).get(key)
        return hit

    def put(self, key: str, params: dict, attempt: int, response: dict, request_id: str | None) -> None:
        rec = {"key": key, "attempt": attempt, "request_id": request_id, "params": params, "response": response}
        append_jsonl(self.root / f"{key[:2]}.jsonl", rec)
        self._shards.setdefault(key[:2], {})[key] = rec


class Runner:
    def __init__(self, experiment_id: str, guard: BudgetGuard, client=None, cfg: AgentConfig = AgentConfig(),
                 concurrency: int = 16, results_root: Path = ROOT / "results", replay_dir: Path | None = None,
                 ledger_path: Path = LEDGER, sdk_max_retries: int = 4):
        if client is None:
            import anthropic  # deferred so tests and spend_watch never need the network client
            client = anthropic.AsyncAnthropic(max_retries=sdk_max_retries)
        self.experiment_id = experiment_id
        self.guard = guard
        self.client = client
        self.cfg = cfg
        self.sem = asyncio.Semaphore(concurrency)
        self.dir = Path(results_root) / experiment_id
        self.ckpt_path = self.dir / "checkpoints.jsonl"
        self.batches_path = self.dir / "batches.jsonl"  # submitted batch ids, so a restart re-attaches
        self.replay = ReplayCache(replay_dir or Path(results_root) / "_replay")
        self.ledger_path = ledger_path
        self.done = {self._key(r["seed"], r["round_t"], r["agent_id"]): r for r in read_jsonl(self.ckpt_path)}
        self.last_batch_latency_s: float | None = None

    # ---------- shared pieces ----------

    def _key(self, seed, round_t, agent_id) -> tuple:
        return (self.experiment_id, seed, round_t, agent_id)

    def _upper_bound(self, params: dict, batch: bool) -> float:
        """Conservative cost of one call for the pre-call budget check: every prompt token uncached
        (~3 chars/token over the JSON of system + messages, an overestimate) and max_tokens of output."""
        chars = len(json.dumps(params["system"])) + len(json.dumps(params["messages"]))
        return cost_from_tokens(params["model"], batch, chars // 3 + 1, params["max_tokens"])

    def _bill(self, model: str, usage, batch: bool, request_id: str | None) -> tuple[float, dict]:
        tokens = usage_tokens(usage)
        cost = cost_usd(model, usage, batch)
        record_call(self.experiment_id, model, batch, tokens, cost, request_id, self.ledger_path)
        self.guard.add(cost)
        return cost, tokens

    def _attempt(self, key, msg, reason, request_id, cost, tokens, replayed) -> dict:
        stop_details = (msg or {}).get("stop_details") or {}  # set only on refusals, e.g. category "reasoning_extraction"
        return {"request_key": key, "request_id": request_id, "stop_reason": msg.get("stop_reason") if msg else None,
                "refusal_category": stop_details.get("category"), "reason": reason, "replayed": replayed,
                "cost_usd": cost, "tokens": tokens}

    def _finish(self, seed, round_t, req: AgentRequest, belief, reason, attempts, batch) -> dict:
        rec = {"experiment_id": self.experiment_id, "seed": seed, "round_t": round_t, "agent_id": req.agent_id,
               "status": "ok" if belief else "missing", "missing_reason": None if belief else reason,
               "private_reasoning": None, "public_message": None, "calibrated_belief_prob": None,
               "belief_vector": None, "model": self.cfg.model, "batch": batch, "n_attempts": len(attempts),
               "cost_usd": sum(a["cost_usd"] for a in attempts), "attempts": attempts, "ts": time.time()}
        if belief:
            rec.update(belief)
        append_jsonl(self.ckpt_path, rec)
        self.done[self._key(seed, round_t, req.agent_id)] = rec
        return rec

    def _split_done(self, seed, round_t, requests):
        ids = [r.agent_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate agent_id in round")
        done = {r.agent_id: self.done[k] for r in requests if (k := self._key(seed, round_t, r.agent_id)) in self.done}
        todo = [r for r in requests if r.agent_id not in done]
        return done, todo

    # ---------- realtime ----------

    async def _call(self, params: dict, attempt: int):
        key = request_key(params, attempt)
        hit = self.replay.get(key)
        if hit is not None:
            return key, hit["response"], hit["request_id"], 0.0, usage_tokens(hit["response"]["usage"]), True
        async with self.sem:
            self.guard.check(self._upper_bound(params, batch=False))
            msg = await self.client.messages.create(**params)
        rid = getattr(msg, "_request_id", None)
        d = _to_dict(msg)
        cost, tokens = self._bill(params["model"], d["usage"], False, rid)
        self.replay.put(key, params, attempt, d, rid)
        return key, d, rid, cost, tokens, False

    async def _run_agent(self, seed, round_t, req: AgentRequest) -> dict:
        params = build_params(req, self.cfg)
        attempts, belief, reason = [], None, None
        for attempt in range(MAX_ATTEMPTS):
            key, d, rid, cost, tokens, replayed = await self._call(params, attempt)
            belief, reason = parse_belief(d, req.q)
            attempts.append(self._attempt(key, d, reason, rid, cost, tokens, replayed))
            if belief is not None or reason == "refusal":
                break
        return self._finish(seed, round_t, req, belief, reason, attempts, batch=False)

    async def run_round(self, seed: int, round_t: int, requests: list[AgentRequest]) -> dict:
        """Realtime path. Returns {agent_id: checkpoint record} for every request in the round."""
        out, todo = self._split_done(seed, round_t, requests)
        if not todo:
            return out
        self.guard.check(sum(self._upper_bound(build_params(r, self.cfg), False) for r in todo))
        results = await asyncio.gather(*(self._run_agent(seed, round_t, r) for r in todo), return_exceptions=True)
        errors = {}
        for r, res in zip(todo, results):
            if isinstance(res, BaseException):
                errors[r.agent_id] = res
            else:
                out[r.agent_id] = res
        for e in errors.values():
            if isinstance(e, BudgetExceeded):
                raise e
        if errors:
            raise RoundError(errors)
        return out

    # ---------- Message Batches ----------

    async def _submit_batch(self, params_by_aid: dict, keys_by_aid: dict, poll_s: float) -> dict:
        """Submit (or re-attach to) one batch; return {agent_id: (message dict | None, error, request_id, cost,
        tokens)}. Results arrive in any order, so they are matched back by custom_id only."""
        cid_by_aid = {aid: f"r{i}" for i, aid in enumerate(params_by_aid)}  # custom_id must match [A-Za-z0-9_-]{1,64}
        aid_by_cid = {c: a for a, c in cid_by_aid.items()}
        sig = sorted(keys_by_aid.values())  # identifies the exact set of requests in this batch
        prior = [b for b in read_jsonl(self.batches_path) if b["sig"] == sig]
        if prior:  # a previous process submitted exactly these requests; collect that batch instead
            key_to_cid = prior[-1]["cid_by_key"]
            aid_by_cid = {key_to_cid[keys_by_aid[a]]: a for a in params_by_aid}
            batch = await self.client.messages.batches.retrieve(prior[-1]["batch_id"])
        else:
            batch = await self.client.messages.batches.create(
                requests=[{"custom_id": cid_by_aid[a], "params": p} for a, p in params_by_aid.items()])
            append_jsonl(self.batches_path, {"batch_id": batch.id, "sig": sig, "ts": time.time(),
                                             "cid_by_key": {keys_by_aid[a]: cid_by_aid[a] for a in params_by_aid}})
        t0 = time.monotonic()
        while batch.processing_status != "ended":
            await asyncio.sleep(poll_s)
            batch = await self.client.messages.batches.retrieve(batch.id)
        self.last_batch_latency_s = time.monotonic() - t0
        out = {}
        async for res in await self.client.messages.batches.results(batch.id):
            aid = aid_by_cid[res.custom_id]
            rid = f"{batch.id}:{res.custom_id}"
            if res.result.type == "succeeded":
                d = _to_dict(res.result.message)
                cost, tokens = self._bill(params_by_aid[aid]["model"], d["usage"], True, rid)
                out[aid] = (d, None, rid, cost, tokens)
            else:  # errored / canceled / expired are not billed
                out[aid] = (None, f"batch_{res.result.type}", rid, 0.0, usage_tokens(None))
        for aid in params_by_aid:
            out.setdefault(aid, (None, "batch_missing_result", None, 0.0, usage_tokens(None)))
        return out

    async def run_round_batch(self, seed: int, round_t: int, requests: list[AgentRequest],
                              poll_s: float = 15.0) -> dict:
        """Batch path: all uncached calls of the round go in one batch; malformed replies are retried in a
        second batch. Returns {agent_id: checkpoint record}."""
        out, todo = self._split_done(seed, round_t, requests)
        reqs = {r.agent_id: r for r in todo}
        params = {a: build_params(r, self.cfg) for a, r in reqs.items()}
        attempts = {a: [] for a in reqs}
        final: dict = {}  # agent_id -> (belief, reason)
        pending = list(reqs)
        for attempt in range(MAX_ATTEMPTS):
            keys = {a: request_key(params[a], attempt) for a in pending}
            fresh = {}
            for a in pending:
                hit = self.replay.get(keys[a])
                if hit is not None:
                    belief, reason = parse_belief(hit["response"], reqs[a].q)
                    attempts[a].append(self._attempt(keys[a], hit["response"], reason, hit["request_id"], 0.0,
                                                     usage_tokens(hit["response"]["usage"]), True))
                    final[a] = (belief, reason)
                else:
                    fresh[a] = params[a]
            if fresh:
                self.guard.check(sum(self._upper_bound(p, True) for p in fresh.values()))
                got = await self._submit_batch(fresh, {a: keys[a] for a in fresh}, poll_s)
                for a, (d, err, rid, cost, tokens) in got.items():
                    if d is not None:
                        self.replay.put(keys[a], params[a], attempt, d, rid)
                        belief, reason = parse_belief(d, reqs[a].q)
                    else:
                        belief, reason = None, err
                    attempts[a].append(self._attempt(keys[a], d, reason, rid, cost, tokens, False))
                    final[a] = (belief, reason)
            pending = [a for a in pending if final[a][0] is None and final[a][1] != "refusal"]
            if not pending:
                break
        errors = {}
        for a, r in reqs.items():
            belief, reason = final[a]
            if belief is None and reason.startswith("batch_"):  # transport failure: leave unrecorded for a restart
                errors[a] = reason
                continue
            out[a] = self._finish(seed, round_t, r, belief, reason, attempts[a], batch=True)
        if errors:
            raise RoundError(errors)
        return out
