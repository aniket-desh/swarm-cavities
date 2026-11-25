"""engine/ tests with a fake Anthropic client (no network)."""

import asyncio
import copy
import json
import re
import threading
from types import SimpleNamespace

import pytest
from anthropic.types import Usage

from engine.agent import AgentConfig, AgentRequest, build_params, parse_belief, validate_belief
from engine.budget import BudgetExceeded, BudgetGuard
from engine.ledger import append_jsonl, read_jsonl, record_call, total_api_spend
from engine.pricing import cost_usd, usage_tokens
from engine.runner import RoundError, Runner

USAGE = {"input_tokens": 1000, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 3000,
         "output_tokens": 400}


# ---------------- pricing ----------------

def test_cost_haiku_hand_computed():
    # (1000*0.10 + 2000*1.25*0.10 + 3000*0.01 + 400*0.50) / 1e6 = (100 + 250 + 30 + 200) / 1e6
    assert cost_usd("claude-haiku-5-5", USAGE, batch=False) == pytest.approx(580e-6, rel=1e-12)
    assert cost_usd("claude-haiku-5-5", USAGE, batch=True) == pytest.approx(290e-6, rel=1e-12)


def test_cost_from_sdk_usage_object_matches_dict():
    u = Usage(**USAGE)
    assert cost_usd("claude-haiku-5-5", u, batch=False) == pytest.approx(580e-6, rel=1e-12)


def test_cost_opus_1h_cache_split():
    # 1000 cache-write tokens of which 600 are 1h: 100*4 + 400*1.25*4 + 600*2*4 + 0 + 50*20 = 400+2000+4800+1000
    u = {"input_tokens": 100, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 0,
         "output_tokens": 50, "cache_creation": {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 600}}
    assert cost_usd("claude-opus-5-5", u, batch=False) == pytest.approx(8200e-6, rel=1e-12)
    assert usage_tokens(u)["cache_write_tokens"] == 1000


def test_cost_sonnet_fable_and_haiku_long_prompt():
    u = {"input_tokens": 10_000, "cache_read_input_tokens": 20_000, "output_tokens": 1000}
    assert cost_usd("claude-sonnet-5-5", u, False) == pytest.approx((10_000 * 2 + 20_000 * 0.1 + 1000 * 10) / 1e6)
    assert cost_usd("claude-fable-5-2", u, True) == pytest.approx(0.5 * (10_000 * 8 + 20_000 * 0.2 + 1000 * 40) / 1e6)
    # Haiku above 100K prompt tokens: $0.50 / $2.50
    assert cost_usd("claude-haiku-5-5", {"input_tokens": 150_000, "output_tokens": 100}, False) == \
        pytest.approx((150_000 * 0.5 + 100 * 2.5) / 1e6)
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", u, False)


# ---------------- ledger ----------------

def test_ledger_concurrent_appends(tmp_path):
    path = tmp_path / "ledger.jsonl"
    tokens = usage_tokens(USAGE)
    n_threads, per_thread = 16, 200
    long_id = "x" * 3000  # lines longer than a pipe buffer still must not interleave

    def worker(i):
        for j in range(per_thread):
            record_call(f"{long_id}-{i}", "claude-haiku-5-5", False, tokens, 0.001, f"req_{i}_{j}", path)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = path.read_text().splitlines()
    assert len(lines) == n_threads * per_thread
    recs = [json.loads(line) for line in lines]  # every line parses: no interleaving
    assert len({r["request_id"] for r in recs}) == n_threads * per_thread
    assert total_api_spend(path) == pytest.approx(0.001 * n_threads * per_thread)
    assert total_api_spend(path, f"{long_id}-3") == pytest.approx(0.001 * per_thread)


def test_ledger_skips_partial_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_jsonl(path, {"cost_usd": 1.5})
    with path.open("a") as f:
        f.write('{"cost_usd": 9')  # crash mid-line
    assert total_api_spend(path) == 1.5


# ---------------- budget ----------------

def test_budget_guard_refuses(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_jsonl(path, {"experiment_id": "a", "cost_usd": 10.0})
    append_jsonl(path, {"experiment_id": "b", "cost_usd": 2.0})
    g = BudgetGuard(cap_usd=20.0, ledger_path=path, pod_spend_fn=lambda: 5.0)
    assert g.spent() == pytest.approx(17.0)
    g.check(3.0)
    with pytest.raises(BudgetExceeded):
        g.check(3.5)
    g.add(2.0)  # spend this process just made counts before the next ledger re-read
    with pytest.raises(BudgetExceeded):
        g.check(1.5)
    ge = BudgetGuard(cap_usd=1e6, experiment_cap_usd=3.0, experiment_id="b", ledger_path=path,
                     pod_spend_fn=lambda: 0.0)
    ge.check(1.0)
    with pytest.raises(BudgetExceeded):
        ge.check(1.5)


def test_budget_project(tmp_path, capsys):
    g = BudgetGuard(ledger_path=tmp_path / "l.jsonl", pod_spend_fn=lambda: 0.0)
    usd = g.project(1000, in_tokens=500, cached_tokens=1500, out_tokens=600, model="claude-haiku-5-5", batch=True)
    assert usd == pytest.approx(1000 * 0.5 * (500 * 0.10 + 1500 * 0.01 + 600 * 0.50) / 1e6)
    assert "projection" in capsys.readouterr().out


# ---------------- fake client ----------------

def belief_json(p, q=None):
    d = {"rationale": "r", "public_message": f"I think {p}", "calibrated_belief_prob": p}
    if q:
        d["belief_vector"] = [1.0 / q] * q
    return json.dumps(d)


def agent_prob(params):
    """Responder default: agent k answers k/10, read back from its evidence text."""
    k = int(re.search(r"agent (\d+)", params["system"][1]["text"]).group(1))
    return k / 10


def message(text, stop="end_turn"):
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "stop_reason": stop, "stop_details": {"type": "refusal", "category": "reasoning_extraction"}
            if stop == "refusal" else None, "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
            "usage": dict(USAGE)}


class FakeMsg:
    def __init__(self, d, rid):
        self._d, self._request_id = d, rid

    def to_dict(self):
        return copy.deepcopy(self._d)


class FakeBatches:
    def __init__(self, parent):
        self.parent, self.store, self.created = parent, {}, []

    async def create(self, requests):
        bid = f"msgbatch_{len(self.created)}"
        self.created.append([r["custom_id"] for r in requests])
        self.store[bid] = [(r["custom_id"], r["params"]) for r in requests]
        return SimpleNamespace(id=bid, processing_status="in_progress")

    async def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended")

    async def results(self, bid):
        items = list(reversed(self.store[bid]))  # results come back in a different order than submitted

        async def gen():
            for cid, params in items:
                if self.parent.batch_error and self.parent.batch_error(params):
                    yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored"))
                    continue
                self.parent.n += 1
                msg = FakeMsg(self.parent.responder(params, self.parent.n), None)
                yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg))
        return gen()


class FakeClient:
    def __init__(self, responder=None, batch_error=None):
        self.batch_error = batch_error
        self.responder = responder or (lambda params, n: message(belief_json(agent_prob(params))))
        self.n = 0
        self.calls = []
        self.messages = SimpleNamespace(create=self._create, batches=FakeBatches(self))

    async def _create(self, **params):
        assert "temperature" not in params
        self.calls.append(params)
        self.n += 1
        return FakeMsg(self.responder(params, self.n), f"req_{self.n}")


def requests(n, q=None):
    return [AgentRequest(agent_id=k, system_rules="RULES " * 50, evidence=f"evidence for agent {k}",
                         dynamic="peer messages", q=q) for k in range(n)]


def make_runner(tmp_path, client, cap=100.0, exp="exp1"):
    ledger = tmp_path / "budget" / "ledger.jsonl"
    guard = BudgetGuard(cap_usd=cap, ledger_path=ledger, pod_spend_fn=lambda: 0.0)
    return Runner(exp, guard, client=client, results_root=tmp_path / "results", ledger_path=ledger), ledger


# ---------------- agent ----------------

def test_build_params_shape():
    p = build_params(requests(1)[0], AgentConfig())
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral"}  # breakpoint ends the stable prefix
    assert p["messages"] == [{"role": "user", "content": "peer messages"}]
    assert p["output_config"]["effort"] == "low"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in p and "thinking" not in p


def test_validate_belief():
    ok = json.loads(belief_json(0.3, q=4))
    assert validate_belief(ok, 4)["calibrated_belief_prob"] == 0.3
    for bad in ({**ok, "calibrated_belief_prob": 1.2}, {**ok, "calibrated_belief_prob": True},
                {**ok, "belief_vector": [0.5, 0.6, 0.0, 0.0]}, {**ok, "belief_vector": [1.0]},
                {**ok, "extra": 1}):
        with pytest.raises(ValueError):
            validate_belief(bad, 4)
    assert parse_belief(message("not json"), None) == (None, "malformed_json")
    assert parse_belief(message(belief_json(0.5), stop="refusal"), None) == (None, "refusal")


# ---------------- runner: realtime ----------------

def test_realtime_round_ledger_and_checkpoint(tmp_path):
    client = FakeClient()
    runner, ledger = make_runner(tmp_path, client)
    out = asyncio.run(runner.run_round(0, 0, requests(5)))
    assert {a: r["calibrated_belief_prob"] for a, r in out.items()} == {k: k / 10 for k in range(5)}
    assert len(client.calls) == 5
    assert len(list(read_jsonl(ledger))) == 5
    assert total_api_spend(ledger) == pytest.approx(5 * 580e-6)
    assert len(list(read_jsonl(runner.ckpt_path))) == 5


def test_checkpoint_resume_skips_done(tmp_path):
    client = FakeClient()
    runner, ledger = make_runner(tmp_path, client)
    asyncio.run(runner.run_round(0, 0, requests(3)))
    # restart: a new Runner over the same files, with two new agents in the same round
    client2 = FakeClient()
    runner2, _ = make_runner(tmp_path, client2)
    out = asyncio.run(runner2.run_round(0, 0, requests(5)))
    assert len(out) == 5
    assert len(client2.calls) == 2  # only agents 3 and 4 were called
    assert len(list(read_jsonl(runner2.ckpt_path))) == 5
    assert len(list(read_jsonl(ledger))) == 5


def test_replay_identical_request_is_free(tmp_path):
    client = FakeClient()
    runner, ledger = make_runner(tmp_path, client)
    asyncio.run(runner.run_round(0, 0, requests(3)))
    # a different experiment sending identical params (e.g. the unperturbed agents of a cavity probe)
    client2 = FakeClient(lambda p, n: pytest.fail("should replay, not call"))
    runner2, _ = make_runner(tmp_path, client2, exp="exp1-replay")
    out = asyncio.run(runner2.run_round(0, 0, requests(3)))
    assert all(r["cost_usd"] == 0 and r["attempts"][0]["replayed"] for r in out.values())
    assert [out[k]["calibrated_belief_prob"] for k in range(3)] == [0.0, 0.1, 0.2]
    assert len(list(read_jsonl(ledger))) == 3  # no new ledger lines
    # changing one agent's peer messages only pays for that agent
    client3 = FakeClient()
    runner3, _ = make_runner(tmp_path, client3, exp="exp1-perturbed")
    reqs = requests(3)
    reqs[1] = AgentRequest(1, reqs[1].system_rules, reqs[1].evidence, "perturbed peer messages")
    asyncio.run(runner3.run_round(0, 0, reqs))
    assert len(client3.calls) == 1


def test_malformed_retry_then_ok(tmp_path):
    def responder(params, n):
        return message("{not json" if n == 1 else belief_json(0.7))
    client = FakeClient(responder)
    runner, ledger = make_runner(tmp_path, client)
    rec = asyncio.run(runner.run_round(0, 0, requests(1)))[0]
    assert rec["status"] == "ok" and rec["calibrated_belief_prob"] == 0.7
    assert rec["n_attempts"] == 2 and rec["attempts"][0]["reason"] == "malformed_json"
    assert len(list(read_jsonl(ledger))) == 2  # both attempts billed
    assert rec["cost_usd"] == pytest.approx(2 * 580e-6)


@pytest.mark.parametrize("text,stop,reason,calls", [
    ("{not json", "end_turn", "malformed_json", 2),
    (belief_json(1.7), "end_turn", "invalid: calibrated_belief_prob=1.7 not in [0, 1]", 2),
    ('{"rationale": "trunc', "max_tokens", "max_tokens", 2),
    ("", "refusal", "refusal", 1),
])
def test_bad_output_recorded_missing(tmp_path, text, stop, reason, calls):
    client = FakeClient(lambda p, n: message(text, stop))
    runner, _ = make_runner(tmp_path, client)
    rec = asyncio.run(runner.run_round(0, 0, requests(1)))[0]
    assert rec["status"] == "missing" and rec["missing_reason"] == reason
    assert rec["calibrated_belief_prob"] is None
    assert len(client.calls) == calls
    if stop == "refusal":
        assert rec["attempts"][0]["refusal_category"] == "reasoning_extraction"


def test_budget_refusal_blocks_calls(tmp_path):
    client = FakeClient()
    runner, ledger = make_runner(tmp_path, client, cap=1e-4)
    with pytest.raises(BudgetExceeded):
        asyncio.run(runner.run_round(0, 0, requests(4)))
    assert client.calls == [] and not ledger.exists()


# ---------------- runner: batch ----------------

def test_batch_results_reordered_and_retried(tmp_path):
    def responder(params, n):
        k = agent_prob(params)
        # agent 2's first reply is malformed; the retry (attempt 1, second batch) is valid
        if k == 0.2 and not hasattr(responder, "seen"):
            responder.seen = True
            return message("garbage")
        return message(belief_json(k))
    client = FakeClient(responder)
    runner, ledger = make_runner(tmp_path, client)
    out = asyncio.run(runner.run_round_batch(0, 0, requests(5, q=None), poll_s=0))
    assert {a: r["calibrated_belief_prob"] for a, r in out.items()} == {k: k / 10 for k in range(5)}
    assert client.messages.batches.created == [["r0", "r1", "r2", "r3", "r4"], ["r0"]]
    assert out[2]["n_attempts"] == 2 and out[2]["status"] == "ok"
    led = list(read_jsonl(ledger))
    assert len(led) == 6 and all(r["batch"] for r in led)
    assert total_api_spend(ledger) == pytest.approx(6 * 290e-6)  # batch = half price
    # resume: nothing left to do, no new batch
    runner2, _ = make_runner(tmp_path, FakeClient())
    asyncio.run(runner2.run_round_batch(0, 0, requests(5), poll_s=0))
    assert runner2.client.messages.batches.created == []


def test_batch_q_class(tmp_path):
    client = FakeClient(lambda p, n: message(belief_json(agent_prob(p), q=3)))
    runner, _ = make_runner(tmp_path, client)
    out = asyncio.run(runner.run_round_batch(1, 2, requests(3, q=3), poll_s=0))
    assert all(r["belief_vector"] == [1 / 3] * 3 for r in out.values())
    schema = client.messages.batches.store["msgbatch_0"][0][1]["output_config"]["format"]["schema"]
    assert "belief_vector" in schema["required"]


def test_batch_errored_entries_not_checkpointed(tmp_path):
    client = FakeClient(batch_error=lambda p: agent_prob(p) == 0.1)  # agent 1 errors in both batches
    runner, ledger = make_runner(tmp_path, client)
    with pytest.raises(RoundError):
        asyncio.run(runner.run_round_batch(0, 0, requests(3), poll_s=0))
    assert sorted(r["agent_id"] for r in read_jsonl(runner.ckpt_path)) == [0, 2]
    assert len(list(read_jsonl(ledger))) == 2  # errored entries are not billed
    # restart once the error clears: only agent 1 is submitted
    runner2, _ = make_runner(tmp_path, FakeClient())
    out = asyncio.run(runner2.run_round_batch(0, 0, requests(3), poll_s=0))
    assert runner2.client.messages.batches.created == [["r0"]] and out[1]["status"] == "ok"
