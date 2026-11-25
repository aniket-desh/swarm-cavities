"""envs/vllm_backend.py without a GPU: prompt rendering, the digit-path value distribution on hand-made
next-token distributions, the missing-mass rule, parse_sample, and one end-to-end call through a fake vLLM engine
(character-level tokenizer) that checks the interleaving of the three engine steps and checkpoint replay."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from envs.evidence_game import evidence_for_field
from envs.prompts import FORBIDDEN_WORDS, AgentContext, InboxMsg, dynamic, rules
from envs.vllm_backend import (FORMAT_NOTE, GRID, SAMPLE_REGEX, VLLMBackend, belief_from_masses, expand_digits,
                               parse_sample, render_messages, render_prompt, token_kind, value_distribution)


def _ctx(round_t=1, agree=(), disagree=(), agent_id=0, session=0):
    ev = evidence_for_field(0.4, 3, np.random.default_rng(1))
    hist = ((0.6, "I lean HEADS."),) if round_t > 0 else ()
    return AgentContext(agent_id=agent_id, session=session, round_t=round_t, h_priv=ev.h_priv, evidence=ev.text(),
                        history=hist, agree=agree, disagree=disagree)


# ---------------------------------------------------------------- rendering

def test_render_messages_is_the_claude_agent_plus_format_note():
    ctx = _ctx(agree=(InboxMsg(0.7, "Mostly HEADS here."),), disagree=(InboxMsg(0.2, "TAILS."),))
    msgs = render_messages(ctx, 1, "inverted")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == f"{rules(1, 'inverted')}\n\n{ctx.evidence}\n\n{FORMAT_NOTE}"
    assert msgs[1]["content"] == dynamic(ctx, 1)
    text = (msgs[0]["content"] + msgs[1]["content"]).lower()
    assert not any(w in text for w in FORBIDDEN_WORDS)          # guardrail 4
    # the format note names the three fields in the order the sampling regex enforces
    order = [FORMAT_NOTE.index(k) for k in ("rationale", "public_message", "calibrated_belief_prob")]
    assert order == sorted(order)
    assert [SAMPLE_REGEX.index(k) for k in ("rationale", "public_message", "calibrated_belief_prob")] == sorted(
        SAMPLE_REGEX.index(k) for k in ("rationale", "public_message", "calibrated_belief_prob"))


def test_render_prompt_disables_thinking():
    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            seen.update(kw, messages=messages)
            return "PROMPT"

    assert render_prompt(Tok(), _ctx(), 1) == "PROMPT"
    assert seen["enable_thinking"] is False and seen["add_generation_prompt"] is True and seen["tokenize"] is False


# ---------------------------------------------------------------- parse_sample

def test_parse_sample():
    good = '{"rationale": "Two of three sensors say HEADS.", "public_message": "I lean HEADS, 0.6.", "calibrated_belief_prob": '
    assert parse_sample(good) == ("Two of three sensors say HEADS.", "I lean HEADS, 0.6.")
    assert parse_sample(good + "0.6}") is None                              # number must not be sampled
    assert parse_sample(good[:-1]) is None                                  # missing the trailing space
    assert parse_sample(good.replace("HEADS.", 'HEADS "x".', 1)) is None   # quote inside a field
    assert parse_sample('{"rationale": "", "public_message": "a", "calibrated_belief_prob": ') is None  # empty field
    assert parse_sample(" " + good) is None


def test_token_kind():
    assert [token_kind(t) for t in ("0", "7", ".", "}", "},", ",", " ", "\n", "\n}", "05", "a", "", "٣")] == \
        ["0", "7", ".", "end", "end", "end", "end", "end", "end", "other", "other", "other", "other"]


# ---------------------------------------------------------------- value distribution

def test_value_distribution_by_hand():
    """Paths of the module docstring on a hand-made distribution:
        A (after x):        "0" 0.90, "1" 0.04, "x" 0.06
        B (after x"0"):     "." 0.80, "}" 0.15, "a" 0.05
        C (after x"0."):    "3" 0.50, "7" 0.45, "9" 0.05 (9 not expanded)
        D3 (after x"0.3"):  "5" 0.60, "}" 0.30, "\\n" 0.10
        D7 (after x"0.7"):  "0" 0.70, "2" 0.20, "q" 0.10
    """
    dA = {"0": 0.90, "1": 0.04, "x": 0.06}
    dB = {".": 0.80, "}": 0.15, "a": 0.05}
    dC = {"3": 0.50, "7": 0.45, "9": 0.05}
    dD = {3: {"5": 0.60, "}": 0.30, "\n": 0.10}, 7: {"0": 0.70, "2": 0.20, "q": 0.10}}
    m, info = value_distribution(dA, dB, dC, dD)
    pre_dot = 0.90 * 0.80
    want = np.zeros(101)
    want[100] = 0.04
    want[0] = 0.90 * 0.15
    want[35] = pre_dot * 0.50 * 0.60
    want[30] = pre_dot * 0.50 * (0.30 + 0.10)
    want[70] = pre_dot * 0.45 * 0.70          # "0.70" (digit 0) lands on the same grid point as "0.7" + end
    want[72] = pre_dot * 0.45 * 0.20
    want[95] = pre_dot * 0.05                 # unexpanded d1 = 9 -> midpoint 0.95
    np.testing.assert_allclose(m, want, atol=1e-12)
    assert info["skipped_mass"] == pytest.approx(pre_dot * 0.05)
    assert info["p_zero_token"] == 0.90 and info["p_one_token"] == 0.04 and info["p_dot"] == 0.80

    b = belief_from_masses(m)
    valid = 0.04 + 0.90 * 0.15 + pre_dot * (0.50 + 0.45 * 0.90 + 0.05)
    assert b["valid_mass"] == pytest.approx(valid)
    assert b["p_logprob"] == pytest.approx(float(GRID @ want) / valid)
    assert b["ok"] is False                   # valid ~ 0.83 < 0.95
    assert expand_digits(dC, 0.06) == [3, 7]


def test_value_distribution_merges_tokens_of_one_kind():
    """Two token strings of the same kind ("}" and "},") add their probabilities."""
    m, _ = value_distribution({"0": 1.0}, {".": 1.0}, {"4": 1.0}, {4: {"}": 0.5, "},": 0.3, "2": 0.2}})
    assert m[40] == pytest.approx(0.8) and m[42] == pytest.approx(0.2) and m.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("valid,ok", [(0.9499, False), (0.95, True), (0.999, True), (0.0, False)])
def test_missing_mass_rule(valid, ok):
    m = np.zeros(101)
    m[60] = valid
    b = belief_from_masses(m, 0.95)
    assert b["ok"] is ok
    if valid > 0:
        assert b["p_logprob"] == pytest.approx(0.60)   # the mean is renormalized by the valid mass
    else:
        assert np.isnan(b["p_logprob"])


# ---------------------------------------------------------------- fake engine, end to end

class CharTok:
    """One token per character (ids = code points); 0 is the eos token."""
    eos_token_id = 0
    all_special_ids = [0]

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids if i != 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


SAMPLED = '{"rationale": "Sensors lean HEADS.", "public_message": "HEADS, about 0.6.", "calibrated_belief_prob": '
# next-token distributions of the fake model, chosen so the belief is known exactly:
#   after x: "0" 0.98, "1" 0.02;  after x"0": "." 1;  after x"0.": "6" 0.7, "5" 0.3
#   after x"0.6": "}" 0.5, "5" 0.5 -> 0.60, 0.65;   after x"0.5": "}" 1 -> 0.50
FAKE = {"A": {"0": 0.98, "1": 0.02}, "B": {".": 1.0}, "C": {"6": 0.7, "5": 0.3},
        "D6": {"}": 0.5, "5": 0.5}, "D5": {"}": 1.0}}
P_EXACT = 0.02 * 1.0 + 0.98 * (0.7 * (0.5 * 0.60 + 0.5 * 0.65) + 0.3 * 0.50)


class FakeLLM:
    def __init__(self, malformed_first=False):
        self.calls = []
        self.malformed_first = malformed_first

    def get_tokenizer(self):
        return CharTok()

    def _dist(self, ids):
        text = "".join(chr(i) for i in ids)
        assert SAMPLED in text, "belief prefixes must contain the sampled text"
        tail = text[text.rindex(SAMPLED) + len(SAMPLED):]
        key = {"": "A", "0": "B", "0.": "C", "0.6": "D6", "0.5": "D5"}[tail]
        return FAKE[key]

    def generate(self, prompts, params, use_tqdm=False):
        params = params if isinstance(params, list) else [params] * len(prompts)
        self.calls.append([p.kw for p in params])
        outs = []
        for pr, sp in zip(prompts, params):
            ids = pr["prompt_token_ids"]
            if sp.kw.get("logprobs"):
                lp = {ord(t): SimpleNamespace(logprob=float(np.log(p))) for t, p in self._dist(ids).items()}
                comp = SimpleNamespace(token_ids=[next(iter(lp))], logprobs=[lp], finish_reason="length")
            else:
                assert "structured" in sp.kw, "sampling must be constrained"
                bad = self.malformed_first and sp.kw["seed"] % 2 == 0
                text = SAMPLED.replace('"rationale"', '"reason"') if bad else SAMPLED
                comp = SimpleNamespace(token_ids=[ord(c) for c in text] + [0], logprobs=None, finish_reason="stop")
            outs.append(SimpleNamespace(outputs=[comp]))
        return outs


class SP:
    def __init__(self, **kw):
        self.kw = kw


FAKE_PARAMS = (SP, lambda regex: {"structured": regex})


def test_backend_end_to_end_and_replay(tmp_path):
    llm = FakeLLM()
    be = VLLMBackend(llm, "unit", results_root=tmp_path, params=FAKE_PARAMS, d1_thresh=1e-3)
    ctxs = [_ctx(agent_id=i, agree=(InboxMsg(0.7, "HEADS."),)) for i in range(3)]
    res = be.respond_sync(0, 1, ctxs, 1)
    assert len(llm.calls) == 3              # sample, stage 1, stage 2
    assert len(llm.calls[1]) == 9           # 3 prefixes per call
    assert len(llm.calls[2]) == 6           # d1 in {5, 6} expanded per call
    for r in res:
        assert r.status == "ok" and r.message == "HEADS, about 0.6."
        assert r.prob == pytest.approx(P_EXACT)
        assert r.record["valid_mass"] == pytest.approx(1.0)
        assert r.record["p_sampled"] in (0.5, 0.6, 0.65, 1.0)
    rows = [json.loads(line) for line in open(tmp_path / "unit" / "vllm_calls.jsonl")]
    assert [(r["seed"], r["round_t"], r["agent_id"]) for r in rows] == [(0, 1, i) for i in range(3)]

    # a restarted backend replays the checkpoint without touching the engine
    llm2 = FakeLLM()
    be2 = VLLMBackend(llm2, "unit", results_root=tmp_path, params=FAKE_PARAMS)
    res2 = be2.respond_sync(0, 1, ctxs, 1)
    assert llm2.calls == [] and [r.prob for r in res2] == [r.prob for r in res]


def test_backend_retries_malformed_once_then_missing(tmp_path):
    llm = FakeLLM(malformed_first=True)
    be = VLLMBackend(llm, "unit2", results_root=tmp_path, params=FAKE_PARAMS)
    res = be.respond_sync(0, 1, [_ctx(agent_id=i) for i in range(8)], 1)
    seeds0 = [kw["seed"] for kw in llm.calls[0]]
    seeds1 = [kw["seed"] for kw in llm.calls[1]] if llm.calls[1] and "seed" in llm.calls[1][0] else []
    for r in res:
        assert r.status in ("ok", "missing")
        if r.status == "missing":
            assert r.reason == "malformed_sample" and r.record["attempts"] == 2 and r.prob is None
        else:
            assert r.prob == pytest.approx(P_EXACT)
    # every call that failed its first attempt was retried with a different seed
    assert len(seeds1) == sum(s % 2 == 0 for s in seeds0)
    assert not set(seeds1) & set(seeds0)


def test_low_valid_mass_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(FAKE, "A", {"0": 0.90, "x": 0.10})       # 10% of the mass leaves the number
    be = VLLMBackend(FakeLLM(), "unit3", results_root=tmp_path, params=FAKE_PARAMS)
    (r,) = be.respond_sync(0, 1, [_ctx()], 1)
    assert r.status == "missing" and r.reason == "low_valid_mass" and r.prob is None
    assert r.record["valid_mass"] == pytest.approx(0.90)
