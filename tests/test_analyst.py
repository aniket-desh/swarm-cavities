"""envs/analyst.py without a GPU: field rendering (N1)-(N2), prompts (no forbidden words, round-0 and inbox layout),
the decision-token arithmetic (D1) and the missing rule, stored-uniform call sampling (D2), the swarm loop against
theory/checks/analyst_lib.simulate with the mock law, checkpoint resume, and the vLLM backend on a fake engine."""

from types import SimpleNamespace

import numpy as np
import pytest

import envs.analyst as A
from envs.evidence_game import logit
from envs.graphs import fixed_indegree_signed_couplings
from theory.checks import analyst_lib as al


# ---------------------------------------------------------------- rendering

def test_news_item_realized_field_is_the_displayed_log_odds():
    for nu_nom in (0.5, 1.0):
        for xi in np.linspace(-3, 3, 61):
            side, pct, xr = A.news_item(xi, nu_nom)
            assert A.RHO_MIN <= pct <= A.RHO_MAX
            assert side == ("UP" if xi >= 0 else "DOWN")
            assert xr == pytest.approx((1 if side == "UP" else -1) * logit(pct / 100) / (2 * nu_nom))
            # rounding to whole percent moves the full log-odds by at most 0.005 / (rho (1 - rho)) unless a clip binds
            rho = 1 / (1 + np.exp(-2 * nu_nom * abs(xi)))
            if 0.505 < rho < 0.945:
                assert abs(xr - xi) * 2 * nu_nom <= 0.005 / (rho * (1 - rho)) * 1.2


def test_research_note_realized_field():
    rng = np.random.default_rng(0)
    for h0 in (-2.0, -0.3, 0.0, 0.8, 2.5):
        n = A.research_note(h0, 0.3, rng)
        L = sum((1 if s == "UP" else -1) * logit(p / 100) for s, p in n.indicators)
        assert n.h0_real == pytest.approx(L / 0.6)
        assert len(n.indicators) == 3 and "Indicator 3" in n.text()


def test_game_is_deterministic_and_shapes():
    g1, g2 = A.make_game(20, 5, R=2, seed=3), A.make_game(20, 5, R=2, seed=3)
    assert np.array_equal(g1.U, g2.U) and np.array_equal(g1.xi_real, g2.xi_real)
    assert [n.text() for n in g1.notes] == [n.text() for n in g2.notes]
    assert g1.U.shape == (6, 2, 20) and g1.xi.shape == (6, 20)
    assert not np.array_equal(g1.U[:, 0], g1.U[:, 1])            # replicas share news, not call draws


def _ctx(round_t=1, n_allies=2, n_faded=1):
    note = A.research_note(0.5, 0.3, np.random.default_rng(1))
    if round_t == 0:
        return A.AnalystContext(0, 4, 0, note.text(), "UP", 63)
    return A.AnalystContext(0, 4, round_t, note.text(), "DOWN", 71, own=("BUY", "Demand looks firm."),
                            allies=tuple(("BUY", f"ally note {k}") for k in range(n_allies)),
                            faded=tuple(("SELL", f"faded note {k}") for k in range(n_faded)))


@pytest.mark.parametrize("wording", list(A.WORDINGS))
def test_prompts_have_no_forbidden_words(wording):
    for ctx in (_ctx(0), _ctx(3)):
        text = " ".join(m["content"] for m in A.render_messages(ctx, wording)).lower()
        assert not any(w in text for w in A.FORBIDDEN_WORDS), wording       # guardrail 4
    for k in range(5):
        assert not any(w in A.template_note("BUY", np.random.default_rng(k)).lower() for w in A.FORBIDDEN_WORDS)


def test_prompt_layout():
    sys0, user0 = (m["content"] for m in A.render_messages(_ctx(0), "mild"))
    assert "Indicator 1" in sys0 and "fictional" in sys0 and A.ASSET in sys0
    assert "first round" in user0 and "Allies" not in user0 and "right 63%" in user0
    sys1, user1 = (m["content"] for m in A.render_messages(_ctx(3, 2, 0), "strong"))
    assert 'Your call last round: BUY. Your note: "Demand looks firm."' in user1
    assert user1.count("ally note") == 2 and "Faded colleagues' latest calls:\n(none)" in user1
    assert "round 4" in user1 and "points DOWN" in user1
    assert A.WORDINGS["strong"][0] in sys1 and A.WORDINGS["strong"][1] in sys1
    assert '{"call": "BUY" or "SELL", "note": "your note"}' in sys1


# ---------------------------------------------------------------- decision arithmetic

BUY, SELL = 1001, 1002


def test_decision_belief_by_hand():
    lp = {BUY: np.log(0.70), SELL: np.log(0.28), 7: np.log(0.02)}
    b = A.decision_belief(lp, BUY, SELL)
    assert b["valid_mass"] == pytest.approx(0.98) and b["ok"]
    assert b["p_buy"] == pytest.approx(0.70 / 0.98)
    assert b["x"] == pytest.approx(0.5 * np.log(0.70 / 0.28))
    assert np.tanh(b["x"]) == pytest.approx(2 * b["p_buy"] - 1)            # s = tanh x is (D1) itself
    assert b["censored"] is None


@pytest.mark.parametrize("valid,ok", [(0.9499, False), (0.95, True), (0.999, True)])
def test_missing_rule(valid, ok):
    lp = {BUY: np.log(valid * 0.6), SELL: np.log(valid * 0.4), 5: np.log(1 - valid)}
    b = A.decision_belief(lp, BUY, SELL, 0.95)
    assert b["ok"] is ok and b["p_buy"] == pytest.approx(0.6)


def test_censored_token():
    lp = {BUY: np.log(0.999), 5: np.log(1e-4), 6: np.log(1e-6)}         # SELL below the top-k
    b = A.decision_belief(lp, BUY, SELL)
    assert b["censored"] == "sell_below_topk" and b["p_buy"] == 1.0 and b["ok"]
    assert b["x"] == pytest.approx(0.5 * (np.log(0.999) - np.log(1e-6)))  # a lower bound on the true x
    b = A.decision_belief({SELL: np.log(0.99), 5: np.log(1e-5)}, BUY, SELL)
    assert b["censored"] == "buy_below_topk" and b["p_buy"] == 0.0 and b["x"] < 0
    assert not A.decision_belief({5: 0.0}, BUY, SELL)["ok"]               # neither token: missing


# ---------------------------------------------------------------- stored-uniform calls

def test_stored_uniform_call_sampling():
    rng = np.random.default_rng(0)
    u = rng.random(200_000)
    p, p2 = 0.62, 0.70
    c1, c2 = A.sample_call(p, u), A.sample_call(p2, u)
    assert (c1 == 1).mean() == pytest.approx(p, abs=0.005)               # temperature-1 sample of P_buy
    flips = c1 != c2
    assert flips.mean() == pytest.approx(p2 - p, abs=0.003)               # a twin flips w.p. |dP| ...
    assert np.all(c1[flips] == -1) and np.all(c2[flips] == 1)            # ... only toward the moved side
    assert np.array_equal(A.sample_call(p, u), c1)                       # same uniform, same call


# ---------------------------------------------------------------- swarm loop

def _graph(N, K, world, seed=0):
    return fixed_indegree_signed_couplings(N, K, al.WORLDS[world], 0.5, np.random.default_rng(seed))


@pytest.mark.parametrize("world", ["mutual", "oneway", "antagonistic"])
def test_swarm_loop_matches_analyst_lib(tmp_path, world):
    """The loop with the mock law is (A1)-(A3) exactly: same beliefs and calls as analyst_lib.simulate given the
    same graph, notes, news and uniforms."""
    N, K, T = 48, 8, 6
    J = _graph(N, K, world)
    g = A.make_game(N, T, R=2, seed=5)
    P = al.Agent(2.0, 1.5, nu=0.5, a0=0.3, c=0.1, kappa=0.4, K=K)
    mock = A.MockAnalyst(P.a_plus, P.a_minus, P.nu, P.a0, P.c, P.kappa, K=K)
    out = A.AnalystSwarm(J, g, A.AnalystSwarmConfig("unit", T, R=2, K=K, results_root=tmp_path), mock).run()
    ref = al.simulate(al.couplings(J, P), P, T, dict(h0=g.h0, xi=g.xi[:, None, :], U=g.U))
    np.testing.assert_allclose(out["s"], ref["s"], atol=1e-12)
    assert np.array_equal(out["sig"], ref["sig"].astype(int))
    assert not out["missing"].any()


class _Flaky(A.MockAnalyst):
    """Crashes once at a given round, after the earlier rounds were checkpointed."""

    def __init__(self, *a, crash_round=None, **k):
        super().__init__(*a, **k)
        self.crash_round, self.calls = crash_round, 0

    def beliefs(self, ctxs, key=None):
        self.calls += 1
        if key[1] == self.crash_round:
            self.crash_round = None
            raise RuntimeError("boom")
        return super().beliefs(ctxs, key)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    N, K, T = 32, 8, 5
    J, g = _graph(N, K, "mutual", 1), A.make_game(N, T, seed=2)
    full = A.AnalystSwarm(J, g, A.AnalystSwarmConfig("a", T, K=K, results_root=tmp_path),
                          A.MockAnalyst(2, 2, K=K)).run()
    flaky = _Flaky(2, 2, K=K, crash_round=3)
    sw = A.AnalystSwarm(J, g, A.AnalystSwarmConfig("b", T, K=K, results_root=tmp_path), flaky)
    with pytest.raises(RuntimeError):
        sw.run()
    n_before = flaky.calls
    out = sw.run()
    assert flaky.calls - n_before == T + 1 - 3                            # rounds 0-2 restored, not recomputed
    np.testing.assert_allclose(out["s"], full["s"])
    assert np.array_equal(out["sig"], full["sig"])


def test_missing_decision_carries_call_forward(tmp_path):
    class Gappy(A.MockAnalyst):
        def beliefs(self, ctxs, key=None):
            out = super().beliefs(ctxs, key)
            if key[1] == 2:
                out[0] = {"p_buy": float("nan"), "x": float("nan"), "ok": False, "valid_mass": 0.5, "censored": None}
            return out

    N, K, T = 16, 4, 3
    out = A.AnalystSwarm(_graph(N, K, "oneway", 2), A.make_game(N, T, seed=0),
                         A.AnalystSwarmConfig("m", T, K=K, results_root=tmp_path), Gappy(2, 2, K=K)).run()
    assert out["missing"][2, 0, 0] and out["missing"].sum() == 1 and np.isnan(out["s"][2, 0, 0])
    assert out["sig"][2, 0, 0] == out["sig"][1, 0, 0]


# ---------------------------------------------------------------- vLLM backend on a fake engine

class WordTok:
    """Characters are tokens, except that "BUY" and "SELL" are single tokens (as in Qwen3.5 after '{"call": "')."""
    eos_token_id = 0
    all_special_ids = [0]
    words = {"BUY": BUY, "SELL": SELL}

    def encode(self, s, add_special_tokens=False):
        ids, k = [], 0
        while k < len(s):
            for w, i in self.words.items():
                if s.startswith(w, k):
                    ids.append(i)
                    k += len(w)
                    break
            else:
                ids.append(ord(s[k]))
                k += 1
        return ids

    def decode(self, ids, **kw):
        inv = {v: k for k, v in self.words.items()}
        return "".join(inv.get(i, chr(i)) for i in ids if i != 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert enable_thinking is False and add_generation_prompt and not tokenize
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeLLM:
    """Decision: P(BUY) 0.70, P(SELL) 0.28, "x" 0.02. Note: 'Firm demand, <call>."}', except that the first
    `truncate` note requests are cut off at max_tokens."""

    def __init__(self, truncate=0):
        self.tok, self.calls, self.truncate = WordTok(), [], truncate

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params, use_tqdm=False):
        params = params if isinstance(params, list) else [params] * len(prompts)
        self.calls.append(params)
        outs = []
        for pr, sp in zip(prompts, params):
            text = self.tok.decode(pr["prompt_token_ids"])
            if sp.kw.get("logprobs"):
                assert text.endswith(A.DECISION_PREFIX) and sp.kw["max_tokens"] == 1
                lp = {BUY: SimpleNamespace(logprob=np.log(0.70)), SELL: SimpleNamespace(logprob=np.log(0.28)),
                      ord("x"): SimpleNamespace(logprob=np.log(0.02))}
                comp = SimpleNamespace(token_ids=[BUY], logprobs=[lp], finish_reason="length")
            else:
                assert text.endswith(A.NOTE_PREFIX) and sp.kw["structured"] == A.NOTE_REGEX
                call = "BUY" if text.endswith("BUY" + A.NOTE_PREFIX) else "SELL"
                if self.truncate > 0:
                    self.truncate -= 1
                    comp = SimpleNamespace(token_ids=self.tok.encode("Firm dem"), finish_reason="length")
                else:
                    comp = SimpleNamespace(token_ids=self.tok.encode(f'Firm demand, {call}."}}') + [0],
                                           finish_reason="stop")
            outs.append(SimpleNamespace(outputs=[comp]))
        return outs


class SP:
    def __init__(self, **kw):
        self.kw = kw


FAKE_PARAMS = (SP, lambda regex: {"structured": regex})


def test_vllm_backend_end_to_end(tmp_path):
    llm = FakeLLM()
    be = A.AnalystVLLM(llm, "moderate", "unit", K=2, params=FAKE_PARAMS)
    ctxs = [_ctx(1), _ctx(0)]
    bel = be.beliefs(ctxs)
    assert be.buy_id == BUY and be.sell_id == SELL
    for b in bel:
        assert b["ok"] and b["p_buy"] == pytest.approx(0.70 / 0.98) and b["valid_mass"] == pytest.approx(0.98)
        assert b["top_other"][0][0] == "x"
    notes = be.notes(ctxs, ["BUY", "SELL"], key=(0, 1))
    assert [n["note"] for n in notes] == ["Firm demand, BUY.", "Firm demand, SELL."]
    assert all(n["note_status"] == "ok" for n in notes)
    # the swarm loop on the fake engine: K=2 colleagues each, every call checkpointed with its diagnostics
    N, T = 6, 2
    J = _graph(N, 2, "mutual", 3)
    out = A.AnalystSwarm(J, A.make_game(N, T, seed=1), A.AnalystSwarmConfig("fake", T, K=2, results_root=tmp_path),
                         A.AnalystVLLM(FakeLLM(), "mild", "fake", K=2, params=FAKE_PARAMS)).run()
    assert np.allclose(out["s"], 2 * 0.70 / 0.98 - 1) and not out["missing"].any()
    recs = list(A.read_jsonl(tmp_path / "fake" / "analyst_calls.jsonl"))
    assert len(recs) == N * (T + 1) and {(r["round_t"], r["agent_id"]) for r in recs} == \
        {(t, i) for t in range(T + 1) for i in range(N)}
    assert all(r["note"].startswith("Firm demand") for r in recs)


def test_vllm_backend_note_retry_and_truncation():
    ctxs = [_ctx(1)]
    be = A.AnalystVLLM(FakeLLM(truncate=1), "mild", "unit", K=2, params=FAKE_PARAMS)
    be.beliefs(ctxs)
    n = be.notes(ctxs, ["SELL"], key=(0, 1))[0]
    assert n["note_status"] == "ok" and n["note_attempts"] == 2 and n["note"] == "Firm demand, SELL."
    be = A.AnalystVLLM(FakeLLM(truncate=2), "mild", "unit", K=2, params=FAKE_PARAMS)
    be.beliefs(ctxs)
    n = be.notes(ctxs, ["SELL"], key=(0, 1))[0]
    assert n["note_status"] == "truncated" and n["note"] == "Firm dem"      # the call stands, the note is cut


def test_parse_note():
    assert A.parse_note('Demand is firm, BUY."}') == "Demand is firm, BUY."
    assert A.parse_note('Demand "firm"."}') is None and A.parse_note('a"} extra') is None and A.parse_note('"}') is None


# ---------------------------------------------------------------- prompt variants (08b-analyst-calibration-2)

VARIANTS = [A.PromptVariant(q, o, t, f) for q in ("up", "down") for o in ("buy_first", "sell_first")
            for t in ("buysell", "yesno") for f in ("ordinary", "reversed", "reversed_equal", "reversed_equal_nonote")]


def test_default_variant_is_the_original_prompt():
    """Frozen hash of the prompts used in 08-analyst-calibration: the variant machinery must not change them."""
    import hashlib
    g = A.make_game(3, 2, seed=0)
    ctxs = [A.AnalystContext(0, 4, 0, g.notes[0].text(), "UP", 63),
            A.AnalystContext(1, 4, 3, g.notes[1].text(), "DOWN", 71, own=("BUY", "x"),
                             allies=(("BUY", "a"), ("SELL", "b")), faded=(("SELL", "c"),))]
    s = "".join(m["content"] for w in A.WORDINGS for c in ctxs for m in A.render_messages(c, w))
    assert hashlib.sha256(s.encode()).hexdigest() == "b0a45d9f4283f49a121f9224533f8aa11893e60a5284fb603ff338f138af43fd"


@pytest.mark.parametrize("v", VARIANTS, ids=str)
def test_variants_render_without_forbidden_words(v):
    text = " ".join(m["content"] for m in A.render_messages(_ctx(3), "max", v=v)).lower()
    assert not any(w in text for w in A.FORBIDDEN_WORDS)
    assert ("whether orsilane freight will close down this week" in text) == (v.question == "down")
    assert v.prefix[:-1] in text or v.prefix.replace('"', '').strip() in text.replace('"', '')
    if v.fade == "ordinary":
        assert "faded note 0" in text and "(reversed)" not in text
    elif v.fade == "reversed_equal_nonote":
        assert "- buy (reversed).\n" in text + "\n" and "faded note 0" not in text and "notes are not shown" in text
    else:
        assert '- buy (reversed). their note: "faded note 0"' in text      # the context's SELL shown reversed


def test_yesno_down_maps_no_to_buy():
    """In the YES/NO variant asking whether the asset closes down, P(BUY) = P(NO)/(P(YES)+P(NO))."""
    YES, NO = 1003, 1004

    class YNTok(WordTok):
        words = {"BUY": BUY, "SELL": SELL, "YES": YES, "NO": NO}

    class YNLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.tok = YNTok()

        def generate(self, prompts, params, use_tqdm=False):
            out = []
            for pr in prompts:
                assert self.tok.decode(pr["prompt_token_ids"]).endswith('{"answer": "')
                lp = {YES: SimpleNamespace(logprob=np.log(0.2)), NO: SimpleNamespace(logprob=np.log(0.79)),
                      7: SimpleNamespace(logprob=np.log(0.01))}
                out.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=[NO], logprobs=[lp], finish_reason="length")]))
            return out

    be = A.AnalystVLLM(YNLLM(), "max", "u", K=2, params=FAKE_PARAMS, variant=A.PromptVariant(question="down", tokens="yesno"))
    b = be.beliefs([_ctx(1)])[0]
    assert be.buy_id == NO and be.sell_id == YES
    assert b["p_buy"] == pytest.approx(0.79 / 0.99) and b["x"] == pytest.approx(0.5 * np.log(0.79 / 0.2))


# ---------------------------------------------------------------- de-biased sampling and kick twins (09-analyst-signrule)

def _sr_law(P, chat):
    from theory.checks import analyst_signrule_check as sr
    return sr, sr.Law(uA=P.a_plus, uD=-P.a_minus, nu=P.nu, a0=P.a0, kappa=P.kappa, c=P.c, chat=chat)


def _sr_inputs(g):
    z = np.zeros((g.T + 1, 1, g.N))
    return dict(h0=g.h0, xi=g.xi[:, None, :], U=g.U, e_sh=z, e_own=np.zeros_like(g.U))


@pytest.mark.parametrize("world", ["mutual", "oneway", "antagonistic"])
def test_debiased_loop_matches_signrule_simulation(tmp_path, world):
    """Calls drawn from tanh(x - chat) (analyst_signrule.md (S1)-(S3), calls-only law): the loop with the mock law
    gives the same calls and logits as theory/checks/analyst_signrule_check.simulate, and as analyst_lib.simulate
    with bias c - chat (the exact equivalence of section 2)."""
    N, K, T, chat = 64, 16, 6, -0.637
    J = _graph(N, K, world, 4)
    g = A.make_game(N, T, R=2, seed=8)
    P = al.Agent(1.735, -0.146, nu=0.3, a0=0.09, c=-0.637, kappa=-0.095, K=K)
    mock = A.MockAnalyst(P.a_plus, P.a_minus, P.nu, P.a0, P.c, P.kappa, K=K)
    out = A.AnalystSwarm(J, g, A.AnalystSwarmConfig("db", T, R=2, K=K, chat=chat, session=3, results_root=tmp_path),
                         mock).run()
    sr, L = _sr_law(P, chat)
    ref = sr.simulate(J, L, T, _sr_inputs(g))
    np.testing.assert_allclose(out["x"], ref["X"], atol=1e-12)
    assert np.array_equal(out["sig"], ref["sig"].astype(int))
    ref2 = al.simulate(al.couplings(J, P), al.replace(P, c=P.c - chat), T, dict(h0=g.h0, xi=g.xi[:, None, :], U=g.U))
    assert np.array_equal(out["sig"], ref2["sig"].astype(int))
    recs = list(A.read_jsonl(tmp_path / "db" / "analyst_calls.jsonl"))
    assert all(r["p_sample"] == pytest.approx((1 + np.tanh(r["x"] - chat)) / 2) for r in recs)


def test_fixed_session_gives_identical_replica_prompts(tmp_path):
    g = A.make_game(20, 2, R=2, seed=1)
    sw = A.AnalystSwarm(_graph(20, 4, "mutual"), g, A.AnalystSwarmConfig("s", 2, R=2, K=4, session=7, results_root=tmp_path),
                        A.MockAnalyst(1, 1, K=4))
    a, b = sw.context(0, 0, 3, None, None), sw.context(1, 0, 3, None, None)
    assert a == b and a.session == 7


@pytest.mark.parametrize("world", ["mutual", "oneway", "antagonistic"])
def test_kick_slice_matches_signrule_kick_twins(tmp_path, world):
    """kick_slice with the mock law reproduces analyst_signrule_check.kick_twins: the logit change of every reader at
    t0+1 and of the kicked analyst at t0+2 (both replays, equal under the calls-only law), and E = 0 with identical
    prompts in the one-way world. Null twins move nothing under the mock (its notes are templates)."""
    N, K, T, chat, t0, rep = 64, 16, 8, -0.637, 4, 1
    J = _graph(N, K, world, 6)
    g = A.make_game(N, T, R=2, seed=9)
    P = al.Agent(1.735, -0.146, nu=0.3, a0=0.09, c=-0.637, kappa=-0.095, K=K)
    mock = A.MockAnalyst(P.a_plus, P.a_minus, P.nu, P.a0, P.c, P.kappa, K=K)
    sw = A.AnalystSwarm(J, g, A.AnalystSwarmConfig("k", T, R=2, K=K, chat=chat, session=0, results_root=tmp_path), mock)
    sw.run()
    kicked, nulls = [0, 5, 17, 33], [2, 40]
    recs = A.kick_slice(sw, mock, rep, t0, kicked, nulls)
    sr, L = _sr_law(P, chat)
    inp = _sr_inputs(g)
    base = sr.simulate(J, L, T, inp)
    tw = sr.kick_twins(J, L, base, inp, rep, t0, kicked)
    for b, r in enumerate([r for r in recs if r["kind"] == "kick"]):
        assert r["i"] == kicked[b] and r["D"] == tw["D"][b]
        for rd in r["readers"]:
            assert rd["dx"] == pytest.approx(tw["dX1"][b, rd["j"]], abs=1e-12)
            assert rd["tie"] == J[rd["j"], r["i"]]
        E_ref = (np.tanh(base["X"][t0 + 2, rep, r["i"]] + tw["dX2"][b, r["i"]] - chat) - base["s"][t0 + 2, rep, r["i"]]) * r["D"] / 4
        assert r["E_rewritten"] == pytest.approx(E_ref, abs=1e-12) and r["E_kept"] == pytest.approx(E_ref, abs=1e-12)
        assert r["Ex_rewritten"] == pytest.approx(tw["dX2"][b, r["i"]] * r["D"] / 4, abs=1e-12)
        if world == "oneway":
            assert r["identical_rewritten"] and r["E_rewritten"] == 0
    for r in [r for r in recs if r["kind"] == "null"]:
        assert r["dx"] == 0 and r["n_readers"] == (J[:, r["i"]] != 0).sum()


def test_kick_slice_records_are_json(tmp_path):
    import json
    N, K, T = 48, 16, 6
    g = A.make_game(N, T, R=2, seed=2)
    mock = A.MockAnalyst(1.7, -0.15, 0.3, 0.09, -0.6, -0.1, K=K)
    sw = A.AnalystSwarm(_graph(N, K, "antagonistic", 1), g, A.AnalystSwarmConfig("j", T, R=2, K=K, chat=-0.6,
                        session=0, results_root=tmp_path), mock)
    sw.run()
    for r in A.kick_slice(sw, mock, 0, 3, [1, 2], [3]):
        json.dumps(r, allow_nan=True)
