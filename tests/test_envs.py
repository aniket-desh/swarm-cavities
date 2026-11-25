"""envs/: fixed in-degree graphs, the evidence game, the shared prompts, and the swarm (mock and fake-client
LLM backends). Zero cost: no network."""

import asyncio
import json
import re

import numpy as np
import pytest

from engine.agent import belief_schema
from engine.ledger import read_jsonl
from envs.evidence_game import evidence_for_field, logit, make_game
from envs.graphs import (fixed_indegree_signed_couplings, gaussian_couplings, sparse_signed_couplings,
                         spectral_log)
from envs.prompts import FORBIDDEN_WORDS, AgentContext, InboxMsg, dynamic, make_request, rules
from envs.swarm import LLMBackend, MockBackend, Response, Swarm, SwarmConfig, run_swarm
from tests.test_engine import FakeClient, belief_json, make_runner, message

# ---------------------------------------------------------------- graphs


@pytest.mark.parametrize("N,K,gamma,p_neg", [(1000, 8, 0.5, 0.3), (32, 8, -0.5, 0.5), (64, 10, 1.0, 0.2),
                                             (500, 12, 0.0, 0.4), (40, 8, 0.25, 0.3)])
def test_fixed_indegree_constraints(N, K, gamma, p_neg):
    """claim1.md section 5 (i)-(iii): exact in-degree, exact reciprocated count per agent, sign agreement
    +1 / 0 on reciprocated ties, negative fraction p_neg, and the realized gamma of eq. (13)."""
    J = fixed_indegree_signed_couplings(N, K, gamma, p_neg, np.random.default_rng(3))
    assert J.dtype == np.int8 and np.all(np.diag(J) == 0) and set(np.unique(J)) <= {-1, 0, 1}
    A = J != 0
    n_r = round(abs(gamma) * K)
    assert np.all(A.sum(1) == K)
    assert np.all((A & A.T).sum(1) == n_r)
    log = spectral_log(J, eigs=False)
    assert log["in_degree_min"] == log["in_degree_max"] == K
    assert log["recip_per_agent_min"] == log["recip_per_agent_max"] == n_r
    assert log["frac_neg"][0][0] == pytest.approx(p_neg, abs=0.5 / A.sum() + 1e-3)
    if n_r:
        assert log["sign_agreement"][0][0] == (1.0 if gamma >= 0 else 0.0)
    # Pearson gamma over ordered pairs with unit couplings (docstring formula)
    p, mu_s, r = K / (N - 1), 1 - 2 * log["frac_neg"][0][0], n_r / K
    a = p * mu_s**2
    expect = (r - a) / (1 - a) if gamma >= 0 else -(r + a) / (1 - a)
    assert log["gamma"][0][0] == pytest.approx(expect, abs=0.01)


def test_fixed_indegree_reproducible_and_rejects_infeasible():
    a = fixed_indegree_signed_couplings(50, 8, 0.5, 0.3, np.random.default_rng(9))
    b = fixed_indegree_signed_couplings(50, 8, 0.5, 0.3, np.random.default_rng(9))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        fixed_indegree_signed_couplings(31, 8, 0.375, 0.3, np.random.default_rng(0))   # N * n_r odd
    with pytest.raises(ValueError):
        fixed_indegree_signed_couplings(50, 8, -1.0, 0.2, np.random.default_rng(0))    # discordant r > 2 p_neg


def test_existing_generators_unchanged():
    """Same seeds give the same matrices as graphs.py at commit c03cc00 (values computed from that file)."""
    J = sparse_signed_couplings(200, 4.0, 0.5, 0.3, np.random.default_rng(2))
    G = gaussian_couplings([30, 20], 0.5, 1.0, 0.3, np.random.default_rng(1))
    assert (J != 0).sum() == 785 and int(J.sum()) == 323
    assert G[0, 1] == pytest.approx(0.13173453293212117, rel=1e-12)
    assert G.sum() == pytest.approx(18.593554751525048, rel=1e-12)


# ---------------------------------------------------------------- evidence game


def test_evidence_field_is_exact_bayes_logodds():
    game = make_game(200, mu=0.6, sigma=1.1, k=3, seed=4, truth=1)
    for e in game.evidence:
        expect = sum((1 if c.side == "HEADS" else -1) * logit(c.reliability / 100) for c in e.clues)
        assert e.h_priv == pytest.approx(expect, abs=1e-12)
        assert len(re.findall(r"right (\d+)% of the time\) reports (HEADS|TAILS)", e.text())) == 3
    # realized fields track the N(truth * mu, sigma^2) targets up to display rounding
    assert np.abs(game.h_priv - game.h_target).max() < 0.2
    assert game.h_priv.mean() == pytest.approx(0.6, abs=3 * 1.1 / np.sqrt(200))
    assert game.h_priv.std() == pytest.approx(1.1, rel=0.15)


def test_evidence_game_reproducible():
    a, b = make_game(30, seed=7), make_game(30, seed=7)
    assert [e.text() for e in a.evidence] == [e.text() for e in b.evidence] and a.truth == b.truth
    assert [e.text() for e in make_game(30, seed=8).evidence] != [e.text() for e in a.evidence]
    e = evidence_for_field(1.0, 4, np.random.default_rng(0))
    assert e.h_priv == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------- prompts


def _ctx(**kw):
    base = dict(agent_id=0, session=3, round_t=2, h_priv=0.4, evidence=make_game(1, seed=0).evidence[0].text(),
                history=((0.6, "I lean HEADS."), (0.55, "Still HEADS, weakly.")),
                agree=(InboxMsg(0.7, "HEADS seems likely."),), disagree=(InboxMsg(0.2, "I think TAILS."),))
    base.update(kw)
    return AgentContext(**base)


def test_prompts_have_no_forbidden_wording():
    """CLAUDE.md guardrail 4: nothing the model reads calls a field or text private, hidden or secret."""
    texts = [rules(0), rules(3), dynamic(_ctx(), 1), dynamic(_ctx(round_t=0, history=(), agree=(), disagree=()), 2),
             _ctx().evidence, json.dumps(belief_schema())]
    for t in texts:
        for w in FORBIDDEN_WORDS:
            assert w not in t.lower(), (w, t[:80])


def test_prompt_layout_and_memory_window():
    req = make_request(_ctx(), H=1)
    assert req.system_rules == rules(1) and "sensor" in req.evidence.lower()
    d = req.dynamic
    assert d.startswith("Discussion 3, round 3.")
    assert "Still HEADS, weakly." in d and "I lean HEADS." not in d          # only the last H = 1 answer
    assert "Round 2: probability of HEADS 0.55" in d
    assert d.index("tend to agree with") < d.index("0.70") < d.index("tend to disagree with") < d.index("0.20")
    d2 = dynamic(_ctx(), H=2)
    assert "Round 1: probability of HEADS 0.60" in d2
    d0 = dynamic(_ctx(), H=0)
    assert "recent rounds" not in d0 and "recent rounds" not in rules(0)
    first = dynamic(_ctx(round_t=0, history=(), agree=(), disagree=()), H=1)
    assert "first round" in first and "tend to agree" not in first
    empty = dynamic(_ctx(agree=()), H=1)
    assert "(none this round)" in empty


# ---------------------------------------------------------------- swarm: mock backend


def _small(N=24, K=4, gamma=0.5, p_neg=0.25, seed=0):
    return (fixed_indegree_signed_couplings(N, K, gamma, p_neg, np.random.default_rng(seed)),
            make_game(N, mu=0.4, sigma=0.8, seed=seed, truth=1))


def test_mock_swarm_sync_matches_direct_iteration(tmp_path):
    """Sync swarm with the deterministic mock equals s(t+1) = tanh(beta (b J) s(t) + h_priv + kappa s(t))
    iterated directly on the numbers, up to the 2-decimal display of peer probabilities (not used here:
    the mock reads InboxMsg.prob, which the swarm stores unrounded)."""
    J, game = _small()
    beta, b, kappa = 0.9, 0.5, 0.3
    out = run_swarm(J, game, SwarmConfig("e", T=5, H=1, replicas=1, results_root=tmp_path),
                    MockBackend(beta, b, b, kappa=kappa))
    s = np.tanh(game.h_priv)
    ref = [s]
    for _ in range(5):
        s = np.tanh(beta * (b * J) @ s + game.h_priv + kappa * s)
        ref.append(s)
    np.testing.assert_allclose(out["s"][0], np.array(ref), atol=1e-12)
    assert out["p"].shape == (1, 6, 24) and not out["missing"].any()
    meta = json.loads((tmp_path / "e" / "swarm_meta.json").read_text())
    assert meta["graph"]["in_degree_min"] == 4 and meta["config"]["T"] == 5


def test_mock_swarm_async_reads_latest(tmp_path):
    """Async control: agents update in a random order and read messages already updated this round
    (a Gauss-Seidel sweep), so it differs from sync but equals a direct sequential iteration."""
    J, game = _small()
    beta, b = 0.9, 0.5
    cfg = SwarmConfig("a", T=3, H=0, replicas=1, mode="async", results_root=tmp_path)
    sw = Swarm(J, game, cfg, MockBackend(beta, b, b))
    out = asyncio.run(sw.run())
    s = np.tanh(game.h_priv)
    for t in range(1, 4):
        for i in sw._rng(0, t, 2**31 - 1, 0).permutation(24):
            s[i] = np.tanh(beta * (b * J[i]) @ s + game.h_priv[i])
        np.testing.assert_allclose(out["s"][0, t], s, atol=1e-12)
    sync = run_swarm(J, game, SwarmConfig("s", T=3, H=0, replicas=1, results_root=tmp_path), MockBackend(beta, b, b))
    assert np.abs(sync["s"][0, 3] - out["s"][0, 3]).max() > 1e-3


class _Crash(MockBackend):
    async def respond(self, seed, round_t, contexts, H):
        if round_t == 3 and seed == 1:
            raise RuntimeError("simulated crash")
        return await super().respond(seed, round_t, contexts, H)


def test_mock_swarm_resume_reproduces_uninterrupted_run(tmp_path):
    """Stochastic mock, 3 replicas sharing round 0: a crash in round 3 of replica 1, then a restart over the
    same checkpoint file, gives exactly the uninterrupted result."""
    J, game = _small()
    kw = dict(beta=1.0, b_plus=0.6, b_minus=0.4, kappa=0.2, stochastic=True, seed=5)
    full = run_swarm(J, game, SwarmConfig("full", T=5, replicas=3, results_root=tmp_path), MockBackend(**kw))
    cfg = SwarmConfig("cr", T=5, replicas=3, results_root=tmp_path)
    with pytest.raises(RuntimeError):
        run_swarm(J, game, cfg, _Crash(**kw))
    rows = list(read_jsonl(tmp_path / "cr" / "swarm_rounds.jsonl"))
    assert {(r["replica"], r["round_t"]) for r in rows} >= {(1, 0), (1, 1), (1, 2)}
    resumed = run_swarm(J, game, cfg, MockBackend(**kw))
    np.testing.assert_array_equal(resumed["s"], full["s"])
    assert np.all(full["s"][:, 0] == full["s"][0, 0])                  # shared initial answers
    assert np.abs(full["s"][0, 1:] - full["s"][1, 1:]).sum() > 0        # replicas diverge afterwards


# ---------------------------------------------------------------- swarm: LLM backend with a fake client


def _mean_agree_responder(refuse_call=None):
    """Fake model: p = mean stated probability in the agree group, else 0.6; refuses call number refuse_call."""
    def responder(params, n):
        if n == refuse_call:
            return message(belief_json(0.5), stop="refusal")
        user = params["messages"][0]["content"]
        agree = user.split("tend to agree with:")[1].split("Messages from sources")[0] if "tend to agree" in user else ""
        ps = [float(x) for x in re.findall(r"Stated probability of HEADS ([0-9.]+)", agree)]
        return message(belief_json(round(float(np.mean(ps)), 4) if ps else 0.6))
    return responder


def test_llm_swarm_through_runner(tmp_path):
    """N=6, T=2, two replicas through engine.Runner with a fake client: the rendered inbox carries the right
    senders, round 0 is paid once (shared), every call is in the ledger, and a refusal becomes a NaN
    belief with the previous message carried forward."""
    N = 6
    J = fixed_indegree_signed_couplings(N, 2, 0.0, 0.0, np.random.default_rng(1))   # all agree ties
    game = make_game(N, seed=1)
    client = FakeClient(_mean_agree_responder(refuse_call=9))
    runner, ledger = make_runner(tmp_path, client, exp="llm-swarm")
    cfg = SwarmConfig("llm-swarm", T=2, H=1, replicas=2, results_root=tmp_path / "results")
    out = run_swarm(J, game, cfg, LLMBackend(runner))
    assert out["p"].shape == (2, 3, N)
    assert len(client.calls) == N + 2 * N * 2                             # round 0 once, then 2 replicas x 2 rounds
    assert len(list(read_jsonl(ledger))) == len(client.calls)
    assert np.allclose(out["p"][:, 0], 0.6)
    assert out["missing"].sum() == 1 and np.isnan(out["p"][out["missing"]]).all()
    # the replica without the refusal: round 1 everyone reads 0.6, round 2 the mean of 0.6s again
    clean = int(out["missing"][0].any())
    np.testing.assert_allclose(out["p"][clean, 1:], 0.6)
    # prompts are replica-specific from round 1 on (otherwise the replay cache would merge replicas)
    users = [c["messages"][0]["content"] for c in client.calls]
    assert any(u.startswith("Discussion 0, round 2") for u in users) and any(u.startswith("Discussion 1, round 2") for u in users)
    assert all("tend to agree with" in u for u in users if "round 1." not in u.split("\n")[0])
    # restart: everything replays from the checkpoints, no new calls
    client2 = FakeClient(lambda p, n: pytest.fail("should not call"))
    runner2, _ = make_runner(tmp_path, client2, exp="llm-swarm")
    out2 = run_swarm(J, game, cfg, LLMBackend(runner2))
    np.testing.assert_array_equal(np.nan_to_num(out2["p"], nan=-1), np.nan_to_num(out["p"], nan=-1))


def test_async_with_batch_backend_refused(tmp_path):
    J, game = _small(N=8, K=2)
    runner, _ = make_runner(tmp_path, FakeClient())
    with pytest.raises(ValueError):
        Swarm(J, game, SwarmConfig("x", T=1, mode="async", results_root=tmp_path), LLMBackend(runner, batch=True))


# ---------------------------------------------------------------- integration: mock swarm vs DMFT


@pytest.mark.parametrize("gamma,p_neg", [(0.5, 0.4), (-0.5, 0.5)])
def test_mock_swarm_N1000_matches_dmft(gamma, p_neg, tmp_path):
    """Deterministic mock swarm at N=1000 on fixed in-degree K=16 graphs, couplings b J with b = 1/sqrt(K)
    (g = 1), against the single-site DMFT of claim1.md eqs. (4)-(8) fed the realized block statistics
    (Jbar = K b (1 - 2 p_neg), g^2, gamma from spectral_log) and the realized h_priv mean and s.d.

    Reference: theory/checks/toy.py::dmft_mc.
    TODO: switch to theory/dmft_solver.py once it lands (its interface was not fixed when this was written).

    Tolerance 0.025 on m(t) and C(t,t) over t <= 8. The disorder s.e. of the 8-graph average is about
    0.007, and a sparse graph at K = 16 carries O(1/K) corrections that the dense theory leaves out
    (non-Gaussian field, and with a fixed in-degree the noise covariance g^2 C - (Jbar^2/K) m m,
    claim5.md section 2): measured residuals were 0.004 (m) / 0.006 (C) at gamma=0.5 and 0.012 / 0.018 at
    gamma=-0.5. The same run without the reaction term misses by 0.05-0.14, so the test resolves it.
    """
    from theory.checks.toy import dmft_mc

    N, K, T, beta, G = 1000, 16, 8, 1.2, 8
    b = 1 / np.sqrt(K)
    m_sim, C_sim, stats, hps = [], [], [], []
    for g in range(G):
        J = fixed_indegree_signed_couplings(N, K, gamma, p_neg, np.random.default_rng(g))
        game = make_game(N, mu=0.3, sigma=0.5, k=3, seed=g, truth=1)
        hps.append(game.h_priv)
        s = run_swarm(J, game, SwarmConfig(f"g{g}", T=T, H=0, replicas=1, results_root=tmp_path),
                      MockBackend(beta, b, b))["s"][0]
        m_sim.append(s.mean(1))
        C_sim.append((s**2).mean(1))
        L = spectral_log(np.where(J > 0, b, np.where(J < 0, -b, 0.0)), eigs=False)
        stats.append([L["N_mean"][0][0], L["N_var"][0][0], L["gamma"][0][0]])
    Jbar, g2, gam = np.mean(stats, 0)
    assert gam == pytest.approx(gamma, abs=0.01)
    hp = np.concatenate(hps)
    args = (np.random.default_rng(0), [1.0], [[Jbar]], [[np.sqrt(g2)]], [[gam]], [hp.mean()], [hp.std()],
            [beta], [0.0], 0.0, T, 100_000)
    ref, ref0 = dmft_mc(*args), dmft_mc(*args, reaction=False)
    m, C = np.mean(m_sim, 0), np.mean(C_sim, 0)
    err_m, err_C = np.abs(m - ref["m"][0]).max(), np.abs(C - np.diag(ref["C"][0])).max()
    assert err_m < 0.025 and err_C < 0.025, (err_m, err_C)
    assert max(np.abs(m - ref0["m"][0]).max(), np.abs(C - np.diag(ref0["C"][0])).max()) > 0.04
