"""cavity/probes.py on the mock agent, where the true answers are known, and through engine.Runner with a
fake client. Zero cost."""

import asyncio
import re

import numpy as np
import pytest

import cavity.probes as P
from engine.budget import BudgetGuard
from envs.prompts import make_request
from envs.swarm import LLMBackend, MockBackend
from tests.test_engine import FakeClient, belief_json, make_runner, message

K = 8
B = 1 / np.sqrt(K)


@pytest.fixture
def guard(tmp_path):
    return BudgetGuard(ledger_path=tmp_path / "ledger.jsonl", pod_spend_fn=lambda: 0.0)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- synthetic inboxes


@pytest.mark.parametrize("K_", [8, 12])
def test_synthetic_inbox_realizes_target_field(K_):
    """Every composition carries the same nominal field: (sum_agree s - sum_disagree s)/K = hbar, exactly
    before display rounding and to 0.01 after it; group sizes follow claim1.md section 4.1."""
    for lev in (-2, -1, 0, 1, 2):
        hbar = 0.1 * lev
        for comp in P.COMPOSITIONS:
            a, d = P.synth_states(hbar, comp, K_)
            assert len(a) + len(d) == K_
            assert (sum(a) - sum(d)) / K_ == pytest.approx(hbar, abs=1e-12), (comp, lev)
            assert max(map(abs, a + d)) <= P.S_MAX
            msgs = P.to_msgs(a, np.random.default_rng(0)), P.to_msgs(d, np.random.default_rng(0))
            fA = sum(2 * m.prob - 1 for m in msgs[0]) / K_
            fD = sum(2 * m.prob - 1 for m in msgs[1]) / K_
            assert fA - fD == pytest.approx(hbar, abs=0.0101)
        a, d = P.synth_states(hbar, "route_A", K_)
        assert len(a) == len(d) == K_ // 2
        pa, _ = P.synth_states(hbar, "polarized", K_)
        assert sum(abs(x) == pytest.approx(P.S_POLAR) for x in pa) >= K_ - 1
    with pytest.raises(ValueError):
        P.synth_states(1.5, "uniform", 8)          # a peer would have to state |s| > S_MAX
    assert P.synth_states(0.5, "few_strong", 8)[0].count(0.0) == 8 - 5   # n = ceil(4 / 0.96) = 5 peers


def test_sufficiency_design_shape_and_text():
    pr = P.sufficiency_design()
    assert len(pr) == 5 * 6 * 2 * 2 * 25 == 3000
    assert len({p.ctx.agent_id for p in pr}) == 3000
    assert len({make_request(p.ctx, 1).dynamic for p in pr}) == 3000   # distinct params: no replay collisions
    by_id = {p.ctx.agent_id: p for p in pr}
    u = by_id["suff|ev0.4|hist1|h2|uniform|3"].ctx
    sh = by_id["suff|ev0.4|hist1|h2|shuffled|3"].ctx
    assert sorted(m.text for m in u.agree) == sorted(m.text for m in sh.agree)   # same messages, new order
    assert [m.text for m in u.agree] != [m.text for m in sh.agree]
    assert len({p.ctx.evidence for p in pr}) == 2                      # two evidence levels, cached prefixes


# ---------------------------------------------------------------- (i) eta^2 on the mock


def test_eta2_zero_for_scalar_mock_and_detects_channel_asymmetry(guard, capsys):
    rec = run(P.run_probe_set(P.sufficiency_design(n=2), MockBackend(0.5, B, B, kappa=0.3), guard))
    assert "[budget] projection: 240 calls" in capsys.readouterr().out
    e = P.eta2_comp(rec)
    assert e["eta2"] < 1e-6 and e["max_contrast_frac"] < 1e-3         # display rounding is the only effect
    rec = run(P.run_probe_set(P.sufficiency_design(n=2), MockBackend(0.5, B, 0.3 * B, kappa=0.3), guard))
    e = P.eta2_comp(rec)
    assert e["eta2"] > 0.05 and e["max_contrast_frac"] > 0.2
    assert np.all(np.sign(e["route_A_minus_B"]) == np.sign(e["levels"]))   # b_- < b_+: route B is weaker


def test_eta2_stochastic_mock_noise_correction(guard):
    """+-1 mock with a steep response: the raw eta^2 carries sampling noise, the corrected one is ~0 for
    b_+ = b_- and clearly positive for b_- = 0."""
    sym = P.eta2_comp(run(P.run_probe_set(P.sufficiency_design(), MockBackend(3.0, B, B, stochastic=True, seed=1), guard)))
    asym = P.eta2_comp(run(P.run_probe_set(P.sufficiency_design(), MockBackend(3.0, B, 0.0, stochastic=True, seed=1), guard)))
    assert sym["eta2_corrected"] < 0.03 and sym["max_contrast_frac"] < 0.2
    assert asym["eta2_corrected"] > 0.1


# ---------------------------------------------------------------- (ii), (iii) slopes


def test_conformity_and_gains_recover_mock_slopes(guard):
    beta = 0.5
    conf = P.fit_conformity(run(P.run_probe_set(P.conformity_design(n=2), MockBackend(beta, B, B, kappa=0.3), guard)))
    assert conf["beta_resp"] == pytest.approx(beta * B * K, rel=1e-6)          # beta_resp = beta b K
    gains = P.fit_channel_gains(run(P.run_probe_set(P.gains_design(n=2), MockBackend(beta, B, 0.3 * B), guard)))
    assert gains["a_plus"] == pytest.approx(beta * B * K, rel=1e-6)
    assert gains["a_minus"] == pytest.approx(beta * 0.3 * B * K, rel=1e-6)
    # stochastic: within 4 standard errors
    st = P.fit_conformity(run(P.run_probe_set(P.conformity_design(n=150), MockBackend(beta, B, B, stochastic=True, seed=2), guard)))
    assert abs(st["beta_resp"] - beta * B * K) < 4 * st["beta_resp_se"]
    sg = P.fit_channel_gains(run(P.run_probe_set(P.gains_design(n=150), MockBackend(beta, B, -0.5 * B, stochastic=True, seed=2), guard)))
    assert abs(sg["a_minus"] - beta * -0.5 * B * K) < 4 * sg["a_minus_se"]    # a_- < 0: pulled toward distrusted


# ---------------------------------------------------------------- (iv) white-probe kernel


def test_white_probe_matches_tangent_kernel(guard, capsys):
    """Mock with own-history memory (H = 1, kappa = 0.4). The exact response along each delivered
    trajectory is the tangent recursion
        d s(t+1)/d hbar(t') = (1 - s(t+1)^2) [beta b K delta_{t t'} + kappa d s(t)/d hbar(t')],
    and by Stein's lemma the white-probe estimate is unbiased for its average (theta is Gaussian), so the
    two agree within the estimator's s.e. Lag 2 is nonzero only through the agent's own memory."""
    beta, kappa, M, T = 0.5, 0.4, 3000, 6
    out = run(P.run_white_probe(MockBackend(beta, B, B, kappa=kappa), guard, M=M, T=T, eps=0.3, bg_sd=0.3, H=1))
    assert f"[budget] projection: {M * (T + 1)} calls" in capsys.readouterr().out
    est = P.white_probe_kernel(out)
    s, hb = out["s"], out["hbar"]
    tan = np.zeros((M, T + 1, T))                     # tan[m, t, t'] = d s(t) / d hbar(t')
    for t in range(T):
        g = 1 - s[:, t + 1] ** 2
        tan[:, t + 1] = g[:, None] * kappa * tan[:, t]
        tan[:, t + 1, t] += g * beta * B * K
    R_true = tan.mean(0)
    for t in range(1, T + 1):
        for tp in range(t):
            assert abs(est["R"][t, tp] - R_true[t, tp]) < 4.5 * est["se"][t, tp] + 0.02, (t, tp)
    lag1 = np.mean([R_true[t, t - 1] for t in range(1, T + 1)])
    lag2 = np.mean([R_true[t, t - 2] for t in range(2, T + 1)])
    assert est["R_lag"][0] == pytest.approx(lag1, abs=4 * est["R_lag_se"][0] + 0.01)
    assert est["R_lag"][1] == pytest.approx(lag2, abs=4 * est["R_lag_se"][1] + 0.01)
    assert lag2 > 0.1                                 # memory path is resolved


# ---------------------------------------------------------------- (v) surrogate


def test_surrogate_recovers_mock_and_reports_heldout(guard):
    beta, kappa = 0.5, 0.3
    mk = lambda st: MockBackend(beta, B, 0.3 * B, kappa=kappa, stochastic=st, seed=3)
    designs = lambda n: P.sufficiency_design(n=n) + P.conformity_design(n=n) + P.gains_design(n=n)
    fit = P.fit_surrogate(run(P.run_probe_set(designs(2), mk(False), guard)))
    true = {"a_A": beta * B * K, "a_D": beta * 0.3 * B * K, "a_p": 1.0, "a_u": kappa}
    for k, v in true.items():
        assert fit["params"][k] == pytest.approx(v, rel=1e-5, abs=1e-6)
    assert fit["rmse_heldout_cellmeans"] < 1e-6 and fit["n_test_cells"] > 10
    st = P.fit_surrogate(run(P.run_probe_set(designs(10), mk(True), guard)))
    for k, v in true.items():
        assert abs(st["params"][k] - v) < 4 * st["params_se"][k] + 0.02, k
    assert st["rmse_heldout_cellmeans"] < 3 * st["cellmean_sampling_floor"]
    single = P.fit_surrogate(run(P.run_probe_set(designs(2), mk(False), guard)), two_channel=False)
    assert single["rmse_heldout_cellmeans"] > 10 * fit["rmse_heldout_cellmeans"] + 1e-3   # misfit when b_+ != b_-


# ---------------------------------------------------------------- through engine.Runner (fake client)


def test_probes_through_runner_with_fake_client(tmp_path, guard, capsys):
    """A probe set on the LLM backend: one Runner call per probe, keyed by probe id, prompts from
    envs/prompts.py, beliefs parsed back, and a rerun replays from checkpoints."""
    def responder(params, n):
        user = params["messages"][0]["content"]
        ps = [float(x) for x in re.findall(r"Stated probability of HEADS ([0-9.]+)", user)]
        return message(belief_json(round(float(np.mean(ps)), 4) if ps else 0.5))
    client = FakeClient(responder)
    runner, ledger = make_runner(tmp_path, client, exp="probe-fake")
    probes = P.gains_design(n=1, xs=(-0.4, 0.4))
    rec = run(P.run_probe_set(probes, LLMBackend(runner), guard))
    assert "[budget] projection: 4 calls" in capsys.readouterr().out
    assert len(client.calls) == 4 and all(r["status"] == "ok" for r in rec)
    assert {r["id"]: r["p"] for r in rec}["gain|ev0.4|hist1|agree|x0.4|0"] == pytest.approx(0.7)
    assert all(c["system"][0]["text"] == P.make_request(probes[0].ctx, 1).system_rules for c in client.calls)
    runner2, _ = make_runner(tmp_path, FakeClient(lambda p, n: pytest.fail("should replay")), exp="probe-fake")
    rec2 = run(P.run_probe_set(probes, LLMBackend(runner2), guard))
    assert [r["p"] for r in rec2] == [r["p"] for r in rec]
