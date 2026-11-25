"""Kick twins (envs/kick.py): exact zero echo without reciprocity, echo sign set by the reciprocated ties, and
only agents downstream of the kick reach the API (everything else replays from the cache, free)."""

import asyncio

import numpy as np
import pytest

from engine.ledger import read_jsonl
from envs.evidence_game import make_game
from envs.graphs import fixed_indegree_signed_couplings
from envs.kick import base_rows, compatible_kicks, kick_twin, mirror
from envs.swarm import LLMBackend, MockBackend, Swarm, SwarmConfig, run_swarm
from tests.test_engine import FakeClient, make_runner
from tests.test_envs import _mean_agree_responder


def _base(tmp_path, gamma, p_neg, backend, N=16, K=2, T=4, exp="base"):
    J = fixed_indegree_signed_couplings(N, K, gamma, p_neg, np.random.default_rng(3))
    game = make_game(N, seed=4)
    cfg = SwarmConfig(exp, T=T, H=1, replicas=1, results_root=tmp_path / "results")
    run_swarm(J, game, cfg, backend)
    return J, game, cfg


def _twin(J, game, cfg, backend, i, t0=1, n_rounds=2):
    sw = Swarm(J, game, cfg, backend)
    rows = base_rows(cfg.results_root / cfg.experiment_id)
    return asyncio.run(kick_twin(sw, rows, 0, t0, [i], n_rounds))


def test_mirror():
    assert mirror(0.72, "About 72% HEADS; heads looks likely, not Tails.") == \
        (0.28, "About 72% TAILS; tails looks likely, not Heads.")


@pytest.mark.parametrize("gamma,p_neg", [(0.0, 0.0), (0.0, 0.5)])
def test_echo_is_exactly_zero_without_reciprocity(tmp_path, gamma, p_neg):
    mock = MockBackend(beta=0.8, b_plus=1.0, b_minus=0.6)
    J, game, cfg = _base(tmp_path, gamma, p_neg, mock)
    for i in range(len(J)):
        out = _twin(J, game, cfg, mock, i)
        # round t0+1: the kicked agent itself is unchanged; its readers moved
        assert out["twin_p"][0, i] == out["base_p"][0, i]
        readers = np.flatnonzero(J[:, i])
        assert np.all(out["twin_p"][0, readers] != out["base_p"][0, readers])
        # round t0+2: no reciprocated tie, so nothing returns to i
        assert out["twin_p"][1, i] == out["base_p"][1, i]


@pytest.mark.parametrize("gamma,sign", [(1.0, +1), (-1.0, -1)])
def test_echo_sign_follows_reciprocated_ties(tmp_path, gamma, sign):
    """b_+ = b_- > 0: the coupling is b J, so the echo through a reciprocated pair has the sign of J_ij J_ji
    times the kick: concordant pairs return the kick, discordant pairs oppose it."""
    mock = MockBackend(beta=0.8, b_plus=1.0, b_minus=1.0)
    J, game, cfg = _base(tmp_path, gamma, 0.5, mock)
    n_checked = 0
    for i in range(len(J)):
        out = _twin(J, game, cfg, mock, i)
        p0, p1 = out["kick"][i]
        echo = out["twin_p"][1, i] - out["base_p"][1, i]
        assert np.sign(echo) == sign * np.sign(p1 - p0)
        n_checked += 1
    assert n_checked == len(J)


def test_only_downstream_agents_are_paid(tmp_path):
    """LLM backend through engine.Runner: the twin sends exactly the agents whose prompts changed; every other
    call replays from the shared cache with no ledger line."""
    N, K = 12, 2
    J = fixed_indegree_signed_couplings(N, K, 0.0, 0.0, np.random.default_rng(5))   # all agree ties, no reciprocity
    game = make_game(N, seed=6)
    runner, ledger = make_runner(tmp_path, FakeClient(_mean_agree_responder()), exp="base")
    cfg = SwarmConfig("base", T=4, H=1, replicas=1, results_root=tmp_path / "results")
    run_swarm(J, game, cfg, LLMBackend(runner))
    n_base = len(list(read_jsonl(ledger)))

    client = FakeClient(_mean_agree_responder())
    runner2, _ = make_runner(tmp_path, client, exp="base-kick")
    i, t0 = 3, 1
    out = _twin(J, game, cfg, LLMBackend(runner2), i, t0=t0)
    readers = set(np.flatnonzero(J[:, i]))
    assert set(np.flatnonzero(out["sent"][0])) == readers           # round t0+1: only i's readers are resampled
    assert not out["sent"][0, i]
    # round t0+2: the readers again (their own history changed) and their readers; never anyone unaffected
    affected2 = readers | {a for j in readers for a in np.flatnonzero(J[:, j])}
    assert set(np.flatnonzero(out["sent"][1])) == affected2
    assert len(client.calls) == int(out["sent"].sum())
    assert len(list(read_jsonl(ledger))) == n_base + len(client.calls)  # replays add no ledger lines


def test_compatible_kicks_do_not_overlap():
    J = fixed_indegree_signed_couplings(64, 2, 1.0, 0.5, np.random.default_rng(7))
    ks = compatible_kicks(J, range(64), n_max=10)
    R = J != 0
    for a in ks:
        for b in ks:
            if a != b:
                assert not (R[:, a] & R[:, b]).any()          # no common reader
                assert not (R[a] & R[:, b]).any()             # a reads nobody who reads b
                assert not R[a, b] and not R[b, a]
    assert len(ks) >= 3
