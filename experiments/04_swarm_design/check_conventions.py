"""Check that design_lib's coupling matrix W (rows receive; agree ties a_+/K, disagree ties -a_-/K) is the field that
envs.swarm.Swarm + MockBackend delivers on the same graph, in both update modes' synchronous limit.

Runs the real Swarm class with a noise-free mock agent s(t+1) = tanh(field + hpriv) and compares every belief with
the matrix recursion s(t+1) = tanh(W s(t) + hpriv).

Run: uv run python experiments/04_swarm_design/check_conventions.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.evidence_game import make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402
from envs.swarm import MockBackend, SwarmConfig, run_swarm  # noqa: E402

N, K, T = 16, 4, 5
a_plus, a_minus = 0.99, 0.57
rng = np.random.default_rng(0)
for gamma in (-0.5, 0.5):
    J = fixed_indegree_signed_couplings(N, K, gamma, 0.5, rng)
    game = make_game(N, seed=3, truth=1)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SwarmConfig(experiment_id="conv", T=T, H=1, replicas=1, results_root=Path(tmp))
        out = run_swarm(J, game, cfg, MockBackend(beta=1.0, b_plus=a_plus / K, b_minus=a_minus / K))
    W = (a_plus * (J > 0) - a_minus * (J < 0)) / K
    s = np.tanh(game.h_priv); ref = [s]
    for t in range(T):
        s = np.tanh(W @ s + game.h_priv); ref.append(s)
    print(f"gamma={gamma:+.1f}: max |Swarm/MockBackend - matrix recursion| = {np.abs(out['s'][0] - np.array(ref)).max():.2e}")
