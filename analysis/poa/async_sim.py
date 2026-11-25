"""Task 2, model side: what the fitted two-channel agent predicts for the period index under synchronous and
asynchronous updates, on the exact GPT-4o-mini async episodes (statement, graph, round-0 state).

Agent (field_sufficiency.py, M2 or M2avg, fitted on the synchronous GPT-4o-mini data): K = 5 votes, each +1 with
probability sigmoid(u_i), u_i = alpha_i + sum_j W_ij msg_j, W = b+ [agree tie] - b- [disagree tie] (common.
coupling_matrix); M2avg divides the sum by the number of messages present. The spin is the majority of the 5 votes.

  sync : msg_j = s_j(t) for every neighbour; s_i(t+1) from u_i(t)            (lib/datagen/dynamics.py)
  async: PoA's schedule (lib/datagen_async): each step is 1000 cells, every agent fires in a cell w.p. 0.0005,
         same-cell collisions in random order. At a firing agent i first posts msg_i = its current spin (seen by
         its neighbours until it fires again), then redraws its 5 votes from the messages currently posted on its
         incoming edges. Boards start empty at step 1, so a neighbour that has not yet fired contributes nothing.
         The recorded raw row of a silent agent is its carried spin repeated 5 times, as in the release.

Observables over rounds (steps) 3..8, as in flicker.py: spin flip rates a_k, period index P = a2/a1, and the
belief E2/E1 computed from the recorded raw rows (so the async value carries the same silent-row contamination
as the data). Comparing the simulated and observed values tests (i) whether the fitted agent reproduces the
synchronous flicker and (ii) whether the async data match what the same agent does under PoA's schedule.

Run: uv run python analysis/poa/async_sim.py [--reps 20]
Outputs: results/poa/async_sim.csv, async_sim_log.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.poa.common import ASYNC_PATH, K_VOTES, RESULTS, cells, coupling_matrix, fit_m2, sigmoid  # noqa: E402
from analysis.poa.flicker import F1, F2, belief_displacements, ratio_se, spin_flips  # noqa: E402

N_CELLS, P_CELL, T_STEPS = 1000, 0.5 / 1000, 8


def majority(npos):
    return np.where(npos > K_VOTES // 2, 1, -1)


def run_sync(alpha, W, s0, avg, rng):
    """(T+1, N) spins and (T+1, N) beliefs (vote means) of one synchronous run."""
    deg = (W != 0).sum(1)
    S, B = [s0.copy()], [s0.astype(float)]
    s = s0.astype(float)
    for _ in range(T_STEPS):
        field = W @ s
        if avg:
            field = np.where(deg > 0, field / np.maximum(deg, 1), 0.0)
        npos = rng.binomial(K_VOTES, sigmoid(alpha + field))
        s = majority(npos).astype(float)
        S.append(s.copy())
        B.append(2 * npos / K_VOTES - 1)
    return np.array(S), np.array(B)


def run_async(alpha, W, s0, avg, rng):
    """(T+1, N) spins and recorded beliefs of one run on PoA's firing schedule (see module docstring)."""
    N = len(alpha)
    nbrs = [np.nonzero(W[i])[0] for i in range(N)]
    wrow = [W[i, nbrs[i]] for i in range(N)]
    cur = s0.astype(float).copy()
    msg = np.zeros(N)                     # posted message; 0 = nothing posted yet on that edge
    posted = np.zeros(N, bool)
    S, B = [cur.copy()], [cur.copy()]
    for _ in range(T_STEPS):
        fires = rng.random((N_CELLS, N)) < P_CELL
        raw_belief = cur.copy()           # silent agents: carried spin x 5 -> belief +-1
        for c in np.nonzero(fires.any(1))[0]:
            agents = np.nonzero(fires[c])[0]
            for i in rng.permutation(agents):
                msg[i], posted[i] = cur[i], True
                nb = nbrs[i]
                field = wrow[i] @ msg[nb]
                if avg:
                    k = posted[nb].sum()
                    field = field / k if k else 0.0
                npos = rng.binomial(K_VOTES, sigmoid(alpha[i] + field))
                cur[i] = 1.0 if npos > K_VOTES // 2 else -1.0
                raw_belief[i] = 2 * npos / K_VOTES - 1
        S.append(cur.copy())
        B.append(raw_belief)
    return np.array(S), np.array(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    C = cells()
    A = cells(ASYNC_PATH)
    rows, lines = [], []
    for mode in ("objective", "subjective"):
        c, ca = C[("gpt-4o-mini", mode)], A[("gpt-4o-mini", mode)]
        sidx = {name: k for k, name in enumerate(c["stmt_names"])}
        N = c["S"].shape[2]
        for form, avg in (("M2", False), ("M2avg", True)):
            bp, bm, alpha = fit_m2(c, averaged=avg)
            lines.append(f"{mode} {form}: b+ = {bp:.3f}, b- = {bm:.3f}")
            for r in range(len(ca["stmt"])):
                stmt = ca["stmt_names"][ca["stmt"][r]]
                a_i = alpha[sidx[stmt] * N + np.arange(N)]
                W = coupling_matrix(ca["J"][r], bp, bm)
                s0 = ca["S"][r, 0].astype(float)
                s0 = np.where(s0 == 0, 1.0, s0)
                for sched, run in (("sync", run_sync), ("async", run_async)):
                    for rep in range(args.reps):
                        S, Bel = run(a_i, W, s0, avg, rng)
                        (A1, E1, A2, E2), _ = belief_displacements(Bel)
                        a1, a2 = spin_flips(S)
                        rows.append(dict(mode=mode, form=form, schedule=sched, statement=stmt, episode=r, rep=rep,
                                         E1=E1, E2=E2, spin_a1=a1, spin_a2=a2))
            print(f"done {mode} {form}", flush=True)
    df = pd.DataFrame(rows)
    out = []
    for (mode, form, sched), g in df.groupby(["mode", "form", "schedule"]):
        cl = g.statement.values
        P, Pse = ratio_se(g.spin_a2, g.spin_a1, cl)
        e, ese = ratio_se(g.E2, g.E1, cl)
        null = F2 / F1 if sched == "async" else 1.0
        out.append(dict(mode=mode, form=form, schedule=sched, n_runs=len(g), spin_a1=g.spin_a1.mean(),
                        spin_a2=g.spin_a2.mean(), P=P, P_se=Pse, P_corrected=P / null, E2_over_E1=e,
                        E2_over_E1_se=ese))
    out = pd.DataFrame(out)
    obs = pd.read_csv(RESULTS / "flicker_async.csv")
    obs = obs[obs.group == "all random"].set_index(["mode", "schedule"])
    out["obs_spin_a1"] = [obs.loc[(m, s), "spin_a1"] for m, s in zip(out["mode"], out.schedule)]
    out["obs_P_corrected"] = [obs.loc[(m, s), "P_corrected"] for m, s in zip(out["mode"], out.schedule)]
    out["obs_P_corrected_se"] = [obs.loc[(m, s), "P_corrected_se"] for m, s in zip(out["mode"], out.schedule)]
    out["obs_E2_over_E1"] = [obs.loc[(m, s), "E2_over_E1"] for m, s in zip(out["mode"], out.schedule)]
    out.to_csv(RESULTS / "async_sim.csv", index=False)
    lines.append(f"\nSimulated fitted agent ({args.reps} runs per episode, 480 async episodes) against observed; "
                 "P_corrected = (a2/a1)/null, null 1 (sync) or f2/f1 = 1.61 (async)")
    for _, r in out.iterrows():
        lines.append(f"  {r['mode']:10s} {r.form:5s} {r.schedule:5s} | sim a1 {r.spin_a1:.3f} (obs {r.obs_spin_a1:.3f}) "
                     f"P_corr {r.P_corrected:.2f} (obs {r.obs_P_corrected:.2f} +- {r.obs_P_corrected_se:.2f}) | "
                     f"belief E2/E1 sim {r.E2_over_E1:.2f} (obs {r.obs_E2_over_E1:.2f})")
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "async_sim_log.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
