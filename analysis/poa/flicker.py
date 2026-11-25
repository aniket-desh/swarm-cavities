"""Task 2: period-2 flicker under asynchronous updates, and the degree confound between the two lattices.

Part A, asynchronous control (claim2.md sec. 3 "Asynchronous control", sec. 4). The async release
(HF physics-of-agents/agent-opinions-async) is GPT-4o-mini only, 480 episodes = 60 statements x the 8 training
random signed graphs x 1 replica, and contains no lattices (PoA excluded them from the async schedule). Every
(statement, J) pair also appears in the synchronous GPT-4o-mini data, with 4 replicas.

Async mechanics (third_party/physics-of-agents/lib/datagen_async/dynamics_seq.py): in each of the 8 steps an
agent fires Poisson(0.5) times (1000 cells of Bernoulli(0.0005)); at a firing it posts a message from its current
spin, then resamples K = 5 votes from the latest message on each incoming edge. A silent agent's raw row is its
carried spin repeated K times, so its "belief" (vote mean) jumps to +-1 even if its last fired belief was 0.2,
which corrupts the belief-based displacement of check_poa_zero_cost.py. We therefore report:

  * the belief-based E2/E1 exactly as in check_poa_zero_cost.py (requested; contaminated, see above),
  * the spin-based flip rates A_k = P(s_i(t) != s_i(t-k)), t = 3..8, for async and for the matching sync
    episodes (same statements and graphs), with the period ratio A2/A1. Null for sync (no temporal structure):
    A2/A1 = 1; flicker: < 1. Under async a change needs at least one firing in the lag, probability
    f_k = 1 - (1-0.0005)^(1000 k) (f1 = 0.394, f2 = 0.632), so the memoryless null is A2/A1 = f2/f1 = 1.61
    and we report the corrected ratio (A2/A1)/(f2/f1): 1 = no temporal structure. Even an agent that flips at
    every one of its own firings gives P(odd number of firings) = (1-e^{-k})/2, i.e. a corrected ratio of 0.85,
    so under async the corrected ratio cannot fall below 0.85 by flicker; values well below that would mean the
    firing model is wrong.

Part B, degree against bipartiteness (synchronous data, all models). Mean degrees: square lattice 3.25,
random signed 3.5, triangular lattice 4.56 (4 x 8 open boundary). Per agent we compute the floor-corrected
belief displacements E1_i, E2_i (check_poa_zero_cost.py definition, rounds 3..8, averaged over the agent's
communities) and tabulate them against the agent's degree and the number of triangles through it, in the full
graph and in the agree (J = +1) subgraph, then fit E1_i - E2_i on degree and triangles within random signed
graphs (s.e. clustered by statement). The agree subgraph matters because disagree ties turn out to carry almost no
coupling (field_sufficiency.py, b- ~ 0), so the effective interaction graph is the agree subgraph.

Run: uv run python analysis/poa/flicker.py
Outputs (results/poa/): flicker_async.csv, flicker_async_by_round.csv, flicker_degree_table.csv,
flicker_degree_regression.csv, flicker_log.txt (printed tables).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/swarm-cavity")
from analysis.poa.common import (ASYNC_PATH, CLASS_SHORT, K_VOTES, RESULTS, SHORT, graph_class,  # noqa: E402
                                 triangles_per_node)
from analysis.poa_data import load_poa  # noqa: E402

P_NOFIRE = (1 - 0.5 / 1000) ** 1000               # per step
F1, F2 = 1 - P_NOFIRE, 1 - P_NOFIRE**2
T0 = 3                                            # rounds 3..8, as in check_poa_zero_cost.py


def belief_displacements(b):
    """check_poa_zero_cost.py, test 1, for one community b (T+1, N): (A1, E1, A2, E2) over rounds 3..8, with
    floor v = (1 - b^2)/(K-1) the unbiased variance of a K-vote mean. Also returns per-agent arrays."""
    v = (1 - b**2) / (K_VOTES - 1)
    out, per_agent = [], []
    for k in (1, 2):
        d2 = (b[T0:] - b[T0 - k:-k]) ** 2
        fl = v[T0:] + v[T0 - k:-k]
        out += [np.nanmean(d2), np.nanmean(d2) - np.nanmean(fl)]
        per_agent.append(np.nanmean(d2 - fl, axis=0))
    return out, per_agent


def spin_flips(s):
    """Spin flip rates A_k = P(s(t) != s(t-k)) over rounds 3..8 for one community s (T+1, N)."""
    return [np.mean(s[T0:] != s[T0 - k:-k]) for k in (1, 2)]


def cluster_mean_se(x, cl):
    """Mean of x with s.e. from cluster means (equal-size clusters here)."""
    m = pd.Series(x).groupby(cl).mean()
    return float(np.mean(x)), float(m.std(ddof=1) / np.sqrt(len(m)))


def ratio_se(a, b, cl):
    """Ratio of means sum(a)/sum(b) with cluster linearization s.e."""
    A = pd.Series(a).groupby(cl).sum()
    B = pd.Series(b).groupby(cl).sum()
    r = A.sum() / B.sum()
    n = len(A)
    se = np.sqrt(n / (n - 1) * np.sum((A - r * B) ** 2)) / B.sum()
    return float(r), float(se)


def part_a(lines):
    sync = load_poa()
    asyn = load_poa(ASYNC_PATH)
    rows, by_round = [], []
    for label, data, sched in (("sync", sync, "sync"), ("async", asyn, "async")):
        for (model, mode, statement), d in data.items():
            if model != "gpt-4o-mini" or (mode, statement) not in {(k[1], k[2]) for k in asyn}:
                continue
            for r in range(len(d["graph"])):
                J = d["J"][r]
                if (J >= 0).all():
                    continue                          # lattices: sync only, no async counterpart
                (A1, E1, A2, E2), _ = belief_displacements(d["beliefs"][r])
                a1, a2 = spin_flips(d["spins"][r])
                rows.append(dict(schedule=sched, mode=mode, statement=statement, J=J.tobytes().hex()[:16],
                                 A1=A1, E1=E1, A2=A2, E2=E2, spinA1=a1, spinA2=a2))
                s = d["spins"][r]
                for t in range(1, s.shape[0]):
                    by_round.append(dict(schedule=sched, mode=mode, statement=statement, t=t,
                                         flip1=np.mean(s[t] != s[t - 1]),
                                         mixed=np.mean(np.abs(d["votes"][r][t].sum(-1)) < K_VOTES)))
    df = pd.DataFrame(rows)
    res = []
    for (sched, mode), g in df.groupby(["schedule", "mode"]):
        cl = g["statement"].values
        e_ratio, e_se = ratio_se(g["E2"].values, g["E1"].values, cl)
        s_ratio, s_se = ratio_se(g["spinA2"].values, g["spinA1"].values, cl)
        null = F2 / F1 if sched == "async" else 1.0
        res.append(dict(schedule=sched, mode=mode, n_episodes=len(g), n_statements=len(set(cl)),
                        E1=g.E1.mean(), E2=g.E2.mean(), E2_over_E1=e_ratio, E2_over_E1_se=e_se,
                        spinA1=g.spinA1.mean(), spinA2=g.spinA2.mean(),
                        spinA2_over_A1=s_ratio, spinA2_over_A1_se=s_se, null_ratio=null,
                        corrected_ratio=s_ratio / null, corrected_ratio_se=s_se / null,
                        flips_per_firing_lag1=g.spinA1.mean() / (F1 if sched == "async" else 1.0)))
    out = pd.DataFrame(res)
    out.to_csv(RESULTS / "flicker_async.csv", index=False)
    br = pd.DataFrame(by_round).groupby(["schedule", "mode", "t"])[["flip1", "mixed"]].mean().reset_index()
    br.to_csv(RESULTS / "flicker_async_by_round.csv", index=False)

    lines.append("Part A: GPT-4o-mini on the 8 training random signed graphs, 60 statements; rounds/steps 3..8")
    lines.append(f"  async firing: P(fire >= 1 in lag 1) = {F1:.4f}, lag 2 = {F2:.4f}, memoryless null A2/A1 = {F2/F1:.3f}")
    for _, r in out.iterrows():
        lines.append(f"  {r.schedule:5s} {r['mode']:10s} n={r.n_episodes:4d} | belief E1 {r.E1:.4f} E2 {r.E2:.4f} "
                     f"E2/E1 {r.E2_over_E1:.2f} ({r.E2_over_E1_se:.2f}) | spin A1 {r.spinA1:.4f} A2 {r.spinA2:.4f} "
                     f"A2/A1 {r.spinA2_over_A1:.2f} ({r.spinA2_over_A1_se:.2f}) null {r.null_ratio:.2f} "
                     f"-> corrected {r.corrected_ratio:.2f} ({r.corrected_ratio_se:.2f}); "
                     f"flips per firing (lag 1) {r.flips_per_firing_lag1:.3f}")
    lines.append("  spin flip rate s(t) != s(t-1) and fraction of mixed vote rows, by round t (sync | async):")
    for mode in ("objective", "subjective"):
        a = br[(br["mode"] == mode) & (br.schedule == "sync")].set_index("t")
        b = br[(br["mode"] == mode) & (br.schedule == "async")].set_index("t")
        lines.append(f"    {mode:10s} flip  sync " + " ".join(f"{x:.3f}" for x in a.flip1) +
                     " | async " + " ".join(f"{x:.3f}" for x in b.flip1))
        lines.append(f"    {mode:10s} mixed sync " + " ".join(f"{x:.3f}" for x in a.mixed) +
                     " | async " + " ".join(f"{x:.3f}" for x in b.mixed))


def part_b(lines):
    data = load_poa()
    gains = pd.read_csv(RESULTS / "field_sufficiency_gains.csv")
    rows = []
    cache = {}
    for (model, mode, statement), d in data.items():
        for r in range(len(d["graph"])):
            J = d["J"][r]
            kb = J.tobytes()
            if kb not in cache:
                Jp = np.where(J > 0, 1, 0)
                cache[kb] = dict(cls=CLASS_SHORT[graph_class(J)], deg=(J != 0).sum(1), deg_a=Jp.sum(1),
                                 tri=triangles_per_node(J), tri_a=triangles_per_node(Jp))
            gi = cache[kb]
            (_, E1, _, E2), (e1, e2) = belief_displacements(d["beliefs"][r])
            for i in range(J.shape[0]):
                rows.append((SHORT[model], mode, statement, gi["cls"], i, kb.hex()[:16], gi["deg"][i], gi["deg_a"][i],
                             gi["tri"][i], gi["tri_a"][i], e1[i], e2[i]))
    df = pd.DataFrame(rows, columns=["model", "mode", "statement", "cls", "agent", "J", "deg", "deg_a", "tri",
                                     "tri_a", "E1", "E2"])
    # average each agent over its 4 replicas (same J, statement, persona)
    df = df.groupby(["model", "mode", "statement", "cls", "J", "agent", "deg", "deg_a", "tri", "tri_a"],
                    as_index=False)[["E1", "E2"]].mean()
    df["flick"] = df.E1 - df.E2

    # table: E2/E1 by class x degree x (triangles > 0), pooled over models and tasks, and per model
    df["tri_any"] = np.where(df.tri > 0, "tri>0", "tri=0")
    tab = []
    for keys, g in df.groupby(["model", "mode", "cls", "deg", "tri_any"]):
        if len(g) < 20:
            continue
        r_, se_ = ratio_se(g.E2.values, g.E1.values, g.statement.values) if g.statement.nunique() > 2 else (np.nan, np.nan)
        tab.append(dict(zip(["model", "mode", "cls", "deg", "tri_any"], keys), n_agents=len(g),
                        E1=g.E1.mean(), E2=g.E2.mean(), E2_over_E1=r_, se=se_))
    pd.DataFrame(tab).to_csv(RESULTS / "flicker_degree_table.csv", index=False)

    # regression within random signed graphs: flick ~ 1 + deg_a + deg_d + tri_a (+ tri_mixed), per model x task
    reg = []
    for (model, mode), g in df[df.cls == "random signed"].groupby(["model", "mode"]):
        X = np.column_stack([np.ones(len(g)), g.deg_a, g.deg - g.deg_a, g.tri_a, g.tri - g.tri_a])
        names = ["const", "agree degree", "disagree degree", "agree triangles", "other triangles"]
        y = g.flick.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        # cluster-robust (statement) sandwich
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = np.zeros((X.shape[1],) * 2)
        for _, idx in g.reset_index(drop=True).groupby("statement").indices.items():
            sc = X[idx].T @ res[idx]
            meat += np.outer(sc, sc)
        ncl = g.statement.nunique()
        V = XtX_inv @ meat @ XtX_inv * ncl / (ncl - 1)
        for n, b_, s_ in zip(names, beta, np.sqrt(np.diag(V))):
            reg.append(dict(model=model, mode=mode, term=n, coef=b_, se=s_, n_agents=len(g), mean_flick=y.mean()))
    reg = pd.DataFrame(reg)
    reg.to_csv(RESULTS / "flicker_degree_regression.csv", index=False)

    lines.append("\nPart B: degree vs bipartiteness (sync, all models). Agent-level E2/E1 pooled over models x tasks:")
    pooled = []
    for keys, g in df.groupby(["cls", "deg_a", "tri_any"]):
        if len(g) < 200:
            continue
        r_, se_ = ratio_se(g.E2.values, g.E1.values, (g.model + g["mode"] + g.statement).values)
        pooled.append((keys, len(g), g.E1.mean(), r_, se_))
    lines.append(f"  {'class':20s} {'agree deg':>9s} {'triangles':>9s} {'n':>6s}   E1     E2/E1 (s.e.)")
    for (cls, dg, ta), n, e1, r_, se_ in pooled:
        lines.append(f"  {cls:20s} {dg:9d} {ta:>9s} {n:6d}  {e1:.3f}   {r_:.2f} ({se_:.2f})")
    lines.append("  Within random signed graphs: OLS of E1_i - E2_i on agree/disagree degree and triangles "
                 "(s.e. clustered by statement)")
    for (model, mode), g in reg.groupby(["model", "mode"]):
        lines.append(f"    {model:14s} {mode:10s} mean {g.mean_flick.iloc[0]:.3f} | " + "  ".join(
            f"{t}: {c:+.3f} ({s:.3f})" for t, c, s in zip(g.term, g.coef, g.se) if t != "const"))
    # gains are used only to explain why the agree subgraph is the relevant one
    lines.append("  (b-/b+ from field_sufficiency_gains.csv: " + ", ".join(
        f"{r.model} {r['mode'][:4]} {r.ratio:+.2f}" for _, r in gains.iterrows()) + ")")


if __name__ == "__main__":
    lines = []
    part_a(lines)
    part_b(lines)
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "flicker_log.txt").write_text(txt + "\n")
