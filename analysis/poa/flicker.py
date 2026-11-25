"""Task 2: period-2 flicker under asynchronous updates, and degree against bipartiteness.

Observables (check_poa_zero_cost.py, test 1; claim2.md sec. 5). For beliefs b_i(t) (mean of the K = 5 votes), over
rounds t = 3..8,
    A_k = <(b_i(t) - b_i(t-k))^2>,  floor_k = <v_i(t) + v_i(t-k)>,  v = (1 - b^2)/(K-1),  E_k = A_k - floor_k,
and E2/E1 < 1 is the period-2 signature (no temporal structure: 1; drift: > 1). For majority spins the same lag
statistic is the flip rate a_k = P(s_i(t) != s_i(t-k)), and the period index is P = a2/a1.

Part A, asynchronous control (claim2.md sec. 3, "Asynchronous control"). The async release
(HF physics-of-agents/agent-opinions-async, 10.9 MB) is GPT-4o-mini only, 480 episodes = 60 statements x 8 random
signed graphs x 1 replica, no lattices. Every (statement, J) also appears in the synchronous GPT-4o-mini data with
4 replicas, so the two schedules are compared on identical statements and graphs.

Async mechanics (third_party/physics-of-agents/lib/datagen_async/dynamics_seq.py, make_schedule.py): each step is
1000 cells in which every agent fires with probability 0.0005 (Poisson(0.5) firings per step); at a firing the agent
posts a message from its current spin and then resamples K = 5 votes from the latest message on each incoming edge.
A silent agent's raw row is its carried spin repeated K times, so its "belief" jumps to +-1 even if its last fired
belief was 0.2. This contaminates the belief-based E_k (reported because it was asked for, flagged as such). The
spin flip rates are clean, but need a different null: a change needs at least one firing in the lag,
f_k = 1 - (1 - 0.0005)^(1000 k) (f1 = 0.394, f2 = 0.632), so an agent that redraws an independent spin at every
firing has a2/a1 = f2/f1 = 1.61. We report the corrected index P_async / (f2/f1): 1 = no temporal structure,
< 1 = period-2 flicker, > 1 = drift. The model-based prediction for both schedules, which also accounts for the
relaxation toward consensus, is in async_sim.py.

Part B, degree against bipartiteness (synchronous data, all four models). The two lattices differ in both
bipartiteness and degree (square: 3.25, bipartite; triangular: 4.56, not bipartite). The random signed graphs give a
cleaner contrast: every one has exactly 112 directed edges (degree 3.5), but because the disagree channel carries
little coupling (field_sufficiency.py) the graph the agents actually interact through is the agree subgraph, and that
subgraph is bipartite in some random graphs (gid 3, 4, 5, 7: every agent in a bipartite agree component) and mostly
not in others (gid 0, 6, 10, 11, 12: at most 25% of agents). We report
  B1  E2/E1 per graph, with its degree, agree degree and agree-bipartite fraction;
  B2  E2/E1 on agree-bipartite against agree-non-bipartite random graphs (same total degree);
  B3  agent level, random signed graphs: E1_i - E2_i on [agree component bipartite], agree degree, disagree degree
      and log component size, s.e. clustered by statement;
  B4  lattices at matched node degree (square against triangular nodes of degree 3 and 4).

Run: uv run python analysis/poa/flicker.py
Outputs (results/poa/): flicker_async.csv, flicker_async_by_round.csv, flicker_graphs.csv, flicker_bip_contrast.csv,
flicker_agent_regression.csv, flicker_lattice_degree.csv, flicker_log.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.poa.common import (ASYNC_PATH, CLASS_SHORT, K_VOTES, RESULTS, SHORT, agree_components,  # noqa: E402
                                 graph_class, triangles_per_node)
from analysis.poa_data import load_poa  # noqa: E402

P_NOFIRE = (1 - 0.5 / 1000) ** 1000               # per step
F1, F2 = 1 - P_NOFIRE, 1 - P_NOFIRE**2
T0 = 3                                            # rounds 3..8, as in check_poa_zero_cost.py
BIP_GRAPHS_MIN, NONBIP_GRAPHS_MAX = 1.0, 0.25     # agree-bipartite fraction defining the two groups of B2


def belief_displacements(b):
    """check_poa_zero_cost.py, test 1, for one community b (T+1, N): (A1, E1, A2, E2) over rounds 3..8, with
    floor v = (1 - b^2)/(K-1) the unbiased variance of a K-vote mean; also per-agent (E1_i, E2_i)."""
    v = (1 - b**2) / (K_VOTES - 1)
    out, per_agent = [], []
    for k in (1, 2):
        d2 = (b[T0:] - b[T0 - k:-k]) ** 2
        fl = v[T0:] + v[T0 - k:-k]
        out += [np.nanmean(d2), np.nanmean(d2) - np.nanmean(fl)]
        per_agent.append(np.nanmean(d2 - fl, axis=0))
    return out, per_agent


def spin_flips(s):
    """Spin flip rates a_k = P(s(t) != s(t-k)) over rounds 3..8 for one community s (T+1, N)."""
    return [np.mean(s[T0:] != s[T0 - k:-k]) for k in (1, 2)]


def ratio_se(a, b, cl):
    """Ratio of sums sum(a)/sum(b) with a cluster-linearization s.e. (clusters cl)."""
    A = pd.Series(np.asarray(a, float)).groupby(np.asarray(cl)).sum()
    B = pd.Series(np.asarray(b, float)).groupby(np.asarray(cl)).sum()
    r = A.sum() / B.sum()
    n = len(A)
    se = np.sqrt(n / (n - 1) * np.sum((A - r * B) ** 2)) / abs(B.sum())
    return float(r), float(se)


def ols_cluster(X, y, cl):
    """OLS with a cluster-robust sandwich s.e. (small-sample factor G/(G-1))."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1],) * 2)
    for idx in pd.Series(np.arange(len(y))).groupby(np.asarray(cl)).indices.values():
        sc = X[idx].T @ res[idx]
        meat += np.outer(sc, sc)
    G = len(np.unique(cl))
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
    return beta, np.sqrt(np.diag(V))


# ------------------------------------------------------------------------------------------------ graph facts

_GRAPH_CACHE: dict = {}


def graph_facts(J):
    """Class, degrees and agree-subgraph bipartiteness of one J (cached by bytes)."""
    kb = J.tobytes()
    if kb not in _GRAPH_CACHE:
        ac = agree_components(J)
        # spectral period-2 index -lambda_min / lambda_max: 1 on a bipartite graph, smaller with short odd cycles
        ev = np.linalg.eigvalsh((J != 0).astype(float))
        ev_a = np.linalg.eigvalsh((J > 0).astype(float))
        _GRAPH_CACHE[kb] = dict(cls=CLASS_SHORT[graph_class(J)], deg=(J != 0).sum(1), **ac,
                                tri=triangles_per_node(J), frac_bip=float(ac["bip"].mean()),
                                spec=-ev.min() / ev.max(), spec_a=-ev_a.min() / ev_a.max(), gid=len(_GRAPH_CACHE))
    return _GRAPH_CACHE[kb]


def bip_group(frac_bip, cls):
    if cls != "random signed":
        return cls
    if frac_bip >= BIP_GRAPHS_MIN:
        return "random, agree-bipartite"
    if frac_bip <= NONBIP_GRAPHS_MAX:
        return "random, agree-non-bipartite"
    return "random, mixed"


# ------------------------------------------------------------------------------------------------ part A

def part_a(lines, sync):
    asyn = load_poa(ASYNC_PATH)
    async_keys = {(k[1], k[2]) for k in asyn}
    rows, by_round = [], []
    for sched, data in (("sync", sync), ("async", asyn)):
        for (model, mode, statement), d in data.items():
            if model != "gpt-4o-mini" or (mode, statement) not in async_keys:
                continue
            for r in range(len(d["graph"])):
                J = d["J"][r]
                gf = graph_facts(J)
                if gf["cls"] != "random signed":
                    continue                          # lattices: sync only, no async counterpart
                (A1, E1, A2, E2), _ = belief_displacements(d["beliefs"][r])
                a1, a2 = spin_flips(d["spins"][r])
                rows.append(dict(schedule=sched, mode=mode, statement=statement, gid=gf["gid"],
                                 group=bip_group(gf["frac_bip"], gf["cls"]),
                                 A1=A1, E1=E1, A2=A2, E2=E2, spin_a1=a1, spin_a2=a2))
                s, v = d["spins"][r], d["votes"][r]
                for t in range(1, s.shape[0]):
                    by_round.append(dict(schedule=sched, mode=mode, statement=statement, t=t,
                                         flip1=np.mean(s[t] != s[t - 1]),
                                         mixed=np.mean(np.abs(v[t].sum(-1)) < K_VOTES)))
    df = pd.DataFrame(rows)
    res = []
    for (sched, mode, group), g in pd.concat([df, df.assign(group="all random")]).groupby(
            ["schedule", "mode", "group"]):
        cl = g["statement"].values
        e_ratio, e_se = ratio_se(g["E2"], g["E1"], cl)
        s_ratio, s_se = ratio_se(g["spin_a2"], g["spin_a1"], cl)
        null = F2 / F1 if sched == "async" else 1.0
        res.append(dict(schedule=sched, mode=mode, group=group, n_episodes=len(g), n_statements=len(set(cl)),
                        E1=g.E1.mean(), E2=g.E2.mean(), E2_over_E1=e_ratio, E2_over_E1_se=e_se,
                        spin_a1=g.spin_a1.mean(), spin_a2=g.spin_a2.mean(),
                        P=s_ratio, P_se=s_se, P_null=null, P_corrected=s_ratio / null, P_corrected_se=s_se / null))
    out = pd.DataFrame(res)
    out.to_csv(RESULTS / "flicker_async.csv", index=False)
    br = pd.DataFrame(by_round).groupby(["schedule", "mode", "t"])[["flip1", "mixed"]].mean().reset_index()
    br.to_csv(RESULTS / "flicker_async_by_round.csv", index=False)

    lines.append("Part A: GPT-4o-mini, same 60 statements x 8 random signed graphs; rounds (steps) 3..8")
    lines.append(f"  async: P(>=1 firing in lag 1) = {F1:.3f}, lag 2 = {F2:.3f}; memoryless null a2/a1 = {F2/F1:.3f}")
    lines.append("  belief E2/E1 (async value contaminated by silent rows) | spin index P = a2/a1, "
                 "corrected = P / null (1: no structure, <1: flicker)")
    for _, r in out.iterrows():
        lines.append(f"  {r.schedule:5s} {r['mode']:10s} {r.group:28s} n={r.n_episodes:4d} | E1 {r.E1:.3f} E2 "
                     f"{r.E2:.3f} E2/E1 {r.E2_over_E1:.2f} ({r.E2_over_E1_se:.2f}) | a1 {r.spin_a1:.3f} a2 "
                     f"{r.spin_a2:.3f} P {r.P:.2f} ({r.P_se:.2f}) -> corrected {r.P_corrected:.2f} "
                     f"({r.P_corrected_se:.2f})")
    lines.append("  spin flip rate s(t) != s(t-1), and fraction of rows with mixed votes, by round t = 1..8:")
    for mode in ("objective", "subjective"):
        for sched in ("sync", "async"):
            a = br[(br["mode"] == mode) & (br.schedule == sched)]
            lines.append(f"    {mode:10s} {sched:5s} flip  " + " ".join(f"{x:.3f}" for x in a.flip1)
                         + "   mixed " + " ".join(f"{x:.3f}" for x in a.mixed))
    return out


# ------------------------------------------------------------------------------------------------ part B

def agent_table(sync):
    """Per (model, mode, statement, graph, agent): floor-corrected E1_i, E2_i averaged over the 4 replicas."""
    rows = []
    for (model, mode, statement), d in sync.items():
        for r in range(len(d["graph"])):
            gf = graph_facts(d["J"][r])
            _, (e1, e2) = belief_displacements(d["beliefs"][r])
            for i in range(len(e1)):
                rows.append((SHORT[model], mode, statement, gf["gid"], gf["cls"], bip_group(gf["frac_bip"], gf["cls"]),
                             gf["frac_bip"], gf["spec"], gf["spec_a"], i, gf["deg"][i], gf["deg_a"][i],
                             gf["deg_d"][i], gf["bip"][i], gf["comp_size"][i], gf["tri"][i], e1[i], e2[i]))
    df = pd.DataFrame(rows, columns=["model", "mode", "statement", "gid", "cls", "group", "frac_bip", "spec",
                                     "spec_a", "agent", "deg", "deg_a", "deg_d", "bip", "comp_size", "tri", "E1", "E2"])
    keys = [c for c in df.columns if c not in ("E1", "E2")]
    df = df.groupby(keys, as_index=False)[["E1", "E2"]].mean()
    df["flick"] = df.E1 - df.E2
    df["cell"] = df.model + " | " + df["mode"]
    return df


def part_b(lines, sync):
    df = agent_table(sync)

    # B1: per graph
    g1 = []
    for (cell, gid), g in df.groupby(["cell", "gid"]):
        r, se = ratio_se(g.E2, g.E1, g.statement)
        g1.append(dict(cell=cell, gid=gid, cls=g.cls.iloc[0], group=g.group.iloc[0], frac_bip=g.frac_bip.iloc[0],
                       spec=g.spec.iloc[0], spec_a=g.spec_a.iloc[0], tri=g.tri.mean(),
                       E1=g.E1.mean(), E2=g.E2.mean(),
                       E2_over_E1=r, se=se, n_statements=g.statement.nunique()))
    deg_of = df.groupby("gid")[["deg", "deg_a"]].mean()
    g1 = pd.DataFrame(g1).join(deg_of, on="gid")
    g1.to_csv(RESULTS / "flicker_graphs.csv", index=False)

    # B2: agree-bipartite against agree-non-bipartite random graphs, and the two lattices, per cell
    b2 = []
    for (cell, group), g in df.groupby(["cell", "group"]):
        r, se = ratio_se(g.E2, g.E1, g.statement)
        b2.append(dict(cell=cell, group=group, n_graphs=g.gid.nunique(), deg=g.deg.mean(), deg_a=g.deg_a.mean(),
                       E1=g.E1.mean(), E2=g.E2.mean(), E2_over_E1=r, se=se))
    b2 = pd.DataFrame(b2)
    b2.to_csv(RESULTS / "flicker_bip_contrast.csv", index=False)
    lines.append("\nPart B: degree against bipartiteness (sync, all models). E2/E1 (s.e. over statements) by graph group")
    groups = ["square lattice", "random, agree-bipartite", "random, mixed", "random, agree-non-bipartite",
              "triangular lattice"]
    hdr = "  " + f"{'cell':26s}" + "".join(f"{g[:22]:>24s}" for g in groups)
    lines.append(hdr)
    lines.append("  " + f"{'(mean degree / agree degree)':26s}" + "".join(
        f"{'%.2f / %.2f' % tuple(b2[b2.group == g][['deg', 'deg_a']].iloc[0]):>24s}" for g in groups))
    for cell, g in b2.groupby("cell"):
        g = g.set_index("group")
        lines.append("  " + f"{cell:26s}" + "".join(
            f"{'%.2f (%.2f)' % (g.loc[k, 'E2_over_E1'], g.loc[k, 'se']):>24s}" for k in groups))

    sp = g1.drop_duplicates("gid").set_index("gid").sort_values("spec")
    lines.append("  spectral index -lam_min/lam_max of the unsigned graph (agree subgraph), mean triangles per node:")
    lines.append("    " + ", ".join(f"gid {k} {r.cls[:6]} {r.spec:.2f} ({r.spec_a:.2f}) tri {r.tri:.2f}"
                                  for k, r in sp.iterrows()))
    lines.append("  correlation over the 12 random graphs of E2/E1 with the spectral index of the full (agree) graph:")
    for cell, g in g1[g1.cls == "random signed"].groupby("cell"):
        lines.append(f"    {cell:26s} r = {np.corrcoef(g.spec, g.E2_over_E1)[0, 1]:+.2f} "
                     f"({np.corrcoef(g.spec_a, g.E2_over_E1)[0, 1]:+.2f}); E2/E1 range {g.E2_over_E1.min():.2f}"
                     f"-{g.E2_over_E1.max():.2f}")

    # B3: agent-level regression within random signed graphs
    reg = []
    names = ["const", "agree component bipartite", "agree degree", "disagree degree", "log agree component size",
             "triangles"]
    for cell, g in df[df.cls == "random signed"].groupby("cell"):
        X = np.column_stack([np.ones(len(g)), g.bip.astype(float), g.deg_a, g.deg_d, np.log(g.comp_size), g.tri])
        beta, se = ols_cluster(X, g.flick.values, g.statement.values)
        for n, b_, s_ in zip(names, beta, se):
            reg.append(dict(cell=cell, term=n, coef=b_, se=s_, n_agents=len(g), mean_flick=g.flick.mean(),
                            mean_E1=g.E1.mean()))
    reg = pd.DataFrame(reg)
    reg.to_csv(RESULTS / "flicker_agent_regression.csv", index=False)
    lines.append("  B3, random signed graphs, agent level: OLS of E1_i - E2_i (s.e. clustered by statement)")
    for cell, g in reg.groupby("cell"):
        lines.append(f"    {cell:26s} mean E1-E2 {g.mean_flick.iloc[0]:.3f} | " + "  ".join(
            f"{t}: {c:+.3f} ({s:.3f})" for t, c, s in zip(g.term, g.coef, g.se) if t != "const"))

    # B4: lattices at matched node degree
    b4 = []
    lat = df[df.cls.isin(["square lattice", "triangular lattice"])]
    for (cell, cls, deg), g in lat.groupby(["cell", "cls", "deg"]):
        r, se = ratio_se(g.E2, g.E1, g.statement)
        b4.append(dict(cell=cell, cls=cls, deg=deg, n_nodes=g.agent.nunique(), E1=g.E1.mean(), E2=g.E2.mean(),
                       E2_over_E1=r, se=se))
    b4 = pd.DataFrame(b4)
    b4.to_csv(RESULTS / "flicker_lattice_degree.csv", index=False)
    common_deg = sorted(set(b4[b4.cls == "square lattice"].deg) & set(b4[b4.cls == "triangular lattice"].deg))
    lines.append(f"  B4, lattices at matched node degree {common_deg}: E2/E1 square | triangular")
    for cell, g in b4.groupby("cell"):
        g = g.set_index(["cls", "deg"])
        lines.append(f"    {cell:26s} " + "   ".join(
            f"deg {k}: {g.loc[('square lattice', k), 'E2_over_E1']:.2f} ({g.loc[('square lattice', k), 'se']:.2f})"
            f" | {g.loc[('triangular lattice', k), 'E2_over_E1']:.2f} ({g.loc[('triangular lattice', k), 'se']:.2f})"
            for k in common_deg))
    nn = b4.drop_duplicates(["cls", "deg"]).set_index(["cls", "deg"]).n_nodes
    lines.append("    nodes per degree: " + ", ".join(f"{c[:3]} {d}: {n}" for (c, d), n in nn.items()))
    return df


if __name__ == "__main__":
    sync = load_poa()
    lines = []
    part_a(lines, sync)
    part_b(lines, sync)
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "flicker_log.txt").write_text(txt + "\n")
