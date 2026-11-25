"""Loader for the Physics of Agents dataset (El et al., arXiv:2608.16578; HF physics-of-agents/agent-opinions).

One parquet row is one community: N=32 agents, T=8 synchronous rounds (T+1 = 9 states including the
round-0 answer before any messages), K=5 sampled binary votes per agent per round.

J orientation. In PoA's data generator (third_party/physics-of-agents/lib/datagen/dynamics.py,
``_build_inboxes`` and ``_run_message_phase``) J[i, j] is the edge from sender i to receiver j: the
message of i lands in j's agree/disagree inbox according to sign J[i, j]. Our convention
(envs/graphs.py, docs/spec.md) is the opposite, J[i, j] = influence of j on i with h = J @ s, so the
loader returns ``J = J_poa.T``. PoA's own fitting code (res/utils.py ``drives``, einsum "eij,ej->ei")
uses rows as receivers, i.e. our convention, without transposing; that is harmless only because
every released J is symmetric, which ``summarize`` checks (all 9,600 are).

Beliefs. ``beliefs`` is the mean of the parsed votes (+-1) in each round, i.e. 2 p_agree - 1 in
[-1, 1]; unparsed votes (0) are dropped and a round with no parsed vote is NaN. PoA's
``res/utils.py: opinions`` instead averages all K entries, counting failures as 0. ``spins`` is
PoA's majority vote (ties broken by the first vote, 0 if nothing parsed).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA_PATH = Path("/workspace-global/swarm-cavity/data/poa/agent-opinions.parquet")


def _nested_to_array(col, shape) -> np.ndarray:
    """Fixed-shape nested list column -> ndarray (rows, *shape), checking every list length."""
    col = col.combine_chunks()
    for depth, n in enumerate(shape):
        lens = pc.list_value_length(col).to_numpy()
        if not np.all(lens == n):
            raise ValueError(f"ragged list at depth {depth}: lengths {np.unique(lens)}, expected {n}")
        col = pc.list_flatten(col)
    return col.to_numpy().reshape((-1,) + tuple(shape))


def load_poa(path: Path | str = DATA_PATH) -> dict[tuple[str, str, str], dict]:
    """Return {(model, mode, statement): arrays} with, for that key's R communities (R = 40: 10 graphs x
    4 replicas):
        J        (R, N, N) int8, our orientation (receiver row): J = J_poa.T
        spins    (R, T+1, N) int8 in {-1, 0, +1}
        votes    (R, T+1, N, K) int8 raw votes, 0 = unparsed
        beliefs  (R, T+1, N) float, mean of parsed votes, NaN if none parsed
        replica  (R,) PoA replica index (0..3) within a graph
        graph    (R,) int, index of the distinct J within this key (communities with equal ``graph``
                 share J and statement and differ only in sampling)
        ground_truth, political_lean: str or None
    Rows within a key are sorted by (graph, replica).
    """
    t = pq.read_table(path, columns=["model", "mode", "statement", "J", "spins_history",
                                     "spins_raw_history", "replica", "ground_truth", "political_lean"])
    n_rows = t.num_rows
    N = len(t.column("J")[0])
    T1 = len(t.column("spins_history")[0])
    K = len(t.column("spins_raw_history")[0][0][0])
    J = _nested_to_array(t.column("J"), (N, N)).astype(np.int8).transpose(0, 2, 1)
    spins = _nested_to_array(t.column("spins_history"), (T1, N)).astype(np.int8)
    votes = _nested_to_array(t.column("spins_raw_history"), (T1, N, K)).astype(np.int8)
    n_parsed = (votes != 0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beliefs = np.where(n_parsed > 0, votes.sum(axis=-1) / n_parsed, np.nan)
    meta = t.select(["model", "mode", "statement", "replica", "ground_truth", "political_lean"]).to_pandas()

    out = {}
    for key, idx in meta.groupby(["model", "mode", "statement"], sort=True).indices.items():
        gkeys = [J[i].tobytes() for i in idx]
        gid_of = {gk: n for n, gk in enumerate(dict.fromkeys(gkeys))}
        graph = np.array([gid_of[gk] for gk in gkeys])
        order = np.lexsort((meta.replica.values[idx], graph))
        idx, graph = idx[order], graph[order]
        first = meta.iloc[idx[0]]
        out[key] = dict(
            J=J[idx], spins=spins[idx], votes=votes[idx], beliefs=beliefs[idx],
            replica=meta.replica.values[idx], graph=graph,
            ground_truth=first.ground_truth if isinstance(first.ground_truth, str) else None,
            political_lean=first.political_lean if isinstance(first.political_lean, str) else None,
        )
    assert sum(len(v["graph"]) for v in out.values()) == n_rows
    return out


def summarize(data: dict) -> dict:
    """Print and return a short summary of a ``load_poa`` result."""
    keys = list(data)
    J = np.concatenate([data[k]["J"] for k in keys]).astype(int)
    spins = np.concatenate([data[k]["spins"] for k in keys])
    votes = np.concatenate([data[k]["votes"] for k in keys])
    n_comm, T1, N, K = votes.shape
    nz = J != 0
    # Majority of parsed votes with PoA's tie-break (first vote), to check spins against votes.
    vs = votes.sum(-1)
    maj = np.where(vs > 0, 1, np.where(vs < 0, -1, votes[..., 0]))
    group_sizes = Counter(int(c) for k in keys for c in np.bincount(data[k]["graph"]))
    graphs_per_key = Counter(int(data[k]["graph"].max()) + 1 for k in keys)
    pool = {j.tobytes(): j for j in J.astype(np.int8)}
    n_unsigned = sum(bool(np.all(j >= 0)) for j in pool.values())    # PoA's lattices are all-positive
    graph_sets = {frozenset(j.tobytes() for j in data[k]["J"]) for k in keys}
    models_per_statement_J = Counter()
    for k in keys:
        for j in {j.tobytes() for j in data[k]["J"]}:
            models_per_statement_J[(k[1], k[2], j)] += 1
    s = dict(
        communities=n_comm, keys=len(keys), N=N, rounds_T=T1 - 1, votes_K=K,
        per_model_mode={f"{m} | {md}": (sum(1 for k in keys if k[:2] == (m, md)),
                                         sum(len(data[k]["graph"]) for k in keys if k[:2] == (m, md)))
                        for m, md in sorted({k[:2] for k in keys})},
        mean_degree=float(nz.sum(axis=2).mean()),
        degree_range=(int(nz.sum(axis=2).min()), int(nz.sum(axis=2).max())),
        frac_negative_edges=float((J[nz] < 0).mean()),
        all_symmetric=bool(np.all(J == J.transpose(0, 2, 1))),
        zero_diagonal=bool(np.all(np.einsum("cii->ci", J) == 0)),
        distinct_J=len(pool), distinct_J_all_positive=n_unsigned, distinct_graph_sets=len(graph_sets),
        models_per_statement_and_J=dict(Counter(models_per_statement_J.values())),
        graphs_per_key=dict(graphs_per_key),
        replicas_sharing_J_and_statement=dict(group_sizes),
        frac_unparsed_votes=float((votes == 0).mean()),
        frac_zero_spins=float((spins == 0).mean()),
        frac_rounds_no_parsed_vote=float(((votes != 0).sum(-1) == 0).mean()),
        spins_equal_majority_of_votes=float((maj == spins).mean()),
        frac_spins_plus=float((spins == 1).mean()),
    )
    print(f"PoA: {s['communities']} communities, {s['keys']} (model, mode, statement) keys, N={N}, "
          f"T={s['rounds_T']} rounds (+ round 0), K={K} votes")
    print("  (model | mode): statements, communities")
    for mm, (nq, nc) in s["per_model_mode"].items():
        print(f"    {mm}: {nq}, {nc}")
    print(f"  mean degree {s['mean_degree']:.3f} (range {s['degree_range']}), "
          f"negative edges {s['frac_negative_edges']:.3f}, symmetric: {s['all_symmetric']}, "
          f"zero diagonal: {s['zero_diagonal']}")
    print(f"  distinct J: {len(pool)} ({n_unsigned} all-positive lattices), in {len(graph_sets)} sets of "
          f"graphs; graphs per key: {s['graphs_per_key']}")
    print(f"  communities sharing (model, J, statement): {s['replicas_sharing_J_and_statement']} "
          f"(group size: count); models run on each (statement, J): {s['models_per_statement_and_J']}")
    print(f"  unparsed votes {s['frac_unparsed_votes']:.4f}, zero spins {s['frac_zero_spins']:.4f}, "
          f"rounds with no parsed vote {s['frac_rounds_no_parsed_vote']:.4f}, "
          f"spins == majority(votes) {s['spins_equal_majority_of_votes']:.4f}, "
          f"P(spin=+1) {s['frac_spins_plus']:.3f}")
    return s


if __name__ == "__main__":
    summarize(load_poa())
