"""Zero-cost first look at two predictions on the Physics of Agents release (README.md, section "Physics of Agents").

Data: /workspace-global/swarm-cavity/data/poa/agent-opinions.parquet via analysis.poa_data.load_poa
(N = 32, T = 8, K = 5 votes per agent per round, symmetric signed graphs, synchronous rounds, and an opinion prompt
that does not show the agent its own previous opinion).

Test 1, period-2 flicker (claim2.md section 4). Lag displacements of the beliefs b_i(t) (mean of the K votes):
    A_k = <(b_i(t) - b_i(t-k))^2> over agents and rounds t = 3..8,  k = 1, 2,
    floor = <v_i(t) + v_i(t-k)>, v = (1 - b^2)/(K-1): the part of A_k due to sampling K votes,
    excess E_k = A_k - floor.
    No temporal structure gives E_2 = E_1; drift gives E_2 > E_1; a period-2 component gives E_2 < E_1.
    Reported per model, task type and graph class (random signed; all-positive bipartite = square lattice;
    all-positive non-bipartite = triangular lattice).
Test 2, coupling-induced overlap (claim2.md section 5). For each (model, task type, statement):
    dq(t) = q_same-graph(t) - q_other-graph(t), q = (1/N) sum_i b_i^r(t) b_i^r'(t) over pairs of distinct communities.
    Personas and statement are identical across the ten graphs, so dq isolates what the graph adds. dq(0) must
    vanish, because round 0 is answered before any message is read.

Run: uv run python theory/checks/check_poa_zero_cost.py
"""
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/workspace/swarm-cavity")
from analysis.poa_data import load_poa

data = load_poa()
K = next(iter(data.values()))["votes"].shape[-1]


def graph_class(J):
    if (J < 0).any():
        return "random signed"
    A = J != 0
    color = -np.ones(len(A), int)
    for root in range(len(A)):
        if color[root] >= 0:
            continue
        color[root] = 0; stack = [root]
        while stack:
            i = stack.pop()
            for j in np.nonzero(A[i])[0]:
                if color[j] < 0:
                    color[j] = 1 - color[i]; stack.append(j)
                elif color[j] == color[i]:
                    return "lattice, not bipartite"
    return "lattice, bipartite"


rows = defaultdict(list); dq = defaultdict(list)
cls_cache = {}
for (model, mode, statement), d in data.items():
    b = d["beliefs"]                                        # (R, T+1, N)
    v = (1 - b**2) / (K - 1)
    for r in range(len(b)):
        key = d["J"][r].tobytes()
        if key not in cls_cache:
            cls_cache[key] = graph_class(d["J"][r])
        out = []
        for k in (1, 2):
            A = np.nanmean((b[r, 3:] - b[r, 3 - k:-k]) ** 2)
            fl = np.nanmean(v[r, 3:] + v[r, 3 - k:-k])
            out += [A, A - fl]
        rows[(model.split("/")[-1], mode, cls_cache[key])].append(out)
    # replica overlaps; NaN beliefs (no parsed vote) are left out of the agent average
    R = len(b)
    Q = np.array([[np.nanmean(b[r] * b[rp], axis=1) for rp in range(R)] for r in range(R)])   # (R, R, T+1)
    same = d["graph"][:, None] == d["graph"][None, :]
    off = ~np.eye(R, dtype=bool)
    if (same & off).any() and (~same).any():
        dq[(model.split("/")[-1], mode)].append(Q[same & off].mean(0) - Q[~same].mean(0))

print("Test 1: lag displacements over rounds 3..8 (mean over communities; s.e. over communities)")
print(f"{'model':16s} {'task':11s} {'graph class':24s} {'n':>5s} |  A1      A2    | excess E1  E2   |  E1 - E2 (s.e.)    E2/E1")
for key in sorted(rows):
    r = np.array(rows[key]); mu = r.mean(0)
    diff = r[:, 1] - r[:, 3]
    print(f"{key[0]:16s} {key[1]:11s} {key[2]:24s} {len(r):5d} | {mu[0]:.4f}  {mu[2]:.4f} | {mu[1]:.4f}  {mu[3]:.4f} |"
          f" {diff.mean():+.4f} ({diff.std(ddof=1)/np.sqrt(len(r)):.4f})   {mu[3]/mu[1]:.2f}")

print("\nTest 2: dq(t) = q_same-graph - q_other-graph, mean over statements (s.e. over statements)")
print(f"{'model':16s} {'task':11s} {'n':>4s} | t = 0 .. 8")
for key in sorted(dq):
    a = np.array(dq[key]); mu = a.mean(0); se = a.std(0, ddof=1) / np.sqrt(len(a))
    print(f"{key[0]:16s} {key[1]:11s} {len(a):4d} | " + "  ".join(f"{m:+.3f}" for m in mu))
    print(f"{'':33s} | " + "  ".join(f"({s:.3f})" for s in se))
