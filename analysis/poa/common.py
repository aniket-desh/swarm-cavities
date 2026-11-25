"""Shared pieces for the zero-cost Physics of Agents (PoA) analyses (theory/README.md, "Physics of Agents data").

Contents
  * ``cells()``: the synchronous release, one entry per (model, task type), spins and votes put in a common frame.
  * ``transitions()``: one row per (community, agent i, round t -> t+1), with the inbox counts split by the
    senders' state, the agent's own states at t and t-1, and the K = 5 votes at t+1.
  * ``fit_logit_fe()``: binomial logistic regression with one intercept per (statement, agent), by Newton's
    method with the intercepts eliminated through a Schur complement.
  * graph helpers (lattice class, triangles) and the majority-of-5 response phi(u).

Frame. For objective tasks spins and votes are multiplied by tau = +1 (truth A) or -1 (truth B), so +1 always
means "holds the correct answer"; this leaves every model with a symmetric drive (h, two-channel) unchanged up to
the sign of the intercepts, and makes the separate count gains of the four-count model readable as truth
asymmetry. Subjective tasks keep PoA's A = +1 frame.

What the agent sees (third_party/physics-of-agents/lib/datagen/dynamics.py). After spin phase t, every sender j
writes one message conditioned on its majority spin s_j(t) (and its own inbox); the message is delivered to the
agree inbox of i if J_poa[j, i] = +1 and the disagree inbox if -1 (our J[i, j], loader transposes). In spin phase
t+1 agent i reads the two inboxes and its persona, but not its own previous opinion, and K = 5 votes are sampled.
So the counts are taken over neighbors' s_j(t), and s_i(t), s_i(t-1) enter the prompt only through the persona.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # the clone, /workspace-global/swarm-cavity/repo
sys.path.insert(0, str(REPO))
from analysis.poa_data import load_poa  # noqa: E402

RESULTS = REPO / "results" / "poa"
ASYNC_PATH = Path("/workspace-global/swarm-cavity/data/poa/async/agent-opinions-async.parquet")
SHORT = {"gpt-4o-mini": "gpt-4o-mini", "google/gemma-3n-E4B-it": "gemma-3n-E4B",
         "Qwen/Qwen3.5-9B": "Qwen3.5-9B", "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B"}
MODEL_ORDER = ["gpt-4o-mini", "Qwen3.5-9B", "Llama-3.1-8B", "gemma-3n-E4B"]
K_VOTES = 5


# ------------------------------------------------------------------------------------------------ graphs

def graph_class(J: np.ndarray) -> str:
    """'random signed' if any edge is negative, else the lattice's bipartiteness by BFS 2-colouring.
    (Same rule as theory/checks/check_poa_zero_cost.py.) PoA's lattices are 4 x 8 with open boundaries:
    square (mean degree 3.25, bipartite) and triangular (square plus one diagonal, mean degree 4.56)."""
    if (J < 0).any():
        return "random signed"
    A = J != 0
    color = -np.ones(len(A), int)
    for root in range(len(A)):
        if color[root] >= 0:
            continue
        color[root] = 0
        stack = [root]
        while stack:
            i = stack.pop()
            for j in np.nonzero(A[i])[0]:
                if color[j] < 0:
                    color[j] = 1 - color[i]
                    stack.append(j)
                elif color[j] == color[i]:
                    return "lattice, not bipartite"
    return "lattice, bipartite"


CLASS_SHORT = {"lattice, bipartite": "square lattice", "lattice, not bipartite": "triangular lattice",
               "random signed": "random signed"}


def triangles_per_node(J: np.ndarray) -> np.ndarray:
    """Number of (unsigned) triangles through each node: diag(A^3) / 2."""
    A = (J != 0).astype(float)
    return np.einsum("ij,jk,ki->i", A, A, A) / 2


def agree_components(J: np.ndarray) -> dict:
    """Per-node facts about the agree subgraph (J = +1 edges), which is the whole interaction graph when the
    disagree channel carries little coupling (b- ~ 0, field_sufficiency.py):
      comp (N,) component label, comp_size (N,), bip (N,) bool: the node's agree component is bipartite,
      deg_a (N,) agree degree, deg_d (N,) disagree degree."""
    from scipy.sparse.csgraph import connected_components
    Ap = (J > 0).astype(int)
    nc, lab = connected_components(Ap, directed=False)
    bip = np.zeros(nc, bool)
    for k in range(nc):
        idx = np.nonzero(lab == k)[0]
        bip[k] = graph_class(Ap[np.ix_(idx, idx)]) != "lattice, not bipartite"
    return dict(comp=lab, comp_size=np.bincount(lab)[lab], bip=bip[lab], deg_a=Ap.sum(1), deg_d=(J < 0).sum(1))


def coupling_matrix(J: np.ndarray, b_plus: float, b_minus: float) -> np.ndarray:
    """Effective coupling in logit units of the two-channel model M2 (claim1.md sec. 4.2): an agree tie pulls with
    +b+, a disagree tie with -b-, so W_ij = b+ [J_ij = +1] - b- [J_ij = -1] and the logit is alpha_i + (W s)_i."""
    return b_plus * (J > 0) - b_minus * (J < 0)


# ------------------------------------------------------------------------------------------------ data

def cells(path=None) -> dict:
    """{(model_short, mode): dict} with, stacked over the cell's statements (R communities):
    J (R,N,N) our orientation; S (R,T+1,N) majority spins and V (R,T+1,N,K) votes in the frame above;
    stmt (R,) statement index within the cell; graph (R,) graph index within the statement (0..9);
    gid (R,) global graph id (index into the 14 distinct J); gclass (R,) class string; tau (R,);
    stmt_names: list of statement strings, indexed by stmt."""
    data = load_poa() if path is None else load_poa(path)
    out = {}
    gid_of = {}
    for (model, mode, statement), d in data.items():
        key = (SHORT[model], mode)
        c = out.setdefault(key, {k: [] for k in ("J", "S", "V", "stmt", "graph", "gid", "gclass", "tau")})
        n_stmt = c.setdefault("_nstmt", 0)
        tau = {"A": 1, "B": -1}.get(d["ground_truth"], 1) if mode == "objective" else 1
        R = len(d["graph"])
        for r in range(R):
            kb = d["J"][r].tobytes()
            if kb not in gid_of:
                gid_of[kb] = len(gid_of)
            c["gid"].append(gid_of[kb])
        c["J"].append(d["J"])
        c["S"].append(d["spins"] * tau)
        c["V"].append(d["votes"] * tau)
        c["stmt"].append(np.full(R, n_stmt))
        c["graph"].append(d["graph"])
        c["tau"].append(np.full(R, tau))
        c["_nstmt"] = n_stmt + 1
        c.setdefault("stmt_names", []).append(statement)
    cls_of = {}
    for key, c in out.items():
        c.pop("_nstmt")
        for k in ("J", "S", "V", "stmt", "graph", "tau"):
            c[k] = np.concatenate(c[k])
        c["gid"] = np.array(c["gid"])
        for g, J in zip(c["gid"], c["J"]):
            if g not in cls_of:
                cls_of[g] = graph_class(J)
        c["gclass"] = np.array([cls_of[g] for g in c["gid"]])
    return out


def transitions(c: dict, t_min: int = 1) -> dict:
    """Flattened transition table for one cell, rounds t = t_min..T-1 (transition t -> t+1).

    Counts over neighbours j of i, by the sender's state s_j(t) (state 0 = no parsed vote, counted nowhere):
      nap = #{J_ij=+1, s_j=+1}, nam = #{J_ij=+1, s_j=-1}, ndp = #{J_ij=-1, s_j=+1}, ndm = #{J_ij=-1, s_j=-1}.
    Channel fields: m_a = nap - nam (agree inbox), m_d = ndp - ndm (disagree inbox); scalar field
    h_i = sum_j J_ij s_j = m_a - m_d (claim1.md eq. (1)). Response: npos = #votes = +1 at t+1, ntot = #parsed.
    """
    J, S, V = c["J"], c["S"], c["V"]
    R, T1, N = S.shape
    Ap = (J == 1).astype(np.int16)
    Am = (J == -1).astype(np.int16)
    t_idx = np.arange(t_min, T1 - 1)
    s_now = S[:, t_idx]                                       # (R, nt, N)
    up, dn = (s_now == 1).astype(np.int16), (s_now == -1).astype(np.int16)
    nap = np.einsum("rij,rtj->rti", Ap, up)
    nam = np.einsum("rij,rtj->rti", Ap, dn)
    ndp = np.einsum("rij,rtj->rti", Am, up)
    ndm = np.einsum("rij,rtj->rti", Am, dn)
    votes = V[:, t_idx + 1]                                   # (R, nt, N, K)
    npos = (votes == 1).sum(-1)
    ntot = (votes != 0).sum(-1)
    s_prev = S[:, t_idx - 1] if t_min >= 1 else np.zeros_like(s_now)
    shp = s_now.shape
    bc = lambda a: np.broadcast_to(a, shp).reshape(-1)        # noqa: E731
    agent = np.arange(N)[None, None, :]
    n_stmt = c["stmt"].max() + 1
    out = dict(
        nap=nap.reshape(-1).astype(float), nam=nam.reshape(-1).astype(float),
        ndp=ndp.reshape(-1).astype(float), ndm=ndm.reshape(-1).astype(float),
        s_now=s_now.reshape(-1).astype(float), s_prev=s_prev.reshape(-1).astype(float),
        npos=npos.reshape(-1).astype(float), ntot=ntot.reshape(-1).astype(float),
        comm=bc(np.arange(R)[:, None, None]), t=bc(t_idx[None, :, None]), agent=bc(agent),
        stmt=bc(c["stmt"][:, None, None]), graph=bc(c["graph"][:, None, None]),
        gclass=bc(c["gclass"][:, None, None]),
    )
    out["group"] = out["stmt"] * N + out["agent"]             # fixed effect: (statement, agent) = persona x question
    out["n_groups"] = int(n_stmt * N)
    out["m_a"] = out["nap"] - out["nam"]
    out["m_d"] = out["ndp"] - out["ndm"]
    out["k_a"] = out["nap"] + out["nam"]
    out["k_d"] = out["ndp"] + out["ndm"]
    keep = out["ntot"] > 0
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ fitting

def sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))


def fit_logit_fe(X, npos, ntot, group, n_groups, offset=None, lam_beta=1e-3, prior_sd_alpha=5.0,
                 beta0=None, alpha0=None, max_iter=50, tol=1e-9):
    """Maximize sum_rows [npos log p + (ntot-npos) log(1-p)] - lam_beta |beta|^2/2 - |alpha|^2/(2 sd^2),
    logit p = X beta + alpha[group] + offset.

    The alpha ridge (prior sd 5 in logit units) only matters for (statement, agent) groups that vote the same
    way in every training transition, whose MLE intercept is infinite. Newton steps: with w = ntot p(1-p),
    H = [[X'WX, X'WG], [G'WX, diag(G'W1)]] (G the group indicator matrix); alpha is eliminated by the Schur
    complement S = X'WX - X'WG D^-1 G'WX, so each step costs O(n p^2). Step halving keeps the loglik monotone.
    Returns (beta, alpha, loglik_without_penalty).
    """
    n, p = X.shape
    off = 0.0 if offset is None else offset
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    alpha = np.zeros(n_groups) if alpha0 is None else alpha0.copy()
    lam_a = 1.0 / prior_sd_alpha**2

    def objective(b, a):
        u = X @ b + a[group] + off
        ll = float(np.sum(npos * u - ntot * np.logaddexp(0.0, u)))
        return ll - 0.5 * lam_beta * b @ b - 0.5 * lam_a * a @ a, ll, u

    obj, ll, u = objective(beta, alpha)
    for _ in range(max_iter):
        pr = sigmoid(u)
        r = npos - ntot * pr
        w = ntot * pr * (1 - pr)
        gb = X.T @ r - lam_beta * beta
        ga = np.bincount(group, r, n_groups) - lam_a * alpha
        Hbb = (X * w[:, None]).T @ X + lam_beta * np.eye(p)
        Hba = np.stack([np.bincount(group, w * X[:, k], n_groups) for k in range(p)]) if p else np.zeros((0, n_groups))
        Daa = np.bincount(group, w, n_groups) + lam_a
        Schur = Hbb - (Hba / Daa) @ Hba.T
        db = np.linalg.solve(Schur, gb - Hba @ (ga / Daa)) if p else np.zeros(0)
        da = (ga - Hba.T @ db) / Daa
        step = 1.0
        while True:
            nobj, nll, nu = objective(beta + step * db, alpha + step * da)
            if nobj >= obj - 1e-12 or step < 1e-6:
                break
            step *= 0.5
        beta, alpha = beta + step * db, alpha + step * da
        gain = nobj - obj
        obj, ll, u = nobj, nll, nu
        if gain < tol * max(1.0, abs(obj)):
            break
    return beta, alpha, ll


def fit_m2(c: dict, averaged: bool = False):
    """Full-data fit of the two-channel model M2 of field_sufficiency.py for one cell: returns (b_plus, b_minus,
    alpha) with logit P(vote = +1) = alpha[stmt * N + agent] + b+ m_a - b- m_d (frame of ``cells``). With
    averaged=True the M2avg form, m_a and m_d divided by the inbox size k (0 for an empty inbox)."""
    T = transitions(c)
    k = T["k_a"] + T["k_d"]
    scale = np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0) if averaged else 1.0
    X = np.stack([T["m_a"] * scale, T["m_d"] * scale], 1)
    beta, alpha, _ = fit_logit_fe(X, T["npos"], T["ntot"], T["group"], T["n_groups"])
    return float(beta[0]), float(-beta[1]), alpha


def loglik_rows(u, npos, ntot):
    """Per-row Bernoulli log-likelihood of the votes (no binomial coefficient: each vote is one draw)."""
    return npos * u - ntot * np.logaddexp(0.0, u)


# ------------------------------------------------------------------------------------------------ response

def phi_maj(u, K=K_VOTES):
    """E[majority spin of K iid votes], each +1 w.p. p = sigmoid(u): 2 P(Bin(K,p) > K/2) - 1 (K odd)."""
    from scipy.stats import binom
    p = sigmoid(u)
    return 2 * binom.sf(K // 2, K, p) - 1


def dphi_maj(u, K=K_VOTES):
    """d phi_maj / du. For K = 5: dphi/dp = 2 K C(K-1, 2) p^2 (1-p)^2 = 60 p^2 (1-p)^2 and dp/du = p(1-p),
    so dphi/du = 60 [p(1-p)]^3; 0.9375 at p = 1/2, against 0.5 for a single vote (tanh(u/2))."""
    from math import comb
    p = sigmoid(u)
    m = K // 2
    return 2 * K * comb(K - 1, m) * (p * (1 - p)) ** (m + 1)
