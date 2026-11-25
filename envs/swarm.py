"""N-agent swarm on a signed directed graph, T rounds, with a pluggable agent backend.

Dynamics (theory/claim1.md eq. (1), option B of section 5). Agent i reads the senders j with J_ij != 0
(envs/graphs.py convention: rows receive), J_ij = +1 in its "tend to agree with" inbox and -1 in its
"tend to disagree with" inbox. Beliefs are p = stated probability of HEADS, mapped to s = 2p - 1.

    round 0:       every agent answers from its evidence alone                 -> p(0)
    sync, t >= 1:  every agent reads the round-(t-1) messages of its senders   -> p(t) = F(h(t-1))
    async, t >= 1: agents update one at a time in a fresh random order; each reads its senders' latest
                   messages, including those already updated in round t (the asynchronous control of
                   theory/README.md, test 2)

so a run has T+1 belief vectors and costs N (T+1) agent calls per replica (round 0 once if share_initial).
Replicas share the graph, the evidence and (by default) the round-0 answers, and differ only in the agents'
sampling (theory/claim2.md section 5).

Backends take a list of envs.prompts.AgentContext and return one Response each:
    LLMBackend   renders the context with envs/prompts.py and runs it through engine.Runner (realtime or
                 Message Batches), which checkpoints every agent call and never re-spends on restart.
    MockBackend  the mock Ising agent of solver_spec.md section 0 on the numeric field of the same inbox,
                 h_i = b_+ sum_{agree} s_j - b_- sum_{disagree} s_j (+ theta),
                 s_i = tanh(beta h_i + h_priv_i + kappa u_i), or +-1 with P(+1) = (1 + that) / 2,
                 where u_i is the exponential average (decay lam) of the agent's own shown history
                 (theory/checks/toy.py). With b_+ = b_- = b this is h = sum_j (b J_ij) s_j exactly.

Checkpointing: after each completed round of each replica, one line of results/<experiment_id>/swarm_rounds.jsonl
holds the state the dynamics carries forward. A restart rebuilds the state from it and continues; with the
LLM backend, a crash mid-round also replays the finished agent calls from the Runner's checkpoints.

The LLM backend keys Runner checkpoints by (experiment_id, replica, round, agent), so give every
(graph, game) run its own Runner experiment_id. Peers' probabilities are shown to the model with two
decimals; the mock reads them unrounded.

Missing replies (refusal, malformed twice): the belief is NaN in the output and ``missing`` is True, but the
dynamics carries the agent's previous message forward (p = 0.5 and an empty message at round 0).
"""

from __future__ import annotations

import asyncio
import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from engine.ledger import ROOT, append_jsonl, read_jsonl
from envs.evidence_game import Game
from envs.graphs import spectral_log
from envs.prompts import AgentContext, InboxMsg, make_request


@dataclass
class Response:
    prob: float | None
    message: str | None
    status: str = "ok"            # "ok" | "missing"
    reason: str | None = None     # missing_reason from engine.Runner (e.g. "refusal")
    cost_usd: float = 0.0
    record: dict | None = None    # full Runner checkpoint record (LLM backend)


def _stable_int(x) -> int:
    return zlib.crc32(str(x).encode())


class MockBackend:
    """Mock Ising agent on the numeric field of the inbox (module docstring). Zero cost, deterministic in
    (seed, round_t, agent_id) for the stochastic variant, so resumed runs reproduce uninterrupted ones."""

    def __init__(self, beta: float, b_plus: float = 1.0, b_minus: float = 1.0, kappa: float = 0.0,
                 lam: float = 0.0, stochastic: bool = False, seed: int = 0):
        self.beta, self.b_plus, self.b_minus = beta, b_plus, b_minus
        self.kappa, self.lam, self.stochastic, self.seed = kappa, lam, stochastic, seed

    def field(self, ctx: AgentContext) -> float:
        """h = b_+ sum_agree s_j - b_- sum_disagree s_j + theta, with s_j = 2 p_j - 1."""
        return (self.b_plus * sum(2 * m.prob - 1 for m in ctx.agree)
                - self.b_minus * sum(2 * m.prob - 1 for m in ctx.disagree) + ctx.theta)

    def memory(self, ctx: AgentContext) -> float:
        """u = lam u + (1 - lam) s over the shown own history, oldest first (u = last s when lam = 0)."""
        u = 0.0
        for p, _ in ctx.history:
            u = self.lam * u + (1 - self.lam) * (2 * p - 1)
        return u

    def mean_s(self, ctx: AgentContext) -> float:
        return float(np.tanh(self.beta * self.field(ctx) + ctx.h_priv + self.kappa * self.memory(ctx)))

    async def respond(self, seed, round_t: int, contexts: list[AgentContext], H: int) -> list[Response]:
        out = []
        for ctx in contexts:
            s = self.mean_s(ctx)
            if self.stochastic:
                u = np.random.default_rng([self.seed, _stable_int(seed), round_t, _stable_int(ctx.agent_id)]).random()
                s = 1.0 if u < (1 + s) / 2 else -1.0
            p = (1 + s) / 2
            out.append(Response(p, f"My estimate for HEADS is {p:.2f}."))
        return out


class LLMBackend:
    """Renders contexts with envs/prompts.py and runs them through engine.Runner. ``seed`` and ``round_t``
    become the Runner checkpoint key together with each context's agent_id."""

    def __init__(self, runner, batch: bool = False, poll_s: float = 15.0):
        self.runner, self.batch, self.poll_s = runner, batch, poll_s

    async def respond(self, seed, round_t: int, contexts: list[AgentContext], H: int) -> list[Response]:
        reqs = [make_request(c, H) for c in contexts]
        if self.batch:
            recs = await self.runner.run_round_batch(seed, round_t, reqs, poll_s=self.poll_s)
        else:
            recs = await self.runner.run_round(seed, round_t, reqs)
        out = []
        for c in contexts:
            r = recs[c.agent_id]
            ok = r["status"] == "ok"
            out.append(Response(r["calibrated_belief_prob"] if ok else None, r["public_message"] if ok else None,
                                r["status"], r["missing_reason"], r["cost_usd"], r))
        return out


@dataclass
class SwarmConfig:
    experiment_id: str
    T: int
    H: int = 1                   # own past answers shown to each agent (0 = memoryless)
    replicas: int = 2
    mode: str = "sync"           # "sync" | "async"
    seed: int = 0                # inbox shuffles and async orders
    share_initial: bool = True   # replicas share the round-0 answers (computed once, on replica 0)
    results_root: Path = field(default_factory=lambda: ROOT / "results")


class _ReplicaState:
    def __init__(self, N: int):
        self.p = np.full(N, 0.5)          # carried state the peers read
        self.msg = [""] * N
        self.hist: list[list[tuple[float, str]]] = [[] for _ in range(N)]
        self.rows_p: list[list] = []      # recorded beliefs per round (None = missing)
        self.rows_missing: list[list] = []

    def apply(self, i: int, r: Response):
        if r.status == "ok":
            self.p[i], self.msg[i] = r.prob, r.message
        self.hist[i].append((float(self.p[i]), self.msg[i]))

    def close_round(self, got: dict[int, Response]):
        N = len(self.p)
        self.rows_p.append([got[i].prob if got[i].status == "ok" else None for i in range(N)])
        self.rows_missing.append([got[i].status != "ok" for i in range(N)])


class Swarm:
    def __init__(self, J: np.ndarray, game: Game, cfg: SwarmConfig, backend):
        J = np.asarray(J)
        if not set(np.unique(J)) <= {-1, 0, 1}:
            raise ValueError("J must be a sign matrix in {-1, 0, +1}; gains belong to the backend")
        if J.shape != (game.N, game.N):
            raise ValueError(f"J is {J.shape}, game has N={game.N}")
        if cfg.mode not in ("sync", "async"):
            raise ValueError(cfg.mode)
        if cfg.mode == "async" and getattr(backend, "batch", False):
            raise ValueError("async updates need one call at a time; use the realtime LLM backend")
        self.J, self.game, self.cfg, self.backend = J, game, cfg, backend
        self.N = game.N
        self.agree = [np.flatnonzero(J[i] > 0) for i in range(self.N)]
        self.disagree = [np.flatnonzero(J[i] < 0) for i in range(self.N)]
        self.dir = Path(cfg.results_root) / cfg.experiment_id
        self.ckpt = self.dir / "swarm_rounds.jsonl"
        self.evidence_text = [e.text() for e in game.evidence]
        self.h_priv = game.h_priv

    # ---------- contexts ----------

    def _rng(self, *key) -> np.random.Generator:
        return np.random.default_rng([self.cfg.seed, *key])

    def _context(self, st: _ReplicaState, rep: int, t: int, i: int) -> AgentContext:
        agree, disagree = (), ()
        if t > 0:
            # inbox order shuffled per (replica, round, receiver), as in _build_inboxes of
            # third_party/physics-of-agents/lib/datagen/dynamics.py (Apache-2.0)
            rng = self._rng(rep, t, i)
            agree = tuple(InboxMsg(float(st.p[j]), st.msg[j], int(j)) for j in rng.permutation(self.agree[i]))
            disagree = tuple(InboxMsg(float(st.p[j]), st.msg[j], int(j)) for j in rng.permutation(self.disagree[i]))
        hist = tuple(st.hist[i][-self.cfg.H:]) if self.cfg.H > 0 else ()
        return AgentContext(agent_id=i, session=rep, round_t=t, h_priv=float(self.h_priv[i]),
                            evidence=self.evidence_text[i], history=hist, agree=agree, disagree=disagree)

    # ---------- one round ----------

    async def _round_sync(self, st: _ReplicaState, rep: int, t: int) -> dict[int, Response]:
        ctxs = [self._context(st, rep, t, i) for i in range(self.N)]   # all read the round-(t-1) state
        res = await self.backend.respond(rep, t, ctxs, self.cfg.H)
        got = dict(zip(range(self.N), res))
        for i in range(self.N):
            st.apply(i, got[i])
        return got

    async def _round_async(self, st: _ReplicaState, rep: int, t: int) -> dict[int, Response]:
        got = {}
        for i in self._rng(rep, t, 2**31 - 1, 0).permutation(self.N):          # this round's update order
            (r,) = await self.backend.respond(rep, t, [self._context(st, rep, t, int(i))], self.cfg.H)
            got[int(i)] = r
            st.apply(int(i), r)                                         # visible to later agents this round
        return got

    # ---------- checkpoint ----------

    def _save_round(self, rep: int, t: int, st: _ReplicaState):
        append_jsonl(self.ckpt, {"experiment_id": self.cfg.experiment_id, "replica": rep, "round_t": t,
                                 "p_recorded": st.rows_p[-1], "missing": st.rows_missing[-1],
                                 "p_carried": st.p.tolist(), "msg_carried": st.msg})

    def _load(self) -> dict[tuple[int, int], dict]:
        return {(r["replica"], r["round_t"]): r for r in read_jsonl(self.ckpt)}

    def _restore(self, rows: dict, rep: int) -> tuple[_ReplicaState, int]:
        """Rebuild a replica's state from its consecutive saved rounds; returns (state, next round)."""
        st = _ReplicaState(self.N)
        t = 0
        while (rep, t) in rows:
            r = rows[(rep, t)]
            st.p, st.msg = np.array(r["p_carried"], float), list(r["msg_carried"])
            for i in range(self.N):
                st.hist[i].append((float(st.p[i]), st.msg[i]))
            st.rows_p.append(r["p_recorded"])
            st.rows_missing.append(r["missing"])
            t += 1
        return st, t

    def _write_meta(self):
        meta = self.dir / "swarm_meta.json"
        if meta.exists():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        log = spectral_log(self.J, eigs=self.N <= 300)
        log.pop("eigvals", None)
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        meta.write_text(json.dumps({"config": cfg, "graph": log, "backend": type(self.backend).__name__,
                                    "backend_params": {k: v for k, v in vars(self.backend).items()
                                                       if isinstance(v, (int, float, bool, str))},
                                    "game": {"truth": self.game.truth, "mu": self.game.mu, "sigma": self.game.sigma,
                                             "k": self.game.k, "seed": self.game.seed},
                                    "h_priv": self.h_priv.tolist()}, indent=1, default=float))

    # ---------- run ----------

    async def _run_replica(self, rep: int, rows: dict, initial: dict | None) -> _ReplicaState:
        st, t0 = self._restore(rows, rep)
        for t in range(t0, self.cfg.T + 1):
            if t == 0 and initial is not None:                   # shared round-0 answers
                got = initial
                for i in range(self.N):
                    st.apply(i, got[i])
            elif t == 0 or self.cfg.mode == "sync":
                got = await self._round_sync(st, rep, t)
            else:
                got = await self._round_async(st, rep, t)
            st.close_round(got)
            self._save_round(rep, t, st)
        return st

    async def run(self) -> dict:
        """Run (or resume) all replicas. Returns p and s = 2p - 1 shaped (replicas, T+1, N) with NaN for
        missing replies, the missing mask, and the realized h_priv."""
        self._write_meta()
        rows = self._load()
        R = self.cfg.replicas
        initial = None
        if self.cfg.share_initial:
            st0, _ = self._restore(rows, 0)
            if st0.rows_p:                                      # replica 0 already has its round 0
                initial = {i: Response(st0.rows_p[0][i], st0.hist[i][0][1],
                                       "missing" if st0.rows_missing[0][i] else "ok") for i in range(self.N)}
            else:
                st = _ReplicaState(self.N)
                got = await self._round_sync(st, 0, 0)
                st.close_round(got)
                self._save_round(0, 0, st)
                rows = self._load()
                initial = got
        states = await asyncio.gather(*(self._run_replica(r, rows, initial if r > 0 else None) for r in range(R)))
        p = np.array([[[np.nan if x is None else x for x in row] for row in st.rows_p] for st in states], float)
        missing = np.array([st.rows_missing for st in states], bool)
        out = {"p": p, "s": 2 * p - 1, "missing": missing, "h_priv": self.h_priv}
        np.savez(self.dir / "beliefs.npz", **out)
        return out


def run_swarm(J, game: Game, cfg: SwarmConfig, backend) -> dict:
    return asyncio.run(Swarm(J, game, cfg, backend).run())
