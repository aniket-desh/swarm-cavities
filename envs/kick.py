"""Kick twins of a finished synchronous swarm run: perturb one agent's outgoing message once and replay.

A twin restores replica `rep` of a finished run (results/<experiment_id>/swarm_rounds.jsonl) to the state after
round t0, replaces kicked agent i's carried message, as its readers will see it in round t0+1, by the mirrored
message (p -> 1 - p, HEADS <-> TAILS in the text), and runs rounds t0+1 .. t0+n_rounds again. Agent i's own
history keeps its true round-t0 answer: only what others read is altered.

Consequences that the analysis relies on (theory/quiet_regime.md):
  * every agent whose prompt is unchanged sends byte-identical request params, so engine.Runner returns the base
    run's reply from the replay cache (identical, free, no ledger line); only agents downstream of the kick are
    resampled;
  * round t0+1: i itself is unchanged (its inbox holds its in-neighbours' round-t0 messages); its readers j move,
    giving the first-order responses r(j <- i);
  * round t0+2: i's inbox holds its in-neighbours' round-(t0+1) messages, which changed only for in-neighbours that
    also read i. So i's change at t0+2 is the echo through reciprocated ties, exactly zero when i has none
    (gamma = 0), with sign set by the reciprocated ties' signs and the channel gains.

Checkpoint keys: one Runner (its own experiment_id, e.g. "<base>-kick") holds every twin of a base run, with the
backend seed "r<rep>|t<t0>|k<i>" keeping twins apart, while the agents' session tag stays `rep` so prompts match
the base run's.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from engine.ledger import read_jsonl
from envs.swarm import Swarm, _ReplicaState

_SWAP = {"HEADS": "TAILS", "TAILS": "HEADS", "heads": "tails", "tails": "heads", "Heads": "Tails", "Tails": "Heads"}


def mirror(p: float, text: str) -> tuple[float, str]:
    """The same message about the other side of the coin: p -> 1 - p (two decimals, as displayed) and
    HEADS <-> TAILS in the text."""
    return round(1.0 - float(p), 2), re.sub(r"\b(HEADS|TAILS|heads|tails|Heads|Tails)\b",
                                           lambda m: _SWAP[m.group(0)], text)


def base_rows(base_dir: Path) -> dict[tuple[int, int], dict]:
    return {(r["replica"], r["round_t"]): r for r in read_jsonl(Path(base_dir) / "swarm_rounds.jsonl")}


def restore_state(rows: dict, rep: int, t0: int, N: int) -> _ReplicaState:
    """Replica `rep` after round t0: carried (p, msg) and every agent's own history up to t0, built exactly as
    envs.swarm.Swarm._restore does, but stopping at t0."""
    st = _ReplicaState(N)
    for t in range(t0 + 1):
        r = rows[(rep, t)]
        st.p, st.msg = np.array(r["p_carried"], float), list(r["msg_carried"])
        for i in range(N):
            st.hist[i].append((float(st.p[i]), st.msg[i]))
    return st


def compatible_kicks(J: np.ndarray, candidates, n_max: int) -> list[int]:
    """Greedy set of agents that can be kicked in the same twin without contaminating each other's first-order
    responses or echoes over two rounds: for kicked i != k, no agent reads both, i does not read anyone who reads
    k, and neither reads the other. (J[a, b] != 0 means a reads b.)"""
    R = J != 0
    readers = [set(np.flatnonzero(R[:, i])) for i in range(len(J))]
    reads = [set(np.flatnonzero(R[i])) for i in range(len(J))]
    chosen: list[int] = []
    for i in candidates:
        if all(not (readers[i] & readers[k]) and not (reads[i] & readers[k]) and not (reads[k] & readers[i])
               and k not in readers[i] and i not in readers[k] for k in chosen):
            chosen.append(int(i))
        if len(chosen) == n_max:
            break
    return chosen


async def kick_twin(sw: Swarm, rows: dict, rep: int, t0: int, kicked: list[int], n_rounds: int = 2,
                    sham_backend=None) -> dict:
    """Run one twin. `sw` is a Swarm on the base run's J, game and config (same seed and H, mode "sync") whose
    backend writes to the twin's Runner. Returns, for rounds t0+1 .. t0+n_rounds, base and twin carried beliefs
    (n_rounds, N), the twin's missing mask, which agents were sent to the model rather than replayed, and the
    kick itself (p before and after).

    Sham twin (`sham_backend` given, a backend whose Runner has its own replay salt): no message is changed, but at
    round t0+1 the readers of the kicked agents are sent through `sham_backend`, so they draw fresh replies to
    prompts identical to the base run's; later rounds propagate exactly as in a real twin. Regressed on the kick the
    agent would have received, a sham twin's changes measure what resampling alone contributes to the slopes (it
    must be zero within noise). The returned kick is the hypothetical one."""
    if sw.cfg.mode != "sync":
        raise ValueError("kick twins are defined for synchronous runs")
    N = sw.N
    st = restore_state(rows, rep, t0, N)
    kick = {}
    for i in kicked:
        p_new, m_new = mirror(st.p[i], st.msg[i])
        kick[int(i)] = (float(st.p[i]), p_new)
        if sham_backend is None:
            st.p[i], st.msg[i] = p_new, m_new        # what readers see in round t0+1; hist[i] keeps the truth
    sham_set = {int(j) for i in kicked for j in np.flatnonzero(sw.J[:, i])} if sham_backend is not None else set()
    key = f"r{rep}|t{t0}|k{'-'.join(map(str, kicked))}" + ("|sham" if sham_backend is not None else "")
    base_p = np.array([rows[(rep, t)]["p_carried"] for t in range(t0 + 1, t0 + n_rounds + 1)], float)
    twin_p = np.zeros((n_rounds, N))
    missing = np.zeros((n_rounds, N), bool)
    sent = np.zeros((n_rounds, N), bool)
    for k, t in enumerate(range(t0 + 1, t0 + n_rounds + 1)):
        ctxs = [sw._context(st, rep, t, i) for i in range(N)]
        if k == 0 and sham_set:
            fresh = sorted(sham_set)
            rest = [i for i in range(N) if i not in sham_set]
            got = dict(zip(fresh, await sham_backend.respond(key, t, [ctxs[i] for i in fresh], sw.cfg.H)))
            got.update(zip(rest, await sw.backend.respond(key, t, [ctxs[i] for i in rest], sw.cfg.H)))
            res = [got[i] for i in range(N)]
        else:
            res = await sw.backend.respond(key, t, ctxs, sw.cfg.H)
        for i, r in enumerate(res):
            st.apply(i, r)
            missing[k, i] = r.status != "ok"
            rec = r.record or {}
            sent[k, i] = any(not a.get("replayed", True) for a in rec.get("attempts", [])) if rec else True
        twin_p[k] = st.p
    return {"rep": rep, "t0": t0, "kicked": [int(i) for i in kicked], "kick": kick, "sham": sham_backend is not None,
            "base_p": base_p,
            "twin_p": twin_p, "missing": missing, "sent": sent}
