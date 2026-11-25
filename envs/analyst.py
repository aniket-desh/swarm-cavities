"""Analyst scenario (theory/analyst_scenario.md): fictional-asset BUY/SELL desk, prompts, a minimal synchronous swarm
loop, a mock backend that implements the law (A1)-(A3), and a vLLM backend that reads P(BUY) from the decision
token's logprobs.

Story. A desk of analysts covers one fictional asset. Every round each analyst answers the same question, "will the
asset close up this week?", by committing to BUY or SELL plus a short note to the analysts who read it. It holds a
standing research note (quenched field h0) and gets one fresh news item per round (white field xi, known to the
experimenter), and from round 1 on reads the latest call and note of K colleagues: allies ("found reliable",
J_ij = +1) and faded colleagues ("usually wrong; lean against them", J_ij = -1). It also sees its own previous call
and note.

Fields and their rendering (the known log-odds mapping, as in envs/evidence_game.py). The theory works in half
log-odds x, P(BUY) = (1 + tanh x) / 2, so a report that is right with probability rho carries full log-odds
logit(rho) = 2 x. With nominal amplitudes nu_nom (news) and a0_nom (note):

    news item:      side = sign xi,  displayed reliability rho = round(100 sigmoid(2 nu_nom |xi|)) % in [51, 95]
                    xi_real = side * logit(rho) / (2 nu_nom)                                                  (N1)
    research note:  evidence_for_field(2 a0_nom h0, k) of envs/evidence_game (k indicators whose displayed
                    log-odds sum to 2 a0_nom h0 up to rounding),  h0_real = h_priv / (2 a0_nom)               (N2)

so a Bayesian reader who believes the displayed reliabilities has x = nu_nom xi_real + a0_nom h0_real + (inbox).
xi, h0 ~ N(0, 1); the realized values (after rounding and clipping) are what the analysis regresses on.

Decision and belief. The assistant turn is forced to start with the JSON prefix DECISION_PREFIX = '{"call": "'.
With Qwen3.5's tokenizer "BUY" and "SELL" are single tokens right after it (checked in experiments/08_analyst_
calibration), so one forward pass gives

    valid_mass = P(BUY) + P(SELL),   P_buy = P(BUY) / valid_mass,   x = (log P(BUY) - log P(SELL)) / 2         (D1)

recorded missing when valid_mass < 0.95 (CLAUDE.md guardrail 4: never guessed). P_buy is a deterministic function
of the prompt, so the belief s = 2 P_buy - 1 carries no sampling noise. If one of the two tokens falls outside the
returned top-k logprobs, its probability is below the smallest returned one; x is then censored at that bound and
flagged. The call is sampled at temperature 1 from P_buy with a stored uniform per (replica, round, analyst):

    sigma = +1 (BUY) iff u < P_buy                                                                           (D2)

so base runs and kick twins share draws (analyst_scenario.md section 5). The note is then generated conditional on
the sampled call (prefix + call + '", "note": "', constrained to a quote-free string and '"}').

Guardrail 4: no prompt text calls any field or text private, hidden or secret; the research note is "your own
research note". tests/test_analyst.py checks every wording.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine.ledger import ROOT, append_jsonl, read_jsonl
from envs.evidence_game import evidence_for_field, logit
from envs.prompts import FORBIDDEN_WORDS  # noqa: F401  (re-exported for tests)

ASSET = "Orsilane Freight"           # invented name; no real ticker or firm
RHO_MIN, RHO_MAX = 51, 95            # displayed reliability range in percent (as envs/evidence_game.py)
DECISION_PREFIX = '{"call": "'
NOTE_PREFIX = '", "note": "'
NOTE_REGEX = r'[^"\\\n]{1,400}"\}'   # the note: a quote-free one-line string, then the closing '"}'
_NOTE_PARSE = re.compile(r'\A([^"\\\n]{1,400})"\}\Z')
CALLS = {+1: "BUY", -1: "SELL"}

# Trust-intensity wordings, mild to strong; one sentence per group. They scale a_+ and a_- together.
WORDINGS = {
    "mild": ("colleagues whose calls you have found somewhat reliable.",
             "colleagues whose calls have been wrong somewhat more often than right; you may lean a little "
             "against them."),
    "moderate": ("colleagues you have found reliable; give their calls real weight.",
                 "colleagues who are usually wrong; lean against their calls."),
    "strong": ("colleagues you have found highly reliable; lean on their calls heavily.",
               "colleagues who are usually wrong; fade their calls firmly: their BUY is a reason to SELL, and "
               "their SELL is a reason to BUY."),
    "max": ("colleagues who have been right far more often than anyone else on the desk, including you; follow "
            "their calls very heavily.",
            "colleagues who are almost always wrong; treat each of their calls as strong evidence for the "
            "opposite call."),
}

TEMPLATE_NOTES = ("{C}.", "My call is {C}.", "I am calling {C} this round.", "{C} from me this round.",
                  "Going with {C}.")


def template_note(call: str, rng: np.random.Generator) -> str:
    """A neutral note that carries the call and nothing else (the template arm of the calibration)."""
    return str(rng.choice(TEMPLATE_NOTES)).format(C=call)


# ---------------------------------------------------------------- fields -> text

def render_report(L_full: float) -> tuple[str, int, float]:
    """(side, displayed percent, realized full log-odds) of one report with target full log-odds L_full."""
    side = "UP" if L_full >= 0 else "DOWN"
    pct = int(np.clip(round(100 / (1 + np.exp(-abs(L_full)))), RHO_MIN, RHO_MAX))
    return side, pct, (1.0 if side == "UP" else -1.0) * float(logit(pct / 100))


def news_item(xi: float, nu_nom: float) -> tuple[str, int, float]:
    """(N1): side, percent, xi_real."""
    side, pct, L = render_report(2 * nu_nom * xi)
    return side, pct, L / (2 * nu_nom)


@dataclass(frozen=True)
class ResearchNote:
    indicators: tuple[tuple[str, int], ...]    # (side UP/DOWN, displayed percent)
    h0: float                                  # drawn field
    h0_real: float                             # (N2), from the displayed numbers

    def text(self, asset: str = ASSET) -> str:
        lines = [f"- Indicator {j + 1} (right {p}% of the time) points {s}." for j, (s, p) in
                 enumerate(self.indicators)]
        return f"Your own research note on {asset} (it stays the same all week):\n" + "\n".join(lines)


def research_note(h0: float, a0_nom: float, rng: np.random.Generator, k: int = 3) -> ResearchNote:
    """(N2) through envs.evidence_game.evidence_for_field; HEADS/TAILS become UP/DOWN."""
    ev = evidence_for_field(2 * a0_nom * h0, k, rng)
    ind = tuple(("UP" if c.side == "HEADS" else "DOWN", c.reliability) for c in ev.clues)
    return ResearchNote(ind, float(h0), ev.h_priv / (2 * a0_nom))


@dataclass
class AnalystGame:
    """Everything quenched or pre-drawn for one swarm: notes, news (shared by replicas, as in the theory), call
    uniforms (one per replica, round, analyst)."""
    N: int
    T: int
    R: int
    seed: int
    nu_nom: float
    a0_nom: float
    asset: str
    notes: list[ResearchNote]
    xi: np.ndarray          # (T+1, N) drawn news fields
    xi_real: np.ndarray     # (T+1, N) realized (N1)
    news_side: np.ndarray   # (T+1, N) "UP"/"DOWN"
    news_pct: np.ndarray    # (T+1, N)
    U: np.ndarray           # (T+1, R, N) stored call uniforms (D2)

    @property
    def h0(self) -> np.ndarray:
        return np.array([n.h0 for n in self.notes])

    @property
    def h0_real(self) -> np.ndarray:
        return np.array([n.h0_real for n in self.notes])


def make_game(N: int, T: int, R: int = 1, seed: int = 0, nu_nom: float = 0.5, a0_nom: float = 0.3, k_note: int = 3,
              asset: str = ASSET) -> AnalystGame:
    """Deterministic in seed. h0_i, xi_i(t) ~ N(0, 1) i.i.d.; U ~ U(0, 1)."""
    rng = np.random.default_rng([seed, 0xA11A])
    h0 = rng.standard_normal(N)
    notes = [research_note(h, a0_nom, rng, k_note) for h in h0]
    xi = rng.standard_normal((T + 1, N))
    U = rng.random((T + 1, R, N))
    side = np.empty((T + 1, N), dtype=object)
    pct = np.empty((T + 1, N), dtype=int)
    xr = np.empty((T + 1, N))
    for t in range(T + 1):
        for i in range(N):
            side[t, i], pct[t, i], xr[t, i] = news_item(xi[t, i], nu_nom)
    return AnalystGame(N, T, R, seed, nu_nom, a0_nom, asset, notes, xi, xr, side, pct, U)


# ---------------------------------------------------------------- prompts

@dataclass(frozen=True)
class AnalystContext:
    """One analyst update. The text backends render it; the mock reads the numbers in meta."""
    agent_id: int | str
    session: int | str                      # replica / probe tag shown to the model (makes requests distinct)
    round_t: int                            # 0 = first round: no inbox, no own call
    note_text: str                          # ResearchNote.text()
    news_side: str
    news_pct: int
    own: tuple[str, str] | None = None      # own previous (call, note)
    allies: tuple[tuple[str, str], ...] = ()
    faded: tuple[tuple[str, str], ...] = ()
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def rules(wording: str, K: int = 16, asset: str = ASSET) -> str:
    ally, fade = WORDINGS[wording]
    return (
        "You are one of the analysts on a research desk in a trading simulation. Everything in it is fictional: "
        "the asset, the desk, the colleagues and the data.\n\n"
        "The question\n"
        f"- The desk covers one fictional asset, {asset}. The question for everyone is whether {asset} will close "
        "up this week, that is, higher at Friday's close than at Monday's open.\n"
        "- The desk makes its calls over several rounds during the week. In every round each analyst commits to a "
        f"call: BUY if they think {asset} will close up this week, SELL if they think it will close down.\n\n"
        "Your information\n"
        "- Your own research note: a few indicators you prepared before the week started. Each points UP or DOWN "
        "and is right with the probability shown next to it. The note stays the same all week.\n"
        "- One fresh news item each round, with how often items of that kind have been right in past weeks. You "
        "get a new item every round; the other analysts get their own items.\n"
        "- All indicators and news items, yours and the other analysts', are independent of each other.\n"
        f"- From the second round on, you see the latest call and note of {K} colleagues, in two groups:\n"
        f"  - Allies: {ally}\n"
        f"  - Faded colleagues: {fade}\n"
        "  Which colleagues fall in which group stays the same all week. A group can be empty.\n"
        "- You also see your own call and note from the previous round.\n\n"
        "How to answer\n"
        '- call: "BUY" or "SELL".\n'
        "- note: one sentence, at most 25 words, for the analysts who read your calls. They see it next round "
        "together with your call.\n"
        "- Weigh your research note, your news and your colleagues' calls in whatever way you judge best. You may "
        "change your call from round to round.\n"
        '- Reply with one JSON object and nothing else: {"call": "BUY" or "SELL", "note": "your note"}'
    )


def _fmt_calls(msgs) -> str:
    if not msgs:
        return "(none)"
    return "\n".join(f'- {c}. Note: "{n.strip()}"' for c, n in msgs)


def dynamic(ctx: AnalystContext) -> str:
    """The user turn for one analyst and round."""
    parts = [f"Desk session {ctx.session}, round {ctx.round_t + 1}.",
             "Your news item this round:\n"
             f"- A news item that has been right {ctx.news_pct}% of the time in past weeks points {ctx.news_side}."]
    if ctx.own is None:
        parts.append("This is the first round. You have no earlier call, and there are no colleague calls yet.")
    else:
        c, n = ctx.own
        parts.append(f'Your call last round: {c}. Your note: "{n.strip()}"')
        parts.append("Allies' latest calls:\n" + _fmt_calls(ctx.allies))
        parts.append("Faded colleagues' latest calls:\n" + _fmt_calls(ctx.faded))
    parts.append(f"Make your call for round {ctx.round_t + 1}.")
    return "\n\n".join(parts)


def render_messages(ctx: AnalystContext, wording: str, K: int = 16, asset: str = ASSET) -> list[dict]:
    return [{"role": "system", "content": f"{rules(wording, K, asset)}\n\n{ctx.note_text}"},
            {"role": "user", "content": dynamic(ctx)}]


# ---------------------------------------------------------------- logprobs -> belief, call

def decision_belief(lp: dict[int, float], buy_id: int, sell_id: int, min_valid_mass: float = 0.95) -> dict:
    """(D1) from the top-k next-token logprobs after DECISION_PREFIX (token id -> logprob).

    A token missing from the top-k has probability below the smallest returned one (lp_min): then P_buy is computed
    with that token at 0 and x is set to the bound (lp_buy - lp_min)/2 or (lp_min - lp_sell)/2, flagged censored."""
    lb, ls = lp.get(buy_id), lp.get(sell_id)
    lp_min = min(lp.values())
    pb = float(np.exp(lb)) if lb is not None else 0.0
    ps = float(np.exp(ls)) if ls is not None else 0.0
    valid = pb + ps
    out = {"lp_buy": lb, "lp_sell": ls, "lp_min": lp_min, "valid_mass": valid, "censored": None}
    if valid <= 0:
        return {**out, "p_buy": float("nan"), "x": float("nan"), "ok": False}
    if lb is not None and ls is not None:
        x = (lb - ls) / 2
    elif lb is not None:
        x, out["censored"] = (lb - lp_min) / 2, "sell_below_topk"      # true x is larger
    else:
        x, out["censored"] = (lp_min - ls) / 2, "buy_below_topk"       # true x is smaller
    return {**out, "p_buy": pb / valid, "x": float(x), "ok": valid >= min_valid_mass}


def sample_call(p_buy, u):
    """(D2): +1 (BUY) iff u < P_buy. Works on arrays."""
    return np.where(np.asarray(u) < np.asarray(p_buy), 1, -1)


def parse_note(text: str) -> str | None:
    m = _NOTE_PARSE.match(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- backends

class MockAnalyst:
    """The law (A1)-(A2) on the numbers in ctx.meta (h0, xi, own_sigma) and the calls in the inbox; template notes.
    Matches theory/checks/analyst_lib.simulate (sync) exactly when fed the same fields and uniforms."""

    def __init__(self, a_plus, a_minus, nu=0.5, a0=0.0, c=0.0, kappa=0.0, K=16, real_fields=False):
        self.a_plus, self.a_minus, self.nu, self.a0, self.c, self.kappa, self.K = a_plus, a_minus, nu, a0, c, kappa, K
        self.real_fields = real_fields                  # use h0_real / xi_real instead of the drawn fields

    def x(self, ctx: AnalystContext) -> float:
        m = ctx.meta
        h0, xi = (m["h0_real"], m["xi_real"]) if self.real_fields else (m["h0"], m["xi"])
        fA = sum(1 if c == "BUY" else -1 for c, _ in ctx.allies) / self.K
        fD = sum(1 if c == "BUY" else -1 for c, _ in ctx.faded) / self.K
        own = 0.0 if ctx.own is None else (1.0 if ctx.own[0] == "BUY" else -1.0)
        return self.a_plus * fA - self.a_minus * fD + self.a0 * h0 + self.nu * xi + self.c + self.kappa * own

    def beliefs(self, ctxs, key=None) -> list[dict]:
        out = []
        for ctx in ctxs:
            x = self.x(ctx)
            out.append({"p_buy": (1 + np.tanh(x)) / 2, "x": x, "ok": True, "valid_mass": 1.0, "censored": None})
        return out

    def notes(self, ctxs, calls, key=None) -> list[dict]:
        return [{"note": f"My call is {c}.", "note_status": "ok"} for c in calls]


def _stable_seed(*key) -> int:
    from envs.vllm_backend import _stable_seed as s
    return s(*key)


class AnalystVLLM:
    """Qwen on vLLM (envs/vllm_backend.make_llm). One instance per wording; instances may share one llm.

    beliefs(ctxs): one engine call, max_tokens=1 with top-`logprobs` logprobs, on chat prompt + DECISION_PREFIX.
    notes(ctxs, calls): one engine call (plus one retry for samples that hit max_tokens), temperature 1, on
    chat prompt + DECISION_PREFIX + call + NOTE_PREFIX, constrained by NOTE_REGEX.
    ``params`` overrides (SamplingParams, structured-output constructor), for tests."""

    def __init__(self, llm, wording: str, tag: str, K: int = 16, asset: str = ASSET, logprobs: int = 20,
                 min_valid_mass: float = 0.95, note_max_tokens: int = 100, params=None):
        from envs.vllm_backend import _vllm_params
        self.llm, self.wording, self.tag, self.K, self.asset = llm, wording, tag, K, asset
        self.logprobs, self.min_valid_mass, self.note_max_tokens = logprobs, min_valid_mass, note_max_tokens
        self.tok = llm.get_tokenizer()
        self.SP, self._so = params or _vllm_params()
        self.stop_ids = set(getattr(self.tok, "all_special_ids", []) or [])
        if getattr(self.tok, "eos_token_id", None) is not None:
            self.stop_ids.add(self.tok.eos_token_id)
        self.buy_id = self.sell_id = None
        self.note_prefix_ids = None

    def chat_text(self, ctx: AnalystContext) -> str:
        return self.tok.apply_chat_template(render_messages(ctx, self.wording, self.K, self.asset), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def _check_tokens(self, chat: str):
        """BUY and SELL must each be exactly one token after the prefix, and the decision prompt must tokenize as
        the chat prompt + prefix with no merge across the boundary that changes the call token."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        base = enc(chat + DECISION_PREFIX)
        ids = {}
        for w in ("BUY", "SELL"):
            full = enc(chat + DECISION_PREFIX + w)
            if full[:len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(f"{w!r} is not a single token after the decision prefix: {full[len(base) - 2:]}")
            ids[w] = full[-1]
        self.buy_id, self.sell_id = ids["BUY"], ids["SELL"]
        self.note_prefix_ids = {}
        for w in ("BUY", "SELL"):
            full = enc(chat + DECISION_PREFIX + w + NOTE_PREFIX)
            if full[:len(base) + 1] != base + [ids[w]]:
                raise ValueError(f"the note prefix merges with the call token {w!r}")
            self.note_prefix_ids[w] = full[len(base) + 1:]

    def decision_ids(self, ctx: AnalystContext) -> list[int]:
        chat = self.chat_text(ctx)
        if self.buy_id is None:
            self._check_tokens(chat)
        return self.tok.encode(chat + DECISION_PREFIX, add_special_tokens=False)

    def _gen(self, prompts, params):
        return self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)

    def beliefs(self, ctxs, key=None) -> list[dict]:
        t0 = time.time()
        ids = [self.decision_ids(c) for c in ctxs]
        sp = self.SP(max_tokens=1, temperature=1.0, top_p=1.0, logprobs=self.logprobs, seed=0)
        out = []
        for p, o in zip(ids, self._gen(ids, sp)):
            lp = {int(t): float(v.logprob) for t, v in o.outputs[0].logprobs[0].items()}
            b = decision_belief(lp, self.buy_id, self.sell_id, self.min_valid_mass)
            top_other = sorted(((v, t) for t, v in lp.items() if t not in (self.buy_id, self.sell_id)), reverse=True)[:3]
            b["top_other"] = [[self.tok.decode([t]), round(float(np.exp(v)), 6)] for v, t in top_other]
            b["n_prompt_tokens"] = len(p)
            out.append(b)
        self.last_belief_s = time.time() - t0
        return out

    def notes(self, ctxs, calls, key=()) -> list[dict]:
        """calls[i] in {"BUY", "SELL"}; key makes the sampling seeds (tag, *key, agent, attempt)."""
        t0 = time.time()
        prompts = [self.decision_ids(c) + [self.buy_id if k == "BUY" else self.sell_id] + self.note_prefix_ids[k]
                   for c, k in zip(ctxs, calls)]
        res: list[dict | None] = [None] * len(ctxs)
        last_text = [""] * len(ctxs)
        for attempt in (0, 1):                                     # one retry, then keep the truncated text
            idx = [i for i in range(len(ctxs)) if res[i] is None]
            if not idx:
                break
            params = [self.SP(max_tokens=self.note_max_tokens, temperature=1.0, top_p=1.0,
                              seed=_stable_seed(self.tag, *key, ctxs[i].agent_id, attempt), **self._so(NOTE_REGEX))
                      for i in idx]
            for i, o in zip(idx, self._gen([prompts[i] for i in idx], params)):
                c = o.outputs[0]
                gen = [t for t in c.token_ids if t not in self.stop_ids]
                text = self.tok.decode(gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                note = parse_note(text) if c.finish_reason == "stop" else None
                last_text[i] = text
                if note is not None:
                    res[i] = {"note": note.strip(), "note_status": "ok", "note_attempts": attempt + 1,
                              "n_note_tokens": len(gen)}
        for i in range(len(ctxs)):
            if res[i] is None:                                     # the call stands; the note is cut where it ended
                txt = last_text[i].split('"')[0].replace("\\", "").replace("\n", " ").strip() or "(no note)"
                res[i] = {"note": txt, "note_status": "truncated", "note_attempts": 2, "n_note_tokens": None}
        self.last_note_s = time.time() - t0
        return res


# ---------------------------------------------------------------- swarm loop

@dataclass
class AnalystSwarmConfig:
    experiment_id: str
    T: int
    R: int = 1
    seed: int = 0                 # inbox shuffles
    K: int = 16
    results_root: Path = field(default_factory=lambda: ROOT / "results")


def inbox(J: np.ndarray, i: int, calls: np.ndarray, notes: list[str], rng: np.random.Generator):
    """(allies, faded) of reader i, each a tuple of (call, note), in a shuffled order."""
    a, f = np.flatnonzero(J[i] > 0), np.flatnonzero(J[i] < 0)
    return (tuple((CALLS[int(calls[j])], notes[j]) for j in rng.permutation(a)),
            tuple((CALLS[int(calls[j])], notes[j]) for j in rng.permutation(f)))


class AnalystSwarm:
    """Synchronous rounds (A1)-(A3): round 0 from note and news alone; round t >= 1 reads the round-(t-1) calls and
    notes of the colleagues and the analyst's own round-(t-1) call and note, with news xi(t).

    Checkpointing (guardrail 3): one line per (experiment_id, replica, round_t, agent_id) in analyst_calls.jsonl,
    written after the round's notes; a restart restores every complete round and continues. A missing decision
    (valid mass < 0.95) records belief NaN and carries the previous call and note forward (round 0: call from the
    stored uniform at P = 1/2, note "(no note)")."""

    def __init__(self, J: np.ndarray, game: AnalystGame, cfg: AnalystSwarmConfig, backend):
        J = np.asarray(J)
        if J.shape != (game.N, game.N) or not set(np.unique(J)) <= {-1, 0, 1}:
            raise ValueError("J must be an N x N sign matrix")
        if cfg.T > game.T or cfg.R > game.R:
            raise ValueError("game has fewer rounds or replicas than the config")
        self.J, self.game, self.cfg, self.backend, self.N = J, game, cfg, backend, game.N
        self.dir = Path(cfg.results_root) / cfg.experiment_id
        self.ckpt = self.dir / "analyst_calls.jsonl"
        self.note_text = [n.text(game.asset) for n in game.notes]

    def context(self, rep: int, t: int, i: int, calls, notes) -> AnalystContext:
        g = self.game
        meta = dict(h0=float(g.notes[i].h0), h0_real=float(g.notes[i].h0_real), xi=float(g.xi[t, i]),
                    xi_real=float(g.xi_real[t, i]))
        if t == 0:
            return AnalystContext(i, rep, 0, self.note_text[i], g.news_side[t, i], int(g.news_pct[t, i]), meta=meta)
        allies, faded = inbox(self.J, i, calls, notes, np.random.default_rng([self.cfg.seed, rep, t, i]))
        return AnalystContext(i, rep, t, self.note_text[i], g.news_side[t, i], int(g.news_pct[t, i]),
                              own=(CALLS[int(calls[i])], notes[i]), allies=allies, faded=faded, meta=meta)

    def _restore(self, rep: int):
        rows: dict[int, dict[int, dict]] = {}
        for r in read_jsonl(self.ckpt):
            if r["replica"] == rep:
                rows.setdefault(r["round_t"], {})[r["agent_id"]] = r
        done = []
        t = 0
        while t in rows and len(rows[t]) == self.N:
            done.append([rows[t][i] for i in range(self.N)])
            t += 1
        return done

    def _round(self, rep: int, t: int, calls, notes) -> list[dict]:
        ctxs = [self.context(rep, t, i, calls, notes) for i in range(self.N)]
        t0 = time.time()
        bel = self.backend.beliefs(ctxs, key=(rep, t))
        t1 = time.time()
        u = self.game.U[t, rep]
        new_calls = np.empty(self.N, int)
        for i, b in enumerate(bel):
            if b["ok"]:
                new_calls[i] = 1 if u[i] < b["p_buy"] else -1
            else:
                new_calls[i] = (1 if u[i] < 0.5 else -1) if t == 0 else calls[i]
        ok = [i for i in range(self.N) if bel[i]["ok"]]
        got = self.backend.notes([ctxs[i] for i in ok], [CALLS[int(new_calls[i])] for i in ok], key=(rep, t))
        t2 = time.time()
        note_rec = dict(zip(ok, got))
        recs = []
        for i in range(self.N):
            b = bel[i]
            nr = note_rec.get(i, {"note": notes[i] if t > 0 else "(no note)", "note_status": "carried"})
            rec = {"experiment_id": self.cfg.experiment_id, "replica": rep, "round_t": t, "agent_id": i,
                   "status": "ok" if b["ok"] else "missing", "p_buy": b["p_buy"] if b["ok"] else None,
                   "x": b["x"] if b["ok"] else None, "call": int(new_calls[i]), "u": float(u[i]),
                   **{k: v for k, v in b.items() if k not in ("p_buy", "x", "ok")}, **nr,
                   "xi": float(self.game.xi[t, i]), "xi_real": float(self.game.xi_real[t, i]),
                   "p_buy_raw": b["p_buy"]}
            if t > 0:
                c = ctxs[i]
                rec.update(own_call=int(calls[i]), n_allies=len(c.allies), n_faded=len(c.faded),
                           fA=sum(1 if k == "BUY" else -1 for k, _ in c.allies) / self.cfg.K,
                           fD=sum(1 if k == "BUY" else -1 for k, _ in c.faded) / self.cfg.K)
            recs.append(rec)
        for rec in recs:
            append_jsonl(self.ckpt, rec)
        append_jsonl(self.dir / "analyst_timings.jsonl",
                     {"experiment_id": self.cfg.experiment_id, "replica": rep, "round_t": t, "N": self.N,
                      "belief_s": t1 - t0, "note_s": t2 - t1, "total_s": t2 - t0, "n_missing": self.N - len(ok),
                      "ts": time.time()})
        return recs

    def _write_meta(self):
        meta = self.dir / "analyst_meta.json"
        if meta.exists():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        g = self.game
        np.savez(self.dir / "inputs.npz", J=self.J.astype(np.int8), h0=g.h0, h0_real=g.h0_real, xi=g.xi,
                 xi_real=g.xi_real, U=g.U)
        meta.write_text(json.dumps({"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.cfg).items()},
                                    "game": {k: getattr(g, k) for k in ("N", "T", "R", "seed", "nu_nom", "a0_nom", "asset")},
                                    "backend": type(self.backend).__name__,
                                    "wording": getattr(self.backend, "wording", None)}, indent=1))

    def run(self) -> dict:
        """Run or resume all replicas. Returns s = 2 P_buy - 1, x, sig, missing with shape (T+1, R, N)."""
        self._write_meta()
        T, R, N = self.cfg.T, self.cfg.R, self.N
        s, x = np.full((T + 1, R, N), np.nan), np.full((T + 1, R, N), np.nan)
        sig, missing = np.zeros((T + 1, R, N), int), np.zeros((T + 1, R, N), bool)
        for rep in range(R):
            rounds = self._restore(rep)
            calls, notes = None, None
            for t in range(T + 1):
                recs = rounds[t] if t < len(rounds) else self._round(rep, t, calls, notes)
                calls = np.array([r["call"] for r in recs])
                notes = [r["note"] for r in recs]
                for i, r in enumerate(recs):
                    missing[t, rep, i] = r["status"] != "ok"
                    if not missing[t, rep, i]:
                        s[t, rep, i], x[t, rep, i] = 2 * r["p_buy"] - 1, r["x"]
                sig[t, rep] = calls
        out = dict(s=s, x=x, sig=sig, missing=missing)
        np.savez(self.dir / "beliefs.npz", **out)
        return out
