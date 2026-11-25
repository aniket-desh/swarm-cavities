"""Open-weight agent backend: the agent of envs/prompts.py run on a vLLM engine, with the belief read from token
logprobs instead of sampled text.

Same agent as the Claude swarm. The system turn is rules(H, disagree) + evidence (system[0] and system[1] of
engine/agent.py, joined by a blank line) + FORMAT_NOTE, the user turn is prompts.dynamic(ctx, H), rendered with
the model's chat template with thinking disabled (Qwen3.5: the generation prompt ends in an empty
"<think>\\n\\n</think>\\n\\n" block). FORMAT_NOTE stands in for the json_schema output format the Claude API
receives through output_config.format: it names the same three fields in the same order.

One agent call = three engine steps, each batched over every agent of the round:

  1. sample   temperature 1, top_p 1, constrained by SAMPLE_REGEX to
                  {"rationale": "<text>", "public_message": "<text>", "calibrated_belief_prob":
              so the call writes its rationale and message and stops exactly where the number would start.
              The sampled token ids become the belief prefix x. A sample that doesn't complete the pattern
              within max_tokens is retried once with a new seed, then recorded as missing (guardrail 4).
  2. stage 1  next-token distributions (top LOGPROBS logprobs, unconstrained) after
                  x,   x + "0",   x + "0."
  3. stage 2  next-token distribution after x + "0." + d1 for every first decimal d1 with
              P(d1 | x "0.") >= d1_thresh.

Qwen's tokenizer splits numbers into single digits, so a stated probability written 0.d1d2 is the token path
"0" "." d1 d2 and its probability is a product of next-token probabilities along that path. value_distribution()
turns the step-2/3 distributions into a distribution m(v) over the grid v = 0.00, 0.01, ..., 1.00:

    m(1.00)        += P_A("1")                                  (the model starts with "1")
    m(0.00)        += P_A("0") P_B(end)                         ("0" then the number ends)
    m(d1/10)       += P_A("0") P_B(".") P_C(d1) P_D(end | d1)   (one decimal)
    m(d1/10+d2/100)+= P_A("0") P_B(".") P_C(d1) P_D(d2 | d1)    (two decimals; a third digit is truncated)

with "end" any token that starts with "}", "," or is whitespace (the number is over). An unexpanded d1 (below
d1_thresh) puts its whole mass at d1/10 + 0.05, which moves the mean by at most 0.05 x skipped_mass; the
skipped mass is recorded. Then

    valid_mass = sum_v m(v),      belief p = sum_v v m(v) / valid_mass,

and the call is recorded as missing when valid_mass < min_valid_mass (0.95): the remaining mass went to tokens
that don't continue a well-formed probability, and a belief is never guessed. Mass outside the returned top
LOGPROBS tokens counts as invalid, so valid_mass is a lower bound. p is a deterministic function of the sampled
prefix (up to GPU floating-point non-determinism across batch compositions). For comparison, p_sampled is one
draw from m / valid_mass with a seed fixed by the call key, i.e. what a temperature-1 sample of the number
would have stated (truncated at two decimals).

Checkpointing (CLAUDE.md guardrail 3): every finished call is appended to
results/<experiment_id>/vllm_calls.jsonl keyed by (experiment_id, seed, round_t, agent_id), and a restarted
backend replays those calls instead of recomputing them. GPU calls have no per-call price; pod spend is
tracked by orch (engine.budget.pod_spend), so nothing is written to the API ledger.
"""

from __future__ import annotations

import json
import re
import time
import zlib
from pathlib import Path

import numpy as np

from engine.ledger import ROOT, append_jsonl, read_jsonl
from envs.prompts import AgentContext, make_request
from envs.swarm import Response

MODEL = "Qwen/Qwen3.5-9B"
LOGPROBS = 20          # top-k logprobs per step (vLLM's default max_logprobs)
GRID = np.round(np.arange(101) / 100, 2)

FORMAT_NOTE = ('Answer format: reply with one JSON object with exactly these keys, in this order: "rationale" '
               '(a string), "public_message" (a string), "calibrated_belief_prob" (a number between 0 and 1).')

# Structural pattern of the sampled part. Strings exclude quotes, backslashes and newlines, so the two text
# fields are read back by this same pattern without any guessing; the number is never sampled here.
SAMPLE_REGEX = (r'\{"rationale": "[^"\\\n]+", "public_message": "[^"\\\n]+", '
                r'"calibrated_belief_prob": ')
_PARSE = re.compile(r'\{"rationale": "([^"\\\n]+)", "public_message": "([^"\\\n]+)", "calibrated_belief_prob": \Z')


# ---------------------------------------------------------------- rendering

def render_messages(ctx: AgentContext, H: int, disagree: str = "label") -> list[dict]:
    req = make_request(ctx, H, disagree)
    system = f"{req.system_rules}\n\n{req.evidence}\n\n{FORMAT_NOTE}"
    return [{"role": "system", "content": system}, {"role": "user", "content": req.dynamic}]


def render_prompt(tokenizer, ctx: AgentContext, H: int, disagree: str = "label") -> str:
    """Chat-template string ending in the assistant generation prompt, thinking disabled."""
    return tokenizer.apply_chat_template(render_messages(ctx, H, disagree), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def parse_sample(text: str) -> tuple[str, str] | None:
    """(rationale, public_message) if the sampled text is exactly the structural prefix, else None."""
    m = _PARSE.match(text)
    return (m.group(1), m.group(2)) if m else None


# ---------------------------------------------------------------- logprobs -> belief

def token_kind(text: str) -> str:
    """'0'..'9' for a single digit, '.' for the decimal point, 'end' for a token that closes the number
    (starts with '}' or ',' or is whitespace), else 'other'."""
    if len(text) == 1 and text.isdigit() and text.isascii():
        return text
    if text == ".":
        return "."
    if text.strip() == "" and text != "":
        return "end"
    if text.lstrip()[:1] in ("}", ","):
        return "end"
    return "other"


def _by_kind(dist: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for text, p in dist.items():
        k = token_kind(text)
        out[k] = out.get(k, 0.0) + p
    return out


def value_distribution(dA: dict[str, float], dB: dict[str, float], dC: dict[str, float],
                       dD: dict[int, dict[str, float]]) -> tuple[np.ndarray, dict]:
    """m(v) on GRID from the next-token distributions (token text -> probability) after x, x"0", x"0." and
    x"0."d1 (dD keyed by the expanded d1). See the module docstring for the four paths."""
    A, B, C = _by_kind(dA), _by_kind(dB), _by_kind(dC)
    m = np.zeros(101)
    m[100] += A.get("1", 0.0)
    pre = A.get("0", 0.0)
    m[0] += pre * B.get("end", 0.0)
    pre_dot = pre * B.get(".", 0.0)
    skipped = 0.0
    for d1 in range(10):
        base = pre_dot * C.get(str(d1), 0.0)
        if d1 in dD:
            D = _by_kind(dD[d1])
            m[10 * d1] += base * D.get("end", 0.0)
            for d2 in range(10):
                m[10 * d1 + d2] += base * D.get(str(d2), 0.0)
        else:
            m[10 * d1 + 5] += base
            skipped += base
    return m, {"p_zero_token": pre, "p_one_token": A.get("1", 0.0), "p_dot": B.get(".", 0.0),
               "skipped_mass": skipped}


def belief_from_masses(m: np.ndarray, min_valid_mass: float = 0.95) -> dict:
    """Expected stated probability and the missing-mass rule (guardrail 4)."""
    valid = float(m.sum())
    mean = float(GRID @ m / valid) if valid > 0 else float("nan")
    return {"valid_mass": valid, "p_logprob": mean, "ok": valid >= min_valid_mass}


def expand_digits(dC: dict[str, float], thresh: float) -> list[int]:
    """First decimals whose conditional probability after x"0." reaches thresh (stage 2 expands these)."""
    C = _by_kind(dC)
    return [d for d in range(10) if C.get(str(d), 0.0) >= thresh]


# ---------------------------------------------------------------- backend

def _stable_seed(*key) -> int:
    return zlib.crc32("|".join(map(str, key)).encode()) % (2**31 - 1)


def _vllm_params():
    """(SamplingParams, structured-output constructor) for the installed vLLM version."""
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams, lambda regex: {"structured_outputs": StructuredOutputsParams(regex=regex)}
    except ImportError:  # vLLM < 0.11
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams, lambda regex: {"guided_decoding": GuidedDecodingParams(regex=regex)}


class VLLMBackend:
    """envs/swarm.py backend interface: ``await respond(seed, round_t, contexts, H) -> list[Response]``.

    ``llm`` is a vllm.LLM (or anything with ``generate(prompts, params, use_tqdm=False)`` and
    ``get_tokenizer()`` returning vLLM-shaped outputs, as in tests/test_vllm_backend.py). ``params`` overrides
    the (SamplingParams, structured-output) pair, for tests.
    """

    def __init__(self, llm, experiment_id: str, disagree: str = "label", max_tokens: int = 320,
                 d1_thresh: float = 1e-3, min_valid_mass: float = 0.95, results_root: Path | None = None,
                 params=None):
        self.llm, self.experiment_id, self.disagree = llm, experiment_id, disagree
        self.max_tokens, self.d1_thresh, self.min_valid_mass = max_tokens, d1_thresh, min_valid_mass
        self.tok = llm.get_tokenizer()
        self.SP, self._so = params or _vllm_params()
        self.id0, self.id_dot = self._single_id("0"), self._single_id(".")
        self.id_digit = [self._single_id(str(d)) for d in range(10)]
        self.stop_ids = set(getattr(self.tok, "all_special_ids", []) or [])
        if getattr(self.tok, "eos_token_id", None) is not None:
            self.stop_ids.add(self.tok.eos_token_id)
        self._text_cache: dict[int, str] = {}
        self.dir = Path(results_root or ROOT / "results") / experiment_id
        self.ckpt = self.dir / "vllm_calls.jsonl"
        self.done = {self._key(r["seed"], r["round_t"], r["agent_id"]): r for r in read_jsonl(self.ckpt)}
        self.timings: list[dict] = []

    # ----- helpers

    def _single_id(self, s: str) -> int:
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token for this tokenizer ({ids}); the digit-path belief "
                             "of this module assumes one token per digit")
        return ids[0]

    def _text(self, tid: int) -> str:
        if tid not in self._text_cache:
            self._text_cache[tid] = self.tok.decode([tid], skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
        return self._text_cache[tid]

    @staticmethod
    def _key(seed, round_t, agent_id) -> tuple[str, str, str]:
        return (str(seed), str(round_t), str(agent_id))

    def _gen(self, prompts: list[list[int]], params) -> list:
        return self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)

    def _next_dists(self, prefixes: list[list[int]]) -> list[dict[str, float]]:
        """Top-LOGPROBS next-token distribution (token text -> probability) after each prefix."""
        if not prefixes:
            return []
        sp = self.SP(max_tokens=1, temperature=1.0, top_p=1.0, logprobs=LOGPROBS)
        out = []
        for o in self._gen(prefixes, sp):
            d: dict[str, float] = {}
            for tid, lp in o.outputs[0].logprobs[0].items():
                t = self._text(tid)
                d[t] = d.get(t, 0.0) + float(np.exp(lp.logprob))
            out.append(d)
        return out

    # ----- the three steps

    def _sample(self, prompt_ids: list[list[int]], seeds: list[int]) -> list[dict | None]:
        params = [self.SP(max_tokens=self.max_tokens, temperature=1.0, top_p=1.0, seed=s, **self._so(SAMPLE_REGEX))
                  for s in seeds]
        res = []
        for o in self._gen(prompt_ids, params):
            c = o.outputs[0]
            ids = list(c.token_ids)
            while ids and ids[-1] in self.stop_ids:
                ids.pop()
            text = self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            fields = parse_sample(text) if c.finish_reason == "stop" else None
            res.append(None if fields is None else {"gen_ids": ids, "text": text, "rationale": fields[0],
                                                     "public_message": fields[1]})
        return res

    def _beliefs(self, prefixes: list[list[int]]) -> list[dict]:
        t0 = time.time()
        stage1 = []
        for x in prefixes:
            stage1 += [x, x + [self.id0], x + [self.id0, self.id_dot]]
        d1 = self._next_dists(stage1)
        dA, dB, dC = d1[0::3], d1[1::3], d1[2::3]
        t1 = time.time()
        jobs = [(i, d) for i in range(len(prefixes)) for d in expand_digits(dC[i], self.d1_thresh)]
        d2 = self._next_dists([prefixes[i] + [self.id0, self.id_dot, self.id_digit[d]] for i, d in jobs])
        dD: list[dict[int, dict]] = [{} for _ in prefixes]
        for (i, d), dist in zip(jobs, d2):
            dD[i][d] = dist
        t2 = time.time()
        self._last_stage_s = (t1 - t0, t2 - t1, len(stage1), len(jobs))
        out = []
        for i in range(len(prefixes)):
            m, info = value_distribution(dA[i], dB[i], dC[i], dD[i])
            out.append({"masses": m, **info, **belief_from_masses(m, self.min_valid_mass),
                        "expanded_d1": sorted(dD[i])})
        return out

    # ----- interface

    async def respond(self, seed, round_t: int, contexts: list[AgentContext], H: int) -> list[Response]:
        return self.respond_sync(seed, round_t, contexts, H)

    def respond_sync(self, seed, round_t: int, contexts: list[AgentContext], H: int) -> list[Response]:
        keys = [self._key(seed, round_t, c.agent_id) for c in contexts]
        todo = [i for i, k in enumerate(keys) if k not in self.done]
        if todo:
            self._run(seed, round_t, [contexts[i] for i in todo], H)
        return [self._response(self.done[k]) for k in keys]

    def _run(self, seed, round_t, ctxs: list[AgentContext], H: int):
        t0 = time.time()
        prompt_ids = [self.tok.encode(render_prompt(self.tok, c, H, self.disagree), add_special_tokens=False)
                      for c in ctxs]
        samples: list[dict | None] = [None] * len(ctxs)
        attempts = [0] * len(ctxs)
        for attempt in (0, 1):                      # guardrail 4: one retry, then missing
            idx = [i for i in range(len(ctxs)) if samples[i] is None]
            if not idx:
                break
            seeds = [_stable_seed(self.experiment_id, seed, round_t, ctxs[i].agent_id, attempt) for i in idx]
            for i, s in zip(idx, self._sample([prompt_ids[i] for i in idx], seeds)):
                samples[i], attempts[i] = s, attempt + 1
        t1 = time.time()
        ok = [i for i in range(len(ctxs)) if samples[i] is not None]
        beliefs = dict(zip(ok, self._beliefs([prompt_ids[i] + samples[i]["gen_ids"] for i in ok])))
        t2 = time.time()
        for i, c in enumerate(ctxs):
            rec = {"experiment_id": self.experiment_id, "seed": seed, "round_t": round_t, "agent_id": c.agent_id,
                   "model": MODEL, "disagree": self.disagree, "attempts": attempts[i],
                   "n_prompt_tokens": len(prompt_ids[i])}
            s = samples[i]
            if s is None:
                rec.update(status="missing", missing_reason="malformed_sample", calibrated_belief_prob=None,
                           p_sampled=None, public_message=None, rationale=None)
            else:
                b = beliefs[i]
                m = b["masses"]
                rng = np.random.default_rng(_stable_seed(self.experiment_id, seed, round_t, c.agent_id, "num"))
                p_samp = float(rng.choice(GRID, p=m / m.sum())) if m.sum() > 0 else None
                rec.update(status="ok" if b["ok"] else "missing",
                           missing_reason=None if b["ok"] else "low_valid_mass",
                           calibrated_belief_prob=b["p_logprob"] if b["ok"] else None,
                           p_logprob_raw=b["p_logprob"], p_sampled=p_samp, valid_mass=b["valid_mass"],
                           p_zero_token=b["p_zero_token"], p_one_token=b["p_one_token"], p_dot=b["p_dot"],
                           skipped_mass=b["skipped_mass"], expanded_d1=b["expanded_d1"],
                           masses={str(k): round(float(m[k]), 6) for k in np.flatnonzero(m > 1e-5)},
                           rationale=s["rationale"], public_message=s["public_message"],
                           n_gen_tokens=len(s["gen_ids"]))
            append_jsonl(self.ckpt, rec)
            self.done[self._key(seed, round_t, c.agent_id)] = rec
        st1, st2, n1, n2 = getattr(self, "_last_stage_s", (0.0, 0.0, 0, 0))
        tm = {"experiment_id": self.experiment_id, "seed": seed, "round_t": round_t, "n_calls": len(ctxs),
              "sample_s": t1 - t0, "stage1_s": st1, "stage2_s": st2, "n_stage1": n1, "n_stage2": n2,
              "total_s": t2 - t0, "prompt_tokens": int(sum(map(len, prompt_ids))),
              "gen_tokens": int(sum(len(s["gen_ids"]) for s in samples if s)), "ts": time.time()}
        self.timings.append(tm)
        append_jsonl(self.dir / "vllm_timings.jsonl", tm)

    @staticmethod
    def _response(r: dict) -> Response:
        ok = r["status"] == "ok"
        return Response(r["calibrated_belief_prob"] if ok else None, r["public_message"] if ok else None,
                        r["status"], r["missing_reason"], 0.0, r)


def make_llm(model: str = MODEL, **overrides):
    """vllm.LLM for Qwen3.5-9B as a text-only model with prefix caching. Keyword overrides go straight to
    vllm.LLM (e.g. gpu_memory_utilization, max_num_seqs)."""
    from vllm import LLM

    kw = dict(model=model, enable_prefix_caching=True, max_model_len=8192, gpu_memory_utilization=0.90,
              max_logprobs=LOGPROBS, seed=0, language_model_only=True)
    kw.update(overrides)
    return LLM(**kw)


def summarize_calls(path: Path) -> dict:
    """Counts and mass diagnostics over a vllm_calls.jsonl (for smoke checks)."""
    rs = list(read_jsonl(path))
    ok = [r for r in rs if r["status"] == "ok"]
    vm = np.array([r["valid_mass"] for r in rs if r.get("valid_mass") is not None])
    return {"n": len(rs), "ok": len(ok),
            "missing": {k: sum(r.get("missing_reason") == k for r in rs) for k in ("malformed_sample", "low_valid_mass")},
            "valid_mass_min": float(vm.min()) if vm.size else None,
            "valid_mass_median": float(np.median(vm)) if vm.size else None,
            "skipped_mass_max": max((r.get("skipped_mass", 0.0) for r in ok), default=None),
            "mean_abs_logprob_minus_sampled": float(np.mean([abs(r["calibrated_belief_prob"] - r["p_sampled"])
                                                             for r in ok])) if ok else None}


if __name__ == "__main__":  # print one rendered prompt with a stub tokenizer-free layout, for eyeballing
    from envs.evidence_game import evidence_for_field

    ev = evidence_for_field(0.4, 3, np.random.default_rng(0))
    ctx = AgentContext(agent_id=0, session=0, round_t=0, h_priv=ev.h_priv, evidence=ev.text())
    print(json.dumps(render_messages(ctx, 1), indent=1))
