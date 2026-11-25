"""One agent's request for one round, and validation of the structured belief it returns.

Prompt layout (render order system -> messages, so the first two blocks form the cached prefix):

    system[0]  static rules, identical for all agents       <- cache breakpoint (shared across agents)
    system[1]  this agent's private evidence                <- cache breakpoint (stable across rounds)
    user       this round's peer messages (dynamic, uncached)

The reply is constrained by output_config.format to the belief schema below. Numeric bounds are not supported
by the JSON-schema subset the API accepts, so [0, 1] and sum-to-1 are checked here. A reply that fails any
check is retried once by the runner and then recorded as missing; nothing is ever pattern-matched out of
free text.
"""

import json
import math
from dataclasses import dataclass

PROB_SUM_TOL = 1e-2  # |sum(belief_vector) - 1| allowed; the raw vector is stored unnormalized


@dataclass(frozen=True)
class AgentRequest:
    agent_id: int | str
    system_rules: str  # identical for every agent in the experiment
    evidence: str      # this agent's private evidence, fixed across rounds
    dynamic: str       # peer messages received this round (and anything else that changes)
    q: int | None = None  # number of classes for q-class tasks; adds a required belief_vector


@dataclass(frozen=True)
class AgentConfig:
    model: str = "claude-haiku-5-5"
    max_tokens: int = 2048  # thinking counts toward this; effort "low" keeps it short
    effort: str = "low"
    cache_ttl: str | None = None  # None = 5-minute default, or "1h"


def belief_schema(q: int | None = None) -> dict:
    props = {
        "private_reasoning": {"type": "string", "description": "Your private reasoning; never shown to peers."},
        "public_message": {"type": "string", "description": "The message your peers will read next round."},
        "calibrated_belief_prob": {"type": "number",
                                   "description": "Your calibrated probability (0 to 1) that the claim is true."},
    }
    if q is not None:
        props["belief_vector"] = {"type": "array", "items": {"type": "number"},
                                  "description": f"Probabilities over the {q} options, in order, summing to 1."}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_params(req: AgentRequest, cfg: AgentConfig) -> dict:
    """Exact Messages API params. Deterministic in (req, cfg), so their hash keys the replay cache."""
    cc = {"type": "ephemeral"} if cfg.cache_ttl is None else {"type": "ephemeral", "ttl": cfg.cache_ttl}
    return {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": [
            {"type": "text", "text": req.system_rules, "cache_control": cc},
            {"type": "text", "text": req.evidence, "cache_control": cc},
        ],
        "messages": [{"role": "user", "content": req.dynamic}],
        "output_config": {"effort": cfg.effort, "format": {"type": "json_schema", "schema": belief_schema(req.q)}},
    }


def _is_prob(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


def validate_belief(obj, q: int | None) -> dict:
    """Return the belief fields or raise ValueError naming the first violated check."""
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    keys = set(belief_schema(q)["properties"])
    if set(obj) != keys:
        raise ValueError(f"keys {sorted(obj)} != {sorted(keys)}")
    for k in ("private_reasoning", "public_message"):
        if not isinstance(obj[k], str):
            raise ValueError(f"{k} is not a string")
    if not _is_prob(obj["calibrated_belief_prob"]):
        raise ValueError(f"calibrated_belief_prob={obj['calibrated_belief_prob']!r} not in [0, 1]")
    if q is not None:
        v = obj["belief_vector"]
        if not (isinstance(v, list) and len(v) == q and all(_is_prob(x) for x in v)):
            raise ValueError(f"belief_vector is not {q} probabilities")
        if abs(sum(v) - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"belief_vector sums to {sum(v):.4f}")
    return dict(obj)


def parse_belief(message: dict, q: int | None) -> tuple[dict | None, str | None]:
    """(belief, None) on success, else (None, reason) with reason in
    refusal | max_tokens | no_text | malformed_json | invalid: <check>."""
    stop = message.get("stop_reason")
    if stop == "refusal":
        return None, "refusal"
    if stop == "max_tokens":
        return None, "max_tokens"
    text = "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")
    if not text.strip():
        return None, "no_text"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "malformed_json"
    try:
        return validate_belief(obj, q), None
    except ValueError as e:
        return None, f"invalid: {e}"
