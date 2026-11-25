"""Anthropic API prices and the dollar cost of one response, computed from its `usage`.

Prices are $/MTok, verified 2026-11-25 against the claude-api skill (see CLAUDE.md, "API facts"):

    cost = [ input * p_in + cache_write_5m * 1.25 p_in + cache_write_1h * 2 p_in
             + cache_read * p_read + output * p_out ] / 1e6  x  (0.5 if batch else 1)

`usage.input_tokens` counts only the uncached part of the prompt, so the three prompt terms are disjoint.
Thinking tokens are billed as output and are already inside `output_tokens`.
"""

from dataclasses import dataclass

CACHE_WRITE_5M = 1.25  # x input price, 5-minute TTL (the default)
CACHE_WRITE_1H = 2.0   # x input price, 1-hour TTL
BATCH_DISCOUNT = 0.5   # Message Batches: 50% off every token class


@dataclass(frozen=True)
class Price:
    input: float       # $/MTok, uncached prompt tokens
    output: float      # $/MTok, including thinking
    cache_read: float  # $/MTok


PRICES = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01),   # prompts <= 100K tokens
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.10),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20),
    "claude-fable-5-2": Price(8.00, 40.00, 0.20),
}

# Haiku 5.5 above 100K prompt tokens is $0.50 / $2.50 (claude-api skill model table). The cache-read price for
# that tier is not in CLAUDE.md; we assume it keeps the 10%-of-input ratio of the base tier. Agent prompts in this
# project are far below 100K, so this branch exists to over- rather than under-count if one ever crosses it.
HAIKU_LONG_THRESHOLD = 100_000
HAIKU_LONG = Price(0.50, 2.50, 0.05)


def _get(obj, name, default=0):
    """Read a field from an SDK object or a plain dict (replayed responses are dicts)."""
    if obj is None:
        return default
    v = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if v is None else v


def usage_tokens(usage) -> dict:
    """Token counts by billing class. Cache writes are split by TTL when the API reports the split."""
    cw = int(_get(usage, "cache_creation_input_tokens"))
    split = _get(usage, "cache_creation", None)
    cw_1h = int(_get(split, "ephemeral_1h_input_tokens"))
    cw_5m = cw - cw_1h  # anything not reported as 1h is billed at the 5-minute rate
    return {
        "input_tokens": int(_get(usage, "input_tokens")),
        "cache_write_tokens": cw,
        "cache_write_5m_tokens": cw_5m,
        "cache_write_1h_tokens": cw_1h,
        "cache_read_tokens": int(_get(usage, "cache_read_input_tokens")),
        "output_tokens": int(_get(usage, "output_tokens")),
    }


def price_for(model: str, prompt_tokens: int) -> Price:
    if model not in PRICES:
        raise KeyError(f"no verified price for {model!r}; add it to engine/pricing.py from the claude-api skill")
    if model == "claude-haiku-5-5" and prompt_tokens > HAIKU_LONG_THRESHOLD:
        return HAIKU_LONG
    return PRICES[model]


def cost_from_tokens(model: str, batch: bool, input_tokens: int, output_tokens: int,
                     cache_read_tokens: int = 0, cache_write_5m_tokens: int = 0,
                     cache_write_1h_tokens: int = 0) -> float:
    prompt = input_tokens + cache_read_tokens + cache_write_5m_tokens + cache_write_1h_tokens
    p = price_for(model, prompt)
    usd = (input_tokens * p.input
           + cache_write_5m_tokens * CACHE_WRITE_5M * p.input
           + cache_write_1h_tokens * CACHE_WRITE_1H * p.input
           + cache_read_tokens * p.cache_read
           + output_tokens * p.output) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


def cost_usd(model: str, usage, batch: bool) -> float:
    """Actual dollar cost of one response from its `usage` (SDK object or dict)."""
    t = usage_tokens(usage)
    return cost_from_tokens(model, batch, t["input_tokens"], t["output_tokens"], t["cache_read_tokens"],
                            t["cache_write_5m_tokens"], t["cache_write_1h_tokens"])
