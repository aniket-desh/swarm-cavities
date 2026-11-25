# swarm-cavity

Dynamical mean-field (cavity) theory of LLM agent swarms: derive single-agent effective processes for N-agent
belief dynamics with asymmetric couplings and context-window memory, then test the predictions on Claude swarms
(API) and open-weight swarms (GPU). The source program is `docs/spec.md`; what prior work actually shows is in
`docs/literature.md`; the derivations are in `theory/`. Order of work and limits: `PLAN.md`. Status: `HANDOFF.md`.

## Environment

- The network volume is near its quota, so the venv and uv cache live on the container disk. Before any `uv` command:
  `source scripts/env.sh` (sets `UV_CACHE_DIR=/root/.cache/uv-sc`, `UV_PROJECT_ENVIRONMENT=/root/venvs/swarm-cavity`).
  If the container restarted, `uv sync` rebuilds it. Don't write large files under `/workspace`; put bulky outputs
  (activations, transcripts beyond a few hundred MB) on `/workspace-global` or the worker.
- Run orch as `cd /workspace/swarm-cavity && orch ...`. Never touch `/workspace/rr-trial` or other projects.
- `.env` holds `ANTHROPIC_API_KEY`. Load it inside Python (`dotenv.load_dotenv()`); a hook blocks shell commands that
  both read `.env` and call a non-Runpod API.
- Pods are prefixed `aniket-sc`. Never delete the coordinator or a volume.

## API facts (verified 2026-11-25 against the claude-api skill)

| model | input $/MTok | output $/MTok | cache read | notes |
|---|---|---|---|---|
| `claude-haiku-5-5` | 0.10 | 0.50 | 0.01 | prices for prompts <=100K tokens; thinking on by default at effort `medium` |
| `claude-sonnet-5-5` | 2.00 | 10.00 | 0.10 | |
| `claude-opus-5-5` | 4.00 | 20.00 | 0.20 | thinking cannot be disabled; default effort `medium` |
| `claude-fable-5-2` | 8.00 | 40.00 | 0.20 | |

Cache writes cost 1.25x input (5-min TTL) or 2x (1-h TTL). Batch API is 50% off everything. The minimum cacheable
prefix is 512 tokens. Thinking tokens are billed as output, so agent calls should set `output_config={"effort": "low"}`
and keep `max_tokens` modest, and cost must be computed from `response.usage`, never estimated. The Claude API
returns **no logprobs**: beliefs come from the structured `calibrated_belief_prob` field (and on open-weight models,
from token logprobs). `temperature` is not accepted on current models, so per-agent stochasticity comes from the
model itself, and replay determinism comes from caching transcripts, not from seeds.

## Guardrails (non-negotiable)

1. **3-stage gate before spending.** (a) `uv run pytest` passes, including the mock deterministic Ising agent
   `s_i(t+1) = tanh(beta * h_i(t) + h_priv_i)`, for which `theory/dmft_solver.py` must reproduce the analytic or
   large-N-simulated DMFT curves to <1% error; (b) a <=$5 smoke test on Haiku 5.5 at N=8, T=4; (c) every sweep
   expected to cost more than $50 prints a token-cost projection to stdout and logs it before launch. Any sweep over
   $300 first runs a pilot of at most 10% of it.
2. **Budget.** $2,000 total for experiment spend (Anthropic API from experiment code + Runpod pods). Every API call
   appends its actual cost to `budget/ledger.jsonl` as one JSON line with at least
   `{"ts", "experiment_id", "model", "batch": bool, "input_tokens", "cache_write_tokens", "cache_read_tokens",
   "output_tokens", "cost_usd"}`. `engine.BudgetGuard` refuses new calls once ledger + pod spend reaches the cap it is
   given (default $1,900). `scripts/spend_watch.py` (tmux session `sc-spend`) logs each $100 crossed to
   `budget/alerts.log`.
3. **Atomic checkpointing.** Every round of every trajectory is appended to JSONL keyed by
   `(experiment_id, seed, round_t, agent_id)` and flushed immediately, so a crash resumes without re-spending.
4. **Structured beliefs.** Agents return JSON with `private_reasoning`, `public_message`, `calibrated_belief_prob`
   (validated; a malformed reply is retried once, then recorded as missing, never regex-guessed).
5. **Falsification first.** Every comparison plots DMFT prediction vs. empirical with residuals, and flags where the
   Gaussian cavity assumption fails (bimodal fields, heavy tails, small N). Negative results are reported plainly.

## Code conventions

Use the `research-code` skill for code and the `paper-writing` skill for figures and write-ups. Keep equations
visible in code, with comments naming the equation in `theory/` they implement. Reuse code from the papers'
repositories under `third_party/` after cleaning it up to these conventions; note the source in a comment.
Results go in `results/<experiment>/` (data dirs come back from workers outside git); figures in `paper/figures/`.
Push to `origin main` after each milestone.
