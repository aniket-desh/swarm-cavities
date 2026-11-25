---
name: 08b-analyst-calibration-2
title: Analyst scenario, calibration 2: polarity counterbalancing, experimenter-reversed fading, and gains at K = 8, 4, 2 (Qwen3.5-9B, real notes)
status: draft
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S
count: 1
min_gpu_gb: 40
hours: 1.5
budget_usd: 5
disk_gb: 120
data:
secrets:
---

# Analyst scenario, calibration 2: polarity counterbalancing, experimenter-reversed fading, and gains at K = 8, 4, 2 (Qwen3.5-9B, real notes)

## Question

`experiments/08-analyst-calibration` found three obstacles to running the analyst scenario near its stability
boundary (`theory/analyst_scenario.md`):
- a common SELL bias of c ≈ −0.7 half log-odds that saturates beliefs and cuts Λ_mutual by about a factor of 9;
- faded colleagues attract instead of repelling (a₋ < 0 in every wording);
- weak gains, ā ≤ 1.24 at K = 16, with Λ_mutual ≤ 0.015.

The coordinator asked three questions, and this experiment answers them:

1. **Bias.** Is the bias attached to the position of an option in the format sentence, to the framing of the
   question, to the decision token, or to the content (the asset going down)? Does some variant remove it, and what
   are the gains then?
2. **Fading.** If the desk shows faded colleagues' calls already reversed, with a label that explains it, does the
   faded channel repel with a gain near the ally gain (a₋/a₊ → 1)?
3. **Small K.** With the best combination of 1 and 2, what are the gains and Λ_mutual at K = 8, 4 and 2? Could a
   small-K swarm reach Λ_mutual ≥ 0.7?

The theory's predictions are frozen for K = 16 only. Nothing is re-frozen here; the small-K numbers are reported as
they are.

## Hypothesis

- **Bias.** No firm prior on where it comes from. A position or token bias would flip sign with the order or token
  variant; a content bias would not.
- **Fading.** Reversal by the experimenter turns the faded channel into an ally-like one, so r > 0.3, and plausibly
  r near 1.
- **Small K.** 08 showed readers partly average within a group, so the gain per unit field should stay roughly
  constant as K falls and g² = ā²/K should rise. From 08's numbers, the solver puts Λ_mutual at 0.4 to 0.66 for
  K = 4 and 0.8 to 0.9 for K = 2 once c = 0. Counting against this: gains per unit field that fall with K, a bias
  that survives every variant, or a reversed faded channel that is still ignored.

## Method

The agent is identical to 08's: `envs/analyst.py`, Qwen/Qwen3.5-9B on vLLM 0.32.0, thinking off, the decision read
from the first token after the forced prefix with top-100 logprobs, missing when valid mass < 0.95. It uses the "max"
wording everywhere (08's strongest ally gain).

The new prompt variants are `envs/analyst.PromptVariant`. The default variant renders 08's prompt byte for byte; a
test checks a frozen hash of it.
- **question:** "close up" (default), or "close down" ("whether the asset will close down this week, that is, lower
  at Friday's close than at Monday's open").
- **order:** BUY listed first (default), or SELL listed first, in the call sentence, the answer line and the JSON
  format example.
- **tokens:** BUY/SELL (default), or YES/NO answering the question. The desk records YES as BUY for "close up" and YES
  as SELL for "close down", and states this in the rules. The prefix becomes `{"answer": "`; YES and NO are single
  tokens (checked offline).
- **fade:** ordinary (default); "reversed", where the rules say the desk reverses faded colleagues' calls before
  showing them, the group header reads "reversed by the desk", and each line reads `- BUY (reversed). Their note:
  "…"` with the original note unchanged; or "reversed_equal", which also says "A reversed call is as informative as an
  ally's call."
- The context always holds the true calls, and x is the half log-odds of BUY. So a₋ > 0 always means leaning against
  the faded colleague's actual call.

The probes are 08's base probes: same functions, same seeds, same pilot note pool. The pool is exported to
`note_pool.json` here in the same order, so the worker doesn't need 08's results. Each probe is a K-slot inbox (each
slot faded with probability 1/2, group consensus uniform), real pilot notes, a random own previous call, h0 at
nominal 0.3, and ξ at nominal 0.5 or 1.0. Real notes only (no template arm).

1. **Phase A** (K = 16, 1,000 base probes per cell, 8,000 decisions):
   - P0 default. It repeats 08's max/real probes exactly, as a reproducibility check.
   - P1 SELL first. P2 "close down". P3 "close down" with SELL first.
   - P4 YES/NO to "close up". P5 YES/NO to "close down".
   - F1 reversed fading. F2 reversed_equal.
2. **Selection** (on the worker, by rule): polarity* is the cell among P0–P5 with the smallest |c|; fade* is the cell
   among P0, F1, F2 with the largest a₋/a₊. Both use M1 point estimates, and cells with ≥ 5% missing are excluded.
   The best variant is polarity* combined with fade*.
3. **Phase B**: the best variant at K = 16 (skipped if it is already a phase-A cell), 8, 4 and 2, with 1,000 base
   probes each (up to 4,000 decisions). The base probes at K < 16 come from the same generator with K slots; they
   are new draws, not subsets of the K = 16 probes.
4. **Analysis** (`analyze.py`, fixed before data): M1 of 08 by OLS on x per cell, bootstrap over base probes
   (B = 1,000).
   - ā, r = a₋/a₊ and J̄ = (a₊ − a₋)/2 per cell.
   - Λ_mutual three ways. Solver: `analyst_lib.dmft` with the cell's K, also at c = 0, s.e. from 100 bootstrap
     draws. Tree: formula (A13), no news or bias. Graph: 1 − 1/A from `analyst_lib.chi_sg` on simulated runs of the
     law with the measured parameters on `fixed_indegree_signed_couplings(1024, K, +1, 1/2)` graphs, 3 graphs × 4
     replicas, T = 60, also at c = 0.
   - Polarity decomposition. Order effect = mean c(SELL first) − mean c(BUY first) over P0–P3. Framing effect =
     mean c("down") − mean c("up") over P0–P3. Token part (c(P5) − c(P4))/2 and content part (c(P4) + c(P5))/2 from
     the YES/NO pair.
   - Reproducibility: P0 against 08 max/real, |Δx| per probe.
   - `analyze.py selftest`: the graph estimate gives 0.585 at gains (4, 4) in the reference design (theory table 3.3:
     0.58), and 0.207 against the tree's exact 0.232 at (2, 2) without news.

## Inputs

Qwen/Qwen3.5-9B from the Hub, via `experiments/06-qwen-smoke/setup_vllm.sh`. `note_pool.json` is in this folder:
7,680 notes from 08's pilot (959 BUY, 6,721 SELL). No data tars.

## Commands

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/08b-analyst-calibration-2/run.py tok phaseA select phaseB 2>&1 | tee results/analyst-calib2.log
```

Coordinator: `uv run python experiments/08b-analyst-calibration-2/analyze.py` and `plot.py`.

## Expected outputs

- `results/analyst-calib2-20261125/`: `{cell}_K{K}.jsonl` (one line per probe: P_buy, x, valid mass, censoring, top
  invalid tokens, variant), `design_K{K}.json` and `selection.json`. About 10 MB.
- `results/analyst-calib2.log`.
- Coordinator: `results/analyst-calib2/summary.json`, figures in `paper/figures/analyst_calib2_*.{pdf,png}`, and
  `RESULT.md` here.

## Controls

- P0 repeats 08 exactly. The per-probe |Δx| is expected at most 0.0625 (bf16), with a₊ and c within their s.e.; if
  not, something changed in the prompt or the engine.
- P0 is also the within-run baseline that every variant is compared against, on paired base probes.
- The YES/NO pair separates a bias toward a token from a bias toward the content, and the 2×2 of question × order
  separates framing from position.
- Three Λ estimates. The graph simulation is the reference at K ≤ 8, where the Gaussian closure isn't expected to
  hold; the solver is the theory's own map; the tree form is an exact check without news or bias.
- Unit tests: 48, including the frozen default-prompt hash, no forbidden words in any of the 24 variants, and the
  YES/NO token mapping.

## Success and failure criteria

The run fails if vLLM doesn't load or a decision word isn't a single token. A cell with ≥ 5% missing probes is
reported with its missing rate, excluded from the selection, and labelled inconclusive.

Each criterion below is three-way on the bootstrap s.e., with threshold τ: **holds** iff the estimate is on the right
side of τ by more than 2 s.e.; **fails** iff it is on the wrong side by more than 2 s.e.; otherwise **inconclusive**.
1. **Bias removed** by a variant iff |c| < 0.2.
2. **Reversed fading works** iff r > 0.3, the theory's contrast threshold. Reported in addition: whether r lies
   within 0.7–1.3, which is "near the ally gain".
3. **Small-K boundary reachable** iff the graph estimate Λ ≥ 0.7 at some K for the best variant at its measured
   bias, with the solver value reported beside it. Also reported: the same with the bias removed by the experimenter
   (c = 0), which would correspond to sampling calls from tanh(x − ĉ).

Overall: supported if 2 holds and 3 holds at some K; not supported if 2 fails or 3 fails at every K; inconclusive
otherwise. Criterion 1 is reported, and enters the overall label only through the measured c used in 3.

## Time and cost

Measured in 08 on an L40S: 1,000 K = 16 probes in ~100 s, setup with tests 7 min, first engine start included.
Phase A ≈ 14 min; phase B ≈ 5 min (K < 16 prompts are shorter); tok and selection 2 min. Total ≈ 30 min:
≈ $0.30 on an A40 ($0.59/h) or ≈ $0.55 on an L40S ($1.09/h). Cap 1.5 h (3× the estimate); budget $5. One process,
same memory as 08 (fits a 46–48 GB card).

## If something fails

- A decision word isn't a single token (`tok` raises): drop that variant (amendment) and continue with the rest.
- Out of memory: as in 08 (`max_num_seqs` 128).
- Time: every batch of 512 probes is appended as it finishes, and a rerun resumes. If the cap nears during phase B,
  keep the finished K values.
- An amendment that changes prompt text bumps TAG, so old and new records never mix.

## Fixed rules

- The missing rule (valid mass 0.95), and no belief parsed from text.
- The cells, probe seeds, selection rule and analysis above are fixed before data.
- Nothing is re-frozen in `theory/`. No swarm runs are made.

## Questions asked before launch

None needed. The coordinator's request fixed the three questions, the reuse of the pilot pool, the constraints, the
spend protocol and the ~$5 budget. Choices made here:
- the "max" wording;
- the YES/NO pair as the token-versus-content control, beyond the suggested order swap and "close down" framing;
- a second reversed variant that states the reversed call is as informative as an ally's;
- K = 8 added between 16 and 4;
- graph-simulated Λ as the small-K reference.

## Review

<fill: reviewer findings>

## Amendments

None.

## Outcome

Not run yet.
