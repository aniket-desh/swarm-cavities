---
name: 08b-analyst-calibration-2
title: Analyst scenario, calibration 2: polarity counterbalancing and experimenter-reversed fading at K = 16 (Qwen3.5-9B, real notes)
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

# Analyst scenario, calibration 2: polarity counterbalancing and experimenter-reversed fading at K = 16 (Qwen3.5-9B, real notes)

## Question

`experiments/08-analyst-calibration` found two obstacles at K = 16 that the coordinator asked to address. Both bear
on the frozen go rule of `theory/analyst_scenario.md`:
- a common SELL bias of c ≈ −0.7 half log-odds, which saturates beliefs and cuts Λ_mutual by about a factor of 9;
- faded colleagues attract instead of repelling (a₋ < 0 in every wording), which makes the three-world contrast a
  no-go.

This experiment asks, at K = 16:
1. **Bias.** Is the bias attached to an option's position in the format sentence, to the framing of the question, to
   the decision token, to the content (the asset going down), or to the pilot's note text? Does some variant remove
   it, and what are the gains then?
2. **Fading.** If the desk shows faded colleagues' calls already reversed, with a label that explains it, does the
   faded channel repel with a gain near the ally gain (a₋/a₊ → 1)? The ordinary faded wording is the comparison.
3. **Go rule.** With each variant's gains, what does the frozen rule say? The contrast is a go iff ā ≥ 1 and
   a₋/a₊ > 0.3. The predicted echo is resolvable iff |Γ|(1−q)² > 0.03. ρ₂ (mutual and antagonistic) and Λ_mutual come
   from `analyst_lib`.

Reversing the faded calls is done by the experimenter. By construction it tests the sign structure of the reaction
term given negative couplings; it does not test whether agents act on distrust.

## Hypothesis

- **Bias.** No firm prior on where it comes from. A position or token bias would flip sign with the order or token
  variant; a content bias would not. A bias carried by the bearish pilot notes would shrink in the template cell.
- **Fading.** Reversal turns the faded channel into an ally-like one, so r > 0.3, and plausibly r near 1, at least
  once the faded colleagues' bearish notes are removed (F3).
- **Go rule.** With r near 1 and ā ≈ 1.2 to 1.7, the contrast rule passes, but the predicted echoes stay small
  (Γ = g² ≈ 0.1). So the echo-resolvability criterion is expected to pass only if the bias is also near 0.

## Method

The agent is identical to 08's: `envs/analyst.py`, Qwen/Qwen3.5-9B on vLLM 0.32.0, thinking off, the decision read
from the first token after the forced prefix with top-100 logprobs, missing when valid mass < 0.95. It uses the "max"
wording everywhere (08's strongest ally gain) and K = 16 everywhere.

The new prompt variants are `envs/analyst.PromptVariant`. The default variant renders 08's prompt byte for byte; a
test checks a frozen hash of it.
- **question:** "close up" (default), or "close down" ("whether the asset will close down this week, that is, lower
  at Friday's close than at Monday's open").
- **order:** BUY listed first (default), or SELL listed first, in the call sentence, the answer line and the JSON
  format example.
- **tokens:** BUY/SELL (default), or YES/NO answering the question. The desk records YES as BUY for "close up" and YES
  as SELL for "close down", and states this in the rules. The prefix becomes `{"answer": "`; YES and NO are single
  tokens (checked offline).
- **fade:**
  - ordinary (default);
  - "reversed": the rules say the desk reverses faded colleagues' calls before showing them, the header reads
    "reversed by the desk", and each line reads `- BUY (reversed). Their note: "…"` with the original note;
  - "reversed_equal": also says "A reversed call is as informative as an ally's call.";
  - "reversed_equal_nonote": as reversed_equal, but the faded notes are omitted and the rules say "Their notes are
    not shown".
- The context always holds the true calls, and x is the half log-odds of BUY. So a₋ > 0 always means leaning against
  the faded colleague's actual call.

The probes are 08's base probes: same functions, same seeds, same pilot note pool. The pool is exported to
`note_pool.json` here in the same order, so the worker doesn't need 08's results. Each probe is a 16-slot inbox (each
slot faded with probability 1/2, group consensus uniform), real pilot notes, a random own previous call with a pilot
note, h0 at nominal 0.3, and ξ at nominal 0.5 or 1.0.

1. **Phase A** (1,000 base probes per cell, 10,000 decisions):
   - P0 default. It repeats 08's max/real probes exactly, as a reproducibility check.
   - P1 SELL first. P2 "close down". P3 "close down" with SELL first.
   - P4 YES/NO to "close up". P5 YES/NO to "close down".
   - F1 reversed. F2 reversed_equal. F3 reversed_equal_nonote.
   - T0: P0 with template notes everywhere, including the analyst's own previous note, so no pilot text appears.
2. **Selection** (on the worker, by rule). Cells with ≥ 5% missing, and cells whose ā falls below 0.8 × ā(P0), are
   excluded. polarity* is the cell among P0–P5 with the smallest |c|; fade* is the cell among P0, F1, F2, F3 with the
   largest a₋/a₊. Both use M1 point estimates. The best variant is polarity* combined with fade*.
3. **Best**: the best variant with 1,000 base probes (skipped if it is already a phase-A cell).
4. **Analysis** (`analyze.py`, fixed before data): M1 of 08 by OLS on x per cell, bootstrap over base probes
   (B = 1,000).
   - The go-rule numbers per cell, at the measured parameters and again with c = 0. The second set is the bias
     removed by any means, for example by sampling calls from tanh(x − ĉ). The numbers are ā, r, J̄; Γ and q per world
     with |Γ|(1−q)²; ρ₂ per world (solver, 50,000 sites, T = 30); and Λ_mutual (3,000 sites × 64 copies, T = 40).
     The s.e. come from 100 bootstrap draws.
   - Polarity decomposition. Order effect = mean c(SELL first) − mean c(BUY first) over P0–P3. Framing effect =
     mean c("down") − mean c("up") over P0–P3. Token part (c(P5) − c(P4))/2 and content part (c(P4) + c(P5))/2.
     T0 − P0 measures the pilot note text.
   - Reproducibility: P0 against 08 max/real, |Δx| per probe.
   - `analyze.py selftest`: at gains (4, 4) in the reference design, Λ_mutual = 0.589 (note 0.60), ρ₂ = +0.256 and
     −0.257 (note +0.257 and −0.258). It passes.

## Inputs

Qwen/Qwen3.5-9B from the Hub, via `experiments/06-qwen-smoke/setup_vllm.sh`. `note_pool.json` is in this folder:
7,680 notes from 08's pilot (959 BUY, 6,721 SELL). No data tars.

## Commands

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/08b-analyst-calibration-2/run.py tok phaseA select best 2>&1 | tee results/analyst-calib2.log
```

Coordinator: `uv run python experiments/08b-analyst-calibration-2/analyze.py` and `plot.py`.

## Expected outputs

- `results/analyst-calib2-20261125/`: `{cell}_K16.jsonl` (one line per probe: P_buy, x, valid mass, censoring, top
  invalid tokens, variant), `design_K16.json` and `selection.json`. About 8 MB.
- `results/analyst-calib2.log`.
- Coordinator: `results/analyst-calib2/summary.json`, figures `paper/figures/analyst_calib2_{bias,fade,go}.{pdf,png}`
  (`plot.py`, tested on fake data), and `RESULT.md` here.

## Controls

- P0 repeats 08 exactly. The per-probe |Δx| is expected at most 0.0625 (bf16), with a₊ and c within their s.e.; if
  not, something changed in the prompt or the engine.
- P0 is also the within-run baseline that every variant is compared against, on paired base probes.
- The YES/NO pair separates a bias toward a token from a bias toward the content, and the 2×2 of question × order
  separates framing from position.
- T0 removes all pilot text, which separates a bias carried by the bearish pilot notes from one in the prompt itself.
- F3 separates "the reversal is ignored" from "the bearish faded notes override the reversed label".
- Unit tests: 56 in `tests/test_analyst.py` (61 with `tests/test_vllm_backend.py`), including the frozen
  default-prompt hash, no forbidden words in any of the 32 variants, and the YES/NO token mapping.

## Success and failure criteria

The run fails if vLLM doesn't load or a decision word isn't a single token. A cell with ≥ 5% missing probes is
reported with its missing rate, excluded from the selection, and labelled inconclusive.

Each criterion is three-way on the bootstrap s.e., with threshold τ: **holds** iff the estimate is on the right side
of τ by more than 2 s.e.; **fails** iff it is on the wrong side by more than 2 s.e.; otherwise **inconclusive**. The
labels are computed in `analyze.py`.
1. **Bias removed** in a cell iff |c| < 0.2. Judged per phase-A cell and for the best cell.
2. **Reversed fading works** in a cell iff r > 0.3. Judged on F1, F2, F3 and the best cell. Also reported: whether r
   lies within 0.7–1.3, which is "near the ally gain".
3. **Go rule**, reported per cell (no three-way label; these are decisions):
   - contrast go iff ā ≥ 1 and r > 0.3;
   - echo resolvable iff |Γ|(1−q)² > 0.03 in a world;
   - boundary go iff Λ_mutual ≥ 0.7;
   - herding first iff J̄ > 1.3.
   The rule is computed at the measured parameters and with c = 0.

Overall: supported if criterion 2 holds in some reversed cell and the best cell is a contrast go with a resolvable
echo in both reciprocal worlds at its measured c. Not supported if 2 fails in every reversed cell. Inconclusive
otherwise.

## Time and cost

Measured in 08 on an L40S: 1,000 K = 16 probes in ~100 s, setup with tests 7 min. Here 10 to 11 cells take ≈ 18 min;
tok and selection 2 min. Total ≈ 30 min: ≈ $0.30 on an A40 ($0.59/h) or ≈ $0.55 on an L40S ($1.09/h). Cap 1.5 h
(3× the estimate); budget $5. One process, same memory as 08 (fits a 46–48 GB card).

## If something fails

- A decision word isn't a single token (`tok` raises): drop that variant (amendment) and continue with the rest.
- Out of memory: as in 08 (`max_num_seqs` 128).
- Time: every batch of 512 probes is appended as it finishes, and a rerun resumes.
- An amendment that changes prompt text bumps TAG, so old and new records never mix.

## Fixed rules

- The missing rule (valid mass 0.95), and no belief parsed from text.
- The cells, probe seeds, selection rule and analysis above are fixed before data.
- Nothing is re-frozen in `theory/`. No swarm runs are made. K = 16 only.

## Questions asked before launch

None needed. The coordinator's request, as amended by the user's scope change, fixed the questions, the reuse of
the pilot pool, the constraints, the spend protocol and the ~$5 budget. Choices made here:
- the "max" wording;
- the YES/NO pair as the token-versus-content control, beyond the suggested order swap and "close down" framing;
- reversed variants that state the reversed call is as informative as an ally's, with and without the faded notes;
- a template-notes cell.

## Review

First pass (general-purpose agent following `agents/experiment-reviewer.md`, definition path only), on the version
that still had small-K cells: needs revision.
- **P1, criterion 3 couldn't be computed** (no s.e. for the graph Λ, no code for the labels), and the graph
  estimator was unvalidated at small K. Moot after the scope change: small K is dropped and the graph estimator
  removed. The remaining labels are computed in `analyze.py`. The finding was still useful: at small K, 1 − 1/A
  overstates Λ, because A = 1 + KΛ/((K−1)(1−Λ)) on a K-regular tree. Recorded for the deferred small-K experiment.
- **P1, hypothesis numbers from the wrong estimator at small K**: moot (scope change).
- **P1, reversed cells can't separate "reversal ignored" from "bearish notes override it"**: fixed. Added F3.
- **P1, selection ignored gains**: fixed. Cells with ā < 0.8 ā(P0) are excluded from the selection.
- **P2, the content bias may be pilot-note text**: fixed. Added T0.
- **P2, ambiguities**: criterion 2 is now judged on F1–F3 and the best cell. "At some K" no longer applies. If the
  best cell has ≥ 5% missing, it is reported as inconclusive and the phase-A cells still answer questions 1 and 2.

## Amendments

- 2026-11-25 18:55 UTC, before approval and launch: scope change from the user (relayed by the coordinator). Part 3,
  gains at K = 8, 4, 2, is dropped and deferred to a separate experiment. Parts 1 and 2 are kept at K = 16. The frozen
  go rule is re-applied with the new gains (contrast; echo resolvable, |Γ|(1−q)² > 0.03; ρ₂ and Λ_mutual from
  `analyst_lib`), and RESULT.md must state that experimenter-inverted fading tests the sign structure of the
  reaction term given negative couplings, not whether agents act on distrust. Phase B, the graph-simulated Λ and
  the small-K figure were removed from the scripts; F3 and T0 were added from the review.

## Outcome

Not run yet.
