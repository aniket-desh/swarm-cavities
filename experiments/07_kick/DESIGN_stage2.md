# Kick and echo, stage 2: do echoes add over ties, survive at K = 4, and is the confirm/disconfirm asymmetry real?

Status: revision 2 (after blind review 1), frozen by commit before any stage-2 data. Stage 1
(`experiments/07_kick/RESULT.md`) confirmed the sign structure of the reaction term at K = 2. Stage 2 asks the
questions stage 1 couldn't, using only the inverted wording (the label wording's echoes at K = 4 are predicted at
0.003–0.012 and stage 1 found label single-message responses at half their predicted size, so those cells would not
resolve within this budget).

## Questions, in order of importance

1. **Additivity.** The reaction term sums over reciprocated ties, so at fixed K and gains the echo at γ = ±0.5 (one
   reciprocated tie per agent at K = 2) is half the echo at γ = ±1 (two ties): R = E(±0.5)/E(±1) = 0.5. This needs no
   gains and no single-agent law (`quiet_regime.md` section 3.2; mock 0.48–0.52; anchoring law 0.40–0.49 at K = 2).
   Failure would mean the echo doesn't decompose over ties (for instance, an agent only responds when both partners
   move).
2. **K = 4.** Does the sign structure hold at K = 4 (γ = ±1), and do responses still compose (F against F_open)?
3. **The confirm/disconfirm asymmetry**, found but not pre-registered in stage 1: second-step inputs that confirm an
   agent's lean moved it about 0.6 as much as inputs that contradict it (pooled open-path ratio 0.61 ± 0.05 with
   stage 1's estimator, 0.60 ± 0.05 with this stage's lagged slope factors), against 0.90–0.93 for noisy tanh agents
   and 0.83 for the anchoring law at K = 2. Stage 2 tests it on new data, separately at K = 2 and K = 4.
4. **Single-agent law** at the new cells (F, F_open against the refrozen linear and anchoring values).
5. **Early kicks** (exploratory): kicks of round-0 messages (raw sensor reports) at K = 4.

## Design

| | |
|---|---|
| model | claude-haiku-5-5, effort low, realtime API for base runs and twins |
| base runs | `experiments/05_swarm/cells.py kick2`, tag `kick2-20261125`: inverted wording, p₋ = 0.5, N = 64, T = 10, H = 1, 2 replicas; cells K = 2 × γ ∈ {−0.5, +0.5} and K = 4 × γ ∈ {−1, 0, +1}; 8 graphs each (graphs 0–7; 40 base runs) |
| settled kicks | t0 ∈ {6, 8}, both replicas, every agent with \|s(t0)\| ≥ 0.1, one kick per twin, in the four cells with reciprocated ties; up to 2,048 twins per cell |
| sham | 16 twins per base run (128 per cell) in the same four cells; pooled minimum detectable offset on E ≈ 0.01 (3 s.e.), comparable to the smallest expected echo (K = 4, γ = −1), so a sham pass excludes only offsets of that size |
| γ = 0, K = 4 | t0 = 6, replica 0: up to 512 twins (echo zero by construction; first-order and open-path statistics) |
| early kicks | K = 4, γ = ±1, t0 ∈ {0, 1}, replica 0, \|s(t0)\| ≥ 0.1: up to 1,024 twins per cell, a separate tag so they never mix with the settled kicks, analysed separately per t0 |
| kick | the mirrored message, as in stage 1 (`envs/kick.py`) |

Commands (repo root, each in tmux; caps cumulative over each tag prefix):

```
# 1. base runs; rerun until the log reports 40/40 cells done (at most 8 concurrent runs: the API's output-token limit)
uv run python experiments/05_swarm/cells.py kick2 kick2-20261125 --cap 25 --max-runs 8
# 2. saturation-only null for the kick-size ratio, from base runs only
uv run python experiments/07_kick/analyze_kick.py kick2-20261125 --null
# 3. one-twin replay check; require "non-readers-sent@t0+1 0" before step 4
uv run python experiments/07_kick/kick.py kick2-20261125 kick2k-20261125 --t0 6 --agents first-eligible --min-abs-s 0.1 --cap 1 --cells inverted-K4-pn0.5-g+1.0-N64-T10-sync-G0
uv run python experiments/07_kick/analyze_kick.py kick2k-20261125 --methods
# 4. settled twins (one tag) and early twins (another tag)
#    (each kick command is rerun until it reports every twin done: a rate-limit RoundError stops it without losing data)
uv run python experiments/07_kick/kick.py kick2-20261125 kick2k-20261125 --t0 6 --rep 0 --min-abs-s 0.1 --cap 6 --parallel 32 --cells K4-pn0.5-g+0.0
uv run python experiments/07_kick/kick.py kick2-20261125 kick2k-20261125 --t0 6 8 --rep 0 1 --min-abs-s 0.1 --sham 16 --cap 60 --parallel 32 --cells K2-pn0.5-g-0.5 K2-pn0.5-g+0.5 K4-pn0.5-g-1.0 K4-pn0.5-g+1.0
uv run python experiments/07_kick/kick.py kick2-20261125 kick2e-20261125 --t0 0 1 --rep 0 --min-abs-s 0.1 --cap 18 --parallel 32 --cells K4-pn0.5-g-1.0 K4-pn0.5-g+1.0
# 5. analysis (lagged slope factors and equal-weight pooling are the defaults; --stage1-frozen reproduces stage 1)
uv run python experiments/07_kick/analyze_kick.py kick2k-20261125
uv run python experiments/07_kick/additivity.py
uv run python experiments/07_kick/analyze_kick.py kick2e-20261125 --t0 0
uv run python experiments/07_kick/analyze_kick.py kick2e-20261125 --t0 1
```

## Estimators

As in stage 1 (`analyze_kick.py`: slopes through the origin, slope-factor factorization K5 with leave-one-out
coefficients, open paths with the same |s| ≥ 0.1 filter, alignment split, graph bootstrap, joint graph bootstrap for
pooled values over every cell with reciprocated ties), with two changes from the review:

- **Lagged slope factors.** Each response's factor comes from the state before it: readers' d_j from s_j(t0), the
  receiver's d at t0+2 from s(t0+1). Stage 1 took them from the same round's base draw, which shares noise with the
  outcome (the base draw is subtracted in Δs) and biases the alignment split (toward 1 at K = 2, away from it at K = 4).
  Stage 1's frozen numbers are reproduced exactly with `--stage1-frozen`; stage 1 re-analysed with lagged factors gives
  pooled F = 1.05 ± 0.08, F_open = 1.12 ± 0.08, F − F_open,matched = −0.05 ± 0.07 and asymmetry ratio 0.60 ± 0.05
  (`results/kick1k-20261125-analysis-lagged-d.txt`), a sensitivity analysis, not a change to its verdicts.
- **Equal-weight pooling**: pooled values are the unweighted mean over cells (the predictions are pooled the same way),
  with s.e. from the joint graph bootstrap, over all four settled cells and separately for the K = 2 and K = 4 pairs.

New:

- **R±** (`additivity.py`): E(±0.5, this stage) / E(±1, stage 1, inverted K = 2), s.e. by the delta method from the two
  graph-bootstrap s.e. (stage-1 values: E(+1) = +0.0567 ± 0.0054, E(−1) = −0.0146 ± 0.0030), and R_norm, R rescaled by
  the ratio of the two cells' slope factors 𝒟 computed from their base runs, which removes the difference in settled
  confidence between them. Expected s.e.(R+) ≈ 0.08, dominated by stage 1's denominator.
- **Asymmetry ratio** `pooled_open_confirm_over_disconfirm`: the pooled open-path confirming slope divided by the
  pooled disconfirming slope, computed per bootstrap draw, for K = 2 and K = 4 separately (open paths have both
  alignment classes in every cell; closed paths have one).

## Predictions (frozen before stage-2 data)

Echoes (`quiet_regime.md` section 3.8, frozen at commit 91e267c; linear ± 1 s.d., anchoring simulated):

| cell | E linear | E anchoring | sign |
|---|---|---|---|
| K = 2, γ = −0.5 | −0.0142 ± 0.0025 | −0.0056 | − |
| K = 2, γ = +0.5 | +0.0290 ± 0.0034 | +0.0313 | + |
| K = 4, γ = −1 | −0.0200 ± 0.0039 | −0.0063 | − |
| K = 4, γ = 0 | 0 exactly | 0 exactly | 0 |
| K = 4, γ = +1 | +0.0300 ± 0.0046 | +0.0132 | + |

Parameter-free, from stage 1's measured echoes and additivity: E(+0.5) ≈ 0.5 × 0.0567 = +0.028, E(−0.5) ≈ −0.007.

Factorization slopes and the asymmetry ratio with this stage's estimators (lagged slope factors;
`refreeze_predictions.py --stage2`, `refrozen_predictions_stage2.json`, revision 2; 3 × 16 simulated graphs per cell;
Monte Carlo s.e. 0.004–0.07 per cell; pooled = unweighted mean of cells):

| law | pooled F | pooled F_open | pooled F − F_open,matched | ratio K = 2 | ratio K = 4 | K = 4 only: F / F_open / F − F_open,matched |
|---|---|---|---|---|---|---|
| linear, noise σ₀ = 0.045 | 0.986 | 0.969 | +0.028 | 0.897 | 1.037 | 1.007 / 0.972 / +0.040 |
| linear, noise σ₀ = 0.063 | 0.968 | 0.971 | −0.010 | 0.934 | 0.996 | 0.984 / 0.987 / −0.025 |
| anchoring | 0.614 | 0.651 | −0.016 | 0.834 | 0.779 | 0.574 / 0.576 / +0.019 |

Stage 1's measured asymmetry at K = 2 (before stage-2 data): 0.60 ± 0.05 (lagged factors).

Early kicks (secondary, in-sample, inverted only; `quiet_regime.md` 3.7, simulated anchoring law with the pilot's
gains): E ≈ −0.019 (γ = −1) and +0.093 (γ = +1) at K = 4. Sign predictions as for settled kicks.

## Decision rules (fixed before the data)

"2 s.e." below means two combined standard errors: the graph-bootstrap s.e. of the measurement and the Monte Carlo
s.e. of the prediction (±0.03 for pooled slopes and differences, ±0.05 for the asymmetry ratio). With 8 graph clusters,
2 s.e. corresponds to about 91% two-sided coverage under t₇ and 3 s.e. to about 98%.

0. **Validity.** As in stage 1: machinery checks hold (kicked agent unchanged at t0+1, no non-reader resent, γ = 0 echo
   exactly zero); pooled sham slopes r_A, r_D, E over the four settled cells each within 3 s.e. of zero, else the
   offset is subtracted with its s.e. propagated and both values reported. Minimum detectable offsets are reported.
1. **Sign at K = 4 and γ = ±0.5.** In each cell the predicted sign holds at ≥ 3 s.e., fails at the opposite sign at
   ≥ 3 s.e., else unresolved. γ = −0.5 is expected to be marginal (≈ −0.007, about 3 s.e.) and is reported either way.
2. **Additivity (primary).** With s.e.(R+) ≈ 0.08, this stage can tell additive echoes (R = 0.5) from an echo that needs
   both ties (R near 0) or that saturates at one tie (R near 1), not from mild saturation (R ≈ 0.6–0.7). So: R+
   **consistent with additivity** if its 2-s.e. interval contains 0.5 and excludes both 0.15 and 0.85; **not
   additive** if the interval excludes 0.5; otherwise unresolved. The same reading for R_norm, reported alongside; R− is
   secondary (its stage-1 denominator has a 20% s.e.).
3. **Composition**, pooled over the four settled cells and separately for the two K = 4 cells. F − F_open,matched
   within 2 s.e. of 0 (all three predictions lie within 0.04 of 0 pooled; at K = 4 within 0.04 of +0.01): responses
   compose as the reaction term assumes. More than 0.2 below the prediction and beyond 2 s.e.: the returning change is
   discounted; more than 0.2 above: amplified. Otherwise unresolved.
4. **Single-agent law**, pooled over the four settled cells, and separately at K = 4. Linear response if F and F_open
   are both within 2 s.e. of the linear values (pooled F 0.97–0.99, F_open 0.97; K = 4: F 0.98–1.01, F_open 0.97–0.99)
   and at least 2 s.e. above the anchoring values (pooled 0.61, 0.65; K = 4: 0.57, 0.58). Anchoring if the reverse.
   Otherwise unresolved. F is also reported per alignment class, since stage 1 showed pooled F mixes slopes near 0.6
   (confirming) and near 1 (disconfirming).
5. **Asymmetry (replication of the stage-1 finding)**, at K = 2 and K = 4 separately. **Replicates** if the pooled
   ratio is below the linear null (K = 2: 0.90–0.93; K = 4: 1.00–1.04) by more than 2 s.e. and below the anchoring law's
   ratio (K = 2: 0.83; K = 4: 0.78) by more than 2 s.e., i.e. stronger than either law produces. **Does not replicate**
   if within 2 s.e. of the linear null. Otherwise unresolved (including "consistent with the anchoring law's ratio").
6. **Magnitude** (secondary): each echo against both frozen columns in combined s.d., as stage 1's rule 4.
7. **Early kicks** (exploratory): sign at ≥ 3 s.e. per t0, reported; t0 = 0 magnitudes compared with the in-sample
   values without a decision.

## Fixed rules

- No change to cells, t0, kicks, estimators or rules after the base runs launch (amendments recorded with time and
  reason; the original analysis still reported).
- Twins are selected only on |s(t0)| ≥ 0.1; sham jobs and early jobs are fixed by the base state only.
- The base command is rerun until 40/40 base runs are complete; `kick.py` refuses an incomplete base run.
- Budget: base runs ≈ $20 (53,760 calls); settled twins ≈ $40 (K = 2 cells ~7.5 calls per twin, K = 4 ~21); γ = 0
  ≈ $4; early ≈ $14; sham ≈ $3: about $80 projected, capped at $25 (base) + $60 (kick2k tag, cumulative) + $18
  (kick2e tag) = $103. On a cap abort, the same command is rerun with the cap raised, recorded as an amendment.

## Not tested by this stage

- The label wording at K = 4 (too small to resolve), self-recognition (as in stage 1), K > 4.

## Review

Blind review 1 (general-purpose agent given only this file): verdict "needs revision". Findings and changes:

- P0, rule 5's null was wrong and the estimator biased by its own outcome (receiver slope factor from the base draw at
  t0+2): lagged slope factors in `analyze_kick.py` and `refreeze_predictions.py`; F, F_open, the matched difference and
  the asymmetry nulls refrozen per K under three laws; rule 5 thresholds set against both the linear null and the
  anchoring ratio. Stage 1 re-analysed with lagged factors as a sensitivity analysis (asymmetry 0.60 ± 0.05 persists).
- P1, inverse-variance pooling of a ratio favours low ratios: the pooled ratio is now the ratio of pooled slopes per
  bootstrap draw, reported per K.
- P1, the additivity rule had no equivalence margin: rule 2 now states what s.e. ≈ 0.08 can and can't distinguish and
  decides on 2-s.e. intervals against 0.15, 0.5 and 0.85.
- P1, no K = 4 readout for composition: K = 4-only pooled values with their own predictions in rules 3 and 4.
- P2s: R normalised by base-state slope factors; equal-weight pooling on both sides with Monte Carlo s.e. in the rules
  and F per alignment class; budget text corrected and cap-abort instruction added; --parallel 32 and rerun-until-done
  for kick commands; early kicks analysed per t0; inputs committed with this file; sham minimum detectable offset
  stated.

## Amendments

(none)
