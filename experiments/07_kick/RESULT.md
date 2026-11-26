# Kick and echo, stage 1: the reaction term acts in Haiku swarms with the predicted sign structure

The cavity theory's reaction term is there, two rounds after a kick, with the sign the effective couplings predict.
On identical graphs and evidence, a discordant reciprocated tie returns a kicked agent's change against it when the
disagree channel repels (inverted wording, E = −0.0146 ± 0.0030, 4.9 s.e.) and with it when the disagree channel
attracts (label wording, E = +0.0137 ± 0.0018, 7.6 s.e.); concordant ties return it with it in both wordings, and
without reciprocated ties the echo is exactly zero. The returning change is composed like any other input of the same
kind (factorization slope matched to open paths: difference +0.05 ± 0.05, predicted +0.06), and the composition is
closer to linear response than to the anchoring law (pooled F = 0.86 ± 0.06, F_open = 0.88 ± 0.04, against 0.98 and
0.945 for linear response and 0.67 and 0.63 for anchoring), though F sits right at the 2 s.e. edge of the linear band.
The echo magnitudes match the frozen linear prediction in the inverted wording and fall short of both predictions in
the label wording, and the shortfall is already in the first step: label-wording readers respond to a single kicked
message about half as strongly as the transfer-check gains predicted.

Design, predictions and decision rules: `DESIGN.md` revision 3 (frozen before any stage-1 data, commit 9056751, after
two blind reviews). Predictions: `theory/quiet_regime.md` section 3.8 and `refrozen_predictions.json`.

## What ran

Claude Haiku 5.5 swarms (effort low, realtime API), N = 64, K = 2, p₋ = 0.5, own last answer shown, in six cells:
wording ∈ {inverted, label} × γ ∈ {−1, 0, +1}. Base runs: 8 graphs per cell, 2 replicas, T = 10 (48 runs, 95,744
calls, 0 missing, $23.04). Kicks: every agent with |s(t0)| ≥ 0.1, mirrored message (p → 1 − p, HEADS ↔ TAILS) seen by
its readers once; t0 ∈ {6, 8} × both replicas at γ = ±1, t0 = 6 × replica 0 at γ = 0; one kick per twin; unaffected
agents replay the base run's replies from the cache. Sham: 32 twins per base run at γ = ±1 in which nothing is changed
and the readers redraw on identical prompts. Totals: 7,146 kick twins at γ = ±1 (1,737–1,839 per cell), 878 at γ = 0,
1,024 sham twins, 63,509 calls sent, 0 missing, $22.89. Uncertainties are bootstrap s.e. over the 8 graphs of a cell
(pooled values: joint bootstrap over graph indices across cells).

## Results against the frozen rules

**Rule 0, validity: passes.** The kicked agent never changed at t0+1 and no agent outside the kicked agent's readers
was resent at t0+1, in all 9,048 twins; every γ = 0 echo is exactly zero (878 of 878). Pooled sham slopes: r_A
+0.0003 ± 0.0010, r_D −0.0006 ± 0.0016, E −0.0008 ± 0.0024, all within 3 s.e. of zero; resampling alone puts no
detectable offset into the slopes (minimum detectable offsets 0.003, 0.005, 0.007).

**Rule 1, sign structure: confirmed.** Both informative cells hold, and so do both γ = +1 cells:

| wording | γ | E measured | s.e. | E / s.e. | predicted sign | E linear (frozen) | E anchoring (frozen) |
|---|---|---|---|---|---|---|---|
| inverted | −1 | −0.0146 | 0.0030 | −4.9 | − | −0.0285 ± 0.0051 | −0.0128 |
| inverted | 0 | 0 exactly | | | 0 | 0 | 0 |
| inverted | +1 | +0.0567 | 0.0054 | +10.5 | + | +0.0581 ± 0.0067 | +0.0672 |
| label | −1 | +0.0137 | 0.0018 | +7.6 | + | +0.0329 ± 0.0065 | +0.0208 |
| label | 0 | 0 exactly | | | 0 | 0 | 0 |
| label | +1 | +0.0151 | 0.0020 | +7.5 | + | +0.0388 ± 0.0075 | +0.0307 |

The label γ = −1 echo is positive with a positive measured disagree response (r_D = +0.049 ± 0.007), as the rule
requires. The sign flip between the two γ = −1 cells comes from the wording alone: same graphs, same evidence, same
kicks; only the sentence telling agents how to treat "sources you tend to disagree with" differs.

**Rule 2, composition: the returning change is processed like comparable inputs.** Pooled F − F_open,matched =
+0.049 ± 0.053 against the predicted +0.046 (linear) or +0.07 (anchoring), well within 2 s.e. The reaction term can be
computed from the population's response. Per cell (secondary): −0.01 ± 0.07, +0.18 ± 0.08, −0.52 ± 0.21, +0.12 ± 0.14
for inverted −1, inverted +1, label −1, label +1; the label γ = −1 value is 2.4 s.e. below zero, but rule 2 is decided
on the pooled value and the per-cell values scatter on both sides.

**Rule 3, single-agent law: linear response, at the edge.** Pooled F = 0.860 ± 0.058 is 1.96 combined s.e. (including
the ±0.02 Monte Carlo uncertainty of the prediction) below the noisy-linear 0.98 and 3.1 above the anchoring 0.67;
F_open = 0.883 ± 0.037 is 1.5 below 0.945 and 6 above 0.63. Both fall inside the linear band and outside the anchoring
one, so the rule's verdict is linear response; F alone would flip to unresolved with a slightly larger error. The
kick-size ratio (reported only) is 0.60 ± 0.04 pooled, below its saturation null computed from the base runs before
any twin (1.05–1.26) and in the range that message content alone produced in the content control (0.62–0.84), so it
says nothing about anchoring, as the design anticipated.

**Rule 4, magnitude.**
- Inverted γ = +1 matches the linear prediction (−0.2 combined s.d.) and the anchoring one.
- Inverted γ = −1 lies 2.4 combined s.d. below the linear magnitude, so it counts as consistent with linear response,
  and it is also within 0.6 s.e. of the anchoring value.
- Label γ = −1 lies 2.9 combined s.d. below linear (consistent with linear by the rule's 3 s.d. criterion, barely) and
  3.9 s.e. below the anchoring value.
- Label γ = +1 lies 3.05 combined s.d. below linear and 7.8 s.e. below anchoring: neither model's magnitude.

The label shortfall is a first-order effect, not a failure of composition: label-wording readers respond to a kicked
message with r_A = 0.115 and r_D = +0.047 (pooled over the three label cells; s.e. ≈ 0.005), about half of the
transfer-based predictions (0.203 ± 0.026 and 0.098 ± 0.015), while inverted-wording readers match theirs (r_A = 0.25
against 0.275 ± 0.020, r_D = −0.065 against −0.057 ± 0.009; figure 1, right). Composing the measured first-order
responses gives the label echoes almost exactly: 2 × (0.115² + 0.047²)/2 ≈ 0.015 at γ = +1 (measured 0.0151) and
2 × 0.115 × 0.047 ≈ 0.011 at γ = −1 (measured 0.0137). The transfer check measured label gains from fresh agents
reading whole inboxes; a settled agent reading one flipped message among K = 2 is apparently half as moved in this
wording.

## A finding the design did not target: confirming inputs move agents less

Splitting the factorization slope by whether the returning or second-step change pushes an agent further the way it
already leans (confirming) or against it (disconfirming), pooled over the four cells: confirming 0.59 ± 0.06 (closed)
and 0.61 ± 0.03 (open); disconfirming 1.13 ± 0.08 (closed) and 1.00 ± 0.06 (open) (figure 2, right). The asymmetry is
the same on closed and open paths, so it belongs to the single agent, not to the loop: relative to a tanh response,
Haiku under-reacts to news that agrees with its lean and reacts fully to news that contradicts it, as a stated-
confidence ceiling would produce. It also explains why the inverted γ = −1 echo, the one cell whose returning change
is confirming, is about half the naive composition of its first-order responses (−0.0146 against −0.033), while the
other three cells, whose returning changes are disconfirming, compose almost exactly. Part of any such asymmetry is
plain tanh curvature (a finite confirming change pushes the agent toward saturation, where its slope is smaller), but
tanh agents run through the same twin code and estimator give only confirming 0.87–0.96 against disconfirming
0.96–1.03 (ratio ≈ 0.9), against Haiku's ≈ 0.55. The first-order responses point the same way without any second
step: agree-tie readers whose lean the kick contradicts respond more than readers it confirms in all six cells
(0.300 vs 0.196, 0.268 vs 0.216, 0.235 vs 0.207 in the inverted cells; 0.124 vs 0.103, 0.118 vs 0.109, 0.109 vs 0.093
in the label cells, s.e. 0.005–0.025). This was not pre-registered and should be tested directly (single-agent probes
with confirming against disconfirming single-message changes at matched size) before it is relied on.

## Sensitivity to the slope-factor convention (added after the stage-2 design review)

Stage 2's review found that taking each response's slope factor from the same round's base draw, as this stage's
frozen estimator does, shares noise with the outcome and biases the alignment split. Re-analysing stage 1 with lagged
factors (readers' d from s(t0), the receiver's from s(t0+1); `analyze_kick.py`, default since stage 2) gives pooled
F = 1.05 ± 0.08, F_open = 1.12 ± 0.08, F − F_open,matched = −0.05 ± 0.07 and an open-path confirm/disconfirm ratio of
0.60 ± 0.05 (`results/kick1k-20261125-analysis-lagged-d.txt`). Composition and the asymmetry are unchanged, and F
moves further from the anchoring value. This is a sensitivity analysis; the verdicts above are the frozen ones.
Stage 2 (`RESULT_stage2.md`) replicated the asymmetry at K = 2 on new data (0.54 ± 0.04).

## Figures

- `paper/figures/kick1_echo.{pdf,png}`: echoes and first-order responses against the frozen linear (open blue) and
  anchoring (orange cross) predictions, with residuals in units of the combined s.d.
- `paper/figures/kick1_composition.{pdf,png}`: pooled F and F_open against the two laws, and the alignment split.
- Plotting script `plot_kick1.py`; numbers `results/kick1k-20261125-summary.json`; full printout
  `results/kick1k-20261125-analysis.txt`.

## What this does and doesn't establish

- It establishes, for Haiku 5.5 in this game at K = 2, that an agent's influence returns to it through reciprocated
  ties two rounds later with the sign set by the reciprocated ties and the channel gains, as the reaction term of
  `theory/claim1.md` eq. (4) says, and that its size is the composition of single-link responses.
- It does not test the self-recognition question: the returning change follows a message the agent didn't write and
  that contradicts its own history (rule 2's caveat).
- It doesn't test K = 4, |γ| = 0.5 (the additivity E(±0.5) = ½E(±1)) or early kicks; that is stage 2.
- The single-agent law verdict is marginal for F and rests on the pooled values; the alignment asymmetry shows that
  neither tanh-linear response nor the anchoring law is the whole story.

## Cost and runtime

Base runs $23.04 (projected $25.80); twins and sham $22.89 (projected about $28); total stage 1 ≈ $45.9 against a
projected $53 and caps of $82. Operational problems, none of which changed the design: the first base launch was
killed by the container's 8 GB memory limit (every Runner held its own copy of the replay cache), the second hit the
API's 2M output-tokens-per-minute limit with 16 concurrent runs, and the twins were slowed by fsync'd checkpoint writes
for free replays and by in-memory record growth; each was fixed in `engine/runner.py` and the runs resumed from their
checkpoints without re-spending.
