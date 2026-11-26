# Kick and echo, stage 2: echoes add over ties and keep their sign at K = 4, but K = 4 responses are half the predicted size

The reaction term's structure held up on new data. At K = 2 the echo through one reciprocated tie is half the echo
through two, as additivity requires (R+ = 0.43 ± 0.07, 2-s.e. interval [0.30, 0.56] around the predicted 0.5), and at
K = 4 the echo keeps the predicted sign in both cells (−0.0035 ± 0.0010 at γ = −1, +0.0080 ± 0.0012 at γ = +1) and is
exactly zero at γ = 0. Responses still compose (pooled F − F_open,matched = −0.09 ± 0.09), and the composition is again
closer to linear response than to the anchoring law, with F once more right at the 2 s.e. edge. The stage-1 finding
that confirming inputs move agents less than disconfirming ones replicates at K = 2 on new data and more strongly
(open-path ratio 0.54 ± 0.04 against the linear null 0.90–0.93 and the anchoring law's 0.83). At K = 4 the ratio is
below the linear null (0.71 ± 0.12) but can't be told apart from the anchoring law. The K = 4 echoes themselves are
4 to 6 times smaller than the frozen linear prediction, which misses them by 4.1 and 4.6 combined s.d. The shortfall is
first-order, as it was for stage 1's label wording: a settled agent at K = 4 responds to one flipped message about half
as strongly as the transfer-check gains predicted (r_A ≈ 0.066 against 0.128 ± 0.012), and composing the measured
responses gives the echoes.

Design, predictions and decision rules: `DESIGN_stage2.md` revision 3 (frozen at commit 653a1ce before any stage-2
data, after a blind review and its re-check; no amendments). Predictions: `theory/quiet_regime.md` section 3.8 and
`refrozen_predictions_stage2.json`.

## What ran

Claude Haiku 5.5 swarms (effort low, realtime API), inverted wording, N = 64, p₋ = 0.5, own last answer shown. Base
runs: K = 2 × γ ∈ {−0.5, +0.5} and K = 4 × γ ∈ {−1, 0, +1}, 8 graphs per cell, 2 replicas, T = 10 (40 runs, 0
missing, $19.58). Settled kicks as in stage 1 (every agent with |s(t0)| ≥ 0.1, mirrored message seen once by its
readers, t0 ∈ {6, 8}, both replicas) in the four cells with reciprocated ties: 1,692–1,802 twins per cell, 7,036 in
all, plus 128 sham twins per cell; 439 kicks at K = 4, γ = 0 (t0 = 6, replica 0). Early kicks at K = 4, γ = ±1,
t0 ∈ {0, 1}, replica 0, under a separate tag: 1,768 twins. Every kick command finished on its first try with 0
missing. Uncertainties are bootstrap s.e. over the 8 graphs of a cell; pooled values are unweighted means over cells
with a joint graph bootstrap. Slope factors are lagged (each response's factor from the state before it), as frozen.

## Results against the frozen rules

**Rule 0, validity: passes.** In all 9,755 twins (settled, sham, γ = 0 and early) the kicked agent never changed at t0+1 and no agent outside its
readers was resent at t0+1; all 439 γ = 0 echoes are exactly zero. Pooled sham slopes over the four settled cells:
r_A −0.0023 ± 0.0035, r_D −0.0022 ± 0.0043, E +0.0002 ± 0.0037, all within 3 s.e. of zero (minimum detectable offsets
0.011, 0.013, 0.011). The largest per-cell sham value, r_A = −0.013 ± 0.005 at K = 2, γ = +0.5, is under 3 s.e. and
decides nothing.

**Rule 1, sign at K = 4 and γ = ±0.5: holds in all four cells.**

| K | γ | E measured | s.e. | E / s.e. | E linear (frozen) | (meas − lin) / s.d. | E anchoring (frozen) | (meas − anch) / s.e. |
|---|---|---|---|---|---|---|---|---|
| 2 | −0.5 | −0.0083 | 0.0017 | −4.8 | −0.0142 ± 0.0025 | +2.0 | −0.0056 | −1.6 |
| 2 | +0.5 | +0.0241 | 0.0029 | +8.3 | +0.0290 ± 0.0034 | −1.1 | +0.0313 | −2.5 |
| 4 | −1 | −0.0035 | 0.0010 | −3.5 | −0.0200 ± 0.0039 | +4.1 | −0.0063 | +2.8 |
| 4 | 0 | 0 exactly | | | 0 | | 0 | |
| 4 | +1 | +0.0080 | 0.0012 | +6.7 | +0.0300 ± 0.0046 | −4.6 | +0.0132 | −4.3 |

The γ = −0.5 cell, expected to be marginal at about 3 s.e., came out at 4.8.

**Rule 2, additivity (primary): consistent with additivity.** R+ = E(+0.5)/E(+1) = 0.426 ± 0.065, 2-s.e. interval
[0.30, 0.56]: it contains 0.5 and excludes both 0.15 (an echo that needs both ties) and 0.85 (one that saturates at a
single tie). Normalizing by the two cells' slope factors from their base runs (𝒟 = 0.460 at γ = +0.5, 0.420 at γ = +1)
gives R_norm = 0.389 ± 0.059, interval [0.27, 0.51], which also contains 0.5, at its edge. R− = 0.572 ± 0.169
(interval [0.23, 0.91]) is secondary and doesn't exclude 0.85, as expected from stage 1's 20% s.e. on E(−1). As the
design stated, this resolution separates additive echoes from both extremes but not from mild saturation (R ≈ 0.6–0.7)
or mild superadditivity; both point estimates of R+ sit a little below 0.5, toward the anchoring law's 0.40–0.49.

**Rule 3, composition: responses compose.** Pooled over the four settled cells F − F_open,matched = −0.090 ± 0.089,
within 2 s.e. of zero (all three predictions lie within 0.05 of zero). At K = 4 alone it is −0.003 ± 0.167, also
within 2 s.e. but with little power. A secondary number the rule doesn't decide on: the K = 2 pair alone gives
−0.178 ± 0.038, 4.7 s.e. below zero, with both cells on the same side (−0.13 ± 0.05 at γ = −0.5, −0.22 ± 0.07 at
γ = +0.5). By the rule's own reading (more than 0.2 below the prediction and beyond 2 s.e. means "discounted") it falls
just short of the 0.2 margin against every law (0.18–0.19 below the linear values). So at K = 2 the returning change
moves an agent about 20% less than a comparable change arriving along an open path. Stage 1, with the same lagged
estimator, gave −0.05 ± 0.07 (K = 2, both wordings), so the two stages agree within 1.6 s.e. This is worth a direct test, because a
discount specific to returning changes is what self-recognition would produce, but it isn't established here.

**Rule 4, single-agent law: linear response, again at the edge for F.**

| | measured | linear σ₀ = 0.045 | linear σ₀ = 0.063 | anchoring | above anchoring by |
|---|---|---|---|---|---|
| pooled F | 0.859 ± 0.122 | 0.986 (−1.0 s.e.) | 0.968 (−0.9) | 0.614 | 2.01 s.e. |
| pooled F_open | 0.973 ± 0.073 | 0.969 (+0.1) | 0.971 (+0.0) | 0.651 | 4.4 s.e. |
| K = 4 F | 1.044 ± 0.208 | 1.007 (+0.2) | 0.984 (+0.3) | 0.574 | 2.3 s.e. |
| K = 4 F_open | 1.020 ± 0.125 | 0.972 (+0.4) | 0.987 (+0.3) | 0.576 | 3.5 s.e. |

Both F and F_open are within 2 s.e. of both linear laws and at least 2 s.e. above the anchoring law, pooled and at
K = 4, so the rule's verdict is linear response in both. Pooled F clears the anchoring law by 2.01 s.e., so it rests
on F_open more than on F. Per alignment class (reported, as the rule asks): pooled F is 0.34 ± 0.29 for confirming
returns and 1.09 ± 0.29 for disconfirming ones, and F_open is 0.70 ± 0.08 and 1.13 ± 0.09.

**Rule 5, asymmetry: replicates at K = 2; unresolved against the anchoring law at K = 4.**

| | measured ratio (open confirming / disconfirming) | linear null | below linear by | anchoring | below anchoring by | verdict |
|---|---|---|---|---|---|---|
| K = 2 | 0.535 ± 0.041 | 0.897 ± 0.016, 0.934 ± 0.028 | 8.2, 8.0 s.e. | 0.834 ± 0.018 | 6.7 s.e. | replicates |
| K = 4 | 0.710 ± 0.116 | 1.037 ± 0.056, 0.996 ± 0.033 | 2.5, 2.4 s.e. | 0.779 ± 0.133 | 0.4 s.e. | unresolved |

At K = 2 the ratio is stronger than either law produces, on graphs and cells stage 1 never saw (stage 1: 0.60 ± 0.05).
At K = 4 the outcome is the one the design anticipated: below the linear null but indistinguishable from the anchoring
law, which also predicts a lower ratio there. This is reported as unresolved against anchoring, not as a replication.

The first-order version of the asymmetry, which stage 1 found in all six cells, replicates at K = 2 and does not
appear at K = 4. Agree-tie readers who leaned the same way as the kicked agent, so that the mirrored message
contradicted them, responded more at K = 2 (0.331 ± 0.024 against 0.226 ± 0.021 at γ = −0.5; 0.244 ± 0.019 against
0.192 ± 0.019 at γ = +0.5). At K = 4 the difference vanished or reversed (0.071 against 0.076 at γ = −1, 0.058 against
0.070 at γ = +1, s.e. 0.004–0.009). So at K = 4 the open-path ratio is below 1 while the first-step split is not.

**Rule 6, magnitude (secondary).** Both K = 2 cells match the linear prediction (+2.0 and −1.1 combined s.d.) and the
parameter-free values from stage 1 and additivity (−0.007 and +0.028); γ = +0.5 is also within 3 s.e. of the
anchoring value (−2.5). The K = 4 echoes are inconsistent with linear magnitude (+4.1, −4.6 s.d.); γ = −1 is within
3 s.e. of the anchoring value (+2.8), γ = +1 matches neither (−4.3).

The K = 4 shortfall is first-order, not a failure of composition. Readers at K = 4 respond to one mirrored message with
r_A = 0.073, 0.070, 0.060 and r_D = −0.020, −0.020, −0.016 in the three cells (s.e. 0.001–0.013), against frozen
linear values of 0.128 ± 0.012 and −0.046 ± 0.008 and anchoring-law values of 0.10 and −0.03. At K = 2 the same
responses match their predictions (r_A = 0.31 and 0.24 against 0.275 ± 0.020; r_D = −0.074 and −0.080 against
−0.057 ± 0.009). Composing each K = 4 cell's measured responses (K(r_A² + r_D²)/2 at γ = +1, K r_A r_D at γ = −1) and scaling by
the frozen prediction's own ratio of echo to that naive composition (0.81 and 0.85, which carry the slope factors)
gives +0.0061 at γ = +1 (measured +0.0080, whose returning changes are disconfirming and slightly amplified, F = 1.15)
and −0.0048 at γ = −1 (measured −0.0035, whose returning changes are confirming and discounted). Both are within a
factor of 1.4 of the measurement, against the factor of 4 to 6 of the frozen linear values. Stage 1's label wording showed the
same thing: transfer-check gains, measured on fresh agents reading whole inboxes, overstate how much a settled agent
moves when one of K messages flips. Here that happens in the inverted wording once K = 4.

**Rule 7, early kicks (exploratory): signs hold at both t0.**

| t0 | γ | E | s.e. | E / s.e. | in-sample prediction (t0 = 0) | r_A | open ratio (pooled over γ) |
|---|---|---|---|---|---|---|---|
| 0 | −1 | −0.0170 | 0.0051 | −3.3 | −0.019 | 0.377 | 1.00 ± 0.11 |
| 0 | +1 | +0.1327 | 0.0087 | +15.3 | +0.093 | 0.366 | |
| 1 | −1 | −0.0084 | 0.0025 | −3.4 | | 0.152 | 0.67 ± 0.14 |
| 1 | +1 | +0.0327 | 0.0049 | +6.7 | | 0.120 | |

The t0 = 0 magnitudes are close to the in-sample values at γ = −1 and larger at γ = +1 (no decision attached). Two
features suggest where the settled-state effects come from, without establishing it. First, a round-0 message moves its
readers five to six times more than a settled message does (r_A ≈ 0.37 against 0.066), falling to about 0.14 by
t0 = 1, so the "half the predicted gain" of settled K = 4 agents is a property of settled agents. Second, the
confirm/disconfirm asymmetry is absent for round-0 kicks (ratio 1.00 ± 0.11) and present by t0 = 1 (0.67 ± 0.14),
which fits a picture in which agents discount confirming news only once they have committed to a lean. Composition
holds at both t0 (F − F_open,matched = +0.03 ± 0.04 and +0.11 ± 0.11). Early-kick factorization slopes are low at
t0 = 0 (F ≈ F_open ≈ 0.44–0.46) because the slope factors from round-0 states don't describe agents that are still
far from settled; they aren't compared with any law.

## Figures

- `paper/figures/kick2_echo.{pdf,png}`: echoes against the frozen linear (open blue) and anchoring (orange cross)
  predictions with residuals in combined s.d.; additivity ratios R± with 2-s.e. bars against 0.5 and the decision-band
  edges.
- `paper/figures/kick2_composition.{pdf,png}`: pooled and K = 4 factorization slopes against the three laws; the
  confirm/disconfirm ratio at K = 2 and K = 4 against the linear nulls and the anchoring law, with stage 1's value.
- Plotting script `plot_kick2.py`; numbers `results/kick2k-20261125-summary.json`, `results/kick2-additivity.json`,
  `results/kick2e-20261125-t0-{0,1}-summary.json`; printouts `results/kick2k-20261125-analysis.txt`,
  `results/kick2-additivity.txt`, `results/kick2e-20261125-t0-{0,1}-analysis.txt`, methods check
  `results/kick2k-20261125-methods.txt`.

## What this does and doesn't establish

- With stage 1, it establishes for Haiku 5.5 in this game that the two-round echo has the sign the reaction term
  predicts in every cell tested (K = 2 and 4; γ = ±0.5, ±1; two wordings at K = 2), is exactly zero without reciprocated
  ties, adds over reciprocated ties within the resolution of an s.e. of 0.07 on R, and is the composition of
  single-link responses.
- It does not make the theory's magnitudes zero-shot. The gains measured in the transfer check predict K = 2 inverted
  responses but overstate K = 4 inverted and K = 2 label responses by about a factor of two, so quantitative DMFT
  curves need gains measured on settled agents receiving single-message changes.
- The confirm/disconfirm asymmetry is now a replicated property of the single agent at K = 2 (two independent data
  sets, both beyond what tanh or the anchoring law produce). Its absence for round-0 kicks is exploratory. The
  single-agent probes proposed in stage 1 (confirming against disconfirming single-message changes at matched size)
  are still the direct test.
- The K = 2 composition gap (−0.18 ± 0.04) isn't a pre-registered quantity and stage 1 didn't show it clearly; it
  should be tested before being read as a sign of self-recognition.
- Not tested: the label wording at K = 4, K > 4, self-recognition.

## Cost and runtime

Base runs $19.58, settled twins and sham $41.50, γ = 0 twins $3.35, early twins $13.96: $78.39 against a projected
$80 and caps totalling $103. The settled twins ran for 5.8 hours at about 5 calls per second, bounded by API latency
at 32 concurrent twins. No cap was hit and no command needed a rerun.
