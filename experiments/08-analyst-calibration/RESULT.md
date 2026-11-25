# 08-analyst-calibration: Qwen3.5-9B analysts couple weakly and follow their faded colleagues, so the swarm sweep is a no-go at K = 16

No trust wording comes near the stability boundary, and none makes the faded channel lean against its colleagues. The
root-mean-square gain is ā = 0.72 to 1.24 half log-odds per unit normalized field (s.e. 0.02 to 0.03; the boundary
needs about 4.6). The fade gain is negative in every wording (a₋ = −0.12 to −0.27, s.e. 0.03 to 0.04), so faded
colleagues pull the reader toward their call even when the prompt says "treat each of their calls as strong evidence
for the opposite call". The solver's Λ_mutual at the measured parameters is 0.010 to 0.015 (s.e. 0.002). By the
theory's rules (`theory/analyst_scenario.md` section 7) the three-world contrast is a no-go, because a₋/a₊ ≤ 0, and the
boundary sweep is a no-go, because Λ_mutual < 0.7. No swarm sweep was run.

A strong SELL bias compounds the weak coupling: c ≈ −0.6 to −0.7 half log-odds in every wording, with symmetric notes
and news. It saturates beliefs (q ≈ 0.7) and on its own cuts Λ by about a factor of 9 (see "What would change the
answer").

Of the law's premises, the log-odds show no majority step, but they don't add over colleagues the way (A1) assumes:
within the ally group the reader partly averages. The leak premise couldn't be tested under the pre-registered rule.
All numbers come from 1,000 base probes per wording and text arm (no missing, no censored values), with bootstrap s.e.
over base probes (B = 1,000, paired across arms and wordings).

## Calibration table (real colleague notes, M1 = the law (A1) by OLS on x)

| wording | a₊ | a₋ | ν at ν_nom = 0.5 | ν at ν_nom = 1 | a0 | c | κ | ρ₁ = κ(1−q) | resid. s.d. |
|---|---|---|---|---|---|---|---|---|---|
| mild | 1.34 (0.04) | −0.27 (0.04) | 0.39 (0.02) | 0.60 (0.02) | 0.11 (0.01) | −0.72 (0.01) | −0.14 (0.01) | −0.045 (0.004) | 0.37 |
| moderate | 1.54 (0.04) | −0.26 (0.03) | 0.34 (0.02) | 0.53 (0.02) | 0.10 (0.01) | −0.70 (0.01) | −0.15 (0.01) | −0.043 (0.004) | 0.36 |
| strong | 1.01 (0.03) | −0.12 (0.03) | 0.19 (0.02) | 0.29 (0.02) | 0.07 (0.01) | −0.53 (0.01) | −0.19 (0.01) | −0.113 (0.007) | 0.31 |
| max | 1.74 (0.04) | −0.15 (0.04) | 0.30 (0.02) | 0.46 (0.02) | 0.09 (0.01) | −0.64 (0.01) | −0.09 (0.01) | −0.025 (0.004) | 0.40 |

Units: half log-odds x, P(BUY) = (1 + tanh x)/2. a₊ and a₋ are per unit of f = Σσ/16; a full 8-colleague ally bloc
moves x by a₊/2. ν and a0 are per unit-s.d. field. a₋ > 0 would mean leaning against faded colleagues. q, used in
ρ₁, comes from the single-site solver at each wording's parameters. Fitting the same model as tanh on s instead of
linear in x changes no coefficient by more than 0.084 (real arm). The 100 repeated probes, run in a different batch, differ in x by 0
(median) and at most 0.0625, which is bf16 logit quantization.

Five things in the table matter beyond the go/no-go:

- **The wording ladder has saturated.** "Strong" is weaker than "moderate" in every coefficient: the explicit
  "their BUY is a reason to SELL" sentence lowers a₊, ν and c together, as if it made the agent hedge overall. "Max"
  raises a₊ by only 0.20 (0.06) over "moderate". No further wording of this kind is likely to reach ā ≈ 4.6.
- **The research note and the news are underweighted.** The research note is rendered at nominal a0 = 0.3 and gets
  a0 ≈ 0.1. The news is rendered at nominal 0.5 and 1.0 and gets ν = 0.19 to 0.39 and 0.29 to 0.60, so doubling its
  displayed strength raises ν by a factor of 1.5 to 1.6, not 2.
- **The own previous call is anti-persistent.** κ < 0 in every wording, so an analyst leans slightly away from its
  last call. The theory's memory flag (ρ₁ > 0.3) is far away. The pilot gives an independent check: the echo function
  in the one-way world, where it measures memory alone, is ρ₁ = −0.080. The probes predict κ(1 − q) = −0.14 × 0.40 =
  −0.056 at the pilot's q = 0.60. That's the same sign and order, from a single run with no s.e.
- **Real notes don't weaken the coupling here, unlike the Haiku coin game.** Real minus template is
  a₊ +0.02 (0.03), +0.10 (0.03), +0.24 (0.03), +0.13 (0.03), and a₋ −0.03, −0.08, −0.17, −0.20 (each s.e. 0.03). So
  real faded notes are followed more than templated ones. With templates, a₋ reaches +0.05 (0.03) in the strong and
  max wordings, which is the only sign of fading anywhere in the data, and it disappears with real notes. κ is
  −0.1 to −0.2 more negative with real notes (s.e. 0.01), even though the analyst's own note is identical in both
  arms.
- **There is a common SELL bias of about 0.7 half log-odds** that no input explains. In the pilot it drives the swarm
  to a mean belief of −0.76 by round 3 in all three worlds.

Figures: `paper/figures/analyst_calib_gains.{pdf,png}` (a₊, a₋ per wording, real against template),
`analyst_calib_linearity.{pdf,png}` (partial residuals of x against the net ally and faded BUY counts),
`analyst_calib_gonogo.{pdf,png}` (Λ_mutual against ā, with the theory's curves).

## Go/no-go per wording (theory section 7)

| wording | ā | r = a₋/a₊ | J̄ = (a₊−a₋)/2 | Λ_mutual (solver) | Λ_mutual at ν_1 | contrast go (ā ≥ 1, r > 0.3) | boundary go (Λ ≥ 0.7) | herding first (J̄ > 1.3) | memory flag (ρ₁ > 0.3) |
|---|---|---|---|---|---|---|---|---|---|
| mild | 0.97 (0.03) | −0.20 (0.03) | 0.81 (0.02) | 0.010 (0.002) | 0.013 | no | no | no | no |
| moderate | 1.11 (0.03) | −0.17 (0.02) | 0.90 (0.02) | 0.010 (0.002) | 0.015 | no | no | no | no |
| strong | 0.72 (0.02) | −0.12 (0.03) | 0.57 (0.02) | 0.015 (0.002) | 0.016 | no | no | no | no |
| max | 1.24 (0.03) | −0.08 (0.02) | 0.94 (0.03) | 0.011 (0.002) | 0.014 | no | no | no | no |

- **Is Λ_mutual ≥ 0.7 reachable (ā ≳ 4.6 in the reference design)?** No. The strongest wording has ā = 1.24, a
  quarter of what is needed, and Λ_mutual = 0.011.
- **Is the three-world contrast a go (ā ≥ 1 and a₋/a₊ > 0.3)?** No. Two wordings pass ā ≥ 1, but r < 0 in all four.
  With a₋ < 0 the antagonistic world has Γ = −a₊a₋/K > 0, so it behaves like a weak mutual world. The rule's fallback
  for 0 < r < 0.3 (keep the mutual-against-one-way half) doesn't apply either. Even if it did, the solver's mutual
  echo at the measured parameters is only ρ₂ = +0.010 (moderate and max), below the 0.016 that 4 runs per world
  resolve (theory section 6).
- **Herding:** J̄ = 0.57 to 0.94 is below 1.3, so by the rule the mutual world doesn't herd first. In practice the bias
  c saturates the swarm anyway (pilot mean s = −0.76 in every world).

## The law's premises (pre-registered three-way tests, real arm)

| wording | in the overall verdict (ā ≥ 1) | linear in the counts | adding, not averaging | call, not confidence |
|---|---|---|---|---|
| mild | no (ā = 0.97) | inconclusive | **fails** | untestable |
| moderate | yes | inconclusive | **fails** | untestable |
| strong | no | fails* | **fails** | untestable |
| max | yes | fails* | **fails** | untestable |

Overall label: **not supported**, because "adding" fails in both wordings with ā ≥ 1.

- **Adding against averaging.** With the group mean g_A = Σ_allies σ/n_allies added to M1, it takes coefficient 0.37,
  0.46, 0.34, 0.66 (s.e. 0.07 to 0.09) while the f_A coefficient falls to 0.65, 0.67, 0.37, 0.49 (0.13 to 0.16). A
  reader therefore responds partly to the share of its allies calling BUY, not only to their number out of 16. In
  the swarm this makes the per-analyst gain vary like 1/n_allies, roughly ±25% at n_allies = 8 ± 2. That variation
  is a per-agent gain spread, which the theory hasn't worked out (section 8). The faded-side group-mean terms are
  small (0.10 to 0.16).
- **Majority rule: none.** The sign(f_A) step has coefficient −0.01 to +0.02 (s.e. 0.03) in every wording, against a
  full-bloc response of 0.5 to 0.9.
- **Saturation: mild.** The ally response is concave (f_A³ coefficient −0.9 to −1.2, s.e. 0.3 to 0.4). The local slope
  is 1.3 to 2.0 at f_A = 0 and about half that at |f_A| = 1/2, and the full-bloc response falls 0.10 to 0.16 half
  log-odds (s.e. about 0.07) short of the slope at the origin. *The pre-registered "fails" for strong and max came from
  the faded-group test. There a₋ ≈ −0.13 makes the 20% relative threshold 0.015 half log-odds, and the rule used the
  s.e. of the bloc response (0.02) instead of the s.e. of its difference from a/2 (0.06 to 0.08). The reviewer flagged
  this as a minor issue before the run, and I left it. Recomputed post hoc with the right s.e.
  (`results/analyst-calib/posthoc_linearity.json`, 400 bootstrap draws), every test is "inconclusive" or "holds"
  except the ally full-bloc test in the strong wording, which fails (−0.13, s.e. 0.06, threshold 0.10). Both labels
  are reported. The post hoc one isn't confirmatory.*
- **Leak.** The pre-registered rule makes this test untestable. The SELL notes in the pool came from senders who were
  nearly all confident, so the gap in mean σ·s between the hesitant and confident terciles is 0.36 for SELL (0.74 for
  BUY), below the fixed 0.4. As a description only: the ally leak coefficient is λ_A = 0.15 to 0.20 (s.e. 0.04 to
  0.05) with real notes, against 0.02 to 0.08 (s.e. 0.06 to 0.11) for the template placebo. So some of the sender's
  confidence probably reaches the reader through the note. λ_D is meaningless here, because a₋ ≈ 0 sits in its
  denominator.

## Pilot (exploratory, excluded from confirmatory use)

N = 256, K = 16, T = 10, mild wording, one graph per world, the same notes, news and call uniforms across worlds,
on an NVIDIA L40S.

- **Missing and refusals.** 0 of 8,448 decisions were missing. The smallest valid mass was 0.984 and the median
  0.9994. 0 of 8,448 notes were truncated, and none matched the refusal or disclaimer regex. Notes have a median of
  14 words. The neutral "your own research note" wording was used from the start, so the "private" variant was never
  tried. The probes' valid mass was at least 0.9978 everywhere.
- **Speed.** A round with inboxes took 50 s at N = 256 (25 s decisions, 25 s notes); round 0 took 28 s. That's
  $0.015 per round at $1.09/h, and $0.008 on an A40 at $0.59/h if the speed is similar.
- **Dynamics.** Mean belief m(t) = −0.26, −0.5, −0.75, then −0.75 to −0.81 from round 3 on, alike in the three worlds,
  with final mean calls of −0.75, −0.79 and −0.80. The echo function ρ₁..₄ is mutual −0.061, −0.022, −0.047, −0.018;
  one-way −0.080, −0.039, −0.034, −0.003; antagonistic −0.060, −0.023, −0.016, 0.000. There's no reciprocity
  signature, as expected for g² ≈ 0.08 at saturated beliefs (predicted mutual ρ₂ ≈ +0.01).
- **In-swarm regression** (theory section 1, each world): a₊ = 0.99 to 1.04, a₋ = −0.45 to −0.53, ν = 0.34 to 0.36,
  a0 = 0.07, κ = −0.11 to −0.12, c = −0.68 to −0.73, residual s.d. 0.33. The probes for the mild wording give a₊ =
  1.34 and a₋ = −0.27. The swarm sits at saturated, mostly SELL inboxes, where the concave response is flatter, which
  is consistent with the lower a₊ in the swarm. So the signs and the bias transfer, and the gains agree to about 30%.
  This comparison is exploratory and has no s.e.

## What would change the answer

These are proposals; none was tested. Λ values are from the single-site solver at the max wording's measured
parameters (3,000 sites × 64 copies).

1. **Remove the SELL bias.** Setting c = 0 at otherwise measured parameters raises Λ_mutual from 0.011 to 0.096, and
   the predicted mutual echo from ρ₂ = +0.010 to +0.074, which one run resolves. Candidate fixes: counterbalance the
   question across analysts (half asked whether the asset closes down, with the call mapping swapped), alternate the
   order of BUY and SELL in the format sentence, or state a 50% base rate. Each needs a probe check for c.
2. **Let the experimenter do the fading.** No wording made a₋ positive with real notes. Showing faded colleagues'
   calls already inverted ("contrarian signal from a faded colleague: implies SELL") realizes W_ij = −a/K exactly,
   provided the reader treats the inverted signal like an ally's. That gives r ≈ 1, and the probes can test it directly.
3. **Lower K rather than strengthening the wording.** Because readers partly average, ā doesn't grow with K, and
   g² = ā²/K. At the max wording's gains with c = 0, Λ_mutual is 0.20, 0.41 and 0.83 at K = 8, 4 and 2 (0.32, 0.66
   and 0.93 with equal gains, as fix 2 would give). This is the only lever here that reaches 0.7. It needs the gains
   recalibrated at the new K, and the theory's predictions re-frozen, since they are for K = 16.
4. **News amplitude doesn't help.** ν is already small (0.3), and lowering it to 0.1 moves Λ_mutual only from 0.096
   to 0.109 at c = 0. The theory's alternative knob for ā > 7 doesn't apply.

## How this was checked, and what wasn't

- `tests/test_analyst.py` (22 tests, no GPU) covers the rendering and realized fields, forbidden words, the decision
  arithmetic, the missing rule, censoring and stored-uniform calls. It also shows the swarm loop equals
  `analyst_lib.simulate` under the mock law in all three worlds, that a resume reproduces an uninterrupted run, and
  that the vLLM backend works on a fake engine. All 22 passed here and on the worker.
- `analyze.py selftest` recovers known gains, a known leak and an averaging agent from synthetic probes, and reproduces
  Λ = 0.60 at (4, 4) in the reference design.
- Checked on the worker: BUY and SELL are single tokens after the prefix, and the decision prompt runs about 800 to
  1,040 tokens.
- **Not verified:**
  - Whether gains measured with mild-wording pilot notes transfer to swarms run in the other wordings, whose notes
    may differ.
  - The leak fraction.
  - The form of the memory beyond the last call.
  - The proposals above.
  - Whether gains per unit f at K = 2 to 4 match those at K = 16.
  - Λ near the boundary, where the solver is truncated. That doesn't matter here, at Λ ≈ 0.01.
- **Deviations from the definition:**
  - The first three commands ran as one worker job.
  - The worker ran on an L40S, because no A40 or A6000 was in stock.
  - The folder is `08-analyst-calibration`, because orch names allow only hyphens.

## Cost

Pod: ≈ $1.0. One L40S at $1.09/h for 0.9 h: setup, tests and smoke 7 min, pilot 26.5 min, probes 12 min. That's
against a projection of $1.5 at 1.4 h and a $4 budget. No API spend.
