# Kick and echo, stage 1: does the reaction term of the cavity theory act in a Haiku swarm?

Status: revision 2, frozen before any stage-1 data. The predictions come from `theory/quiet_regime.md` section 3.8
(commit 91e267c). The kick pilot (`kickpilot-20261125`, pilot graphs 900) is a methods check and is excluded from
every test below; what it already showed is listed under "Known before stage 1".

## Question

The cavity reduction (`theory/claim1.md` eq. (4)) says an agent's own past state returns to it through its
reciprocated ties as a reaction term ΓR: two rounds later, with a sign set by the reciprocated ties' signs and the
channel gains. In the quiet regime where Haiku swarms sit, the oscillation signatures of that term are unmeasurable
(`quiet_regime.md` section 2), so it is measured directly: kick one agent's outgoing message once and read its own
answer two rounds later against an unperturbed twin. Three questions, in order of importance:

1. **Sign structure.** Is the echo positive through concordant ties, and through discordant ties negative when the
   disagree channel repels (inverted wording) but positive when it attracts (label wording)? The informative cells
   are γ = −1 in the two wordings, which run on identical graphs and evidence (the graph and game seeds don't depend
   on the wording), so the wording alone flips the predicted sign. At γ = +1 the echo is a product of two same-type
   responses and is positive for any gains; it is a weaker check.
2. **Composition.** Does an agent respond to the returning change of its partner as it responds to other second-step
   inputs of the same alignment (factorization slope F equal to the open-path slope F_open of matching alignment)?
3. **Which single-agent law.** Linear response with the transfer gains, or the anchoring law (smaller response to
   small input changes)?

## Design

| | |
|---|---|
| model | claude-haiku-5-5, effort low (`engine.agent.AgentConfig` defaults), realtime API for base runs and twins alike |
| cells | wording ∈ {inverted, label} × γ ∈ {−1, 0, +1}; K = 2, p₋ = 0.5, N = 64, own last answer shown (H = 1) |
| base runs | `experiments/05_swarm/cells.py kick1`, tag `kick1-20261125`: 8 graphs per cell (graphs 0–7), T = 10, 2 replicas sharing graph, evidence and round 0 |
| kicks, γ = ±1 | every agent with \|s_i(t0)\| ≥ 0.1, at t0 ∈ {6, 8}, in both replicas, one kick per twin: up to 2,048 twins per cell |
| sham, γ = ±1 | 16 twins per base run (128 per cell), evenly spaced over the eligible kicks: no message is changed, the kicked agent's readers redraw at t0+1 on byte-identical prompts through a Runner with its own replay namespace, and later rounds propagate as in a real twin |
| kicks, γ = 0 | every agent with \|s_i(6)\| ≥ 0.1 in replica 0 at t0 = 6: up to 512 twins per cell (echo zero by construction; first-order and open-path statistics) |
| kick | the mirrored message (p → 1 − p, HEADS ↔ TAILS), seen by the kicked agent's readers in round t0+1 only; its own history stays true (`envs/kick.py`) |

Base runs and twins use the same transport (realtime), the one the pilot validated, so a resampled draw and the base
draw it is compared with come from the same distribution. The sham arm measures what resampling alone puts into the
slopes.

Commands (repo root, each in tmux; caps are cumulative over the tag prefix):

```
# 1. base runs; rerun until the log reports 48/48 cells done
uv run python experiments/05_swarm/cells.py kick1 kick1-20261125 --cap 30 --max-runs 24
# 2. saturation-only null for the kick-size ratio, from base runs only (before any twin exists)
uv run python experiments/07_kick/analyze_kick.py kick1-20261125 --null
# 3. one-twin replay check; require "non-readers-sent@t0+1 0" before step 4
uv run python experiments/07_kick/kick.py kick1-20261125 kick1k-20261125 --t0 6 --agents 1 --min-abs-s 0.1 --cap 1 --cells inverted-K2-pn0.5-g+1.0-N64-T10-sync-G0
uv run python experiments/07_kick/analyze_kick.py kick1k-20261125 --methods
# 4. twins
uv run python experiments/07_kick/kick.py kick1-20261125 kick1k-20261125 --t0 6 --rep 0 --min-abs-s 0.1 --cap 8 --cells g+0.0
uv run python experiments/07_kick/kick.py kick1-20261125 kick1k-20261125 --t0 6 8 --rep 0 1 --min-abs-s 0.1 --sham 16 --cap 50 --cells g-1.0 g+1.0
# 5. analysis
uv run python experiments/07_kick/analyze_kick.py kick1k-20261125
```

## Estimators (`experiments/07_kick/analyze_kick.py`, quiet_regime.md eqs. (K1)–(K5))

With s = 2p − 1, Δ = twin − base, D_i = −2 s_i(t0), d = 1 − s² at the base state of the round in question:

- first-order responses r_A, r_D: slopes through the origin of readers' Δs(t0+1) on D_i, by edge type J_ji; r_A also
  split by whether reader and sender lean the same way at t0;
- echo E: slope of Δs_i(t0+2) on D_i;
- factorization slope F: slope of Δs_i(t0+2) on pred_i = d_i Σ_{j reciprocated} c(J_ij) Δs_j(t0+1), with c(type) the
  first-order response per unit slope factor from the other twins of the cell (eq. (K5));
- open-path slope F_open: the same regression for agents k ≠ i that read a reader of i, are not readers, and have
  |s_k(t0)| ≥ 0.1 (the same selection as kicked agents). F and F_open are each split by alignment, sign(pred · s) of
  the receiver at t0+1: confirming or disconfirming. F_open,matched is the open-path slope of the alignment class the
  cell's closed paths mostly have (disconfirming at γ = +1 and label γ = −1, confirming at inverted γ = −1), and
  F − F_open,matched is estimated inside the bootstrap;
- kick-size ratio: r_A over the smallest third of |D| divided by r_A over the largest third (terciles per cell), and
  the saturation-only null of command 2;
- sham slopes: r_A, r_D and E computed on the sham twins.

Uncertainties: bootstrap over the 8 graphs of a cell (the two replicas of a graph share graph, evidence and round 0,
so they are not independent units). With 8 clusters a 3 s.e. threshold corresponds to a two-sided p ≈ 0.02 under t₇.
Rules 2 and 3 use inverse-variance pooled values over the four |γ| = 1 cells as primary; per-cell values are
secondary. The estimators were checked on deterministic tanh agents through the same twin code (F = 0.97–1.02,
F_open = 0.93–0.98, kick-size ratio 0.98–1.06 at Haiku-like evidence weight) and with i.i.d. noise added to sham
redraws (sham slopes −0.008 to −0.001).

## Frozen predictions (quiet_regime.md section 3.8)

"Linear" is linear response with the one-shot real-message gains of the transfer check, ±1 s.d.; "anchoring" is the
simulated anchoring law. Predicted s.e. of E with this design ≈ 0.0025 (twins treated as independent; the graph
bootstrap will be larger).

| wording | γ | r_A linear | r_D linear | E linear | E anchoring | sign |
|---|---|---|---|---|---|---|
| inverted | −1 | +0.275 ± 0.020 | −0.057 ± 0.009 | −0.0285 ± 0.0051 | −0.0128 | − |
| inverted | 0 | | | 0 exactly | 0 exactly | 0 |
| inverted | +1 | | | +0.0581 ± 0.0067 | +0.0672 | + |
| label | −1 | +0.203 ± 0.026 | +0.098 ± 0.015 | +0.0329 ± 0.0065 | +0.0208 | + |
| label | 0 | | | 0 exactly | 0 exactly | 0 |
| label | +1 | | | +0.0388 ± 0.0075 | +0.0307 | + |

Under the anchoring law r_A, r_D are +0.30, −0.05 (inverted) and +0.20, +0.09 (label).

| relation | noiseless linear (mock) | linear with noise σ₀ = 0.045 (reviewer's run) | anchoring law |
|---|---|---|---|
| F, closed paths | 1 (0.96–1.02) | 1.00 (0.92–0.94 at twice the noise) | 0.75 |
| F_open | 1 (0.92–0.99) | 0.91–0.99 | 0.65 |
| kick-size ratio of r_A | 1 (0.96–1.07) | 0.98–1.11 | 0.64–0.77 |
| E(γ=−1)/E(γ=+1), inverted | −0.49 | | −0.19 |

## Known before stage 1

- **Kick pilot** (inverted, one graph per cell, t0 = 5, twin-level s.e. that understate the uncertainty): the
  kick-size ratio was 0.34–0.49 at K = 2, outside both columns above (seen by the design reviewer, reported here so
  the reading of that ratio can't be adjusted after stage 1). No other pilot echo statistic has been examined.
- **Content control** (`experiments/07_kick/content_control.py`, fresh agents in the transfer check, real messages):
  per unit of stated lean, agents respond to weakly leaning messages (|s| ≤ 0.275, the senders behind small kicks)
  0.76 (0.12) as much as to strongly leaning ones (|s| ≥ 0.5) for inverted K = 2, and 0.62 (0.16) for label K = 2.
  So message content alone can lower the kick-size ratio into the anchoring column's range.
- **Saturation null on the pilot's states**: 0.96–1.23. Saturation does not lower the ratio at Haiku's confidence.

## Decision rules (fixed before the data)

0. **Validity.** The machinery checks must hold (kicked agent unchanged at t0+1, no non-reader resent at t0+1, any
   γ = 0 echo exactly zero). Sham slopes r_A, r_D and E must each be within 2 s.e. of zero in every |γ| = 1 cell; if
   one isn't, the slopes it affects are reported with that offset and no sign or composition conclusion is drawn
   from them.
1. **Sign structure.** In each |γ| = 1 cell the predicted sign **holds** if E has that sign at ≥ 3 s.e., **fails** if
   E has the opposite sign at ≥ 3 s.e., and is **unresolved** otherwise. The effective-coupling picture is confirmed
   if both γ = −1 cells hold (inverted negative, label positive), refuted if either fails, and unresolved otherwise;
   a label γ = −1 echo that is negative while its measured r_D is positive is a failure. The γ = +1 cells are
   reported with the same labels.
2. **Composition.** Pooled F − F_open,matched within 2 s.e. of zero (allowing +0.12 at inverted γ = +1, the tanh
   offset of the noiseless mock): the returning change is processed like other second-step inputs of the same
   alignment, and the reaction term of claim 1 can use the population's response. F − F_open,matched below −0.2 and
   beyond 2 s.e.: the returning change is discounted relative to comparable inputs. The test can't say why (the
   returning change follows a message the agent didn't write and that contradicts its own history, so this is not a
   test of self-recognition), only that the reaction term computed from population responses overstates it.
3. **Single-agent law** (pooled F and F_open). Both within 2 s.e. of the noisy-linear column (F ≥ 0.92, F_open ≥ 0.91)
   and at least 2 s.e. above the anchoring values (0.75, 0.65): linear response. Both within 2 s.e. of the anchoring
   values and at least 2 s.e. below 0.91: the anchoring law. Anything else: unresolved. The kick-size ratio is
   reported against its saturation null, the content control and the two columns, but decides nothing, because
   content alone covers the anchoring range.
4. **Magnitude** (each |γ| = 1 cell, combined s.d. = √(s.e.² + s.d. of the linear prediction²)). |E| above the linear
   column by more than 3 combined s.d.: single-message gains in the swarm exceed the transfer gains. |E| below the
   linear column by more than 3 combined s.d. and within 2 s.e. of the anchoring column: the anchoring magnitude.
   Below linear but not consistent with anchoring: neither model's magnitude. Within 3 combined s.d. of linear:
   consistent with linear response (and with anchoring where the two columns are close, which the table shows).

## Fixed rules

- No change to cells, t0, kick definition, estimators, controls or decision rules after the base runs launch.
  Anything changed goes under Amendments with the time and reason, and the original analysis is still reported.
- Twins are never selected on anything after t0. The only selection is |s(t0)| ≥ 0.1, for kicked and open-path
  agents alike, and the sham jobs are fixed by the base state only.
- The base command is rerun until all 48 base runs are complete; `kick.py` refuses to start on an incomplete one.
- Missing replies stay missing (carried forward by the swarm, as in the base runs); the missing and refusal rates are
  reported per cell.
- Budget: base runs ≈ $25 (64,512 calls at the measured $0.00038), twins and sham ≈ $26 (about 69,000 calls), so
  stage 1 is projected at ≈ $51. The caps above stop it at $30 (base) + $50 (kick tag) = $80.

## Not tested by this stage

- K = 4, |γ| = 0.5 and the early-kick arm (stage 2, decided after stage 1).
- Self-recognition: see rule 2.
- Label-wording swarm parameters are assumed from the inverted wording in the anchoring column; this stage runs the
  first label swarms.

## Review

Blind review 1 (2026-11-25, general-purpose agent given only this file's path): verdict "needs revision". Findings
and what changed:

- P0, base runs on batch and twins realtime with no sham: base runs moved to realtime; sham arm and rule 0 added.
- P1, the γ = 0 command would abort on the cumulative cap: caps made cumulative ($8, then $50), γ = 0 first.
- P1, rule 1 passed on a null echo and its γ = +1 half is automatic: rule 1 now needs the predicted sign at ≥ 3 s.e.,
  names the γ = −1 cells as the informative ones, and labels everything else unresolved.
- P1, bootstrap unit too fine: clustered on graphs (8), with the t₇ reading of 3 s.e. stated.
- P1, rule 3 compared point estimates and the kick-size ratio is confounded: rule 3 now uses 2 s.e. bands on pooled
  F and F_open; the ratio is demoted to reported-only, with the saturation null from the base runs (command 2) and
  the content control (which showed content alone covers the anchoring range).
- P1, closed and open paths differ in alignment, and open agents weren't selected on |s|: F and F_open split by
  alignment, the comparison is against the matched class, |s| ≥ 0.1 applies to open agents, r_A is split by lean,
  and rule 2's conclusion no longer claims self-recognition.
- P2: pooling stated (inverse-variance over the four |γ| = 1 cells); rule 4 covers both directions; the noisy-linear
  column added; the ratio now uses tercile bins like the prediction; a one-twin replay check precedes the full run;
  failed base cells are rerun before any kick.

## Amendments

(none)
