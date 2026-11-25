# Theory for swarm-cavity: the current statement

Last updated 2026-11-25 (night; kick stage 1 result and analyst calibrations added). This file is the one place to read for the state of the theory. The derivations
live in the notes it links (`claim1.md` … `claim6.md`, `quiet_regime.md`), each of which opens with its verdict and
names the scripts in `theory/checks/` that verify its formulas.

**Where things stand.** The cavity reduction itself holds after three corrections, but the swarms we can afford to
run don't sit where its striking predictions live. Haiku 5.5 agents in this game are weakly coupled once real
messages are used (real messages couple 25–50% more weakly than the templates the first calibration used, and the
inverted disagree channel nearly switches off), and they anchor on their own last answer whenever their inbox brings
nothing new. Both push every Haiku cell into a quiet regime (Λ ≈ 0.45 at best) where the oscillation and
amplification signatures of claims 2–3 can't be measured. What still isolates the cavity-specific structure is the
reaction term itself, which a kick-and-echo experiment measures directly. Stage 1 of that experiment found it, with
the predicted sign structure: on identical graphs, the wording that makes disagree ties repel turns the echo through
discordant reciprocated ties negative, and the wording that leaves them attracting turns it positive
(`experiments/07_kick/RESULT.md`). The open-weight track (Qwen3.5-9B) is the route to stronger coupling and large N: in the Physics of
Agents data that model sits beyond the stability boundary at round 0, though with binary votes rather than stated
probabilities.

## 1. What the data have shown (Haiku 5.5)

- **Peers act through a two-channel linear field, not the nominal one** (`experiments/03_calibration/RESULT.md`).
  The nominal field fails (η²_comp = 0.48): sources labelled "you tend to disagree with" still attract at about 40%
  of agree strength. Separate agree and disagree gains fit (held-out RMSE 0.058 against a 0.022 floor), and since the
  two-channel field is linear in the peers' states, the DMFT applies with effective couplings W ≠ J. A wording that
  tells agents disagree sources are miscalibrated makes the disagree gain repel. The Physics of Agents data say the
  same for four other models (`results/poa/README.md`: two gains needed, b₋/b₊ between −0.43 and +0.18, averaging
  slightly better than summing).
- **The template calibration does not transfer to real messages** (transfer check, 600 paired probes per cell,
  `results/transfer/summary.json`). With real swarm messages: inverted K=2 a₊ = 0.80 (0.03), a₋ = 0.15 (0.02);
  inverted K=4 a₊ = 0.66 (0.05), a₋ = 0.22 (0.04); label K=2 a₊ = 0.54 (0.03), a₋ = −0.24 (0.02); label K=4
  a₊ = 0.40 (0.03), a₋ = −0.21 (0.02). Templates gave 25–50% larger a₊ and 2–3× larger inverted a₋. Gains per unit
  normalized field fall from K=2 to K=4: agents average their inbox.
- **The swarm settles into anchored states** (pilot, inverted wording, K = 2 and 4, γ ∈ {−1, 0, +1}, N = 64, T = 10,
  one graph each). Replica differences persist (c1 = 0.68–0.80 against 0.29–0.33 predicted), 22–39% of agents
  repeat their answer exactly in late rounds, P_δ ≈ 1.7–2.4, and c2 shows no sign flip at γ = −1 (it rises with γ:
  K=4 gives 0.33, 0.40, 0.61). The single-agent white probes, whose field changes every round, show little
  persistence, so the anchoring is state-dependent.
- **Physics of Agents data, zero cost** (`results/poa/README.md`): period-2 flicker on synchronous graphs is real and
  disappears under asynchronous updates (period index 0.26–0.29 against 1.17–1.18, predicted 0.24–0.25 and
  1.15–1.16 by the fitted two-channel agent); short odd cycles, not global bipartiteness, suppress it; and coupling
  induces replica overlap (Δq > 0 in every model and task).

## 2. The current single-agent law (`quiet_regime.md` section 1)

The agent moves toward a two-channel tanh target τ by (1 − ρ(u))(τ − s), where u is the mean change of its K inbox
messages since its last answer and ρ(u) = ρ∞ + (ρ0 − ρ∞) e^(−u/u0) falls from about 0.7 (inbox unchanged) to about
0.2 (new inbox). It repeats exactly with a probability that falls with the size of its move, and its noise grows with
the move. Fitted on the white probes alone it closes two thirds of the gap in c1; with three parameters and the gains
taken from the pilot it reproduces ten pilot observables to an r.m.s. of 1.2–2.5 simulated s.d. (the old i.i.d.-noise
surrogate: 2.4–99), which is in-sample. Two consequences: the transfer gains are one-shot gains, (1 − ρ∞) ≈ 0.81 times
the target gains; and the gating scale u0 is barely identified (s.e. as large as the value). Code:
`experiments/04_swarm_design/surrogate_v2.py`. It misses the γ dependence of c2 at K=4 (pilot +0.29 between γ = ±1,
law +0.07 ± 0.07) and the heavy-tailed noise.

## 3. Verdicts on the six claims, and what the data have tested

| claim | verdict from the derivations | tested by data so far |
|---|---|---|
| 1. DMFT reduction (`claim1.md`) | **Holds with three corrections**: full ⟨ss⟩ noise covariance, a two-round reaction term ΣR(t,t′)s(t′), the neighbour population's kernel; matches an N = 4000 simulation to 0.004. | **Scalar-field premise tested and revised**: a two-channel field is needed (Haiku, Physics of Agents). The reaction term is **confirmed in sign structure** by kick stage 1 (section 4): echoes of the predicted sign in all four \|γ\| = 1 cells at 4.9–10.5 s.e., exactly zero at γ = 0, and composed from single-link responses as the reaction term assumes. Zero-shot DMFT curves are no longer zero-shot, since the agent law was tuned on the pilot. |
| 2. Sharp γ_c into frozen polarization (`claim2.md`) | **False as stated.** Boundary d·g(1+\|γ\|) = 1 for memoryless agents; γ < 0 destabilizes too (period 4); no freezing for γ < 1. | **Unmeasurable for Haiku.** Anchoring moves the γ = −1 boundary to d·g = ½√((1+ρ)/(1−ρ)) ≈ 1.19 at ρ ≈ 0.7 and leaves the static one in place (`quiet_regime.md` 2.2); real gains give d·g ≈ 0.4. P_δ = (1−c2)/(1−c1) for any stationary sequence and carries no period information. Period-2 flicker as a synchronous-update effect is **confirmed** in the Physics of Agents data. |
| 3. Two-site cavity and PR (`claim3.md`) | **Theory holds; safety claim false**: PR doesn't fall before a zealot flip, N·Var(m_B) rises 5–20×. | **Untested.** Needs a model and setting with a tipping point (section 5). |
| 4. Zealot threshold and hysteresis (`claim4.md`) | Threshold and hysteresis already known (De Marzo et al. 2026); **the spec's γ effect has the wrong sign**: γ > 0 widens the loop. | **Untested.** Hysteresis needs bistability (steep conformity); Haiku's response is not steep, so it likely has none in this game. |
| 5. Sparse graphs (`claim5.md`) | Holds in its near-tautological form; at small K simulate on the actual graph. | **Consistent**: all current predictions are simulations on actual graphs at K = 2–4. On K = 2, γ = +1 graphs the largest eigenvalue of d·W is 0.73–0.84 with target gains, the one cell with slow modes. |
| 6. Mechanistic bridge (`claim6.md`) | **Not meaningful as written** (confidence confound; the contrast extracts content, not coupling). | **Untested**; a single-agent gain-direction test on an open model comes first. |

Derived and verified numerically: claim 1's equations; the stability boundary with quenched fields and its anchored
form; the two-site system for linear dynamics at any γ and for nonlinear agents at γ = 0; the zealot thresholds; the
PR estimator; the kick-and-echo formulas (K1)–(K5). Derived but not verified: the multi-population forms of claims
2–3, the two-site system for nonlinear agents with γ ≠ 0, a single-site DMFT of the anchored law (it is a two-channel
case of claim 1 eqs. (11)–(12), not worked out).

## 4. Kick and echo: the frozen predictions (`quiet_regime.md` section 3)

Kick one agent's outgoing message once (mirror it: p → 1 − p, HEADS ↔ TAILS) as its readers see it at round t0+1,
leave its own history true, and compare its answer at t0+2 with an unperturbed twin; unaffected agents replay from
the cache, identical and free. The echo is exactly zero without reciprocated ties (γ = 0), and otherwise it is the
reaction term ΓR of claim 1:

E = Σ_types (n_type/K²) a_ij a_ji ⟨d_i d_j⟩  →  positive through concordant ties in both wordings, negative through
discordant ties when the disagree channel repels (inverted) and positive when it attracts (label).

Settled kicks (t0 ∈ {6, 8}), N = 64, p₋ = 0.5. "Linear" uses the real-message gains (±1 s.d.); "anchoring" simulates
the law of section 2.

| wording, K | γ = −1: E linear | E anchoring | γ = +1: E linear | E anchoring |
|---|---|---|---|---|
| inverted, 2 | −0.0285 ± 0.0051 | −0.0128 | +0.0581 ± 0.0067 | +0.0672 |
| inverted, 4 | −0.0200 ± 0.0039 | −0.0063 | +0.0300 ± 0.0046 | +0.0132 |
| label, 2 | +0.0329 ± 0.0065 | +0.0208 | +0.0388 ± 0.0075 | +0.0307 |
| label, 4 | +0.0110 ± 0.0024 | +0.0030 | +0.0123 ± 0.0027 | +0.0047 |

Two relations need no gains at all: E(±0.5) = ½E(±1), and the factorization slope F (the echo regressed on the
partners' measured first-order changes, weighted by slope factors, eq. (K5)) equals 1 if responses compose linearly,
equals the open-path control F_open unless agents discount their own returning influence, and falls to about 0.75
(K = 2) or 0.53 (K = 4) under anchoring. Status: the machinery is verified (exact zero at γ = 0 and the predicted
sign on a mock; only downstream agents are paid), a 384-twin pilot on the pilot graphs passed every methods check,
and stage 1 ran as frozen in `experiments/07_kick/DESIGN.md` (both wordings, K = 2, γ ∈ {−1, 0, +1}, 9,048 twins,
$45.9). **Stage 1 result** (`experiments/07_kick/RESULT.md`):

- *Sign structure confirmed.* E = −0.0146 ± 0.0030 (inverted, γ = −1), +0.0137 ± 0.0018 (label, γ = −1), +0.0567 ±
  0.0054 (inverted, γ = +1), +0.0151 ± 0.0020 (label, γ = +1), exactly 0 at γ = 0; sham controls show no resampling
  offset.
- *Composition holds:* the factorization slope matched to open paths differs by +0.05 ± 0.05 (predicted +0.06), so the
  reaction term can be computed from population responses.
- *Linear response over anchoring, marginally:* pooled F = 0.86 ± 0.06 and F_open = 0.88 ± 0.04 against 0.98/0.945
  (linear) and 0.67/0.63 (anchoring).
- *Magnitudes:* the inverted echoes match the linear prediction; the label echoes are about half, because label
  readers' single-message responses are half the transfer-check gains, and composing the measured responses
  reproduces them.
- *Unplanned finding:* agents under-react to inputs that confirm their lean (slope ≈ 0.6) and react fully to inputs
  that contradict it (≈ 1.0–1.1), on closed and open paths alike, which a stated-confidence ceiling would produce.

## 5. What the open-weight track is for

Haiku can't reach the regimes where claims 2–4 make visible predictions, and the API budget caps Claude swarms near
N = 64. Qwen3.5-9B on vLLM (`envs/vllm_backend.py`; beliefs from digit-token logprobs, so the stated number has no
sampling noise) can run N = 256–1024. The question being measured (`experiments/06_qwen_*`) is whether any
(K, wording) puts a Qwen swarm near the stability boundary (Λ ≳ 0.7) or makes it bistable (conformity slope > 1),
using real-message calibration. If so, that is where the oscillation, zealot-hysteresis and Var(m_B) predictions get
tested. **Measured (`experiments/06-qwen-calibration/RESULT.md`): no.** In the coin game Qwen3.5-9B couples more weakly
than Haiku (inverted K = 2: a₊ = 0.33 against Haiku's 0.80), Λ_max ≤ 0.45 ± 0.05 in every (K, wording) cell, conformity
slopes are 0.36–0.58 with a single fixed point, and its answers carry 2.5× Haiku's noise, almost all from the sampled
rationale and message (the logprob readout removes only about 4% of it). Unlike Haiku it does not freeze (late answers
keep moving by ≈ 0.09 per round). A K = 1,024 round costs $0.14 and 14 minutes on an A40. The analyst scenario was
no-go for the same reason (section 6). So neither model available to the project reaches the regimes where claims 2–4
make visible predictions; the Physics of Agents data suggest that committed binary votes with few, strongly weighted
neighbours (per-neighbour logit gains 1.4–3.4) are where strong coupling lives, which a future design could target.

## 6. The analyst scenario: frozen predictions where the theory should be valid and nontrivial (`analyst_scenario.md`)

Requested so that the cavity theory is tested where its Gaussian closure holds (K = 16 inputs per agent) and the
disorder is order one: N = 256–1,024 Qwen analysts making committed BUY/SELL calls on fictional assets, each with a
standing research note, fresh private news every round, and K = 16 colleagues on a signed trust graph (allies and
faded colleagues), in three worlds of equal mean trust: mutual (γ = +1), one-way (γ = 0), antagonistic (γ = −1).
Mean-field (Curie–Weiss) theory predicts the three worlds are identical. The predictions were frozen before any data
(commit cad1d12; numbers in `checks/analyst_predictions.json`), for an agent whose half log-odds are linear in the
inbox and whose call is sampled from its belief:

- **The boundary exists only in the reciprocal worlds.** Sampling noise in committed calls keeps the one-way world
  below Λ = 2/π at any gain; the mutual world's boundary is the spin-glass transition (ā = 4.23 at no news, about 5
  with news of logit s.d. 0.5, about 5.9 with standing notes); the antagonistic world is the mutual one under a
  period-4 sign pattern (about 5.1). This replaces the (1+|γ|) factor of `claim2.md` for committed calls.
- **The reciprocity signatures are large well before the boundary**: echo ρ₂ = +0.12 / 0 / −0.12 at ā = 2 and
  +0.26 / 0 / −0.26 at ā = 4 (mutual / one-way / antagonistic), against a run-to-run s.d. of about 0.01 at N = 256;
  period-2 and period-4 spectral peaks that an asynchronous schedule removes; amplification 2.4 / 1.4 / 2.2 and
  correlation times of 5–6 rounds at ā = 4; kick echoes of ±0.25 or exactly zero. Even the mean belief under a common
  bias differs between worlds (0.108 / 0.076 / 0.047), which Curie–Weiss can't produce.
- **The Gaussian closure holds at K = 16**: the K = ∞ solver matches graph simulations to 0.002 in ⟨s²⟩ and 1–4% in the
  echo at every N tried, so going from 256 to 1,024 should halve the errors and change nothing else.
- **Go/no-go**: the three-world contrast needs ā ≥ 1 and a₋/a₊ > 0.3 (faded colleagues must repel); reaching the
  boundary needs a wording with ā ≳ 4.6.
- **Calibration result (`experiments/08-analyst-calibration/RESULT.md`): no-go at K = 16.** Qwen3.5-9B analysts couple
  weakly (ā = 0.72–1.24 over four trust wordings; Λ_mutual = 0.010–0.015) and follow their faded colleagues instead
  of fading them (a₋ = −0.12 to −0.27 in every wording), with a SELL bias of c ≈ −0.65 that saturates beliefs. The
  law's premise (A1) is not supported as written: readers partly average within the ally group instead of adding,
  so ā doesn't grow with K. The pilot (N = 256) shows no echo in any world, consistent with the weak coupling.
- **Second calibration (`experiments/08b-analyst-calibration-2/RESULT.md`): still no-go.** Showing faded calls
  already reversed by the experimenter made analysts follow faded colleagues more (a₋ = −0.52), apparently because
  readers undo the reversal and the faded notes argue for the original call; only template notes carrying nothing but
  the call give a positive a₋ (+0.10). No prompt variant removes the SELL bias (best c = −0.38, listing SELL first).
  With the measured bias no cell has a resolvable echo (best |Γ|(1−q)² = 0.024 against 0.03). With the bias removed
  by the experimenter the mutual echo would be resolvable (ρ₂ ≈ +0.06 to +0.09), and since a₋ < 0 the theory's sign
  rule predicts a positive echo in the "antagonistic" graphs; whether to run that test is the user's call. Small K is
  deferred.

## Files

- `claim1.md` … `claim6.md`: the original derivations and red-team, with verdicts; `solver_spec.md` and
  `dmft_solver.py`: the Gaussian DMFT solver (gate a, reference tests in `tests/test_solver.py`).
- `quiet_regime.md`: the anchored agent law, what still discriminates, kick-and-echo derivation and predictions.
- `analyst_scenario.md`: frozen predictions for the K = 16 analyst swarms (three trust worlds), with `checks/analyst_lib.py`.
- `checks/`: one script per verified formula with its printed output; `checks/kick_predictions.json` and
  `checks/kick_surrogate_table.json` hold the frozen kick numbers.
- Data behind section 1: `experiments/03_calibration/`, `experiments/05_swarm/` (pilot, transfer),
  `results/poa/README.md`, and the status log in `HANDOFF.md`.
