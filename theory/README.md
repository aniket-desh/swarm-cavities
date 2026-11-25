# Theory for swarm-cavity: the current statement

Last updated 2026-11-25 (evening). This file is the one place to read for the state of the theory. The derivations
live in the notes it links (`claim1.md` … `claim6.md`, `quiet_regime.md`), each of which opens with its verdict and
names the scripts in `theory/checks/` that verify its formulas.

**Where things stand.** The cavity reduction itself holds after three corrections, but the swarms we can afford to
run don't sit where its striking predictions live. Haiku 5.5 agents in this game are weakly coupled once real
messages are used (real messages couple 25–50% more weakly than the templates the first calibration used, and the
inverted disagree channel nearly switches off), and they anchor on their own last answer whenever their inbox brings
nothing new. Both push every Haiku cell into a quiet regime (Λ ≈ 0.45 at best) where the oscillation and
amplification signatures of claims 2–3 can't be measured. What still isolates the cavity-specific structure is the
reaction term itself, which a kick-and-echo experiment measures directly; its predictions are frozen and stage 1 is
being set up. The open-weight track (Qwen3.5-9B) is the route to stronger coupling and large N: in the Physics of
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
| 1. DMFT reduction (`claim1.md`) | **Holds with three corrections**: full ⟨ss⟩ noise covariance, a two-round reaction term ΣR(t,t′)s(t′), the neighbour population's kernel; matches an N = 4000 simulation to 0.004. | **Scalar-field premise tested and revised**: a two-channel field is needed (Haiku, Physics of Agents). The reaction term is **untested**; kick and echo (section 4) tests it. Zero-shot DMFT curves are no longer zero-shot, since the agent law was tuned on the pilot. |
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
and stage 1 (both wordings, K = 2, γ ∈ {−1, 0, +1}, about 2,000 twins per cell, about $40) is defined in
`experiments/07_kick/DESIGN.md`. **No echo has been compared with these predictions yet.**

## 5. What the open-weight track is for

Haiku can't reach the regimes where claims 2–4 make visible predictions, and the API budget caps Claude swarms near
N = 64. Qwen3.5-9B on vLLM (`envs/vllm_backend.py`; beliefs from digit-token logprobs, so the stated number has no
sampling noise) can run N = 256–1024. The question being measured (`experiments/06_qwen_*`) is whether any
(K, wording) puts a Qwen swarm near the stability boundary (Λ ≳ 0.7) or makes it bistable (conformity slope > 1),
using real-message calibration. If so, that is where the oscillation, zealot-hysteresis and Var(m_B) predictions get
tested. **Not measured yet.**

## Files

- `claim1.md` … `claim6.md`: the original derivations and red-team, with verdicts; `solver_spec.md` and
  `dmft_solver.py`: the Gaussian DMFT solver (gate a, reference tests in `tests/test_solver.py`).
- `quiet_regime.md`: the anchored agent law, what still discriminates, kick-and-echo derivation and predictions.
- `checks/`: one script per verified formula with its printed output; `checks/kick_predictions.json` and
  `checks/kick_surrogate_table.json` hold the frozen kick numbers.
- Data behind section 1: `experiments/03_calibration/`, `experiments/05_swarm/` (pilot, transfer),
  `results/poa/README.md`, and the status log in `HANDOFF.md`.
