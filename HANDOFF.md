# HANDOFF: swarm-cavity

Last updated: 2026-11-25 16:30 UTC. Working copy: `/workspace/swarm-cavity` again (the volume is now 1 TB; the
`/workspace-global` clone is stale). The overnight session died at about 08:25 UTC when its OAuth token was revoked;
this afternoon's session resumed the plan at 14:20 and lost its editor connection at 14:42 (nothing was lost; see
"Afternoon session").

## Spend against the $2,000

| | so far |
|---|---|
| Anthropic API (experiment code, `budget/ledger.jsonl`) | $10.1 (running; see `sc-spend`) |
| Runpod pods (`orch spend`) | ≈ $0.80 (one A40 worker running) |
| **total** | **≈ $11** |

## Afternoon session (14:20–15:00 UTC): pilot and transfer check

- **Infrastructure.** Shared Runpod key switched to the account that owns volume `2we3auziu3` (old key ending
  `uhtn`; backups in `/workspace/.env-backups/`); `orch doctor` is green after a one-line fix to the toolkit's
  `engine/rpctl` (it now reads `/workspace/.env` too; uncommitted in runpod-research). `sc-spend` runs from this copy.
  `engine/ledger.total_api_spend` accepts a trailing `*` so one BudgetGuard cap can cover a sweep of runs.
- **Pilot swarm** (`experiments/05_swarm/cells.py pilot`, tag `pilot-20261125`, $3.02, 0 of 8,448 replies missing):
  inverted wording, p_neg=0.5, K=2 and 4, γ ∈ {−1, 0, +1}, N=64, T=10, one graph each. Excluded from confirmatory
  analysis. The thermal observables disagree with the surrogate's predictions: c1 = 0.68–0.80 (predicted ≈ 0.3),
  c2 = +0.33 to +0.61 with no sign flip at γ=−1 (predicted −0.14 to −0.36), P_delta ≈ 1.7–2.4 everywhere. c2 does
  rise with γ in both K (K=4: 0.33, 0.40, 0.61), so the direction matches but the magnitude doesn't. The swarm
  settles: mean per-round change falls from ≈ 0.2 at round 1 to 0.02–0.04 by round 6, and 30–48% of agents leave
  their answer unchanged in late rounds, so replica differences persist instead of being refreshed each round.
- **Why the surrogate missed it.** The single-agent white-probe trajectories (field redrawn every round, sd 0.34)
  show weak persistence: residual lag-1 autocorrelation 0.21, own-answer weight 0.19, only 2–10% unchanged answers.
  So the stickiness depends on state: the agent repeats itself when its inbox brings nothing new ("the new messages
  don't give me independent evidence, so I'm holding steady"), which the white probes never allowed. The surrogate's
  i.i.d. noise (σ_ξ = 0.10 from repeated identical prompts) is the wrong noise model for a settled swarm.
- **Transfer check** (`experiments/05_swarm/transfer.py`, tag `transfer-20261125`, $1.78, 600 paired probes per
  cell, 0 missing): the same inbox probabilities delivered with real pilot messages against template messages,
  fitted with E[s] = tanh(a₊ f_A − a₋ f_D + a_p h_priv + c); paired bootstrap s.e. in parentheses.

  | wording, K | real a₊ | real a₋ | template a₊ | template a₋ | real − template a₊ | real − template a₋ |
  |---|---|---|---|---|---|---|
  | inverted, 2 | 0.80 (0.03) | 0.15 (0.02) | 1.08 (0.04) | 0.39 (0.03) | −0.28 (0.04) | −0.24 (0.03) |
  | inverted, 4 | 0.66 (0.05) | 0.22 (0.04) | 1.35 (0.06) | 0.52 (0.04) | −0.69 (0.06) | −0.30 (0.05) |
  | label, 2 | 0.54 (0.03) | −0.24 (0.02) | 0.71 (0.03) | −0.25 (0.02) | −0.18 (0.04) | +0.01 (0.02) |
  | label, 4 | 0.40 (0.03) | −0.21 (0.02) | 0.78 (0.03) | −0.37 (0.03) | −0.38 (0.04) | +0.15 (0.03) |

  The template calibration does not transfer: real messages couple 25–50% more weakly, and the inverted disagree
  channel is nearly switched off (a₋ = 0.15–0.22 against the design's assumed 0.76). Real gains per unit field fall
  from K=2 to K=4, so agents average their inbox rather than add it. With real gains, the strongest cell (inverted,
  K=2) has g ≈ 0.41 and effective reciprocity at γ=−1 of only −0.36, so Λ ≈ 0.45: the Haiku swarm is in the quiet
  regime in every cell the design considered, and the γ<0 period-4 signature the design was built around is not
  reachable with this prompt.
- **Not yet done.** The swarm design (DESIGN.md) has to be redone with the real-message gains and a corrected noise
  model before Test 2 spends anything; `predictions.json` in `experiments/04_swarm_design/` uses the old gains.
- **GPU track.** A subagent wrote `envs/vllm_backend.py` (Qwen3.5-9B on vLLM, belief = expected stated probability
  from digit-token logprobs, missing when valid mass < 0.95) and died with the editor before testing it on a GPU.
  Reviewed by reading: complete and consistent; `language_model_only=True` and the vLLM structured-output import
  are unverified until a worker runs it. No pod was launched.
- **Tests.** All 114 pass file by file. The full `uv run pytest` is killed (exit 137) because `tests/test_solver.py`
  peaks at 3.4 GB and the container's 8 GB cap already holds 4.4 GB of editor and Claude processes; that's a
  memory limit, not a regression.

Agent usage (the Fable theory agent, Opus implementation agents) isn't counted here, per your instructions.

## Spend alerts

`scripts/spend_watch.py` runs in tmux session `sc-spend` (`tmux attach -t sc-spend`), checking every 5 minutes. It
sums `cost_usd` over the ledger plus this project's pod spend, taking max(billed, estimated) because Runpod's billed
figure lags by hours, and appends one line to `budget/alerts.log` each time the total passes a new $100 multiple. The
last threshold it alerted on is kept in `budget/alerts_state.json`, so restarts don't re-alert. If spend jumps across
two thresholds between checks, you get one alert for the higher one. This session keeps a Monitor on
`tail -n0 -F budget/alerts.log` and sends a PushNotification for each line.

Test push at 04:06 UTC: the tool reported **"Terminal notification sent. Mobile push requested."** That means it was
handed to Remote Control for your phone. The tool can't confirm it actually arrived, so if you didn't get it, Remote
Control wasn't connected.

## Done

- Repo bootstrapped: `docs/spec.md` (your program verbatim), `PLAN.md`, `CLAUDE.md` (guardrails, verified API
  prices), `orch.toml` (prefix `aniket-sc`, same coordinator and volumes as participation-ratios), `pyproject.toml`.
- API prices in the spec checked against the claude-api skill: all four list prices are correct. Two things the spec
  didn't mention matter for the design. The Claude API returns no logprobs, so beliefs come from a structured
  probability field. And Haiku 5.5 thinks by default with billed thinking tokens, so agent calls will run at low effort.

- **Literature red-team** (`docs/literature.md`). Both cited papers exist, but the spec misdescribes them. Flag Game
  (Pavlova & Tanaka) does have transcript memory in its experiments and a zealot (asymmetric) voter theory, with only
  qualitative fits, and it releases no code. Physics of Agents (El et al.) fits a kinetic logistic Glauber model with
  three scalar couplings on symmetric sparse graphs, not a static Ising model. Its code (Apache-2.0) is in
  `third_party/physics-of-agents/`, and its 9,600 trajectories are on Hugging Face as a free pilot dataset. The
  Hugging Face incident was a spontaneous false belief spreading, not instructed zealots. Verdicts on the claims:
  - Claim 1 is known mathematics. The spec's noise covariance should use the full correlation, not the connected part.
  - Claim 2's sharp γ_c into a frozen phase contradicts Crisanti–Sompolinsky and CKLP. For γ<1 there's no freezing,
    only a correlation time that grows as γ→1. The linear boundary g(1+γ)χ=1 is known (the elliptic law).
  - Claim 3's two-site theory exists already (Clark et al. 2023). PR works only as an ensemble diagnostic, not as a
    per-run detector at N=32.
  - Claim 4's tipping and hysteresis in LLM populations were already shown by De Marzo et al. 2026. Only the
    γ-dependence is new, and the spec's predicted direction is probably backwards.
  - Claims 5 and 6 are plausible but low priority.

- **Engine** (`engine/`, 22 tests pass): pricing from `usage`, an atomic ledger, BudgetGuard, structured beliefs,
  two-breakpoint prompt caching, realtime and Batch API paths, per-round checkpoints and a free replay cache. The live
  check on Haiku 5.5 showed that ledger costs match `usage` exactly and the cached prefix is read on repeat calls. A
  call costs about $0.0002–0.0004, and Haiku spends 150–200 thinking tokens per call even at effort low. A 3-request
  batch took 6 minutes.
- **Refusal fix** (`experiments/00_refusal_ab/`). The spec's `private_reasoning` field drew `reasoning_extraction`
  refusals on 6 of 12 calls. Renamed to `rationale` with neutral wording, it drew 0 of 12, with unchanged beliefs.
  Cost $0.008.

- **Theory** (`theory/`, Fable 5.2; summary in `theory/README.md`). Every formula was checked against direct
  simulation of a toy swarm. Claim 1 holds after three corrections: the noise covariance uses the full correlation,
  the echo is two rounds rather than three, and the kernel is the neighbour population's. The corrected equations
  match an N=4000 simulation to 0.004. The other claims:
  - Claim 2 is false as stated. The boundary is d·g(1+|γ|)=1, and negative γ destabilizes too, into period-4
    oscillation instead of period 2. Freezing happens only at γ=1, after hundreds of rounds, so nothing sharp in γ is
    visible at T=10.
  - Claim 3's PR theory holds, but PR doesn't fall before a zealot flip. N·Var(m_B) rises 5–20× instead.
  - Claim 4 has the sign of the γ effect backwards. γ>0 widens the hysteresis loop and barely moves the forward
    threshold.
  - Claim 6 as written extracts a content direction, not a coupling.
  Two zero-cost predictions already pass on the Physics of Agents data: period-2 flicker on bipartite graphs, and
  coupling-induced replica overlap.

## Revised experiment plan (replaces the spec's Exps 1–3; Exp 4 is on hold)

In order, with estimated costs: (0) zero-cost Physics of Agents analyses; (1) single-agent calibration, testing
scalar-field sufficiency and measuring conformity slope and channel gains, about $2–40; (2) the reciprocity signature
at N=64, period index vs γ with an asynchronous control, about $160; (3) zero-shot DMFT curves at N=32 and 64, about
$25–180; (4) zealot hysteresis vs γ, about $170; (5) PR vs Var(m_B) as an early warning, about $60. That's about $600
in total, leaving room for Sonnet slices and reruns. Test 1 decides whether the couplings are strong enough for tests
2–4 to show anything.

- **DMFT solver** (`theory/dmft_solver.py`, gate a). It reproduces every reference number in `theory/solver_spec.md`
  at the stated tolerance (114 tests pass in total). The variants with the spec's original errors fail the kick test,
  as they should. It also found three errata in the spec's own reference values.
- **Swarm environment and probes** (`envs/`, `cavity/`). The game is a neutral coin with sensor reports. Prompts are
  shared between the swarm and the probes. The swarm reads signed inboxes on directed graphs with fixed in-degree,
  runs synchronously or asynchronously, and has a mock backend for zero-cost tests. The prompt check gave 0 of 20
  refusals.
- **Gate (b) smoke test** (`experiments/02_smoke/`): N=8, T=4 on Haiku, 0 of 102 calls missing, $0.035, and a resume
  costs nothing. The swarm tracks the truth.
- **Test 1, single-agent calibration** (`experiments/03_calibration/RESULT.md`, $2.10).
  - The nominal scalar field fails (η²_comp = 0.48). Peers "you tend to disagree with" still attract at about 40% of
    agree strength, as in the Physics of Agents small models.
  - A two-channel effective field fits, and since it's linear in the peers' states, DMFT still applies with
    effective couplings.
  - A wording that gives agents a reason to invert disagree sources flips their gain to +0.57, which makes J̄ ≈ 0 and
    γ < 0 reachable.
  - The memory kernel is 0.50, 0.15, 0.08 at lags 1–3, carried by the agent's own message content.
  - At K=8 the swarm sits in the quiet regime (Λ ≈ 0.35 even at |γ|=1), so reciprocity effects there are O(g²) ≈ 0.05.

## Evening session (tmux `sc-work`, from 15:00 UTC)

- **Kick twins built and verified** (`envs/kick.py`, `tests/test_kick.py`, 7 tests): restoring a replica at round
  t0, mirroring one agent's outgoing message as its readers see it, and replaying rounds t0+1, t0+2. On the mock
  agent the echo is exactly zero without reciprocity and has the sign of J_ij J_ji with it; through the real Runner
  (fake client) only downstream agents reach the API, and replays add no ledger lines. On a real Haiku base run one
  twin sent exactly the kicked agent's 2 readers at t0+1 and the predicted 5 agents at t0+2 (7 calls, not 128).
- **Kick pilot** (`experiments/07_kick/kick.py`, tag `kickpilot-20261125`, $1.93): every agent of replica 0 of the
  six pilot runs kicked once at t0=5, 384 twins, 5,301 calls, 0 missing; methods checks clean (kicked agent never
  changed at t0+1, no non-reader resent). 7 of 384 kicks were zero because the agent stood at p=0.50, which a mirror
  can't move; confirmatory runs will skip such agents. Echo values not yet examined: they wait until the theory
  agent's predictions are frozen (`experiments/07_kick/analyze_kick.py`, slopes through the origin, leave-one-out
  factorization test, verified on synthetic twins).
- **Theory revised** (`theory/quiet_regime.md`, Fable agent, commit 91e267c): an anchored agent law (own-answer weight
  ≈0.7 when the inbox is unchanged, ≈0.2 when it is new) fits both the white probes and the pilot; period-4 and
  amplification predictions are unmeasurable for Haiku; the reaction term is measured directly by kick and echo,
  with frozen predictions (linear response vs anchoring law). `theory/README.md` rewritten as the single current
  statement of the theory (commit eb379f8).
- **Kick stage 1 designed and frozen** (`experiments/07_kick/DESIGN.md` revision 3, commit 9056751): both wordings,
  K=2, γ ∈ {−1, 0, +1}, 8 graphs × 2 replicas, ~2,000 twins per |γ|=1 cell, 256 sham twins per cell, decision rules
  fixed before data. Two blind reviews: the first found a P0 (batch base runs vs realtime twins with no sham control)
  and five P1s, all fixed; the second found two rule problems (rule 2 centred on the wrong null, an underpowered
  per-cell sham check), fixed by refreezing F, F_open and the matched difference under both laws with the revised
  estimators (`refreeze_predictions.py`) and pooling the sham test. A content control from the transfer data showed
  fresh agents respond 0.62–0.84 as much per unit to weakly leaning messages, so the kick-size ratio is reported
  only. Batch API path checked (works, 50% off, ~6 min per round) but not used for stage 1.
- **Base runs** (tag `kick1-20261125`, 48 runs, ~$26 projected): the first launch was killed by the container's
  8 GB memory limit after 1,534 calls ($0.37; checkpointed, nothing lost) because every Runner held its own copy of
  the 180 MB replay cache. Fixed in `engine/runner.py` (one shared cache per process, slim entries, incremental
  reads; commit after 9056751) and relaunched with 16 concurrent runs (process RSS ≈ 380 MB) in tmux `sc-kick1base`.
- **GPU track** (Opus agent): worker `aniket-sc-06-qwen-calibration` running on an A40 at $0.59/h (8 h cap); pod
  spend so far ≈ $0.80 (orch estimate).
- **Analyst scenario** (new request from the user, relayed by the editor session): a Fable agent is writing frozen
  predictions (`theory/analyst_scenario.md`, `theory/checks/analyst_predictions.json`) for N = 256–1,024 Qwen analysts
  at K = 16 in mutual / one-way / antagonistic trust worlds. Spend tracking for this scenario: projection before every
  launch, worker spend logged here, a message to the editor session at each $50 crossed, stop if a worker exceeds
  1.5× its projection. Predictions frozen (commit cad1d12; summary relayed to the editor session): the boundary exists
  only in the reciprocal worlds (one-way stays below Λ = 2/π), mutual needs ā ≈ 5–5.9, echoes ±0.12 at ā = 2 and
  ±0.26 at ā = 4. Step 2 (Opus agent, `experiments/08_analyst_calibration/`): analyst env, Qwen pilot for real
  notes, calibration and go/no-go.
  Worker log for this scenario: `aniket-sc-08-analyst-calibration` (L40S, $1.09/h), launched ~17:30 UTC, projected
  ~1.4 h ≈ $1.5, worst case $3.27; ran 0.9 h, ≈ $1.0, deleted. **Result: no-go at K = 16**
  (`experiments/08-analyst-calibration/RESULT.md`): ā = 0.72–1.24 (boundary needs ≈ 4.6), Λ_mutual ≤ 0.015, faded
  colleagues attract (a₋ < 0 in all four wordings), SELL bias c ≈ −0.65; pilot (N = 256, 0 missing, 50 s/round on an
  L40S) shows no echo. Relayed to the editor session with a question for the user about a small-K variant. Second
  calibration (counterbalanced polarity, experimenter-inverted fading, K = 4 and 2) launched with the same agent,
  budget ≈ $5. Scenario spend so far: ≈ $1.0.

## Next

1. Theory (Fable agent): revise the single-agent surrogate with the real-message gains and a state-dependent noise
   model (stickiness when the inbox is stable), and re-derive which observables can still discriminate the theory
   in the quiet regime. A kick-and-echo measurement looks like the cleanest: perturb one agent's message once and
   read its own answer two rounds later. The replay cache makes unaffected agents free, the echo is exactly zero
   at γ=0, and its sign is set by γ and the wording. It measures the reaction term without relying on any noise model.
2. Redo DESIGN.md with pre-registered predictions, get an unanchored review, then run Test 2 (pilot first).
3. GPU track: finish validating `envs/vllm_backend.py` on a worker, time large-N rounds, calibrate Qwen3.5-9B with
   real-message inboxes (templates are now known not to transfer).
4. Zealot hysteresis and the Var(m_B) early warning on whichever model shows strong enough coupling.

## Decisions for you to review

- **Belief field renamed** from the spec's `private_reasoning` to `rationale`, because the original name caused 50%
  refusals (see Done). A related risk: Experiment 2's zealots are "secretly instructed" to push a backdoored PR past a
  monitor, and that framing may draw refusals too. I'll pilot its prompts for refusal rate before any sweep, and if
  they're refused I'll reframe the zealots as agents with sincerely wrong evidence (closer to the real Hugging Face
  incident anyway).

- **The volume and the Runpod key are fixed** (resized to 1 TB; shared key switched, see Afternoon session). The
  venv still lives on the container disk (`scripts/env.sh`), which a restart wipes; `uv sync` rebuilds it.

## Blocked

Nothing.
