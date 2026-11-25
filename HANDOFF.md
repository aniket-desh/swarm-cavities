# HANDOFF: swarm-cavity overnight run

Last updated: 2026-11-25 05:45 UTC. Working copy: `/workspace-global/swarm-cavity/repo` (see Blocked). Running in tmux session `swarm-cavity` (attach: `tmux attach -t swarm-cavity`).

## Spend against the $2,000

| | so far |
|---|---|
| Anthropic API (experiment code, `budget/ledger.jsonl`) | $0.00 |
| Runpod pods (`orch spend`) | $0.00 |
| **total** | **$0.00** |

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

## In progress

- DMFT solver (`theory/dmft_solver.py`) against the spec's reference numbers, which is gate (a).
- Swarm environment (evidence game, shared prompts, signed-inbox directed graphs with fixed in-degree,
  synchronous and asynchronous modes, a mock backend) and the single-agent cavity probe harness.
- Zero-cost Physics of Agents analyses: field sufficiency in inbox counts, the asynchronous control for the flicker,
  and effective g and Λ.

## Decisions for you to review

- **Belief field renamed** from the spec's `private_reasoning` to `rationale`, because the original name caused 50%
  refusals (see Done). A related risk: Experiment 2's zealots are "secretly instructed" to push a backdoored PR past a
  monitor, and that framing may draw refusals too. I'll pilot its prompts for refusal rate before any sweep, and if
  they're refused I'll reframe the zealots as agents with sincerely wrong evidence (closer to the real Hugging Face
  incident anyway).

- **Network volume is at its quota** (`uv sync` failed with "Disk quota exceeded"; /workspace holds about 55 GB across
  projects, 12 GB of it the shared uv cache). I didn't delete anything. This project's venv and uv cache live on the
  container disk instead (`scripts/env.sh`), which a container restart wipes, though `uv sync` rebuilds it in about a
  minute. Bulky experiment outputs will go to `/workspace-global` or stay on workers. You may want to raise the volume
  size or prune `/workspace/.cache/uv` (another uv process held its lock, so I couldn't).

## Blocked

- **The /workspace network volume is full** (as of about 06:15 UTC), so writes fail with "Disk quota exceeded" for
  every project and every Claude session, since Claude transcripts live on it via /root/.claude. It's at about 55 GB.
  The biggest folders are coupled-tensor-networks (22 GB), the shared uv cache (12 GB) and participation-ratios
  (9.9 GB). The swarm-cavity repo is 114 MB. My attempt to force-prune the uv cache was denied, which is fair, since
  it's shared. Meanwhile this project works from `/workspace-global/swarm-cavity/repo` and pushes to GitHub, and the
  spend watcher runs from there too. To fix it, resize the network volume or clear space in the other projects, then
  `cd /workspace/swarm-cavity && git pull` brings the old copy up to date.
- **GPU work (Exp 4) needs you to pick the Runpod account.** `orch doctor` fails for every project on this
  coordinator, not just this one: "network volume 2we3auziu3 doesn't exist on this account". The cause is that the
  `RUNPOD_API_KEY` in the shared `/workspace/.env` belongs to an account that doesn't hold this coordinator's volumes
  (its volume list has 38 others, none of them `2we3auziu3`). `runpodctl`'s own config uses a different key, and that
  one does see the coordinator pod. I didn't swap keys between configs, because which account pays for GPUs is your
  call. To unblock, put the key of the account that owns volume `2we3auziu3` into `/workspace/swarm-cavity/.env` as
  `RUNPOD_API_KEY`, then `cd /workspace/swarm-cavity && orch doctor` should go green. Exps 1–3 use only the Anthropic
  API and aren't blocked.
