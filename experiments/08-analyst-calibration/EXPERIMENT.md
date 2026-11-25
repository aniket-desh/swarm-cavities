---
name: 08-analyst-calibration
title: Analyst scenario: Qwen3.5-9B pilot swarm (N=256, K=16) and single-agent calibration of the law (A1)-(A3) with real colleague notes
status: draft
gpu: NVIDIA A40,NVIDIA RTX A6000,NVIDIA L40S
count: 1
min_gpu_gb: 40
hours: 3
budget_usd: 4
disk_gb: 120
data:
secrets:
---

# Analyst scenario: Qwen3.5-9B pilot swarm (N=256, K=16) and single-agent calibration of the law (A1)-(A3) with real colleague notes

## Question

`theory/analyst_scenario.md` froze predictions for swarms of BUY/SELL analysts in three trust worlds (mutual, one-way,
antagonistic, K = 16) under the agent law (A1)–(A3): half log-odds
x = a₊ f_A − a₋ f_D + a0 h0 + ν ξ + c + κ σ_own, belief s = tanh x, call sampled from s. This experiment measures that
law for Qwen3.5-9B with real colleague notes and answers, per trust wording: what are a₊, a₋, ν, a0, c, κ (and
ρ₁ = κ⟨1−s²⟩); do the law's two unverified premises hold (log-odds linear in the ally/faded call counts; readers
respond to the call, not to the sender's confidence leaking through the note); and is the swarm sweep a go under the
theory's rules (section 7: three-world contrast iff ā ≥ 1 and a₋/a₊ > 0.3; boundary sweep iff Λ_mutual ≥ 0.7;
herding first iff (a₊−a₋)/2 > 1.3; memory flag iff ρ₁ > 0.3). The Haiku coin-game transfer check (HANDOFF.md,
afternoon session) showed templated peer messages overstate coupling by 25–50%, so the calibration uses notes
harvested from a real Qwen analyst swarm and measures the template error alongside.

## Hypothesis

No firm prior on the gains. Expectations that the data can contradict: (i) the log-odds are linear in the counts
(no majority-rule step, no saturation within |f| ≤ 1/2) and additive over the 16 colleagues rather than averaged
within each group; (ii) readers respond to the call, so the leak fraction λ is near 0; (iii) gains rise with the
wording from mild to max, with a₋/a₊ > 0.3 at least in the stronger wordings. Counting against the law: a majority
term or a full-bloc response that differs from a/2 by more than 20%, group-mean coefficients that carry the weight,
or λ ≥ 0.25. Whether any wording reaches Λ_mutual ≥ 0.7 is an open question this run answers, not a hypothesis.

## Method

Agent (`envs/analyst.py`): Qwen/Qwen3.5-9B bf16 text-only on vLLM 0.32.0 via `envs/vllm_backend.make_llm`
(validated in `experiments/06-qwen-smoke`), thinking disabled, max_num_seqs 256. Fictional asset "Orsilane Freight";
the system turn states that the asset, desk, colleagues and data are fictional. The analyst holds "your own research
note" (3 indicators with displayed reliabilities whose log-odds sum to 2·a0_nom·h0, a0_nom = 0.3, rendered by
`envs/evidence_game.evidence_for_field`), gets one news item per round (side = sign ξ, reliability
round(100·sigmoid(2·ν_nom·|ξ|))% clipped to 51–95, ν_nom = 0.5 in the pilot), sees its own previous call and note,
and the latest call and note of K = 16 colleagues in two groups, allies and faded colleagues, with a trust wording
(`envs/analyst.WORDINGS`: mild, moderate, strong, max; the strong one says "lean on their calls heavily" and "fade
their calls firmly"). No prompt text describes anything as private, hidden or secret (tested). The assistant turn is
forced to begin `{"call": "`; "BUY" and "SELL" are single tokens there (checked offline with the Qwen3.5 tokenizer
and again in step `tok`). From one forward pass with top-20 logprobs: valid mass = P(BUY) + P(SELL), recorded
missing below 0.95; P_buy = P(BUY)/valid mass; x = (log P(BUY) − log P(SELL))/2, censored at the top-20 bound if one
token is absent (flagged). The call is BUY iff u < P_buy with a stored uniform per (replica, round, analyst); the note
is then sampled at temperature 1 conditional on that call (prefix + call + `", "note": "`, regex: one quote-free
line then `"}`, max 100 tokens, one retry, else the cut text is kept and flagged; the call stands).

1. **Pilot swarm** (`run.py pilot`): `envs/analyst.AnalystSwarm`, synchronous, N = 256, K = 16, T = 10 (11 rounds),
   one replica, mild wording, one graph per world from `envs/graphs.fixed_indegree_signed_couplings(256, 16, γ, 0.5)`
   with γ = +1, 0, −1 (seeds (800, world index)); the same notes, news and call uniforms in the three worlds (game
   seed 800). 3 × 256 × 11 = 8,448 decisions and notes. Purpose: the note pool, missing and refusal rates, seconds
   per round. **Excluded from all confirmatory analysis**; its summary statistics are reported as exploratory.
2. **Calibration probes** (`run.py probes`): 1,000 base probes (seed 8008), each sent with all four wordings and two
   text arms, plus 100 repeats of mild/real probes in a different batch: 8,100 single-token decisions. Design of a
   base probe: each of 16 slots is faded with probability 1/2; group consensus m_A, m_D ~ U(−1, 1) and each call is BUY
   with probability (1+m)/2, so f_A, f_D span their range with every composition (n_allies ~ Bin(16, 1/2)); real arm:
   each slot carries a pilot note (rounds ≥ 1, complete) written after the same call, from the tercile of senders
   least ("hesitant") or most ("confident") sure of that call, one tercile per group with probability 1/2, sender
   belief stored; template arm: the same calls with neutral template notes (`envs/analyst.template_note`); own
   previous call ±1 with probability 1/2 and a pilot note written after it (same in both arms); h0 ~ N(0,1) at
   a0_nom = 0.3; ξ ~ N(0,1) at ν_nom ∈ {0.5, 1.0} with probability 1/2; round uniform in 1–9; one session tag per
   base probe shared by its arms and wordings.
3. **Analysis** (`analyze.py`, coordinator; models fixed in its docstring before the probe data exist): OLS on x per
   wording and arm: M1 (the law; a₊, a₋, ν at both news amplitudes, a0, κ, c), M2 (M1 + leak regressors
   L = Σ_group(s_j − σ_j)/K; λ = coefficient/gain; placebo in the template arm), M3 (M1 + f³ and sign f per group;
   full-bloc response vs a/2), M4 (M1 + group means Σσ/n_group: adding vs averaging). Robustness: M1 without censored
   probes and M1 fitted as tanh on s. Bootstrap over base probes, B = 1,000, paired across arms and wordings. Go/no-go
   per wording from M1 on the real arm: ā, r = a₋/a₊, J̄ = (a₊−a₋)/2, Λ_mutual = max over ω of
   `theory/checks/analyst_lib.dmft(Agent(a₊, a₋, ν_.5, a0, c, κ), "mutual")` (T = 40, 3,000 sites × 64 copies;
   s.e. from 100 bootstrap draws at 1,000 × 32), Λ at ν_1 as the news-amplitude knob, ρ₁ = κ(1 − q) with q from the
   same solver run. `analyze.py selftest` recovers known parameters from probes generated by the law (passes).

## Inputs

Qwen/Qwen3.5-9B from the Hugging Face Hub (Apache-2.0, ungated, 19.3 GB), downloaded on the worker by
`experiments/06-qwen-smoke/setup_vllm.sh`. Everything else is generated by project code from fixed seeds. No data tars.

## Commands

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/08-analyst-calibration/run.py tok smoke 2>&1 | tee results/analyst-calib-smoke.log
set -euo pipefail; uv run --no-sync python experiments/08-analyst-calibration/run.py pilot 2>&1 | tee results/analyst-calib-pilot.log
set -euo pipefail; uv run --no-sync python experiments/08-analyst-calibration/run.py probes 2>&1 | tee results/analyst-calib-probes.log
```

The third command is the timing step (a 3-round swarm at N = 64 and 24 single decisions); the driver reads its output
(missing rate, invalid tokens, notes, seconds per round) before sending the fourth. Coordinator afterwards:
`uv run python experiments/08-analyst-calibration/analyze.py` and `plot.py`.

## Expected outputs

- `results/analyst-pilot-20261125-{mutual,oneway,antagonistic}/`: `analyst_calls.jsonl` (one line per analyst and
  round: P_buy, x, valid mass, censoring, top invalid tokens, call, uniform, note, note status, inbox fields),
  `analyst_timings.jsonl`, `beliefs.npz`, `inputs.npz`, `analyst_meta.json`; ~15 MB in all. Manifest in
  `results/analyst-pilot-20261125/manifest.json`.
- `results/analyst-probes-20261125/`: `design.json` and `<wording>_<arm>.jsonl` (9 files, ~5 MB).
- `results/analyst-calib-*.log`; on the coordinator `results/analyst-calib/summary.json`, figures in
  `paper/figures/analyst_calib_*.{pdf,png}`, and `RESULT.md` in this folder.

## Controls

- Template arm at identical calls: the transfer error of a template calibration, and a placebo for the leak
  regressor (templates carry no sender belief).
- 100 repeated probes in a different batch composition: GPU non-determinism of x.
- `analyze.py selftest`: the fits recover known parameters, a known leak, and detect an averaging agent; the solver
  reproduces Λ = 0.60 at gains (4, 4) in the reference design (theory table 3.3).
- Unit tests (`tests/test_analyst.py`, 22): rendering and realized fields, forbidden words, decision arithmetic,
  missing rule, censoring, stored-uniform calls, the swarm loop equal to `analyst_lib.simulate` under the mock law in
  all three worlds, checkpoint resume, the vLLM backend on a fake engine.
- Exploratory cross-check from the pilot: one-way-world ρ₁ against κ(1 − q), and the in-swarm M1 regression.

## Success and failure criteria

Run-level:
- Failed run: vLLM does not load, BUY/SELL are not single tokens and the format can't be fixed within the run, or
  the pilot's missing rate is ≥ 5% (then the probes are not run; see "If something fails").
- Complete: pilot done with missing < 5%, all 9 probe files with missing < 5% each.

Law premises, per wording, real arm (M2–M4), judged on the bootstrap s.e.:
- Linear in the counts: both sign (majority) coefficients below 0.25 × (a/2) in magnitude or within 2 s.e. of 0, and
  the M3 full-bloc response within 20% of a/2 (or within 2 s.e.).
- Adding, not averaging: each group-mean coefficient below 25% of the corresponding f-coefficient's contribution at
  the typical group size (|g| < 0.25 × a × 8/16) or within 2 s.e. of 0.
- Call, not confidence: λ_A, λ_D below 0.25 or within 2 s.e. of 0.
- Supported: all three premises hold in every wording. Not supported: any premise fails in any wording (named).
  Inconclusive: s.e. too large to decide (2 s.e. above the 25% thresholds).

The go/no-go (theory section 7) is a decision reported per wording with its numbers, not part of the label.

## Time and cost

Setup ~15–20 min (06-qwen-smoke: uv sync, vLLM install, 19 GB download; first engine start 281 s with torch.compile).
Timing basis: in 06-qwen-smoke on an A40, prefills of ~900 tokens ran at ~8,000 tokens/s (690k tokens of stage-1
readout in 85 s at N = 256). Here a decision is one prefill of ~1,040 tokens (measured offline with the Qwen3.5
tokenizer at K = 16) and the note one prefill of ~1,050 tokens (prefix caching covers whole 528-token blocks only)
plus ~40 decoded tokens. Estimate per pilot round at N = 256: ~35 s decisions + ~35 s notes ≈ 70 s, so the pilot is
33 rounds ≈ 40 min; probes 8,100 × 1,040 tokens ≈ 8.4M tokens ≈ 18 min; smoke and tests ~5 min. Total ≈ 1.4 h ≈
$0.83 on an A40 ($0.59/h). Cap 3 h (2× the estimate); budget $4 covers the L40S fallback ($1.09/h) at the cap. One
process; weights 16.7 GiB, KV and linear-attention state ~20 GiB at utilization 0.90 on a 46–48 GB card; CPU light. If
the timing step puts a pilot round above 105 s (1.5×), the pilot runs as defined (it fits the cap up to ~3 min per
round) and N_BASE is cut to 600 before the probe step.

## If something fails

- `tok` raises because BUY or SELL is not a single token after the prefix: stop; fix the format in `envs/analyst.py`
  on the coordinator (amendment), push, rerun.
- Smoke shows decision valid mass < 0.95 in more than 1 of 24 probes: read `top_other`; if the mass goes to case
  variants or a leading space, amend the prompt's format sentence (amendment, committed first), push, rerun the smoke.
  The 0.95 rule is never loosened.
- Notes or decisions show refusal or disclaimer language (regex in `analyze.py`) in the smoke: reword the scenario's
  framing (amendment) before the pilot.
- Out of memory at engine start: `max_num_seqs` 128, then `gpu_memory_utilization` 0.85.
- Pilot missing rate ≥ 5%: stop after the pilot and report; no probes.
- Time running out: every round of the pilot and every probe batch of 512 is appended to JSONL as it finishes; a stop
  loses at most one round or batch, and a rerun resumes from the checkpoints.
- Any other error: read the traceback; fix it if it is a bug in `envs/analyst.py` or `run.py` (amendment), else stop
  and report.

## Fixed rules

- Missing rule: valid mass < 0.95 is missing; no belief is parsed from text.
- Pilot excluded from confirmatory analysis.
- Probe design (1,000 base probes, seed 8008, the distributions above) and the models M1–M4 with the premise
  thresholds above are fixed before any probe data exist.
- Go/no-go thresholds are the theory's (section 7); nothing here launches the swarm sweep.

## Questions asked before launch

None needed: the user's brief (relayed by the main session) fixed the scenario, the pilot size (N = 256, K = 16, one
graph per world, T = 10, mild wording), the calibration targets, the go/no-go rules, the GPU rule (cheapest card that
fits a 9B model) and the spend reporting, and gave a standing go-ahead that covers approval. Choices made here: the
neutral wording "your own research note" from the start (allowed by the brief; avoids a refusal cycle), four wordings
rather than three (the fourth, "max", costs only probes), and a second news amplitude in the probes (the theory names
ν as the alternative control knob).

## Review

<fill: reviewer findings>

## Amendments

None.

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

Not run yet.
