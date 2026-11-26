---
name: 09-analyst-signrule
title: Analyst de-biased sign-rule test: Qwen3.5-9B swarms (N=256, K=16, T=20) in mutual, one-way and antagonistic graphs, echo function and kick twins against frozen predictions
status: done
gpu: NVIDIA A40,NVIDIA L40S,NVIDIA RTX A6000
count: 1
min_gpu_gb: 40
hours: 8
budget_usd: 9
disk_gb: 120
data:
secrets:
approved_utc: 2026-11-25T21:21:53Z
result: supported
finished_utc: 2026-11-26T04:38:34Z
writeup: experiments/09-analyst-signrule/RESULT.md
---

# Analyst de-biased sign-rule test: Qwen3.5-9B swarms (N=256, K=16, T=20) in mutual, one-way and antagonistic graphs, echo function and kick twins against frozen predictions

This definition launches **four workers** from the same contract:
- W1: mutual and one-way synchronous runs;
- W2: antagonistic and one one-way synchronous run;
- W3 and W4: the asynchronous runs (Amendment 1), interleaved across the two worlds.

Each worker has the time cap and budget above, so the worst case is 4 × $9 = $36.

## Question

Does an analyst's own call come back to it through reciprocated ties with the sign of the product of the two
couplings, and not at all without reciprocity? This is the reaction term's sign rule, and `theory/analyst_signrule.md`
froze its predictions on 2026-11-25 (commit 1799988; numbers in `theory/checks/analyst_signrule_predictions.json`)
before any de-biased swarm data existed.

Because faded colleagues attract, the product of the couplings is positive in both reciprocated worlds. So this design
establishes three things, each with its own verdict part:
1. reciprocity is necessary (mutual against one-way);
2. the echo magnitude is ordered as predicted (mutual > antagonistic);
3. the antagonistic echo is positive, the sign of −a₊a₋, but only if it is resolved at 3 s.e. Otherwise that part is
   inconclusive.

A sign-blind reciprocity effect would predict the same signs. Telling it apart needs a cell with a₋ > 0, which no
wording produced.

The test is posed given the couplings Qwen3.5-9B actually has. Faded colleagues attract (a₋ = −0.146, s.e. 0.037),
so the predicted echoes are large in the mutual world, exactly zero in the one-way world, and small and positive in
the "antagonistic" world. The pattern is large, zero, small, not plus, zero, minus. This tests the sign rule given the
measured couplings, not whether agents act on distrust; they don't, in any wording tried (08, 08b). The calls are
de-biased by the experimenter, but the notes are still written from the model's biased belief.

Option (a), a three-world contrast with repelling faded colleagues, was a no-go in 08 and 08b. This is option (b).

## Hypothesis

These are the frozen predictions at the design size (rounds 5–20, sampling belief s = tanh(x − ĉ)). The central law
is M1 (calls only), with M2 (note channel) as the outer bracket. Figures in brackets are calibration s.d.

| statistic | mutual | one-way | antagonistic |
|---|---|---|---|
| X₂ = ρ₂ − ρ₁², M1 | +0.0571 (0.0024) | −0.0005 (0.0003) | +0.0096 (0.0024) |
| X₂, M2 | +0.0539 (0.0085) | −0.0002 (0.0008) | +0.0170 (0.0023) |
| zero-shot interval for X₂ | [0.03, 0.07] | 0 | [0.005, 0.021] |
| ρ₁ (M1) | −0.074 | −0.075 | −0.074 |
| c₂ (M1 / M2) | +0.097 / +0.131 | +0.014 / +0.032 | +0.029 / +0.078 |
| q (M1) | 0.226 | 0.223 | 0.223 |
| m (M1 / M2) | 0.00 / +0.37 | 0.00 / +0.35 | 0.00 / +0.32 |
| kick E per unit kick (M1 / M2; notes kept under M2) | +0.055 / +0.055 | 0 exactly | +0.0092 / +0.0170 (kept +0.0043) |
| first-order reader response w_A, w_D (logit per unit kick, M1) | +0.108, +0.009 | same | same |

What would count against the hypothesis is set out in the falsifiers under "Fixed rules".

## Method

**Agent.** `envs/analyst.py` exactly as in 08b's cell P0:
- Qwen/Qwen3.5-9B bf16 on vLLM 0.32.0, thinking off, top-100 logprobs, missing when valid mass < 0.95;
- wording "max", `PromptVariant()` default (hash-tested to be 08's prompt);
- fictional asset, ν_nom = 0.5, a0_nom = 0.3, K = 16 colleagues in an ally and a faded group, own previous call and
  note shown.

**De-biased calls** (new in `AnalystSwarm`, `cfg.chat`). The model returns x = (log P(BUY) − log P(SELL))/2. The call
is BUY iff u < (1 + tanh(x − ĉ))/2, with ĉ = −0.6366486620518087 (frozen; never changed during the run) and u a
stored uniform per (replica, round, analyst). The model then writes its note, prompted with the sampled call, from
its own prompt.

**Replicas.** The replicas run in lockstep, one engine batch per round. They share the session tag, so their prompts
are identical whenever their states are, and they share graph, notes and news, but each has its own call uniforms.

**Runs.** N = 256, K = 16, p₋ = ½, T = 20 (21 rounds), 2 replicas. Graphs come from
`fixed_indegree_signed_couplings(256, 16, γ, 0.5)`:
- mutual (γ = +1): g = 0, 1;
- one-way (γ = 0): g = 0–4;
- antagonistic (γ = −1): g = 0–4.

Each run has its own graph (seed (9090, world, g)), notes, news and uniforms (game seed 9000 + 100·world + g). That
is 12 runs, 258,048 decisions and as many notes.

**Kick twins** (`envs/analyst.kick_slice`), on settled rounds t0 ∈ {8, 12}, both replicas, one kick per analyst per
slice, analysts drawn without replacement with seed (7070, world, g, rep, t0):
- mutual: 100 kicks and 25 null twins over 8 slices;
- one-way: 20 kicks over 20 slices (protocol check);
- antagonistic: 400 kicks and 25 null twins over 20 slices.

A kick twin works as follows:
- **t0:** after round t0, the readers of analyst i see its call flipped (D = −2σ_i), with the note the model writes for
  the flipped call. i's own history and all uniforms stay the base run's.
- **t0+1:** i's readers decide again; each call is drawn by the stored uniform from the sampling belief, and the
  reader's note is rewritten.
- **t0+2:** i decides again in two replays. "Rewritten": all of the readers' notes are rewritten. "Kept": a reader's
  note is rewritten only if its call flipped.

Every twin decision is paired with a recompute of its base prompt in the same engine batch, and the twin logit is
x_base + (x_twin − x_base,recomputed). This removes bf16 batch non-determinism (up to 0.125 in x in 08b) from r, w and
E. A null twin rewrites i's readers' round-(t0+1) notes with a new seed, with no kick and calls unchanged, and records
the change in i's logit at t0+2: the note noise. Per unit kick (Δ·D/4) the twin gives r_A, r_D and w_A, w_D from the
readers by tie type, and E and E_x from i, in both replays. There is no asynchronous control (the note leaves it out).

**Analysis** (`analyze.py`, fixed before data; its docstring states every estimator):
- Per run, the note's own `analyst_signrule_check.run_stats` on the sampling belief: ρ₁–₄, X₂, c₁, c₂, X^c₂, q, m.
- Per world, the mean, with s.e. = pooled within-world run s.d. / √n_runs (9 d.f.).
- Kick statistics with cluster-bootstrap s.e. over slices and over (run, replica); every decision uses the larger.
- The in-swarm regression (calls only, and with the leak terms; cluster bootstrap over analysts).
- The reaction coefficient from the swarm's own couplings, Ĝ (S4): û from the twins' first-order responses, v̂ from
  the leak regression, d and d^r from the beliefs.
- The falsifiers below, the verdict table, and residuals of every observable against the frozen M1 and M2 values.
- Gaussian-cavity flags (guardrail 5): excess kurtosis and the bimodality coefficient of x − ĉ, conditional on the
  previous call.

Figures come from `plot.py`.

**Validation done before launch:**
- 64 tests in `tests/test_analyst.py` (77 with `tests/test_vllm_backend.py`), including two identity checks:
  - the de-biased loop under the mock law equals `analyst_signrule_check.simulate` exactly, and `analyst_lib.simulate`
    at c − ĉ;
  - `kick_slice` equals the note's `kick_twins` for every reader at t0+1 and the kicked analyst at t0+2, with E = 0
    and identical prompts in the one-way world.
- A full-size dry run of `run.py` with all 12 runs and all twins, plus `analyze.py`, under a mock backend that obeys
  frozen law M1 including its residual (see "Controls"). The harness is committed as `dryrun.py`.

## Inputs

Qwen/Qwen3.5-9B from the Hub, via `experiments/06-qwen-smoke/setup_vllm.sh`. Everything else is generated from fixed
seeds. No data tars.

## Commands

Worker W1 (mutual and one-way):

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py smoke --label W1 2>&1 | tee results/analyst-signrule-smoke-W1.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py base mutual:0 oneway:0 mutual:1 oneway:1 oneway:2 oneway:3 2>&1 | tee results/analyst-signrule-base-W1.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py kicks mutual:0 mutual:1 oneway:0 oneway:1 oneway:2 oneway:3 2>&1 | tee results/analyst-signrule-kicks-W1.log
```

Worker W2 (antagonistic and one one-way run):

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py smoke --label W2 2>&1 | tee results/analyst-signrule-smoke-W2.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py base antagonistic:0 antagonistic:1 antagonistic:2 antagonistic:3 antagonistic:4 oneway:4 2>&1 | tee results/analyst-signrule-base-W2.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py kicks antagonistic:0 antagonistic:1 antagonistic:2 antagonistic:3 antagonistic:4 oneway:4 2>&1 | tee results/analyst-signrule-kicks-W2.log
```

The smoke step is the timing step: two lockstep rounds of N = 256 × 2 replicas. The first base run on W1 is a mutual
one, and its mean belief says which law holds (reported, ĉ unchanged). Coordinator afterwards:
`uv run python experiments/09-analyst-signrule/analyze.py` and `plot.py`.

## Expected outputs

- `results/analyst-signrule-20261125-{world}-g{g}/`: `analyst_calls.jsonl` (one line per replica, round and
  analyst: x, P_buy, p_sample, call, uniform, note, inbox fields, valid mass, censoring), `analyst_timings.jsonl`,
  `inputs.npz`, `beliefs.npz`, `graph.json` and `analyst_meta.json`. About 25 MB per run.
- `results/analyst-signrule-20261125/kicks_{world}_g{g}.jsonl`: one file per run, one line per slice (a single append,
  so a stop never leaves a partial slice), holding that slice's twins.
- `results/analyst-signrule-*.log`.
- Coordinator: `results/analyst-signrule/summary.json`, figures `paper/figures/analyst_signrule_*.{pdf,png}`, and
  `RESULT.md` here.

## Controls

- **One-way world.** It is the null for the echo (X₂ = 0 under any law). Its twins are a protocol check: the kicked
  analyst's t0+2 prompt is identical to the base prompt, so E = 0 exactly.
- **Null twins** measure note-rewrite noise, and the paired base recomputes remove engine noise.
- **X₂** removes the own-call memory (ρ₂ = ρ₁² in the one-way world).
- **The full-size mock dry run** checks every estimator and falsifier against the frozen numbers. A mock analyst obeys
  law M1, with its residual made a deterministic function of the prompt. It runs through `run.py` (12 runs, all
  twins) and `analyze.py`. Two variants were run before launch, and both pass every falsifier:
  - **Residual independent of the inbox** (a function of analyst, session and round only).
    - X₂ = +0.065 (0.005), +0.0015 (0.003), +0.0099 (0.003).
    - E = +0.050 (0.005), 0, +0.0100 (0.0014).
    - w_A = 0.1085 and w_D = 0.0091, exactly u/K.
    - In-swarm Ĝd = 0.058 and 0.0091.
    - Composition ratio 1.04 (antagonistic).
  - **Residual a function of the whole prompt**, the most prompt-sensitive case: every flipped call or rewritten note
    redraws a residual of s.d. 0.40.
    - X₂ = +0.061, −0.0003, +0.015, the same as before.
    - The twins become far noisier: E = +0.036 (0.029) mutual and +0.014 (0.0095) antagonistic. Neither world's twins
      can resolve the frozen E at 3 s.e. That is why the power rule under "Fixed rules" exists.
- **Refusal and missing rates** are recorded per run. A missing decision is a missing belief, filled with the previous
  logit for the estimators and counted.

## Success and failure criteria

The run fails if vLLM doesn't load or the decision tokens fail `tok`. If a run has ≥ 5% missing decisions, it is
reported, and the world's statistics are given with and without it.

The scientific outcome is the set of falsifier outcomes, each reported with its numbers, mapped to a label by the
verdict table under "Fixed rules" (`analyze.verdict`). F2 is reported separately as the memory premise. F5 decides
whether magnitudes are compared zero-shot or with in-swarm numbers, and is reported either way.

## Time and cost

**Timing basis.** From 08 and 08b:
- 08 pilot rounds took 50 s per 256 analysts on an L40S (25 s decisions, 25 s notes).
- An A40 ran K = 16 decisions 1.65 times slower than an L40S (08b: 165 s against 100 s per 1,000).
- A40 estimate: about 130 s per lockstep round (512 decisions and notes), so 21 rounds ≈ 45 min per run. An L40S is
  about 28 min per run.

**Twins.** Each kick costs about 19 decisions (16 readers and 3 own replays, plus the shared base recomputes) and 17
notes. So the antagonistic twins ≈ 0.85 run and the mutual and one-way twins ≈ 0.3 run.

**Per worker, on A40s:**
- W1: 6 runs + 0.3 = 6.3 run-units ≈ 4.7 h, plus setup ≈ 5.0 h ≈ $3.0.
- W2: 6 runs + 0.85 ≈ 5.1 h, plus setup ≈ 5.4 h ≈ $3.2.
- Total ≈ $6.2 (≈ $7 on L40S at about 3.3 h each).

**Caps and stop rule.** Cap 8 h (1.5 × the larger projection), budget $9 per worker. If a worker runs past 1.5 × its
projection, the driver stops it and reports.

**Timing contingency** (fixed): if the smoke's inbox round (round 1) takes more than 170 s, W1 drops oneway:3 and W2
drops oneway:4, together with their twins. That leaves 3 one-way runs. The reviewer's estimate from 08's timing files is
about 165 s per lockstep round on an A40, so W2 ≈ 7 h.

**Memory.** One process per worker, max_num_seqs 256 (a lockstep batch of 512 queues), weights 16.7 GiB, KV about
20 GiB.

## If something fails

- **A step errors:** read the traceback. Fix a bug in `envs/analyst.py` or `run.py` on the coordinator (amendment),
  push, rerun; checkpoints resume.
- **Out of memory:** lower max_num_seqs to 128.
- **A worker hits its cap:** every round and every twin slice is appended as it finishes. The finished runs and slices
  are analysed, and the missing ones are reported.
- **Any change to prompts or decoding mid-run:** not allowed (Fixed rules). Stop and report instead.

## Fixed rules

- ĉ = −0.6366486620518087 throughout. The first run's mean belief is reported (M1: 0.00 ± 0.04; M2: +0.35 ± 0.15),
  and ĉ is not changed whatever it shows.
- The cell, the runs, the seeds, the twin allocation, the estimators in `analyze.py`, the missing rule (valid mass
  0.95), and no belief parsed from text.
- The s.e. definitions: pooled within-world run s.d. / √n for world means.
- **Kick s.e.:** the larger of the cluster bootstraps over slices and over (run, replica).
- **Power rule for the kick parts.** A world's twins count as "powered" iff the frozen M1 kick echo (mutual +0.0553,
  antagonistic +0.0092) is at least 3 × the observed s.e. of E. Unpowered twins are reported, but they don't decide F1
  or F4 (an echo they couldn't have resolved is not evidence against it), and those falsifiers are then decided on X₂.
  A twin E with the wrong sign at > 3 s.e. still falsifies, powered or not. This was added after the most
  prompt-sensitive dry run showed that the literal rule falsifies the true law when twins are noisy.

**Falsifiers** (from `theory/analyst_signrule.md` section 8, made operational). "se" is the s.e. of the stated
comparison from the data.
- **F1, mutual echo.** If X₂(mutual) − X₂(one-way) > 3 se and E_mutual > 3 se (or the mutual twins are unpowered),
  the result is consistent. If either is below −3 se, it is falsified. Otherwise, while the in-swarm Ĝ_M d > 0.03, it
  is falsified ("not positive").
- **F2, one-way echo, which is the one-step memory premise.** |X₂(one-way)| > 3 se is falsified. X₂ = 0 there holds
  only if the own memory is one step in the call. A path through the analyst's own notes (σ_i(t) → note_i(t) →
  note_i(t+1) → x_i(t+2)) would show here, and the kick echo excludes it by holding i's history at the base run's.
  That is why F5, F6 and the Ĝd check use differences from the one-way world.
- **F3, antagonistic sign, conditional on the swarm's couplings.**
  - The prediction is sign(Ĝ_A) of (S4), with û from the first-order kick responses, v̂ from the in-swarm leak
    regression, and d, d^r from the beliefs.
  - If |Ĝ_A d| + 2 s.e. < 0.004, no sign is predicted and the antagonistic world counts as a second null. If
    |Ĝ_A d| < 0.004 but not by 2 s.e., there is no sign test, and the antagonistic part is inconclusive.
  - Otherwise, X₂(antagonistic) − X₂(one-way) or E_antagonistic (rewritten) with the opposite sign at > 3 se is
    falsified.
  - Notes kept: E_kept with the sign opposite to r_A·r_D at > 3 se is falsified.
- **F4, ordering.** X₂(mutual) − X₂(antagonistic) < 3 se, or E_mutual − E_antagonistic < 3 se with both worlds' twins
  powered, is falsified.
- **F5, magnitudes**, on dX₂ = X₂(world) − X₂(one-way). Mutual dX₂ outside [0.03, 0.07], or antagonistic dX₂ outside
  [0.005, 0.021], by > 3 se means the calibration did not transfer. Then dX₂ is compared with Ĝd per world, and
  |dX₂ − Ĝd| > 0.25|Ĝd| + 3 se means the single-site reduction fails.
- **F6, kick against echo function.** |E − dX₂| > 3 se + 0.15|dX₂| in a world whose twins are powered is falsified.
  Unpowered twins give "unpowered", which decides nothing.
- **F7, asynchronous control** (Amendment 1 below; theory note Amendment 1, A5, applied exactly).
  - **Statistics.** D_τ = ρ_τ(mutual) − ρ_τ(one-way), as world means. T₇ = ½·D₂(sync) − D₂(async), with D₂(sync) and
    its s.e. from this experiment's synchronous runs (pooled s.e.), and
    s.e.(T₇)² = ¼·s.e.(D₂ sync)² + σ_m²/n_m + σ_o²/n_o, where σ is the asynchronous run s.d. of ρ₂ in each world.
  - **Decision.** T₇ > 3 se is **confirmed**; T₇ < −3 se is **falsified**; otherwise **not resolved**, reported with
    the ratio D₂(async)/D₂(sync) (predicted 0.21).
  - **Positive check, same runs.** D₁(async) > 3 se passes. D₁(async) < −3 se, or |D₁(async)| < 3 se while
    D₂(sync) > 0.03, is **falsified**, whatever T₇ is.
  - **Secondary:** D₁ − D₂ (asynchronous) > 0, from the per-run ρ₁ − ρ₂.
  - **Reported:** D₁(async)/D₂(sync), predicted 0.71.
  - **Caveat (A4):** if D₂(sync) < 0.04, the lag-1 check is the main test.

**Verdict table** (implemented in `analyze.verdict`; the strings are `analyze.py`'s outcomes):

| condition, checked in this order | label |
|---|---|
| any of F1, F3, F3 notes-kept, F4 or F6 starts with "falsified", or F7's T₇ or positive check is "falsified" | **not supported** (the falsified ones named) |
| F1 "unresolved (G_M d <= 0.03)" | **inconclusive** |
| F3 "consistent (sign resolved)" | **supported**: reciprocity, ordering and the antagonistic sign |
| F3 "no sign predicted (\|G_A d\| + 2 se < 0.004) …" without "but an echo is resolved" | **supported**: reciprocity and ordering; antagonistic world a second null as predicted |
| otherwise (F3 "consistent (sign not resolved at 3 se)", F3 "sign of G_A not resolved …", or an unexpected resolved echo when none is predicted) | **inconclusive (antagonistic part)**: reciprocity and ordering confirmed, antagonistic sign not resolved |

Appended to every label: the F7 outcome (T₇ confirmed or not resolved, positive check, secondary), and
"one-step memory holds" (F2 consistent) or "one-step memory fails (F2)". A falsified F2 does
not by itself make the sign rule "not supported", because F1, F3, F4 and F6 use differences or the kick echo, which are
free of the own-note path. Also appended: the F6 status (consistent, or unpowered) and the F5 reading.

**Recorded, not decided:** m (which law holds); the composition ratios E/(K r_A r_D) (antagonistic) and
E/((K/2)(r_A² + r_D²)) (mutual), which are 1 for calls only and about 5 under M2; missing and refusal rates; and the
null-twin note noise.

## Questions asked before launch

None needed. The coordinator's message (the user's option (b)) and the frozen note fixed the design, the falsifiers,
the constraints, spend tracking ($8–15 expected) and approval, which the user's standing go-ahead covers (the approval below rests on it). Choices made
here:
- two workers (W1 mutual and one-way, W2 antagonistic plus one one-way run), with the mutual run first;
- lockstep replicas with a shared session tag;
- paired base recomputes in the twins;
- the s.e. definitions, and the operational thresholds for the falsifiers' "3 s.e." wording.

## Review

First pass (general-purpose agent following `agents/experiment-reviewer.md`, definition path only): needs revision.
- **P1, raw X₂ mixes the echo with own-note memory**, while F2, F5 and F6 treated it as pure echo. The reviewer
  recomputed the 08 pilot's ρ₂ − ρ₁² as −0.04 to −0.09, against the law's −0.002 to +0.02. Fixed: F5, F6 and the Ĝd
  check now use X₂(world) − X₂(one-way); F2 is reported as the memory premise, and a falsified F2 alone reads
  "one-step memory fails", not "not supported".
- **P1, "Supported" without the antagonistic world contributing, and the coupling sign never varied.** Fixed: the
  claim is reworded to reciprocity, ordering and a resolved-positive antagonistic echo. "Supported" needs the
  antagonistic sign resolved at 3 s.e. (or the second-null case); otherwise the label is "inconclusive (antagonistic
  part)". The sign-blind alternative is stated. The suggested adaptive swap of twins for extra base runs is not
  adopted: the coordinator fixed about 570 twins. If they are unpowered, RESULT.md says so and names extra base runs
  as the follow-up.
- **P1, outcomes not mapped to labels.** Fixed: the verdict table above, implemented in `analyze.verdict`.
- **P2, timing about 25% optimistic.** Fixed: the threshold is 170 s, and W2 is estimated at about 7 h, inside the cap.
- **P2, kick file collision between the workers.** Fixed: per-run files, one line per slice. Smoke tags are per
  worker (`--label`).
- **P2, slice-bootstrap s.e. too small.** Fixed: the s.e. is also clustered by (run, replica), and the larger is used.
- **P2, Gaussian-cavity flags missing.** Fixed: conditional kurtosis and the bimodality coefficient go into
  `summary.json`.
- **P2, dry-run harness not committed.** Fixed: `dryrun.py`. Partial slices can no longer occur (one append per
  slice).
- Both dry variants were re-run after the fixes. Each gives "supported" with "one-step memory holds". In the
  prompt-sensitive variant F6 is "unpowered", and in the white variant it is "consistent".

Re-check (same reviewer, after af11fd3): every earlier P0/P1 addressed and no new P0. One P1 was left: the second-null
route fired on the point estimate of Ĝ_A d. Fixed, so it now needs |Ĝ_A d| + 2 s.e. < 0.004; a small but unresolved
value gives no sign test, and the antagonistic part is then inconclusive. Also fixed: the stale kick-s.e. wording and
the test count. The reviewer said it is ready to run with this fix.

## Amendments

### Amendment 1 (2026-11-25 21:59 UTC): asynchronous control added (falsifier 7)

**Reason and timing.**
- **Why:** a user decision (relayed by the coordinator) to test falsifier 7.
- **Synchronous data collection had already started.** W1 and W2 launched at about 21:22 and 21:28 UTC. Their base
  runs are in progress and continue unchanged. No synchronous data had been read when this amendment was written.
- **The asynchronous predictions were frozen before any asynchronous data:** Amendment 1 of
  `theory/analyst_signrule.md` (commit f613f1b, frozen 21:52 UTC), with numbers in
  `theory/checks/analyst_signrule_async_predictions.json`.

**Schedule** (A1 of the theory amendment, implemented exactly in `AnalystSwarm` as `cfg.blocks = 16`):
- Round 0 is synchronous.
- In each round t ≥ 1, a permutation of the 256 analysts is drawn from the stored seed (order_seed(graph), t), shared
  by the replicas, and cut by `np.array_split` into 16 blocks of 16.
- The blocks update in turn. Each block decides in one engine batch over both replicas, draws its calls from
  tanh(x − ĉ) with the stored uniforms, and writes its notes before the next block.
- An analyst reads each colleague's latest call and note. Its own previous call and note are from round t − 1, and
  its news is ξ_i(t).
- The block index is stored with every record. A round is written only after its last block, so a stop never leaves
  a partial round.

**Tests** (`tests/test_analyst.py`):
- the loop equals the amendment's `simulate_blocks` on the same permutations at B = 16, at B = 4 and at B = 1 (which
  also equals the synchronous simulation);
- at B = N it equals `analyst_lib.simulate(mode="async")`;
- a crash on the second block of a round resumes to an uninterrupted run;
- 70 analyst tests pass in all.

`dryrun.py` adds the asynchronous runs. Under the mock law it gives D₁(async) = +0.037 (frozen +0.041), D₂(async) =
+0.011 (frozen +0.012), T₇ confirmed and the positive check passing.

**Runs.** Following the amendment's sizing, which the user adopted so the comparison resolves:
- 4 asynchronous mutual and 4 asynchronous one-way runs, at 2 replicas, each on its own new graph (g = 10–13, graph
  seed (9090, world, g)), with the cell, ĉ, T = 20 and the statistics of the synchronous runs;
- predicted 3.2 s.e. on falsifier 7 and 9.5 s.e. on the lag-1 check;
- no asynchronous antagonistic run, since the amendment shows it adds nothing resolvable (1.2 to 1.6 s.e.).

**Workers and commands.** Each worker runs its own timing step first, then its asynchronous runs, interleaved so that
a failed or slow worker takes out runs from both worlds equally: W3 = mutual:10, oneway:10, mutual:11, oneway:11 and
W4 = mutual:12, oneway:12, mutual:13, oneway:13. W3's commands:

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py asmoke --label W3 2>&1 | tee results/analyst-signrule-asmoke-W3.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py abase mutual:10 oneway:10 mutual:11 oneway:11 --label W3 --budget-hours 7.3 2>&1 | tee results/analyst-signrule-abase-W3.log
```

W4 runs the same, with `--label W4` and `abase mutual:12 oneway:12 mutual:13 oneway:13`.

**Time and cost.** The block schedule is 16 batches of 32 prompts per round, against one batch of 512. It is timed
by `asmoke` before the projection is sent. If a block round takes s seconds, a run takes about 21 s, and a worker
with 4 runs takes about 84 s plus setup.
- **Clock check** (replaces the earlier 300 s contingency, from the re-check). Before each run, `abase` skips the run
  if 21 × (the last measured seconds per round: the previous run's, else the slowest asmoke round) would end past
  `--budget-hours 7.3` from the step's start. So the 8 h cap can never cut a run partway, and the interleaving keeps
  any shortfall balanced between the two worlds.
- **Completeness check.** `analyze.load_run` uses a run only if all (T+1)·R·N records exist. Incomplete runs, of
  either schedule, are excluded and listed. This was verified on the dry run by truncating one asynchronous run: it
  was excluded, and F7 used the remaining 3 + 4 runs.
- **Reported beside the decision, which stays as frozen:** the Welch degrees of freedom and one-sided t p-values for
  T₇, and T₇'s s.e. with the asynchronous run s.d. pooled over both worlds (the re-check noted that 3 d.f. per world
  make the normal 3-s.e. cut looser than it looks).

**Analysis and figure.** `analyze.f7` and `plot.asynchronous` (`paper/figures/analyst_signrule_async`), as described
under "Fixed rules".

### Re-check of Amendment 1 (same reviewer)

No P0. The schedule code, seeds, f7() and verdict() matched the theory amendment, and 83 tests plus the dry run
passed. One P1: an incomplete run (cut by the cap) would have been analysed as complete. Fixed by the completeness
check in `load_run` and the clock check in `abase`. Two P2s: W3/W4 now interleave the worlds, and the t-based p and
pooled-s.d. s.e. for T₇ are reported. The reviewer said it is ready to run with these fixes.

## Outcome

- Run complete: 12 synchronous and 8 asynchronous runs, 0 of 215,040 decisions missing, 0 refusals, and 545 kick
  twins plus 50 null twins. All four workers finished within cap and were deleted.
- F1 consistent (mutual − one-way X₂ = +0.074 (0.005); E = +0.069 (0.014)).
- F2 consistent (one-way X₂ = +0.002 (0.003)).
- F3 consistent, sign resolved (antagonistic − one-way = +0.031 (0.004); E = +0.031 (0.007); notes kept +0.029
  (0.005); Ĝ_A d = +0.043 (0.002)).
- F4 consistent (mutual − antagonistic = +0.043 (0.005)).
- F5: within 3 s.e. of the zero-shot intervals (antagonistic borderline, 3.3× the central value); consistent with Ĝd
  from the in-swarm couplings.
- F6 consistent where powered (the antagonistic twins were unpowered, with note noise 0.27–0.37 logit).
- F7: T₇ = +0.010 (0.007), not resolved; positive check D₁(async) = +0.044 (0.011) passes.
- Label: supported (reciprocity, ordering and the antagonistic sign); one-step memory holds.
- Recorded: m = −0.23, −0.12, −0.07, matching neither law.
- Cost $10.95 billed ($11.45 estimated) against ≈ $11.2 projected. Write-up: RESULT.md.
