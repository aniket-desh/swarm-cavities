---
name: 09-analyst-signrule
title: Analyst de-biased sign-rule test: Qwen3.5-9B swarms (N=256, K=16, T=20) in mutual, one-way and antagonistic graphs, echo function and kick twins against frozen predictions
status: draft
gpu: NVIDIA A40,NVIDIA L40S,NVIDIA RTX A6000
count: 1
min_gpu_gb: 40
hours: 8
budget_usd: 9
disk_gb: 120
data:
secrets:
---

# Analyst de-biased sign-rule test: Qwen3.5-9B swarms (N=256, K=16, T=20) in mutual, one-way and antagonistic graphs, echo function and kick twins against frozen predictions

This definition launches **two workers** from the same contract: W1 (mutual and one-way) and W2 (antagonistic and one
one-way run). Each worker has the time cap and budget above, so the worst case is 2 × $9 = $18.

## Question

Does an analyst's own call come back to it through reciprocated ties with the sign of the product of the two
couplings, and not at all without reciprocity? This is the reaction term's sign rule, and `theory/analyst_signrule.md`
froze its predictions on 2026-11-25 (commit 1799988; numbers in `theory/checks/analyst_signrule_predictions.json`)
before any de-biased swarm data existed.

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
- Kick statistics with cluster-bootstrap s.e. over slices.
- The in-swarm regression (calls only, and with the leak terms; cluster bootstrap over analysts).
- The reaction coefficient from the swarm's own couplings, Ĝ (S4): û from the twins' first-order responses, v̂ from
  the leak regression, d and d^r from the beliefs.
- The falsifiers below, and residuals of every observable against the frozen M1 and M2 values.

Figures come from `plot.py`.

**Validation done before launch:**
- 64 unit tests, including two identity checks:
  - the de-biased loop under the mock law equals `analyst_signrule_check.simulate` exactly, and `analyst_lib.simulate`
    at c − ĉ;
  - `kick_slice` equals the note's `kick_twins` for every reader at t0+1 and the kicked analyst at t0+2, with E = 0
    and identical prompts in the one-way world.
- A full-size dry run of `run.py` with all 12 runs and all twins, plus `analyze.py`, under a mock backend that obeys
  frozen law M1 including its residual (see "Controls").

## Inputs

Qwen/Qwen3.5-9B from the Hub, via `experiments/06-qwen-smoke/setup_vllm.sh`. Everything else is generated from fixed
seeds. No data tars.

## Commands

Worker W1 (mutual and one-way):

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py smoke 2>&1 | tee results/analyst-signrule-smoke-W1.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py base mutual:0 oneway:0 mutual:1 oneway:1 oneway:2 oneway:3 2>&1 | tee results/analyst-signrule-base-W1.log
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py kicks mutual:0 mutual:1 oneway:0 oneway:1 oneway:2 oneway:3 2>&1 | tee results/analyst-signrule-kicks-W1.log
```

Worker W2 (antagonistic and one one-way run):

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_analyst.py tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/09-analyst-signrule/run.py smoke 2>&1 | tee results/analyst-signrule-smoke-W2.log
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
- `results/analyst-signrule-20261125/kicks_{world}.jsonl`: one line per twin.
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

The scientific outcome is the set of falsifier outcomes below, each reported with its numbers.
- **Supported:** F1, F2, F4 and F6 are consistent, and F3 is consistent with its sign predicted (or the antagonistic
  world is a second null by the |Ĝ_A d| < 0.004 rule).
- **Not supported:** any of F1, F2, F3, F4 or F6 is falsified.

F5 decides whether the comparison is zero-shot or uses in-swarm numbers, and is reported either way.

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

**Timing contingency** (fixed): if the smoke's inbox round (round 1) takes more than 190 s, W1 drops oneway:3 and W2
drops oneway:4, together with their twins. That leaves 3 one-way runs.

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
- The s.e. definitions: pooled within-world run s.d. / √n for world means; the cluster bootstrap over slices for
  twins.
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
- **F2, one-way echo.** |X₂(one-way)| > 3 se is falsified.
- **F3, antagonistic sign, conditional on the swarm's couplings.**
  - The prediction is sign(Ĝ_A) of (S4), with û from the first-order kick responses, v̂ from the in-swarm leak
    regression, and d, d^r from the beliefs.
  - If |Ĝ_A d| < 0.004, no sign is predicted and the antagonistic world counts as a second null.
  - Otherwise, X₂(antagonistic) − X₂(one-way) or E_antagonistic (rewritten) with the opposite sign at > 3 se is
    falsified.
  - Notes kept: E_kept with the sign opposite to r_A·r_D at > 3 se is falsified.
- **F4, ordering.** X₂(mutual) − X₂(antagonistic) < 3 se, or E_mutual − E_antagonistic < 3 se with both worlds' twins
  powered, is falsified.
- **F5, magnitudes.** Mutual X₂ outside [0.03, 0.07], or antagonistic X₂ outside [0.005, 0.021], by > 3 se means the
  calibration did not transfer. Then X₂ is compared with Ĝd per world, and |X₂ − Ĝd| > 0.25|Ĝd| + 3 se means the
  single-site reduction fails.
- **F6, kick against echo function.** |E − X₂| > 3 se + 0.15|X₂| in the mutual or antagonistic world is falsified.
- **F7, asynchronous control.** Not run.

**Recorded, not decided:** m (which law holds); the composition ratios E/(K r_A r_D) (antagonistic) and
E/((K/2)(r_A² + r_D²)) (mutual), which are 1 for calls only and about 5 under M2; missing and refusal rates; and the
null-twin note noise.

## Questions asked before launch

None needed. The coordinator's message (the user's option (b)) and the frozen note fixed the design, the falsifiers,
the constraints, spend tracking ($8–15 expected) and approval, which the user's standing go-ahead covers. Choices made
here:
- two workers (W1 mutual and one-way, W2 antagonistic plus one one-way run), with the mutual run first;
- lockstep replicas with a shared session tag;
- paired base recomputes in the twins;
- the s.e. definitions, and the operational thresholds for the falsifiers' "3 s.e." wording.

## Review

<fill: reviewer findings>

## Amendments

None.

## Outcome

Not run yet.
