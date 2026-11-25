---
name: 06-qwen-calibration
title: Qwen3.5-9B calibration with real Qwen swarm messages: pilot swarm, real-vs-template probes, conformity curve
status: done
gpu: NVIDIA A40,NVIDIA RTX A6000
count: 1
min_gpu_gb: 40
hours: 8
budget_usd: 6
disk_gb: 120
data:
secrets:
approved_utc: 2026-11-25T15:38:29Z
result: inconclusive
finished_utc: 2026-11-25T20:08:50Z
writeup: experiments/06-qwen-calibration/RESULT.md
---

# Qwen3.5-9B calibration with real Qwen swarm messages: pilot swarm, real-vs-template probes, conformity curve

## Question

How strongly does a Qwen3.5-9B agent couple to real swarm messages, and is there any (K, wording) where a Qwen swarm
would sit near or beyond the stability boundary (Lambda >~ 0.7) or be bistable (conformity slope > 1 at its
steepest)? The Haiku 5.5 swarm turned out to be in the quiet regime everywhere (Lambda ~ 0.45 at best, HANDOFF.md
"Afternoon session"), and its template-message calibration did not transfer to real messages (real messages couple
25-50% more weakly), so the open-weight track needs a model whose coupling, measured with its own real messages,
puts the theory's interesting regimes within reach. This decides whether Qwen3.5-9B swarms are worth running at
large N.

## Hypothesis

No strong prior. Qwen3.5-9B is a smaller model; the Physics of Agents small models were more conformist than large
ones, so its gains could be larger than Haiku's. Counting for a strong-coupling regime: Lambda_max >= 0.7 in some
cell, or a steepest conformity slope significantly above 1. Counting against: every cell with Lambda_max well below
0.7 (as for Haiku) and conformity slopes below 1. Either answer is useful.

## Method

Agent: exactly the Claude swarm agent (`envs/prompts.py`) on `envs/vllm_backend.py` (validated in
`experiments/06-qwen-smoke`): Qwen/Qwen3.5-9B bf16 text-only on vLLM 0.32.0, thinking disabled, rationale and message
sampled at temperature 1 under the structural regex, belief = expected stated probability from digit-token logprobs,
missing when valid mass < 0.95, one retry for malformed samples. Own last answer shown (H=1) everywhere.

1. **Pilot swarm** (`run.py pilot`): the Haiku pilot's cell set from `experiments/05_swarm/cells.py` ("pilot"):
   inverted wording, p_neg=0.5, K in {2, 4}, gamma in {-1, 0, +1}, N=64, T=10, 2 replicas sharing graph, evidence
   and round-0 answers, graph 900, the identical graphs and evidence draws as the Haiku pilot (same functions and
   seeds). 6 x 64 x 21 = 8,064 calls. Excluded from confirmatory use; it is the message pool and a first look at
   Qwen swarm dynamics.
2. **Real-vs-template probes** (`run.py transfer`): `experiments/05_swarm/transfer.py`'s design with its functions
   `base_probes`, `context`, `mirror` imported unchanged and its pool function pointed at the Qwen pilot's messages
   (rounds >= 1, parsed). Per wording in {inverted, label} and K in {2, 4, 8}: 600 base probes, each sent with real
   pilot texts and with templated texts (`cavity.probes.peer_text`) at the same displayed probabilities and the
   same session tag. Inbox slots are disagree ties with probability 1/2; peer messages mirrored with probability
   1/2; own evidence h ~ N(+-0.5, 1), k=3; own last answer from the Haiku round-0 map (transfer.py's A_P, C0, kept so
   the design is identical). K=8 draws from the K=4 pool (there is no K=8 pilot cell). 7,200 calls.
3. **Conformity curve** (`run.py conformity`): `cavity.probes.conformity_design` (all K peers in the agree group at
   s = hbar), hbar in {-0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8}, evidence levels 0.4 and 1.4, 20
   replicates, K in {2, 4, 8}, both wordings, two text arms: templated texts, and real texts (each message's text
   replaced by a Qwen pilot message that stated the nearest probability on the same side, mirrored as needed;
   displayed probabilities unchanged). Own last answer 0.70 HEADS in both arms; the real arm also runs with own last
   answer 0.30 TAILS (same inboxes and texts), which measures the weight kappa_u of the agent's own last answer at
   fixed inbox. 7,920 calls.

Analysis (`analyze.py`, on the coordinator, fixed before the run):
- Fit E[s] = tanh(a_+ f_A - a_- f_D + a_p h_priv + c) per (wording, K, arm) with `transfer.fit`; 1,000-sample paired
  bootstrap over base probes (both arms resampled together) for s.e. of each gain and of real - template. Repeated
  with s_sampled (one draw of the stated number from the same digit distribution) to show the noise a sampled
  number adds.
- g = sqrt((a_+^2 + a_-^2) / (2K)) and gamma_eff from `design_lib.effective` at gamma = -1, 0, +1;
  Lambda = d g (1 + |gamma_eff|), Lambda_max = max over the three gammas. d = <1 - s^2> over the real-arm probe answers
  of the same (wording, K) (an upper bound on the swarm's d, since mirrored peers keep answers moderate), with the
  pilot's late-round d (rounds >= 3) as a cross-check. s.e. by the same bootstrap.
- Conformity (own last answer 0.70 HEADS): per (wording, K, arm, evidence level) the level means of s, local slopes
  between adjacent levels with a within-level bootstrap s.e., the steepest slope, beta_resp from
  `cavity.probes.fit_conformity`, and the fixed points of the map hbar -> E[s](hbar): resolved sign changes of
  mean_s - hbar (both neighbouring levels outside 2 s.e.) and the roots of the fitted tanh with its own offset.
- Own-answer memory: E[s] = tanh(beta hbar + kappa_u u + c_ev) on the real arm with both own answers (u = +-0.4);
  kappa_lin = d kappa_u is the linear weight of the last answer, and Lambda / (1 - kappa_lin) for gamma >= 0 is the
  static threshold with memory (theory/claim2.md section 1). Reported as a labelled secondary number.
- Lambda is also reported with sqrt<d^2> in place of <d> (theory/claim2.md section 2) and with the pilot's d; d is
  held fixed in the bootstrap, so the Lambda s.e. covers the gains only.
- Pilot: `design_lib.thermal_observables` per cell (via `experiments/05_swarm/analyze.load_run`), per-round mean
  |delta p|, and fractions of agents whose answer is unchanged from the previous round: displayed probability equal at
  two decimals, sampled number equal, public message identical; the same fractions on the Haiku pilot
  (`results/pilot-20261125-*`) with the same code. For Qwen the displayed probability is an expected value and
  rarely repeats exactly, so the like-for-like comparison with Haiku restating its number is the **modal stated
  number** (argmax of the digit distribution) being unchanged; the displayed-value and identical-message fractions
  are reported beside it.
- Noise floor: within-level s.d. of s and s_sampled across the 20 conformity replicates, against the same statistic
  on the Haiku conformity run (`results/calib-20261125-conf`); residual s.d. of the transfer fits, Qwen against Haiku.

## Inputs

Qwen/Qwen3.5-9B from the Hugging Face Hub (Apache-2.0, ungated), downloaded on the worker. Graphs, evidence and probe
designs come from project code with fixed seeds. The pilot's messages (step 1) are the only data steps 2-3 read.
No data tars.

## Commands

```bash
set -euo pipefail; bash experiments/06-qwen-smoke/setup_vllm.sh
set -euo pipefail; uv run --no-sync pytest -q tests/test_vllm_backend.py
set -euo pipefail; uv run --no-sync python experiments/06-qwen-calibration/run.py pilot 2>&1 | tee results/qwen-calib-pilot.log
set -euo pipefail; uv run --no-sync python experiments/06-qwen-calibration/run.py transfer conformity 2>&1 | tee results/qwen-calib-probes.log
```

Then on the coordinator, after `orch finish`: `uv run python experiments/06-qwen-calibration/analyze.py` and
`uv run python experiments/06-qwen-calibration/plot.py`.

## Expected outputs

- `results/qwen-pilot-20261125/manifest.json` and `results/qwen-pilot-20261125-<cell>/` (vllm_calls.jsonl,
  vllm_timings.jsonl, swarm_rounds.jsonl, swarm_meta.json, beliefs.npz): about 30 MB.
- `results/qwen-transfer-20261125-<wording>/`, `results/qwen-conf-20261125-<wording>/`: per-call checkpoints, ~40 MB.
- `results/qwen-calib/transfer_<wording>.json`, `conformity_<wording>.json`: analysis-ready records, ~10 MB.
- `results/qwen-calib-*.log`. Coordinator side: `results/qwen-calib/summary.json`,
  `paper/figures/qwen_calibration.{pdf,png}`, `paper/figures/qwen_pilot.{pdf,png}`, `RESULT.md` here.

Launch gate: this runs only after 06-qwen-smoke met its own criteria (missing < 5%, median valid mass >= 0.99, the
sampled prefix ending in a lone " " token); see experiments/06-qwen-smoke/RESULT.md. Met at 15:40 UTC from 06-qwen-smoke steps 1-3
(176 calls: 0 missing, median valid mass 0.99998, minimum 0.9996, every prefix ending in `":` then a lone " ");
only that experiment's large-N timing was still running.

## Controls

- Template arm, paired with the real arm on everything but the peer texts: separates the real-message gains from
  the calibration-transfer error, as in the Haiku transfer check (whose numbers are the reference).
- s_sampled against the logprob belief: same calls, so the difference is only the number's sampling noise.
- Haiku pilot on identical graphs and evidence: the stickiness comparison uses the same cells and the same code.
- Missing rate and valid mass per step; tests on the worker before any step.

## Success and failure criteria

Fixed before the run; all six (wording, K) cells reported, none selected after the fact.
- **Near the stability boundary** in a cell (clear): Lambda_max (probe d) - 2 s.e. >= 0.7; beyond it when also
  Lambda_max - 2 s.e. >= 1. Straddling: the interval Lambda_max +- 2 s.e. contains 0.7. Below: Lambda_max + 2 s.e. < 0.7.
- **Bistable** in a cell (evidence 0.4, own last answer HEADS, real texts): steepest local slope - 2 s.e. > 1, AND
  beta_resp > 1, AND three fixed points of hbar -> E[s](hbar) (three resolved crossings of the level means with the
  diagonal, or three roots of the fitted tanh with its offset). A slope above 1 alone is necessary, not sufficient,
  since the evidence and the own answer offset the curve. A steepest slope above 1 that fails any of the other
  conditions is reported as inconclusive (the steepest of 10 slopes is biased upward; the bootstrap mean of the
  maximum is reported next to it).
- **Supported** (a strong-coupling regime is reachable with Qwen): at least one cell clear on Lambda or bistable.
  **Inconclusive**: no such cell, but some cell straddles 0.7 or has an inconclusive bistability call. **Not
  supported**: every cell below on Lambda and no cell bistable or inconclusive. "Not supported" refers to the
  memoryless Lambda of the fixed rule; the with-memory number is secondary and reported beside it.
- Run failed: any step's missing rate above 5%, or a step not completing.

## Time and cost

Measured in 06-qwen-smoke on an A40: a synchronous round of 256 calls with inboxes takes 187 s (sampling 52 s, the
two logprob stages 85 + 50 s; the readout is prefill-bound: about 6.5 prefills of ~1,000 tokens per call, and
prefix caching for this hybrid model caches only in 528-token blocks), i.e. 1.4 calls/s; a batch of 8 takes 8 s.
This run is 23,200 calls (8,064 + 7,200 + 7,920). The A40 is also the cheapest card per call, not only per hour:
the readout is prefill compute-bound, and dense bf16 throughput per dollar is about 150 TFLOPS / $0.59 for the A40
against 990 / $4.39 for an H100 and 312 / $2.19 for an A100, so a faster card buys wall-clock time, not savings.
Projection on the A40 at the measured 1.4 calls/s (1.2 calls/s for the pilot's batches of 64): pilot 1.9 h,
transfer 1.4 h, conformity 1.6 h, setup and two engine starts 0.3 h: about 5.2 h, $3.10. Cap 8 h (1.5x the
projection), worst case $4.72 on the A40 or $5.52 on the A6000 ($0.69/hr), inside the $6 budget. Memory: one vLLM
process, 16.7 GiB weights + 20.4 GiB KV/state on the 44 GiB A40 (measured in 06-qwen-smoke). Timing check: the
pilot's first cell (1,408 calls) is projected at 20 minutes; if it takes over 30 minutes the whole run projects past
the cap, and the driver stops after the pilot and transfer steps, publishes, and reports (conformity then runs on a
second worker).

## If something fails

- A step errors: read the traceback; a bug in run.py may be fixed and pushed (amendment recorded), a failure inside
  vLLM gets one retry with `--max-num-seqs 128`, then stop and report.
- Missing rate above 5% in a step: stop after that step and report (the readout needs work first).
- Time running out: every call is appended to its checkpoint file as soon as its batch finishes and replayed on a
  rerun, so a stop loses at most the current batch; the analysis-ready JSONs of a step are written at its end, but
  can be rebuilt from the checkpoints by rerunning the step on the same worker (replay costs only the engine start).
- The worker is lost: published checkpoints land on the network volume, not in a new worker's clone, so a relaunch
  reruns from scratch (about $5); the replay of finished calls only helps within one worker.

## Fixed rules

- The fit model, the bootstrap, the Lambda formula and d definition, and the thresholds above are not changed after
  seeing data. Extra analyses are labelled exploratory in RESULT.md.
- Missing rule: valid mass < 0.95 or two malformed samples; no belief is parsed from text.
- All six (wording, K) transfer cells and all twelve conformity (wording, K, arm) cells are reported; no arm is
  dropped for time (if time runs short, a whole step moves to a second worker instead).
- Pilot cells, graphs and seeds are the Haiku pilot's.

## Questions asked before launch

Approved under the user's standing go-ahead for GPU spend in this phase (capped at $40).

None needed: the user's brief fixed the design (pilot as in cells.py, transfer.py's paired design for both wordings
at K=2,4,8, the conformity curve with real texts if cheap), the question and its thresholds (Lambda >~ 0.7, slope > 1),
the GPU rule and the $40 cap for this phase, and gave a standing go-ahead for GPU spend.

## Review

experiment-reviewer (general-purpose agent following agents/experiment-reviewer.md), first pass, no P0:
- [P1] slope > 1 is necessary but not sufficient for bistability (Haiku's own conformity curve has a 1.24 local
  slope at the template text switch and one fixed point). Fixed: the rule now also needs beta_resp > 1 and three
  fixed points (resolved crossings or fitted-tanh roots); analyze.py computes both.
- [P1] "near" as a point estimate conflicted with "inconclusive". Fixed: supported needs the lower 2 s.e. bound
  >= 0.7; straddling is inconclusive; not supported needs every upper bound < 0.7.
- [P1] Lambda omits own-answer memory (claim2.md section 1: static threshold d g (1+gamma)/(1-kappa)). Fixed: the
  real conformity arm also runs with own answer TAILS (+2,640 calls over K=2,4,8 and both wordings), giving
  kappa_u; Lambda/(1 - d kappa_u) reported as a labelled secondary; the criteria say "not supported" refers to the
  memoryless Lambda.
- [P2] worker-lost recovery wrong: fixed (rerun from scratch). [P2] time fallback contradicted fixed rules: removed.
  [P2] no launch gate on the smoke test: added. [P2] stickiness comparison not like for like: modal stated number
  added as the comparison. [P2] d fixed in the bootstrap and sqrt<d^2>: both now stated/reported.
Also changed after review (not from it): conformity extended to K=2 (the bistability question covers every K), and
analysis-ready JSONs are written after each K. Second pass: approved, with two P2s: the H100 cost argument used a
sparse/dense ratio (H100 is not cheaper per call), so the GPU list went back to the A40 per the user's rule with an
8 h cap; and a cap stop would have lost the per-wording JSONs, fixed by writing them after each K.

## Amendments

None.

<!-- Each change made during the run, with the time and the reason, recorded and committed before acting on it. -->

## Outcome

- Near the stability boundary: no cell. Lambda_max (probe d) is 0.22-0.45, and the largest upper 2 s.e. bound is 0.54
  (label K=4); all six cells are "below".
- Bistable: no cell. beta_resp is 0.36-0.58, with one fixed point everywhere. Every real-text cell is called
  "inconclusive" by the rule, because its steepest local slope (1.15-2.16) exceeds 1. An exploratory null (fitted tanh
  plus the observed noise) gives a mean steepest slope of 1.3, so the rule's inconclusive branch fires on noise.
- Overall by the fixed rules: inconclusive (driven by that branch); in substance, Qwen is in the quiet,
  noise-dominated regime.
- Run quality: 1 missing of 23,184 calls.
- Cost $2.61 estimated against a $6 budget; 4.4 h against the 8 h cap. Write-up: RESULT.md.
