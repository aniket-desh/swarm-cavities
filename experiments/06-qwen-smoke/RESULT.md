# 06-qwen-smoke: the vLLM backend works on Qwen3.5-9B; a round at N=1024 costs $0.14 on an A40

The backend in `envs/vllm_backend.py` runs as written on a GPU. Over 1,968 agent calls none were missing: the
smallest valid mass was 0.9948 and the median 0.99998, and 15 malformed samples (0.8%) all recovered on their one
retry. Both API points that had only been checked by reading work on vLLM 0.32.0: `language_model_only=True` loads
the text model alone (16.7 GiB of weights), and the `StructuredOutputsParams` import path is the one that's taken. A
synchronous round with inboxes (K=4) takes 187 s at N=256 and 839 s at N=1024 on an A40 at $0.59/hr, which is
**$0.031 per round at N=256 and $0.137 per round at N=1024**. Three quarters of that time is the logprob readout,
not the sampling.

## What ran

The run used one A40 (46 GB, driver 595.112, CUDA 13.2 host), with vLLM 0.32.0+cu129 and torch 2.13.0+cu129
installed by `setup_vllm.sh`. Engine settings were prefix caching on, max_model_len 8192, max_num_seqs 256 and
gpu_memory_utilization 0.90. vLLM chose mamba cache mode "align", with a 528-token attention block, and gave 20.4 GiB
to the KV cache and linear-attention state. The first engine start took 281 s, including torch.compile and the
FlashInfer JIT. A later start took 65 s from the compile cache. The worker ran 31 tests (`tests/test_vllm_backend.py`
and `tests/test_envs.py`) and all passed. The data are in `results/qwen-smoke-*/` and the step records in
`summary.json`.

## Checks

| check | result |
|---|---|
| "0".."9" and "." are single tokens | yes, all 11 |
| generation prompt ends in the empty think block | yes (`<|im_start|>assistant\n<think>\n\n</think>\n\n`) |
| canonical tokenization of `": 0.62}` | `":`, ` `, `0`, `.`, `6`, `2`, `}` |
| sampled prefixes end in `":` then a lone " " | every one of the 1,968 calls |
| probes, 16 per wording (4 at round 0, 12 with a 2+2 inbox) | 32/32 ok; valid mass min 0.99977 |
| swarm N=8, K=2, T=4, 2 replicas, sync | 72/72 calls ok; valid mass min 0.99959 |
| same, async (same graph and evidence) | 72/72 ok; valid mass min 0.99971; 4.4 s per call (one at a time) |
| timing runs, N=256 (3 rounds) and N=1024 (2 rounds) | 1,792/1,792 ok; valid mass min 0.9948 |
| skipped mass (first decimals below 1e-3, put at the midpoint) | at most 0.0028, so the mean moves by at most 0.00014 |
| mean abs(p_logprob - p_sampled) | 0.006-0.023 per run |

The rationales and messages read sensibly. The agent combines its sensors with its inbox and, in the inverted
wording, often says it is discounting the "miscalibrated" sources. Some of its reasoning is wrong, though. In one
probe it called its 52-63% sensors "below 50%, so likely lying" and answered 0.72 against three TAILS reports. In
another, inverted probe it flipped to HEADS against its own TAILS sensors. Messages often run past the 30-word limit
the prompt asks for. In the N=8 sync swarm, the truth-aligned mean m rose from 0.085 at round 0 to 0.283 at round 4,
and the RMS difference between replicas grew from 0 to 0.10-0.18. Both modes ran; their dynamics aren't compared at
this size.

The last column of the table is the noise that sampling the number would add on top of the logprob belief. Haiku's
stated number carries s.d. 0.10 in s (0.05 in p) across replicate calls (`experiments/04_swarm_design/design_lib.py`
SIG_XI). The logprob belief removes the part of that noise that comes from sampling the number itself, but not the
part that comes from the sampled rationale. That split is measured in 06-qwen-calibration.

## Timing (synchronous rounds, K=4, inverted wording, A40)

| N | round | calls | sampling s | logprob stage 1 s | stage 2 s | total s | $ at $0.59/hr |
|---|---|---|---|---|---|---|---|
| 256 | 0 (no inbox) | 256 | 47 | 59 | 35 | 141 | 0.023 |
| 256 | 1 | 256 | 52 | 85 | 52 | 189 | 0.031 |
| 256 | 2 | 256 | 53 | 86 | 48 | 187 | 0.031 |
| 1024 | 0 (no inbox) | 1024 | 163 | 244 | 235 | 642 | 0.105 |
| 1024 | 1 | 1024 | 213 | 347 | 278 | 839 | 0.137 |

A round with inboxes runs at 1.2-1.4 calls/s. A batch of 64 (the pilot's size) took 36-40 s in 06-qwen-calibration,
or 1.7 calls/s. The readout is prefill-bound. Each call costs one sampled generation of about 110 tokens, then three
stage-1 prefills and two to three stage-2 prefills of about 900 tokens each. Prefix caching helps little here,
because for this hybrid (Gated DeltaNet plus attention) model it caches only at 528-token boundaries, so most of
every readout prefill is recomputed.

A faster card wouldn't make a round cheaper. Dense bf16 throughput per dollar is about 254 TFLOPS/$ for the A40,
225 for an H100 and 142 for an A100, so a faster card buys only wall-clock time. Two changes would cut the cost
itself, and neither was tested:

- Raising the stage-2 threshold from 1e-3 to 1e-2 would drop stage 2 roughly by half. It would bias the mean by at
  most 0.05 times the skipped mass.
- Reading dist A, B and C from one request would collapse stage 1 into it.

At $0.137 per N=1024 round, a T=10 run with two replicas costs about $2.9 and takes 4.9 hours on one A40. For
comparison, Haiku costs about $0.0003-0.0004 per call, or $0.3-0.4 per N=1024 round, but it runs at API concurrency.

## Deviations and what wasn't checked

- The N=1024 run had T=2 in the definition. It was stopped after round 1, because round 2 would have repeated round
  1's measurement at another 14 minutes. Its `summary.json` record therefore holds no `timing` entry; the numbers
  above come from `results/qwen-smoke-timing-N*/vllm_timings.jsonl`.
- GPU floating-point non-determinism across batch compositions wasn't measured. The backend says a belief is a
  deterministic function of the sampled prefix only up to it.
- The H100 and other cards weren't timed. The per-dollar comparison above rests on spec-sheet throughput.

Cost: $0.49 (0.83 h of A40), against a $5 budget.
