# Prompt check of envs/prompts.py on claude-haiku-5-5 (2026-11-25 07:23 UTC)

20 realtime calls, one agent (evidence log-odds about 0.4, so about 0.60 on its own; last own answer shown as 0.70;
K = 8 peers, H = 1). Ledger experiment_id `prompt-check-20261125-072310`, $0.0073 total. Mean tokens per call:
347 uncached input, 786 cache read, 593 output (thinking included). Checkpoints are in
`results/prompt-check-20261125-072310/`.

- **Refusals and parse failures:** 0 of 20.
- **Agree-group inboxes** (uniform, peers at p = 0.5 + hbar/2): the response moves with the field. At hbar = -0.6 the
  agent gives 0.37 and 0.45, and at +0.6 it gives 0.66 and 0.72; the fitted slope is dp/dhbar of about 0.30.
- **Disagree-group inboxes:** the label doesn't invert the influence. When sources "you tend to disagree with" state
  0.2 (TAILS), the agent moves *toward* them, to 0.42 and 0.50, from 0.60 on its own evidence. When they state 0.8 it
  gives 0.63 and 0.65. So the disagree channel's gain b- is near zero or negative (it attracts weakly), as Physics of
  Agents found for small models. By theory/claim1.md section 5, that leaves only gamma >= 0 reachable with option B
  and makes the mean coupling dominate. Calibration should test a variant that gives a reason to invert those
  sources, e.g. "their sensors are miscalibrated and right less than half the time".
- **Compositions at hbar = +0.2** (one sample each): polarized 0.65, few-strong 0.66, route A 0.66, route B 0.58.
  The route B gap is a hint of the two-channel effect; the calibration has the samples to resolve it.
- **Qualitative:** the agent often recomputes its own evidence and "corrects" the fake earlier answer of 0.70, and
  it says it discounts peers whose evidence may overlap. Both behaviors fall outside a pure scalar-field model.
