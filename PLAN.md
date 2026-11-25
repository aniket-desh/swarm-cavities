# swarm-cavities: orchestration plan

Source spec: `docs/spec.md` (the user's pasted research program: six theoretical claims, four experiments,
a $2,000 budget, a repo blueprint and guardrails). This file records the order of operations and the limits.

## Limits (set by the user, 2026-11-25)

- Claude Code agent usage for theory and implementation (Fable 5.2 / Opus 5.5 subagents) is **not** budgeted.
- The **$2,000** covers only experiment spend: Anthropic API calls made by experiment code, plus Runpod GPU pods.
  Track it in `budget/ledger.jsonl` (one line per batch/pod with actual cost) and keep a running total in `HANDOFF.md`.
- Pushes to `git@github.com:aniket-desh/swarm-cavities.git` (branch `main`) are allowed and expected at each milestone.
- Never delete the coordinator or any volume. `orch` may delete its own worker pods.
- Pod prefix `aniket-`; GPU work goes through `orch` and the `runpod-research:*` skills.

## Order of operations

1. **Bootstrap.** `orch.toml`, `CLAUDE.md` with the guardrails from the spec, `pyproject.toml` (uv), `docs/spec.md`.
2. **Literature red-team.** Read Pavlova & Tanaka (arXiv:2609.19124) and El et al. (arXiv:2608.16578), find their
   code, and record in `docs/literature.md` what each paper actually shows, what code we can reuse, and which of our
   claims they already cover or contradict. Also check the classical DMFT background (Sompolinsky–Crisanti–Sommers 1988;
   Crisanti–Sompolinsky 1987 on asymmetric couplings; Kadmon–Sompolinsky 2015 on multi-population; Clark–Abbott–Litwin-Kumar
   2023 on two-site cavity / cross-covariances; dynamic cavity / DMP of Kanoria–Montanari, Aurell et al.).
3. **Theory (one Fable 5.2 agent).** Derive and red-team Claims 1–6 → `theory/`. It must say which claims hold as
   stated, which need restating, and which are false, and give the equations `dmft_solver.py` must implement.
4. **Implementation (parallel Opus 5.5 agents, after theory lands).** Modules per the spec blueprint:
   `theory/dmft_solver.py`, `envs/`, `engine/`, `cavity/`, `analysis/`, `interp/`. Each uses the `research-code` skill.
5. **Experiments, gated.** (a) mock Ising agent reproduces analytic DMFT to <1%; (b) $5 Haiku smoke test at N=8, T=4;
   (c) cost projection printed before any sweep over $50. Then Exp 1 → Exp 2 → Exp 3 → Exp 4 (GPU, via `orch experiment`).
6. **Write-up.** Figures and draft with the `paper-writing` skill under `paper/`; predicted-vs-empirical residuals first.
