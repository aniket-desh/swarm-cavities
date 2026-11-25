# HANDOFF: swarm-cavity overnight run

Last updated: 2026-11-25 04:10 UTC. Running in tmux session `swarm-cavity` (attach: `tmux attach -t swarm-cavity`).

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

## In progress

- Literature red-team (background agent): checking whether the two cited arXiv papers exist and say what the spec
  claims, finding their code, and checking each claim against classical DMFT results.

## Decisions for you to review

- **Network volume is at its quota** (`uv sync` failed with "Disk quota exceeded"; /workspace holds about 55 GB across
  projects, 12 GB of it the shared uv cache). I didn't delete anything. This project's venv and uv cache live on the
  container disk instead (`scripts/env.sh`), which a container restart wipes, though `uv sync` rebuilds it in about a
  minute. Bulky experiment outputs will go to `/workspace-global` or stay on workers. You may want to raise the volume
  size or prune `/workspace/.cache/uv` (another uv process held its lock, so I couldn't).

## Blocked

- **GPU work (Exp 4) needs you to pick the Runpod account.** `orch doctor` fails for every project on this
  coordinator, not just this one: "network volume 2we3auziu3 doesn't exist on this account". The cause is that the
  `RUNPOD_API_KEY` in the shared `/workspace/.env` belongs to an account that doesn't hold this coordinator's volumes
  (its volume list has 38 others, none of them `2we3auziu3`). `runpodctl`'s own config uses a different key, and that
  one does see the coordinator pod. I didn't swap keys between configs, because which account pays for GPUs is your
  call. To unblock, put the key of the account that owns volume `2we3auziu3` into `/workspace/swarm-cavity/.env` as
  `RUNPOD_API_KEY`, then `cd /workspace/swarm-cavity && orch doctor` should go green. Exps 1–3 use only the Anthropic
  API and aren't blocked.
