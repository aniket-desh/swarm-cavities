# Physics of Agents: zero-cost analyses

All three questions have answers from the released data, at no API cost. They come with one large caveat: the PoA
swarms are strongly coupled, few-input networks (one neighbour moves an agent's logit by 2 to 7), so the
dense-graph DMFT quantities in item 3 are only qualitative here.

1. **A scalar field suffices, but it's not the graph's field $Js$.** The two-channel model
   $\text{logit}=\alpha_i+b_+m_a-b_-m_d$ is still one linear field $h_i=\sum_jW_{ij}s_j$ with model-specific weights.
   On held-out graphs it captures 86–97% of the peer information that the most flexible count model (M4) captures.
   The literal $h=Js$ captures only 19–61%, because disagree ties don't repel: $b_-\le0$ in 6 of 8 model × task cells,
   with $b_-/b_+$ between $-0.43$ and $+0.18$. So two gains are needed, which is the same as saying the effective
   coupling is $W\neq J$, but no second field is needed. Agents average their inbox slightly better than they sum it
   (6 of 8 cells, by 3–9 millinats per vote). The counts are not exactly additive: saturation and interactions are
   significant everywhere, but together they're worth only 3–14% of the peer information. Separately, the agent's own
   state two rounds back predicts its vote, although the prompt never shows it.
2. **The flicker disappears under asynchronous updates.** On the same 60 statements × 8 random graphs, GPT-4o-mini's
   spin period index is $0.29\pm0.02$ (objective) and $0.26\pm0.02$ (subjective) under synchronous rounds. Under PoA's
   asynchronous schedule it's $1.17\pm0.01$ and $1.18\pm0.01$ after dividing by the schedule's no-structure null, where
   1 means no temporal structure and values below 1 mean period 2. The two-channel agent fitted on synchronous data,
   simulated on PoA's firing schedule, predicts 1.15 and 1.16 (and 0.25, 0.24 under synchronous rounds), so the
   flicker is an artifact of the update schedule. Degree doesn't explain the lattice contrast. The random graphs have
   degree 3.5 and aren't bipartite, yet they flicker almost as much as the bipartite square lattice. At matched node
   degree the square and triangular lattices still differ by a factor of 2–9 in $E_2/E_1$. Within random graphs, more
   agree degree means more flicker, and triangles through an agent mean less. What suppresses the flicker is short
   odd cycles (triangles), not the lack of global bipartiteness.
3. **Every cell is at or beyond the stability boundary when the dynamics start, except Gemma on objective tasks.**
   The effective couplings in logit units are $g=1.3$–$4.5$ and $\bar J=2.0$–$6.0$ on random graphs ($g$ up to 6.8 and
   $\bar J$ up to 15.6 on the lattices), with $\gamma=1$ exactly. The linear map on the actual graphs at the round-0
   state has largest eigenvalue 1.5–4.9 for GPT-4o-mini and 1.1–1.9 for Qwen (beyond the boundary), and 0.8–1.2 for
   Llama and Gemma-subjective (at it). For Gemma-objective it's 0.3 (below; only 14% of communities are unstable).
   By rounds 3–7 every cell has saturated into an attractor where the linearization is stable. That matches the
   $\gamma=1$ synchronous theorem: 2-cycles on the square lattice and random graphs, consensus on the triangular
   lattice. The dense DMFT $\Lambda$ (claim2 eq. 8) has no weak-coupling solution in any cell. It also overstates the
   actual graph's instability by about 3×, so it's not quantitatively usable at $N=32$ and degree 1.3–4.6.

Source: repository at `2983a9d` plus the uncommitted changes in `analysis/poa/` and `results/poa/`. Data:
`/workspace-global/swarm-cavity/data/poa/agent-opinions.parquet` (sync, 9,600 communities) and
`.../poa/async/agent-opinions-async.parquet` (HF `physics-of-agents/agent-opinions-async`, 10.9 MB, 480 episodes;
it was already downloaded). Seeds: bootstrap seed 0 with B = 200 resamples over statements; simulation seed 0 with 20
runs per episode.

## 1. Field sufficiency in counts (`field_sufficiency.py`)

For every transition $t\to t+1$ ($t=1..7$), the K = 5 votes at $t+1$ are regressed on the inbox counts at $t$:
$n_{a\pm}$ ($n_{d\pm}$) counts agree (disagree) neighbours holding $\pm1$, with $m_a=n_{a+}-n_{a-}$ and
$m_d=n_{d+}-n_{d-}$. Every model has one intercept per (statement, agent), which plays the role of the quenched
private field. The nested models are: M0 (no peers), M1 ($\beta\,Js$), M2 ($b_+m_a-b_-m_d$), M2avg (M2 divided by the
inbox size), M2cat (free functions of $m_a$ and $m_d$, additive), M3 (four free counts), M3sq (plus squares), M4 (all
quadratic and interaction terms), and M6 (M4 plus $s_i(t)$ and $s_i(t-1)$). The score is held-out cross-entropy from
5-fold cross-validation by graph, with standard errors clustered over statements. Objective tasks are put in the
truth frame. "Share" below is $(\mathrm{CE}_{M0}-\mathrm{CE}_X)/(\mathrm{CE}_{M0}-\mathrm{CE}_{M4})$, the fraction of the
peer information that model X recovers. Contrasts are in millinats per vote, and negative means the first model is
better.

| cell | $b_+$ [95% CI] | $b_-$ [95% CI] | share M1 / M2 | avg − sum | saturation (M2cat − M2) | interaction (M4 − M3sq) | M4 − M2 | coef. $s_i(t)$ / $s_i(t-1)$ |
|---|---|---|---|---|---|---|---|---|
| GPT-4o-mini obj. | 3.40 [3.16, 3.67] | −0.07 [−0.18, 0.04] | 0.48 / 0.97 | −3.3 ± 1.2 | −1.3 ± 0.4 | −6.2 ± 1.3 | −11.6 ± 1.2 | 0.11 / 0.36 |
| GPT-4o-mini subj. | 3.42 [3.26, 3.62] | +0.62 [0.46, 0.79] | 0.61 / 0.91 | −9.4 ± 2.4 | −14.1 ± 1.5 | −8.6 ± 1.8 | −35.0 ± 3.9 | 0.39 / 0.85 |
| Qwen3.5-9B obj. | 1.57 [1.45, 1.69] | −0.38 [−0.45, −0.33] | 0.28 / 0.92 | −6.2 ± 1.4 | −5.0 ± 0.6 | −8.1 ± 0.8 | −15.0 ± 1.4 | 0.04 / 0.28 |
| Qwen3.5-9B subj. | 1.42 [1.24, 1.60] | +0.08 [0.04, 0.11] | 0.56 / 0.89 | −8.7 ± 1.8 | −9.0 ± 1.0 | −8.7 ± 0.9 | −24.0 ± 3.2 | 0.13 / 0.50 |
| Llama-3.1-8B obj. | 1.37 [1.24, 1.49] | −0.24 [−0.31, −0.17] | 0.35 / 0.94 | −2.8 ± 1.4 | −2.8 ± 0.3 | −6.1 ± 0.8 | −10.0 ± 1.0 | 0.07 / 0.17 |
| Llama-3.1-8B subj. | 2.08 [1.95, 2.24] | −0.31 [−0.43, −0.21] | 0.31 / 0.86 | −8.5 ± 1.6 | −5.5 ± 1.0 | −4.5 ± 0.8 | −20.1 ± 4.1 | 0.29 / 0.73 |
| Gemma-3n-E4B obj. | 1.09 [0.95, 1.27] | −0.47 [−0.59, −0.37] | 0.19 / 0.92 | −0.5 ± 1.8 | −1.6 ± 0.7 | −4.1 ± 0.8 | −5.9 ± 1.4 | 0.17 / 0.25 |
| Gemma-3n-E4B subj. | 1.03 [0.83, 1.34] | −0.14 [−0.23, −0.04] | 0.40 / 0.86 | +8.8 ± 3.7 | −11.4 ± 1.9 | −4.6 ± 1.9 | −16.4 ± 2.9 | 0.42 / 0.66 |

The total peer information (M0 − M4) is 75–405 millinats per vote, and moving from M1 to M2 is worth 55–198 of it
(4.5–16 s.e.).

- **Two gains, one field.** M1 forces disagree ties to repel with the same strength as agree ties attract. In the
  data, a message from a disagree tie pulls the agent toward it ($b_-<0$: Qwen-obj, Llama, Gemma), does nothing
  (GPT-obj), or repels weakly (GPT-subj $b_-/b_+=0.18$, Qwen-subj 0.06). Our gains match PoA's own two-stage fit
  exactly, because it spans the same model (log-likelihood difference below $10^{-3}$). They're also close to PoA's
  `couplings.json`, which was fitted on the 8 training graphs only (GPT-obj 3.03 / −0.18, GPT-subj 3.00 / 0.46).
- **Sum vs average.** Averaging (M2avg) beats summing by 2.8–9.4 millinats in 6 cells (2–5 s.e.). Gemma-subjective
  prefers summing (+8.8 ± 3.7) and Gemma-objective is a tie. Averaging each inbox separately gives mixed results: it's
  worse than averaging the whole inbox in the objective cells and better in GPT- and Qwen-subjective. The preference is real but small, and with node degrees of only 1 to 6 in these
  graphs it can't pin down the $K^{-1/2}$ of claim1 eq. (10).
- **Additivity.** Saturation within a channel and interactions between counts are each significant in every cell,
  at 1–14 and 4–9 millinats respectively. Their combined cost (M4 − M2) is 6–35 millinats, or 3–14% of the peer
  information, and it's largest on subjective tasks.
- **Own state, which fails "memoryless".** The prompt doesn't show the agent its previous opinions, yet adding
  $s_i(t)$ and $s_i(t-1)$ improves held-out CE by a further 0.3–25 millinats. The coefficient on $s_i(t-1)$ (0.17–0.85)
  exceeds the one on $s_i(t)$ in every cell. Two explanations fit: an echo through message text (neighbours' messages
  at $t$ are written after reading $i$'s message at $t-1$), or per-community persistence that the (statement, agent)
  intercept doesn't absorb. This is the "hidden state" failure of `claim1.md` §4.2, and the data can't separate the
  two explanations.

Self-test (`selftest_fit.txt`): on synthetic M2 votes with the real design, the fit recovers $b_+=1.5$ and
$b_-=0.5$ inside the bootstrap CI. Held-out CE ranks M1 worse by 53 ± 2 millinats, and M3, M4 and M6 come out no
better than M2 (within 0.01 millinats).

## 2. Asynchronous control and the degree confound (`flicker.py`, `async_sim.py`)

**Async release.** It contains GPT-4o-mini only, with 60 statements × the 8 training random signed graphs × 1
replica, and no lattices. Every (statement, graph) pair also appears in the synchronous data. In each step an agent
fires Poisson(0.5) times, and at each firing it posts its current spin and then revotes from the latest posted
messages. Silent agents are recorded as their carried spin repeated 5 times. That recording rule breaks the
requested belief-based $E_2/E_1$: it comes out 1.91 and 1.92 under async, and the fitted agent with the same
recording rule gives 2.03 and 2.04, so this number measures the recording artifact, not the dynamics. The clean
statistic is the spin flip rate $a_k=P(s_i(t)\neq s_i(t-k))$ over steps 3–8. A change needs at least one firing in
the lag, so an agent that redraws independently at each firing has $a_2/a_1=f_2/f_1=0.632/0.394=1.61$. We report
$(a_2/a_1)/1.61$.

| GPT-4o-mini, rounds 3–8 | sync observed | sync, fitted agent | async observed | async, fitted agent |
|---|---|---|---|---|
| objective: period index / null | 0.29 ± 0.02 | 0.25 | 1.17 ± 0.01 | 1.15 |
| objective: flip rate $a_1$ | 0.361 | 0.380 | 0.111 | 0.111 |
| subjective: period index / null | 0.26 ± 0.02 | 0.24 | 1.18 ± 0.01 | 1.16 |
| subjective: flip rate $a_1$ | 0.246 | 0.236 | 0.070 | 0.071 |

The async index sits slightly above 1, which means drift as the swarm relaxes toward agreement, and the fitted
agent predicts it to within 0.02. It is the same (1.16–1.20) on random graphs whose agree subgraph is bipartite and
on those whose agree subgraph isn't. The fitted agent reproduces the spin-level flicker, but it overpredicts the
belief-level flicker under sync: synchronous belief $E_2/E_1$ is 0.08 simulated against 0.22 observed. Real agents
have extra round-to-round variance in their vote probability that isn't period 2, which would fit the annealed
message-wording noise of `claim1.md` §4.2. The test can't say whether the flicker also vanishes for the three open
models or on lattices, because no async data exist for them.

**Degree vs bipartiteness** (sync, all 8 cells, belief $E_2/E_1$ with s.e. over statements; `fig_flicker`):

- *Random graphs* (degree exactly 3.5, not bipartite): 0.20–0.73. *Square lattice* (3.25, bipartite): 0.18–0.66.
  *Triangular lattice* (4.56): 0.88–1.31. Degree 3.5 against 3.25 can't explain why the random graphs behave like the
  square lattice and not the triangular one.
- *Matched node degree:* for degree-4 nodes, $E_2/E_1$ is 0.15–0.64 on the square lattice and 0.86–1.35 on the
  triangular lattice, in every cell.
- *Same degree, different agree-subgraph bipartiteness:* four random graphs have a fully bipartite agree subgraph
  and five have at most 25% of agents in bipartite agree components. $E_2/E_1$ is lower on the bipartite ones in 7 of
  8 cells, but only by 0.02–0.12 (for example, Llama-obj 0.37 ± 0.04 against 0.43 ± 0.03). That's a weak effect,
  which fits disagree ties also carrying coupling ($b_-\neq0$).
- *Agent level within random graphs* (OLS of $E_1-E_2$, clustered by statement): each triangle through the agent
  reduces flicker by 0.006–0.077, in all 8 cells and beyond 2 s.e. in 6. Each extra agree tie increases it by
  0.010–0.056 (beyond 2 s.e. in 7). Being in a bipartite agree component adds 0.02–0.33 (beyond 2 s.e. in 2).

On the triangular lattice the agents reach consensus ($E_1$ = 0.10–0.33, against 0.37–1.39 on the agree-bipartite random graphs),
so there's nothing left to flicker. All of this points to short odd cycles as the suppressor. With one triangular graph,
we can't separate its triangle density from its mean degree at the graph level.

## 3. Effective couplings, Λ and regime (`effective_coupling.py`)

The effective coupling is $W_{ij}=b_+[J_{ij}=+1]-b_-[J_{ij}=-1]$ on each actual graph. Mapping it onto claim1 eq. (2)
gives $\bar J=\langle\sum_jW_{ij}\rangle$, $g^2=\langle\sum_jW_{ij}^2\rangle-\bar J^2/N$ and $\gamma=1$, since $W$ is
symmetric. The bare gain of the majority-of-5 spin is $d_i=\phi'(u_i)=60[p(1-p)]^3$, which is at most 0.94. It is
evaluated at three points: $u=\alpha_i$ (a balanced inbox), the round-0 state, and rounds 3–7. The finite-graph
criterion is the eigenvalues of $DW$ for each community. The table uses M2; M2avg gives similar numbers and is in
the CSV.

| cell | $g$ / $\bar J$ (random graphs) | $2g\sqrt{\langle d^2\rangle}$, balanced inbox | largest eig. of $DW$ at round 0: square / random / triangular | communities unstable at round 0 | same, rounds 3–7 | regime at the start |
|---|---|---|---|---|---|---|
| GPT-4o-mini obj. | 4.34 / 6.02 | 6.6 | 2.27 / 1.93 / 4.87 | 67% | 0.14 / 0.43 / 1.07 | beyond |
| GPT-4o-mini subj. | 4.49 / 4.86 | 4.2 | 1.46 / 1.47 / 2.19 | 56% | 0.33 / 0.63 / 0.68 | beyond |
| Qwen3.5-9B obj. | 2.04 / 3.40 | 2.9 | 1.46 / 1.17 / 1.86 | 64% | 0.14 / 0.31 / 0.33 | beyond |
| Qwen3.5-9B subj. | 1.82 / 2.33 | 2.4 | 1.10 / 1.08 / 1.39 | 60% | 0.41 / 0.69 / 0.77 | just beyond |
| Llama-3.1-8B obj. | 1.76 / 2.80 | 2.2 | 0.97 / 0.82 / 1.16 | 43% | 0.15 / 0.30 / 0.31 | at the boundary |
| Llama-3.1-8B subj. | 2.66 / 4.15 | 2.3 | 0.90 / 0.89 / 1.07 | 41% | 0.15 / 0.29 / 0.28 | at the boundary |
| Gemma-3n-E4B obj. | 1.49 / 2.73 | 1.6 | 0.34 / 0.31 / 0.35 | 14% | 0.19 / 0.26 / 0.14 | below |
| Gemma-3n-E4B subj. | 1.32 / 2.03 | 1.5 | 0.99 / 0.87 / 1.13 | 46% | 0.55 / 0.60 / 0.49 | at the boundary |

The round-0 eigenvalue varies widely between communities, with a 10–90% range from about 0 up to 1.2–7.6. That variation
is larger than the effect of the gain uncertainty (5–15%). It follows claim2 §3: a statement on which the round-0
answers are lopsided saturates $d$, so the same model sits below the boundary on one question and above it on
another. That's why the table gives the fraction of unstable communities. The replica check doesn't use the fitted
kernel. It gives amplification $A(0)=0.89$–$1.03$ (the round-0 floor is right), then $A(8)=1.7$–$50$, and
$\hat\Lambda(t\ge1)=0.4$–$0.9$ in every cell. Replicas split in the first round and then stay apart without diverging
further, and they end up disagreeing on 3–33% of agents. On the lattices $A$ keeps rising because the sampling floor
falls as agents saturate, not because the replica distance $D$ grows.

**What fails in item 3:**

- *Dense DMFT.* With $\gamma=1$, claim2 eq. (8) has no weak-coupling solution at the balanced inbox in any cell, and
  none at the round-0 state except Gemma-objective, where it gives 0.26 against 0.31 on the actual graph. At the
  late states the homogeneous estimate $2g\sqrt{\langle d^2\rangle}$ is 0.7–2.8, while the actual graphs give
  0.14–1.07, about 3× lower. These graphs, with $N=32$, degree 1.3–4.6 and saturated agents sparsely mixed with
  responsive ones, are outside the regime the dense theory describes.
- *Many weak inputs* (`claim1.md` §4, assumption 2). One neighbour flipping moves the logit by $2b_+=2$–$7$, so the
  agents act close to threshold units. Linearization works for Gemma-objective, where communities with round-0
  eigenvalue above 1 flicker 3× more than those below 0.5 ($E_1-E_2$ = 0.24 against 0.07). It fails for GPT-4o-mini:
  communities that start linearly stable flicker as much as the unstable ones (1.10 against 0.86), because the
  2-cycle is a large-amplitude nonlinear attractor and doesn't grow from a small perturbation.
- *Graph-level resolution.* Only 14 distinct graphs exist, and only one is triangular.

## Figures

- `fig_field_sufficiency`: (a) held-out cross-entropy of M0, M2, M2avg, M2cat and M4 relative to the literal field
  M1, per model × task (±1 s.e. over statements; below zero is better than M1). (b) $b_+$ and $b_-$ with 95%
  bootstrap CIs over statements. The gap between M1 and the rest is the cost of assuming disagree ties repel.
- `fig_flicker`: (a) GPT-4o-mini spin period index divided by each schedule's no-structure null, observed (±2 s.e.)
  and simulated with the fitted agent, for sync and async. (b) Belief $E_2/E_1$ on four graph groups for all 8
  cells (±2 s.e.).
- `fig_stability`: (a) the largest eigenvalue of $DW$ on the actual graphs at round 0 (filled) and in rounds 3–7
  (open), with the dashed line at 1 marking the linear stability boundary. (b) Replica amplification $A(t)$ by
  graph class for all cells; thick lines are GPT-4o-mini.

## Checks and fixes to the previous agent's code

The earlier partial results (Llama-objective cross-validation, self-test) reproduce exactly. `fit_logit_fe`, the
cluster standard errors, the $J$ orientation in `transitions` and the async firing null were checked and are correct.
Four things were changed:

- `common.py`, `field_sufficiency.py` and `flicker.py` imported and wrote through the stale `/workspace/swarm-cavity`
  copy, which is on the full volume. They now resolve paths from the clone.
- `flicker.py` part B assumed $b_-\approx0$, so that the agree subgraph would be the interaction graph. In fact
  $b_-$ is significantly negative in 5 cells, and the agree-bipartiteness split turned out weak. Part B now tests degree, triangles and
  agree-bipartiteness separately, as described above.
- `field_sufficiency.py` now adds M2avg_sep, M2cat and M3sq, plus contrasts against M2 and M3sq with standard errors,
  so the saturation and interaction questions have their own tests.
- The async belief-$E_2/E_1$ contamination that the previous agent suspected is confirmed by simulation.

## Files

- Scripts in `analysis/poa/`:
  - `common.py`: loader frame, transition table, fixed-effect logit, graph helpers, `fit_m2`.
  - `field_sufficiency.py`: task 1; `--selftest` runs the recovery check.
  - `flicker.py`: task 2, data side.
  - `async_sim.py`: task 2, fitted agent under both schedules.
  - `effective_coupling.py`: task 3.
  - `plot.py`: figures and `field_sufficiency_summary.csv`.
- Results in `results/poa/`:
  - `field_sufficiency_{cv,cv_by_class,gains,summary}.csv`, `field_sufficiency_coefs.json`, `fits_M2.npz`,
    `field_sufficiency_log.txt`, `selftest_fit.txt`.
  - `flicker_{async,async_by_round,graphs,bip_contrast,agent_regression,lattice_degree}.csv`, `flicker_log.txt`.
  - `async_sim.csv`, `async_sim_log.txt`.
  - `effective_coupling{,_amplification}.csv`, `effective_coupling.json`, `effective_coupling_log.txt`.
- Figures: `fig_field_sufficiency`, `fig_flicker`, `fig_stability` (`.pdf` and `.png`).

To reproduce, run `cd /workspace-global/swarm-cavity/repo && source scripts/env.sh`, then
`uv run python analysis/poa/<script>.py` in this order: field_sufficiency (about 35 min with B = 200 on 2 cores),
flicker, async_sim (2 min), effective_coupling (1–2 min), plot.
