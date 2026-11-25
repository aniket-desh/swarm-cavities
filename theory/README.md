# Theory for swarm-cavity: what holds, what changes, what to test

The six claims of `docs/spec.md` were derived, checked against direct simulation of a toy swarm, and red-teamed.
Each note (`claim1.md` … `claim6.md`) opens with its verdict and names the script in `theory/checks/` that verifies
each formula and what it printed; `solver_spec.md` specifies `theory/dmft_solver.py` with reference numbers. Two
of the spec's claims are false as stated (2 and the early-warning part of 3), one has the sign of its main effect
wrong (4), and the rest hold after corrections.

## Verdicts

| claim | verdict | what replaces it |
|---|---|---|
| 1. DMFT reduction | **Holds, with three corrections.** Noise covariance is the full $\langle ss\rangle$; the reaction term is $\sum_{t'<t}R_b(t,t')s_a(t')$ (a two-round echo, not three); the kernel is the neighbour population's. Corrected equations match an $N=4000$ simulation to 0.004; each spec error alone is off by 0.03–0.06. The solution is one causal forward pass, with no fixed-point loop. | Unknown for LLMs: whether peers act only through a scalar field. With text, $g\approx\beta_{\rm resp}\sqrt{\langle w^2\rangle/K}$, so order-one disorder needs few messages or steep conformity. |
| 2. Sharp $\gamma_c$ into frozen polarization | **False as stated.** | Boundary $\Lambda=\max_\omega g^2\langle\lvert R_i(\omega)\rvert^2\rangle=1$; for homogeneous memoryless agents $d\,g(1+\lvert\gamma\rvert)=1$. Under synchronous rounds $\gamma<0$ destabilizes as much as $\gamma>0$ (period 4 against period 2). Beyond it, $0<\gamma<1$ gives slow chaos with period-2 flicker; exact freezing only at $\gamma=1$ (or $\gamma\gtrsim0.9$ at finite $N$ without noise), into 2-cycles, after hundreds of rounds. Nothing sharp in $\gamma$ is visible at $T=10$. |
| 3. Two-site cavity and PR | **Theory holds; safety claim false.** Discrete-time two-site equations verified (PR/N to <1%). PR falls smoothly toward the bulk instability ($1-\Lambda^2$ at $\gamma=0$, more slowly with reciprocity). | Near a zealot saddle-node PR does not fall before the flip: among unflipped replicas it stays flat or rises, while $N_B\mathrm{Var}(m_B)$ rises 5–20×. PR is an ensemble statistic needing $R\approx256$ replicas at $N=32$ with the U-statistic estimator. |
| 4. Zealot $\rho_c$ and hysteresis | **Threshold and hysteresis: already known** (De Marzo et al. 2026 is the $g=0$ limit), and only for steep conformity curves. **$\gamma$ dependence: sign wrong in the spec.** | $\gamma>0$ raises the forward threshold slightly ($+0.12$ to $+0.18\,g^2$ per unit $\gamma$) and lowers the backward one ($-0.4$ to $-0.5\,g^2$), widening the loop: 0.010 → 0.103 from $\gamma=-0.5$ to $1$ at $g=0.3$. Disorder with $\gamma\le0$ narrows the loop and can remove it. No effect on a broadcast board ($g=0$). |
| 5. Sparse graphs | **Holds, in its near-tautological form.** Directed graphs are exactly solvable by population dynamics over trajectories; reciprocated edges need a closure; triangles give the leading loop correction, weighted by their sign. | At $N=32$, $k=3.5$ the graphs are not tree-like; simulate a surrogate on the actual graph. |
| 6. Mechanistic bridge | **Not meaningful as written.** The monotone relation is guaranteed by a confidence confound, and the spec's contrast extracts a content direction whose steering is a bias, not a change of coupling. | A gain direction, tested on one agent first: slope changes, intercept does not, ablation removes the social response (`claim6.md`). |

Derived and verified numerically: claim 1 equations; the stability boundary including quenched fields; the two-site
system for linear dynamics at any $\gamma$ and for nonlinear agents at $\gamma=0$; the zealot fixed points and
thresholds; the PR estimator. Derived but not verified: multi-channel fields, the multi-population forms of claims
2–3, the two-site system for nonlinear agents with $\gamma\neq0$, reciprocity creating a threshold for voter-like
agents. Numerical observation without derivation: the growth of the correlation time with $\gamma$. Conjecture: a
genuine noiseless transition somewhere in $0.8<\gamma<1$.

## Update after the pilot swarm: the quiet, anchored regime (`quiet_regime.md`)

The Haiku 5.5 pilot swarm and the real-message transfer check put every cell in the quiet regime, and the agent
keeps about 0.7 of its own last answer when its inbox has not changed. `quiet_regime.md` gives the revised
single-agent law (`experiments/04_swarm_design/surrogate_v2.py`), shows that anchoring moves the period-4 boundary
from $d\,g(1+|\gamma|)=1$ to $\sqrt{(1+\rho)/(1-\rho)}$ while leaving the static one in place, and replaces tests
2 and 3 of the table below by a direct measurement of the reaction term: kick one agent's message once and read
its own answer two rounds later. The echo is zero at $\gamma=0$ by construction, its sign follows $\gamma$ and the
wording, and its size can be predicted from first-order responses measured in the same experiment. Frozen
predictions are in section 3.8 of that note and in `checks/kick_predictions.json`.

## Physics of Agents data: usable at zero cost, and two predictions already pass

The release is a synchronous, symmetric ($\gamma=1$), memoryless swarm at $N=32$, $T=8$, with four replicas on each
of ten graphs per question. It cannot test PR (four replicas), negative or partial reciprocity (all graphs are
symmetric), memory, or ensemble cavity curves (graphs too small to be tree-like). It can test three things, and
`theory/checks/check_poa_zero_cost.py` ran the first two on all 9,600 communities (`out_poa_zero_cost.txt`):

1. **Period-2 flicker on bipartite graphs** (`claim2.md` §4, `claim5.md` §4). Prediction: one-round changes exceed
   two-round changes, most on bipartite graphs, not on non-bipartite ones. Result, excess displacement ratio
   $E_2/E_1$ over rounds 3–8 after removing vote-sampling noise: 0.18–0.66 on the all-positive bipartite lattices,
   0.22–0.70 on the random signed graphs, and 0.88–1.31 on the all-positive non-bipartite lattices, for all four
   models and both task types (differences $E_1-E_2$ are 4 to 12 standard errors on bipartite lattices and within
   about 0.03 of zero on non-bipartite ones). Caveats: the two lattice types also differ in degree, and the paper
   itself notes oscillation under synchronous updates; the asynchronous release is the control and has not been
   analysed.
2. **Coupling-induced replica overlap** (`claim2.md` §5). $\Delta q(0)=0.000\pm0.001$ as it must be, and
   $\Delta q(t\ge1)>0$ for every model and task: 0.015–0.04 on objective tasks for the three small open models,
   0.07–0.24 on subjective tasks for those models, and for GPT-4o-mini 0.07–0.13 (objective) and 0.25–0.55
   (subjective), alternating between odd and even rounds, which is the same flicker seen across replicas.
3. **Field sufficiency in counts** (`claim1.md` §4.1), not yet run: regress the five votes on agree- and
   disagree-inbox counts with quadratic and interaction terms.

Recommendation: finish item 3 and the asynchronous control before any API spend. These are first-look analyses of
the whole dataset with no tuning, not a fitted model comparison.

## Predictions worth testing on Claude swarms (about \$600 planned, inside \$1,500)

Costs assume roughly \$0.0003 per Haiku 5.5 agent call and \$0.006 per Sonnet 5.5 call (about 1.2k input and 400
output tokens, no batch discount). These are estimates to be replaced by the smoke test's `response.usage`; the plan
leaves a factor of two.

| # | prediction | measurement | expected effect | falsified if | cost |
|---|---|---|---|---|---|
| 1 | Peers act through a scalar field; the model's conformity slope and channel gains fix $g$, $\bar J$ | single-agent probes: equal field, six compositions; conformity curve; agree/disagree gains $b_\pm$ (`claim1.md` §4.1) | composition variance ratio $\eta^2_{\rm comp}<0.05$; $g=\beta_{\rm resp}\sqrt{\langle w^2\rangle/K}$ | a composition contrast above 20% of the response range at fixed field; then use the multi-channel form or stop | 3k–6k calls: \$2 Haiku, \$40 Sonnet |
| 2 | Reciprocity selects the oscillation, with no sharp $\gamma_c$ | $N=64$, $T=10$, $\gamma\in\{-r_{\max},0,0.5,1\}$ at a gain below and one above the predicted boundary, 50 graphs × 2 replicas: period index $\mathcal P=A_2/A_1$, amplification $\mathcal A$, $\hat\Lambda$, $\Delta q$ (`claim2.md` §5); asynchronous control at one $\gamma<0$ | above the boundary $\mathcal P\approx1$ at $\gamma=0$, $\le0.3$ for $\gamma\ge0.5$, $>1.5$ for $\gamma<0$; $\mathcal A\approx1/(1-\Lambda)$ below, $\ge5$ above; toy values in `claim2.md` | $\mathcal P$ independent of $\gamma$ above the boundary; $\mathcal A$ off by more than 2× from the value implied by the measured kernel; oscillation at $\gamma<0$ surviving asynchronous updates | 8 cells × 64k calls: \$160 |
| 3 | Zero-shot DMFT curves from single-agent calibration | $m(t)$, $C(t,t')$ of swarms at $N=32,64$, $\gamma\in\{0,1\}$, 20 graphs × 2 replicas, against the solver fed by test 1 | residual within 3 s.e. and shrinking with $N$; the reaction term shifts $C(t,t-2)$ in the direction of $\gamma$ | residual that does not shrink from $N=32$ to $64$, or the $\gamma=1$ swarm matching the no-reaction prediction better | 77k calls: \$25 Haiku; $N=32$ slice on Sonnet \$155 |
| 4 | Reciprocity widens hysteresis and barely moves the forward threshold | zealot fraction sweep, 8 values × 40 runs at $N=32$ for broadcast ($g=0$), $\gamma=0$, $\gamma=1$; then partial zealot removal to three levels | forward 50%-flip point shifts by less than 0.02 between $\gamma=0$ and $1$; recovery after partial removal less frequent at $\gamma=1$; shifts scale as $g^2$ from test 1 | $\gamma=1$ lowering the forward threshold by more than 0.03 (the spec's claim), or recovery more frequent at $\gamma=1$ | 570k calls: \$170 |
| 5 | PR does not warn before the flip; $\mathrm{Var}(m_B)$ does | three zealot fractions approaching the threshold, $R=256$ replicas, $N=32$, $T=8$; corrected PR and $N_B\mathrm{Var}(m_B)$ over unflipped replicas | PR/$N_B$ of unflipped replicas within 10% of baseline; $N_B\mathrm{Var}(m_B)$ up at least 3× | PR of unflipped replicas down more than 20% at a fraction where fewer than 5% of replicas have flipped | 200k calls: \$60 |

Order: the Physics of Agents analyses, then test 1 (it decides whether $g$ is large enough for tests 2–4 to show
anything; if $\beta_{\rm resp}\sqrt{\langle w^2\rangle/K}\,(1+|\gamma|)$ stays well below 1 the swarm is in the quiet
regime, and test 2 reduces to checking $\mathcal A=1/(1-\Lambda)$), then 2, 3, 4, 5. Test 2 is the highest-value
swarm experiment: it is where the corrected theory and the spec disagree most visibly, and its main observable
already shows the predicted pattern in the Physics of Agents data. The mechanistic bridge (claim 6) is not in this
list; its single-agent version costs a few GPU-hours and should precede any steering of swarms.
