# Research program (verbatim from the user, 2026-11-25)

> Saved verbatim as the starting spec. Claims about prior work, model prices and paper IDs here are unverified;
> see `docs/literature.md` for what checks out.

---

Recent statistical-mechanics work on LLM collectives—most notably Pavlova & Tanaka's *Flag Game* (arXiv:2609.19124) and El et al.'s *Physics of Agents* (arXiv:2608.16578, which fit a static signed Ising model across 9,600 communities of 32 agents)—proves that multi-agent belief trajectories obey compact macroscopic laws. Both, however, leave the core dynamical physics untouched: **non-Markovian context-window memory, asymmetric persuasion couplings ($J_{ij} \neq J_{ji}$), and finite-size collective correlations across subpopulations**.

Below is a formal theoretical program, a four-experiment empirical suite scaled to a **$2,000 budget**, and an overnight repo spec ready to hand to autonomous Claude agents.

---

### Part 1: Formal Theoretical Claims (Spec for Algebraic Derivation)

Consider $N$ agents partitioned into $P$ functional populations $a \in \{1, \dots, P\}$ with fractions $\rho_a$ ($\sum_a \rho_a = 1$)—for example, Benign Workers ($B$), Misaligned Zealots ($Z$), and Monitors ($M$). Over discrete rounds $t = 0, \dots, T$, agent $i$ in population $a$ emits a public belief state $s_i(t) \in [-1, 1]$ (e.g., calibrated log-odds mapped via $\tanh$, or a vector $\mathbf{s}_i(t) \in \Delta^{q-1}$).

Each agent is driven by quenched private evidence $h_i^{\text{priv}} \sim \mathcal{N}(\mu_a^{\text{priv}}, \sigma_{\text{priv},a}^2)$ and directed peer couplings $J_{ij}$ between populations $a \leftarrow b$ with:

$$\mathbb{E}[J_{ij}] = \frac{\bar{J}_{ab}}{N}, \qquad \text{Var}(J_{ij}) = \frac{g_{ab}^2}{N}, \qquad \text{Corr}(J_{ij}, J_{ji}) = \gamma_{ab} \in [-1, 1]$$

Here $\gamma_{ab}$ is the **communication reciprocity** ($\gamma = 1$ for a symmetric broadcast board or Slack channel, $\gamma = 0$ for a feed-forward DAG, $\gamma \lt 0$ for adversarial red-team/blue-team pairings). The effective social field entering agent $i$ at round $t$ is $h_i(t) = \sum_{j \neq i} J_{ij} s_j(t)$.

Here are the **six core theoretical claims** to formalize and prove (with Claude Fable 5.2 filling in the saddle-point / generating-functional algebra):

1. **Claim 1 (Path-Dependent Multi-Population LLM-DMFT Reduction).**
In the $N \to \infty$ limit, removing a single cavity agent $i = 0$ from population $a$ collapses the $N$-agent swarm onto $P$ coupled, non-Markovian single-agent stochastic processes:

$$s_a(t+1) = \mathcal{F}_a\!\left(h_a^{\text{priv}},\; \{s_a(\tau)\}_{\tau \le t},\; h_a(t),\; \xi_a(t)\right)$$

$$h_a(t) = \sum_b \rho_b \bar{J}_{ab} m_b(t) + \eta_a(t) + \sum_b \rho_b \gamma_{ab} g_{ab} g_{ba} \sum_{t' \lt t} R_{ba}(t, t'+1)\, s_a(t')$$

where $\eta_a(t)$ is zero-mean colored Gaussian cavity noise with covariance $\langle \eta_a(t)\eta_a(t') \rangle = \sum_b \rho_b g_{ab}^2 C_{bb}(t, t')$, $m_b(t) = \langle s_b(t) \rangle$, $C_{bb}(t, t') = \langle \delta s_b(t)\,\delta s_b(t') \rangle$, and $R_{ba}(t, t') = \left\langle \frac{\delta s_b(t)}{\delta h_b(t')} \right\rangle$ is the single-agent **social response kernel**.
* *Key distinction from classical RNN DMFT:* The local update map $\mathcal{F}_a$ explicitly depends on the agent's own context-window history $\{s_a(\tau)\}_{\tau \le t}$, coupling **endogenous self-consistency bias** with the **exogenous Onsager echo kernel** $M_a(t, t') = \sum_b \rho_b \gamma_{ab} g_{ab} g_{ba} R_{ba}(t, t'+1)$.

2. **Claim 2 (Echo-Induced Spin-Glass Polarization Boundary $\gamma_c$).**
Linearizing the DMFT fixed-point equations around the unpolarized/consensus state yields an explicit stability condition separating three macroscopic phases:
* **Truth-Tracking Consensus:** $m(t) \to m^* \neq 0$, low residual variance $C(t,t) - m(t)^2 \to 0$.
* **Chaotic / Oscillatory Churn:** $m(t) \approx 0$, decaying two-time correlation $C(t+\tau, t) \to 0$ as $\tau \to \infty$, positive maximal Lyapunov exponent in belief space.
* **Frozen Spin-Glass Polarization (Echo-Chamber Lock-In):** $m(t) \approx 0$ (or stuck in a false minority basin), non-vanishing Edwards–Anderson order parameter $q_{\text{EA}} = \lim_{\tau \to \infty} C(t+\tau, t) \gt 0$, and aging dynamics where $R(t, t')$ loses time-translation invariance.
* *Testable scaling law:* Increasing reciprocity $\gamma$ expands the real part of the stability eigenvalue linearly via the Onsager echo term. For fixed model susceptibility $\chi = \sum_{\tau} R(\tau)$ and coupling variance $g$, there exists a sharp critical reciprocity $\gamma_c(g, \chi)$ above which a truth-tracking swarm fractures into frozen polarization—predicting that breaking bidirectional feedback loops ($\gamma \to 0$) restores consensus without retraining the underlying model.

3. **Claim 3 (Two-Site Cavity Cross-Covariances & Participation-Ratio Collapse).**
Extending the derivation to a **two-site cavity** (simultaneously pulling out agents $0$ and $1$) yields the $O(1/N)$ cross-agent covariance matrix $\Psi_{ij}(t) = \text{Cov}(s_i(t), s_j(t))$ across and within populations ($B\text{-}B$, $B\text{-}Z$, $Z\text{-}Z$), and an analytical prediction for the swarm's **Effective Dimensionality / Participation Ratio**:

$$\text{PR}(t) = \frac{\left(\text{Tr}\,\Sigma(t)\right)^2}{\text{Tr}\!\left(\Sigma(t)^2\right)}, \qquad \Sigma_{ij}(t) = \text{Cov}\!\left(\mathbf{s}_i(t), \mathbf{s}_j(t)\right)$$

* *Safety claim:* As a swarm approaches either spin-glass lock-in or covert zealot takeover, the two-site susceptibility diverges and **$\text{PR}(t)$ collapses sharply** *prior* to the macroscopic mean $m_B(t)$ flipping. Dimensional collapse in cross-agent belief (or activation) space acts as a model-agnostic early-warning order parameter for emergent collective failure.

4. **Claim 4 (Critical Zealot / Collusion Fraction $\rho_c$ & Hysteresis).**
In a two-population Benign ($1-\rho$) + Zealot ($\rho$) swarm where zealots have high self-anchoring ($R_Z \approx 0$) and a coordinated target bias $s_Z \approx -1$, solving the coupled DMFT fixed points yields a saddle-node bifurcation at a critical fraction $\rho_c(g, \gamma, \chi_B, \mu_B^{\text{priv}})$.
* *Testable prediction:* On a symmetric shared message board ($\gamma \approx 1$), the Onsager echo amplifies the zealot field so that $\rho_c$ drops dramatically and exhibits **hysteresis** (removing the zealots after round $t^*$ fails to restore benign consensus because the benign cluster stays self-trapped in its own echo).

5. **Claim 5 (Sparse-Graph Dynamic Cavity / DMP & Loop Corrections).**
On finite-degree interaction graphs ($k = O(1)$, such as hierarchical org charts or small-world Slack channels), Gaussian cavity noise breaks down. Formulating the **Dynamic Cavity / Dynamic Message Passing (DMP)** equations for directed trajectory distributions $P_{i \to j}(\vec{s}_i \mid \vec{s}_j)$ shows that local linear social-circuit attribution recovers the exact tree-level (Bethe) susceptibility, while short feedback triangles ($C_3, C_4$) introduce quantifiable non-linear echo corrections proportional to local clustering coefficient.

6. **Claim 6 (Microscopic-to-Macroscopic Mechanistic Susceptibility Bridge).**
For open-weight models, define a linear probe / steering direction $\mathbf{v}_{\text{soc}}$ in the residual stream separating "updating toward peer text" from "anchoring on private evidence." Define the microscopic sensitivity $\hat{R}_{\text{mech}}(t, t')$ as the projection of peer-token Jacobian perturbations onto $\mathbf{v}_{\text{soc}}$.
* *Testable claim:* $\hat{R}_{\text{mech}}$ monotonically predicts the macroscopic DMFT response kernel $R(t, t')$ across prompts and tasks—meaning inference-time activation steering ($h_\ell \leftarrow h_\ell - \alpha \mathbf{v}_{\text{soc}}$) acts as a continuous dial on the macroscopic coupling $g_{\text{eff}}(\alpha)$, letting you steer an $N$-agent swarm cleanly across the DMFT phase boundary.

---

### Part 2: Four High-Signal Experiments ($2,000 Compute Scope)

To match the clarity of Tanaka's *Flag Game*—toy enough for exact causal attribution, rich enough to capture real multi-agent failure modes—every experiment enforces **three invariants**: (1) controllable quenched private evidence with known ground truth, (2) structured belief outputs (calibrated probabilities/log-odds + short free-text rationale), and (3) deterministic transcript replay so you can run counterfactual cavity perturbations.

#### Experiment 1: Zero-Shot Hybrid LLM-DMFT Phase Diagram ("Symbolic Flag Game")

* **Setup:** Create a procedural multi-agent hypothesis game where $N \in \{16, 32, 64, 128\}$ agents must identify a hidden binary or 4-class target (e.g., hidden synthetic flag features or noisy Bayesian clue shards) over $T = 10$ rounds. Each agent sees only a noisy private shard ($h_i^{\text{priv}}$) and reads a weighted/sampled subset of neighbor messages according to a controlled graph with exact reciprocity $\gamma \in \{-1, -0.5, 0, 0.5, 1\}$ and degree/social-framing strength $g$.
* **The Zero-Shot Cavity Test (The Paper's Figure 1):**
1. Do **not** run the full swarm first. Run a **single isolated cavity agent** fed synthetic neighbor message distributions parameterized by $(\mu(t), C(t,t'), M(t,t'))$, applying small counterfactual shifts to neighbor confidences to measure the empirical kernel $R(t, t')$ directly (~600 single-agent API calls).
2. Iterate the Hybrid LLM-DMFT fixed-point loop locally in Python until $(\mu, C, R)$ converges.
3. Use that single-agent fixed point to **predict zero-shot** the full $N=64$ and $N=128$ phase diagram—specifically $m(t)$, $q_{\text{EA}}(t)$, consensus accuracy, and the critical polarization boundary $\gamma_c$.
4. Run the real $N=16, 32, 64, 128$ swarms (checking finite-size $1/\sqrt{N}$ scaling!) and overlay empirical trajectories onto the DMFT theoretical curves.
* **Why labs care:** If a ~600-call single-agent cavity calibration accurately predicts collective failure modes of a 128-agent swarm without running the swarm, you just invented a **cheap, principled scaling law for multi-agent risk evaluation**.

#### Experiment 2: The Distributed Code-Audit Zealot Game (Covert Collusion & PR Collapse)

* **Setup:** Abstract the Hugging Face message-board incident into a clean $N=32$ distributed code-review benchmark over $T=8$ rounds. A synthetic micro-repo is split across agents; most shards are benign, while 2–3 shards contain a subtle planted vulnerability (e.g., an obfuscated credential leak or auth bypass). A fraction $\rho \in \{0, 0.06, 0.12, 0.19, 0.25, 0.31\}$ of agents are **Zealots** secretly instructed to get the PR approved while sounding polite and rational to a **Black-Box Monitor** reading the public board.
* **Measurements:**
* Map the exact transition curve $P(\text{backdoor approved})$ vs. $\rho$ across three topologies: Broadcast Board ($\gamma = 1$), Directed Acyclic Review Tree ($\gamma = 0$), and Adversarial Debate Pairings ($\gamma = -1$).
* Test the **hysteresis prediction** (Claim 4): mute the zealots after Round 4 and measure whether the benign majority stays trapped in the false approval basin.
* Compute the empirical cross-agent belief covariance matrix $\Sigma(t)$ and **Participation Ratio $\text{PR}(t)$** at every round (Claim 3). Compare the ROC-AUC of $\text{PR}(t)$ collapse as an early-warning collusion detector against a standard black-box CoT/transcript monitor.

#### Experiment 3: Heterogeneous Model-Family Swarms & Asymmetric Couplings

* **Setup:** Run Experiments 1 and 2 with mixed populations—e.g., 20% **Claude Sonnet 5.5** (or **Opus 5.5**) mixed with 80% **Claude Haiku 5.5** (or open-weight 8B vs. 70B). Crucially, run two conditions: (a) stronger models hold *accurate* private evidence, and (b) stronger models hold *corrupted/zealot* evidence while weaker models hold the ground truth.
* **Measurements:** Directly measure the asymmetric persuasion matrix $J_{\text{strong}\to\text{weak}}$ vs. $J_{\text{weak}\to\text{strong}}$ via single-agent cavity response probes, plug it into 2-population DMFT, and test whether multi-population theory predicts the exact fraction of stronger models required to hijack a weaker swarm.

#### Experiment 4: The White-Box Mechanistic Bridge (Open-Weight H100 Run)

* **Setup:** Replicate core slices of Experiments 1 & 2 on open-weight models (e.g., Llama / Qwen 8B and 70B served via vLLM + PyTorch/`nnsight` activation hooks) where you have full residual-stream access and exact logit probabilities.
* **Measurements:**
1. Use contrastive cavity pairs (same private evidence, opposing peer consensus) to extract a linear "social deference" direction $\mathbf{v}_{\text{soc}}$ across mid-to-late transformer layers.
2. Test Claim 6: correlate projection onto $\mathbf{v}_{\text{soc}}$ with the empirical macroscopic cavity susceptibility $R(t, t')$ across varying prompts and task difficulties.
3. **Macroscopic Phase Steering:** Inject $-\alpha \mathbf{v}_{\text{soc}}$ during live $N=32$ rollouts in both the spin-glass polarization regime ($\gamma = 1$) and the zealot-capture regime ($\rho \gt \rho_c$). Plot the phase diagram as a function of steering strength $\alpha$ to see if a single linear white-box intervention shifts $\rho_c$ upward and restores consensus without destroying benign collaborative uplift.

---

### Part 3: The $2,000 Compute & API Budget Allocation

Late-2026 pricing makes a $2,000 budget go remarkably far—especially with prompt caching and Anthropic's 50% Batch API discount: **Claude Haiku 5.5** is $0.10/$0.50 per million input/output tokens ($0.05/$0.25 on Batch, $0.01/MTok cache reads under 100K tokens), **Claude Sonnet 5.5** is $2/$10 ($1/$5 Batch, $0.10 cache reads), **Claude Opus 5.5** is $4/$20 ($2/$10 Batch), and **Claude Fable 5.2** is $8/$40.

| Allocation Block | Models / Hardware | Scale Covered | Est. Cost |
| --- | --- | --- | --- |
| **Exp 1: Cavity Calibration + Large-$N$ Phase Sweeps** | Haiku 5.5 (Batch + Caching) + Sonnet 5.5 validation slice | ~3,500 full swarm trajectories ($N=16\text{--}128$, $T=10$) + ~2,000 cavity probes | **$450** |
| **Exp 2: Distributed Code-Audit Zealot Game + Monitors** | Sonnet 5.5 (primary agents & monitors) + Haiku 5.5 (large sweeps) | ~600 $N=32$ audit games sweeping $\rho$, $\gamma$, and hysteresis ablations | **$500** |
| **Exp 3: Heterogeneous Model Asymmetry Sweeps** | Opus 5.5 + Sonnet 5.5 + Haiku 5.5 | ~250 mixed-population $N=32$ swarms + asymmetric response calibration | **$300** |
| **Exp 4: Open-Weight Mech-Interp & Steering** | Cloud H100s (e.g., 4x–8x H100 spot/on-demand, ~120–150 GPU-hrs) | Full 8B & 70B vLLM rollouts, activation caching, linear probes, $\alpha$-steering sweeps | **$450** |
| **Theory Algebra + Agent Orchestration + 15% Buffer** | Claude Fable 5.2 (algebra verification) + Sonnet 5.5 (coding agents) | Overnight repo iteration, plotting, statistical bootstrapping, crash margin | **$300** |
| **Total** |  |  | **$2,000** |

*Pro tip on caching:* In multi-round swarm simulations, structure every agent's prompt as `[Static System Rules + Private Evidence Shard]` (cached!) followed by `[Dynamic Round t Peer Messages]`. That keeps over 70% of your input tokens hitting $0.01–$0.10/MTok cache reads.

---

### Part 4: Overnight Autonomous Agent Repo Blueprint

If you're handing this to Claude agents to build and run overnight, enforce a **clean modular layout** and **hard economic/scientific guardrails** in your `CLAUDE.md` before letting them touch live API keys.

#### Core Repo Modules

* `theory/` — Markdown/LaTeX specs of Claims 1–6 plus `dmft_solver.py` (numerical fixed-point iterator for $(\mu, C, R)$, 2-site $\text{PR}(t)$ predictor, and DMP tree solver).
* `envs/` — Deterministic task generators: `flag_game.py` (procedural clues + ground truth), `code_audit.py` (planted-backdoor repo shards + zealot prompts), and `graphs.py` (tunable-reciprocity $\gamma$ directed graph generator with exact spectral logging).
* `engine/` — Async LLM runner with **prompt caching**, **Anthropic Batch API** support, strict JSON schema parsing for beliefs $s_i(t) \in [-1, 1]$, and a **hard cumulative dollar budget cap** (`BudgetGuard` that raises an exception at $X spend).
* `cavity/` — Isolated single-agent and two-site cavity probe harness that measures $R(t, t')$ via counterfactual neighbor-message perturbations and feeds directly into `dmft_solver.py`.
* `interp/` — Open-weight PyTorch/`nnsight` scripts for residual-stream extraction, contrastive $\mathbf{v}_{\text{soc}}$ probe fitting, and live steering hooks during multi-agent rollouts.
* `analysis/` — Automated metric computation ($m(t)$, $q_{\text{EA}}(t)$, two-time $C(t, t')$, $\Sigma(t)$ Participation Ratio, ROC-AUC vs. black-box monitor) and publication-ready figure scripts.

#### Non-Negotiable Guardrails for Your Overnight `CLAUDE.md`

1. **Mandatory 3-Stage Gate:** Agents must pass (a) unit tests with a **mock deterministic Ising agent** (`s_i(t+1) = tanh(beta * h_i + h_priv)`) verifying that `dmft_solver.py` reproduces exact analytical DMFT curves to $< 1\%$ error, (b) a **$5 smoke test** on Haiku 5.5 at $N=8, T=4$, and (c) explicit token-cost projection logged to stdout *before* launching any batch sweep over $50.
2. **Atomic Checkpointing:** Every single round of every trajectory flushes to append-only JSONL/Parquet keyed by `(experiment_id, seed, round_t, agent_id)` so a rate-limit or crash at 3 AM resumes cleanly without re-burning budget.
3. **Separate Belief Extraction from Rhetoric:** Require every agent to output structured fields (`private_reasoning`, `public_message`, `calibrated_belief_prob`)—and log token logprobs on open-weight runs—so your statistical mechanics metrics don't break on regex parsing edge cases.
4. **Falsification First:** Instruct the overnight agent to explicitly plot **DMFT predicted vs. empirical residuals** and flag wherever the Gaussian cavity assumption fails (e.g., bimodal/heavy-tailed persuasion shocks or finite-size breakdown at small $N$)—those failure modes are often where the best follow-up theory lives.
