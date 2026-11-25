# Claim 1: multi-population DMFT with memory, in discrete time

**Verdict.** The reduction holds, and it is standard mathematics (Eissfeller & Opper 1994; Coolen 2000; Kadmon &
Sompolinsky 2015 for the block structure). The spec's formula has three errors: the noise covariance uses the connected
correlation where the full one is required, the reaction term is delayed by one round too many, and the response kernel
in it should be that of the neighbour population alone ($R_b$, not $R_{ba}$). With these fixed, the single-site process
reproduces a direct $N=4000$ simulation to within its sampling error (section 6). What is uncertain is not the
derivation but three assumptions about LLM agents (section 4); the first of them, scalar-field sufficiency, has a
direct test costing a few dollars (section 4.1).

Everything in sections 1–3 is derived and checked numerically. Section 5 (how to realize $J_{ij}$ with LLM agents) is
a design argument, with the parts that rest on measurement marked.

## 1. Model

$N$ agents, populations $a=1,\dots,P$ with fractions $\rho_a$. Synchronous rounds $t=0,1,\dots,T$:

$$h_i(t)=\sum_{j\neq i}J_{ij}\,s_j(t),\qquad
s_i(t+1)=F_a\!\big(h_i^{\rm priv},\{s_i(\tau)\}_{\tau\le t},\,h_i(t),\,\xi_i(t)\big). \tag{1}$$

$J_{ij}$ is the influence of $j$ on $i$ (the convention of `envs/graphs.py`). For $i\in a$, $j\in b$,

$$J_{ij}=\frac{\bar J_{ab}}{N}+\frac{g_{ab}}{\sqrt N}\,z_{ij},\qquad \langle z_{ij}\rangle=0,\quad \langle z_{ij}^2\rangle=1,\quad
\langle z_{ij}z_{ji}\rangle=\gamma_{ab}=\gamma_{ba}, \tag{2}$$

with all other pairs of $z$ independent, so that $N\,\mathrm{Cov}(J_{ij},J_{ji})=\gamma_{ab}\,g_{ab}\,g_{ba}$. The private
fields $h_i^{\rm priv}\sim\mathcal N(\mu_a,\sigma_a^2)$ and the couplings are quenched; $\xi_i(t)$ is the agent's own
sampling noise, independent across agents and rounds. The initial state $s_i(0)$ may depend on $h_i^{\rm priv}$ and
$\xi_i$ but not on $J$.

## 2. Cavity derivation

Add an agent $0$ to population $a$, with fresh couplings $J_{0j},J_{j0}$. Write $s_j^{\setminus 0}(t)$ for the
trajectory of agent $j$ in the swarm without agent $0$. The new agent perturbs the field on $j$ by
$J_{j0}s_0(t')=O(N^{-1/2})$, so to first order

$$s_j(t)=s_j^{\setminus 0}(t)+\sum_{t'<t}R_j(t,t')\,J_{j0}\,s_0(t'),\qquad
R_j(t,t')\equiv\frac{\partial s_j(t)}{\partial h_j(t')}. \tag{3}$$

The sum stops at $t'=t-1$ because, with the update convention $s(t+1)=F(h(t))$, the state at round $t$ depends on the
field only up to round $t-1$: $R_j(t,t')=0$ for $t'\ge t$. (Off-diagonal responses $\partial s_j/\partial h_k$ are
$O(N^{-1/2})$ with random signs and drop out of the sum below.) Substituting (3) into $h_0(t)=\sum_jJ_{0j}s_j(t)$ gives
three pieces.

*Mean.* $\sum_j(\bar J_{ab}/N)s_j^{\setminus 0}(t)\to\sum_b\rho_b\bar J_{ab}m_b(t)$ with $m_b(t)=\langle s_b(t)\rangle$.

*Noise.* $\eta_a(t)\equiv\sum_j(g_{ab}/\sqrt N)z_{0j}s_j^{\setminus 0}(t)$. The $z_{0j}$ are independent of the
$s_j^{\setminus0}$, so by the central limit theorem $\eta_a$ is Gaussian with zero mean and

$$\langle\eta_a(t)\eta_a(t')\rangle=\sum_b\rho_b\,g_{ab}^2\,C_b(t,t'),\qquad
C_b(t,t')=\frac1{N_b}\sum_{j\in b}s_j(t)s_j(t')=\langle s_b(t)s_b(t')\rangle. \tag{5}$$

This is the full second moment. The random part of the coupling multiplies $s_j$, not $s_j-m_b$, so the mean does not
subtract out. The spec's $\langle\delta s\,\delta s\rangle$ is wrong whenever $m_b\neq0$, and it is badly wrong for
zealots, whose $s=-1$ has zero connected correlation but contributes $\rho_Zg_{BZ}^2$ of quenched field variance.

*Reaction.* $\sum_jJ_{0j}J_{j0}\sum_{t'<t}R_j(t,t')s_0(t')$. The product $J_{0j}J_{j0}$ has mean
$\gamma_{ab}g_{ab}g_{ba}/N$ (the $\bar J\bar J/N^2$ part is negligible), the sum over $j$ self-averages, and so

$$\text{reaction term}=\sum_b\Gamma_{ab}\sum_{t'<t}R_b(t,t')\,s_0(t'),\qquad \Gamma_{ab}=\rho_b\,\gamma_{ab}\,g_{ab}\,g_{ba}, \tag{6}$$

where $R_b(t,t')=\langle\partial s_b(t)/\partial h_b(t')\rangle$ is the mean response of population $b$, the neighbours
through which the echo travels. It carries one population index: the spec's $R_{ba}$ is not a different object, only
a mislabel.

Collecting terms, the field on a representative agent of population $a$ is

$$\boxed{\;h_a(t)=\sum_b\rho_b\bar J_{ab}\,m_b(t)\;+\;\eta_a(t)\;+\;\sum_b\Gamma_{ab}\sum_{t'=0}^{t-1}R_b(t,t')\,s_a(t')\;} \tag{4}$$

and the agent evolves by its own map,

$$s_a(t+1)=F_a\big(h_a^{\rm priv},\{s_a(\tau)\}_{\tau\le t},h_a(t),\xi_a(t)\big). \tag{7}$$

The order parameters close the system. Averages are over $h^{\rm priv}$, $\eta$, $\xi$ and the initial condition:

$$m_a(t)=\langle s_a(t)\rangle,\qquad C_a(t,t')=\langle s_a(t)s_a(t')\rangle,\qquad
R_a(t,t')=\Big\langle\frac{\partial s_a(t)}{\partial\theta(t')}\Big\rangle\Big|_{\theta=0}, \tag{8}$$

where $\theta(t')$ is a probe added to $h_a(t')$ in (4) with the kernels $m,C,R$ held fixed. The derivative in (8) is
total: it includes the paths through the agent's own memory in $F_a$ and through its own reaction term.

**Time indexing.** The earliest echo in (4) is $R_b(t,t-1)s_a(t-1)$: agent $0$'s state at $t-1$ enters $h_j(t-1)$,
changes $s_j(t)$, and returns in $h_0(t)$, so $s_0(t-1)$ influences $s_0(t+1)$. An echo takes two rounds. The spec
writes $\sum_{t'<t}R(t,t'+1)s(t')$; with $R(t,t')=\delta s(t)/\delta h(t')$ as the spec defines it, the $t'=t-1$ term is
$R(t,t)=0$ and the first surviving term is $R(t,t-1)s(t-2)$, an echo of three rounds. The simulation in section 6
shows the two-round echo directly: for a memoryless population, a kick to $h(t')$ produces a response at $t'+1$,
exactly none at $t'+2$, and the first echo at $t'+3$.

**Memory changes nothing structural.** History dependence in $F_a$ makes the single-site process non-Markovian, which
it already is because of $\eta$ and the reaction term. The two sources of memory add inside $R$: for a linearized map
$s(t+1)=\kappa s(t)+\beta h(t)$ the own-history term is a one-round self-coupling $\kappa$ and the reaction term a
two-round self-coupling $\beta\,\Gamma R(t,t-1)$. Nothing "couples" them beyond this sum.

**The solution is one forward pass, not a fixed-point iteration.** Every quantity needed to draw $h_a(t)$ depends only
on rounds $\le t$: $C(t,t'\le t)$ needs $s(\le t)$, and $R(t,t'<t)$ is the response of a state already computed. So
the kernels can be built round by round over a pool of $M$ sampled single-site trajectories, with no outer loop
(`solver_spec.md`, mode (a)). The spec's "iterate until $(\mu,C,R)$ converges" is needed only if one insists on
stationarity.

## 3. Two replicas of the same swarm

Claims 2 and 3 use replicas $\alpha,\beta$ that share $J$ and $h^{\rm priv}$ and differ in the sampling noise. The
cavity argument goes through for both at once. The noises are jointly Gaussian,

$$\langle\eta_a^\alpha(t)\,\eta_a^\beta(t')\rangle=\sum_b\rho_b\,g_{ab}^2\,C_b^{\alpha\beta}(t,t'),\qquad
C_b^{\alpha\beta}(t,t')=\langle s_b^\alpha(t)s_b^\beta(t')\rangle, \tag{9}$$

and each replica keeps its own reaction term with the same $R_b$, because agent $0$ in replica $\alpha$ perturbs only
replica $\alpha$. The cross-replica correlation $q_b(t,t')=C_b^{\alpha\neq\beta}(t,t')$ is the covariance of the part
of $\eta$ that is frozen by the couplings; $C_b-q_b$ is the part that the agents' own noise regenerates each run. If
the replicas use independent couplings, $\langle\eta^\alpha\eta^\beta\rangle=0$ and the only shared input is
$h^{\rm priv}$ and the mean field.

## 4. Assumptions, and how each can fail for LLM agents

1. **Scalar-field sufficiency.** The response to the peers depends on them only through $h_i(t)=\sum_jJ_{ij}s_j(t)$:
   $F_a$ has no other argument that involves peers. Section 4.1 gives the test and section 4.2 what replaces (4)–(8)
   if it fails.
2. **Many weak inputs.** Each term $J_{ij}s_j$ is small, so that the field is Gaussian and a single peer perturbs an
   agent linearly. With $K$ messages read per round, corrections are $O(1/K)$.
3. **Quenched couplings.** $J_{ij}$ is fixed for the run. If who-reads-whom or the trust labels are redrawn every
   round, the reaction term (6) vanishes and $\eta$ becomes white in time (section 5, option C).

### 4.1 Direct test of scalar-field sufficiency

The test is on a single agent, before any swarm is run. Fix the agent's private evidence and own history, and present
inboxes that have the same scalar field but different composition. Under assumption 1 the response distribution must
be the same.

*Design.* Five field levels $h\in\{-2,-1,0,1,2\}\times h_{\rm unit}$, and at each level six compositions:

| composition | what varies at fixed $h$ |
|---|---|
| uniform | all $K$ peers report the same moderate value |
| polarized | a majority at $+1$, a minority at $-1$, counts set to give $h$ |
| few strong | two peers carry the field, the rest report $0$ |
| sign route A | the field comes from trusted peers who agree with the sign of $h$ |
| sign route B | the same field from distrusted peers who hold the opposite view |
| shuffled | "uniform" with the message order permuted (a null) |

Cross this with two private-evidence levels and two own-history states (the agent's last answer was $+$ or $-$), 25
samples per cell: $5\times6\times2\times2\times25=3000$ single-agent calls, of order a few dollars on Haiku 5.5 (the
exact cost must come from `response.usage`).

*Analysis.* Let $\bar s(h,c)$ be the mean response in composition $c$. Report

$$\eta^2_{\rm comp}=\frac{\mathrm{Var}_c\big[\bar s(h,c)\big]\ \text{averaged over }h}{\mathrm{Var}_h\big[\bar s(h,\cdot)\big]},$$

with the shuffled-vs-uniform contrast as the noise floor. Scalar sufficiency holds for DMFT purposes if
$\eta^2_{\rm comp}<0.05$ and no single contrast at fixed $h$ exceeds 20% of the range of $\bar s$ over the five field
levels. With 100 samples per $(h,c)$ after pooling evidence and history cells, the standard error of $\bar s$ is at
most 0.05 on the $[-1,1]$ scale, so a contrast of 0.15 is resolved at three standard errors.

*What a failure looks like, and what to do.* If "sign route A" and "sign route B" differ, the agree and disagree
channels have different gains, which is what Physics of Agents found (below), and the two-channel theory of section
4.2 applies. If "polarized" differs from "uniform", the response depends on the dispersion of opinions and a third
channel $\sum_j|J_{ij}|\,s_j^2$ is needed. If "few strong" differs, the response is not additive over peers and no
Gaussian-field theory with a small number of channels will be quantitative at this $K$.

*Zero-cost version.* The Physics of Agents release records, for every transition, the inbox composition and five
resampled votes. A logistic regression of the vote on the agree-count and disagree-count separately, with quadratic
terms, tests additivity and saturation in counts for four small models without spending anything (see `README.md`).

### 4.2 What changes when the field is delivered as text

**The field becomes an influence function.** An agent that reads $K$ messages sees the empirical measure
$\nu_i=\frac1K\sum_{j\in\partial i}\delta_{(w_{ij},\,{\rm msg}_j)}$ of (trust label, message) pairs. If its response is a
smooth functional of $\nu_i$, expanding around the population measure $\bar\nu$ gives
$F[\nu_i]\approx F[\bar\nu]+\frac1K\sum_j\varphi(w_{ij},s_j)$, where $\varphi=\delta F/\delta\nu$ is the agent's
influence function. Two consequences follow. First, the scalar field exists but is model-specific:
$h_i=\sum_j\varphi(w_{ij},s_j)$, and it need not be the product $w_{ij}s_j$. Second, the fluctuating part is
$O(K^{-1/2})$: an LLM that averages what it reads, as opposed to summing it, has

$$g\;\simeq\;\frac{\beta_{\rm resp}\,\sqrt{\langle w^2\rangle}}{\sqrt K}, \tag{10}$$

where $\beta_{\rm resp}$ is the slope of its response to the normalized field $\frac1K\sum_jw_{ij}s_j$. Disorder
effects of order one ($g\sim1$) therefore need either few messages or a steep response,
$\beta_{\rm resp}\sim\sqrt K$. With $K=8$ and $\pm1$ labels, $g=0.35\,\beta_{\rm resp}$. This is the practical tension
of the whole program: the Gaussian-field assumption wants large $K$, and visible reciprocity effects want small $K$.

**Several channels.** If the response depends on a few linear statistics
$h^{(k)}_i(t)=\sum_jJ^{(k)}_{ij}\,\psi_k(s_j(t))$, $k=1,\dots,K_{\rm ch}$, the same cavity argument gives jointly Gaussian
fields with

$$\langle\eta^{(k)}_a(t)\eta^{(l)}_a(t')\rangle=\sum_b\rho_b\,G^{kl}_{ab}\,\big\langle\psi_k(s_b(t))\,\psi_l(s_b(t'))\big\rangle,
\qquad G^{kl}_{ab}=N\,\mathrm{Cov}\big(J^{(k)}_{ij},J^{(l)}_{ij}\big), \tag{11}$$

$$\text{reaction on channel }k:\quad\sum_b\rho_b\sum_l\mathcal T^{kl}_{ab}\sum_{t'<t}R^{kl}_b(t,t')\,\psi_l(s_a(t')),\qquad
\mathcal T^{kl}_{ab}=N\,\mathrm{Cov}\big(J^{(k)}_{ij},J^{(l)}_{ji}\big), \tag{12}$$

with $R^{kl}_b(t,t')=\langle\partial\psi_k(s_b(t))/\partial h^{(l)}_b(t')\rangle$. Equations (4)–(8) are the case
$K_{\rm ch}=1$, $\psi(s)=s$. (Derived by the same steps as section 2; not separately simulated.)

Physics of Agents is a two-channel example: the agree inbox and the disagree inbox enter with different fitted gains.
From `third_party/physics-of-agents/res/couplings.json`, the effective coupling of an agree tie is $b_+=1.0$ to $3.0$
across the eight model-by-task fits, and that of a disagree tie is $-b_-$ with $b_-/b_+$ between $-0.59$ and $+0.15$.
In six of the eight fits $b_-<0$: a message from a "source you tend to disagree with" still pulls the agent toward the
message. Their "truth asymmetry" (correct neighbours pull harder) is a state-dependent coupling; for $\pm1$ opinions
it is equivalent to a rescaled coupling plus a quenched field $\propto\tau\sum_jJ_{ij}$, with $\tau$ the true answer, and
fits in (11)–(12) with $\psi(s)=(s+\tau)/2$.

**Annealed components.** The wording of a message varies from round to round at fixed $s_j$. That variation is
redrawn every round, so it adds to the white noise $\xi$ and contributes nothing to the reaction term. Only the part of
the influence that is fixed for the pair $(i,j)$ across rounds counts toward $g$ and $\gamma$.

**Hidden state.** A message carries arguments, not only a scalar. If the receiver responds to argument content that
persists across rounds (a novel argument repeated by its author), the sender's effective state is higher-dimensional
than $s_j$. Novelty discounting ("I have heard this already") is a dependence on the receiver's own history of
inputs, which (7) does not allow: $F_a$ may depend on the agent's past outputs but the field enters only at round
$t$. If the sufficiency test passes with fresh inboxes but swarm predictions fail, this is the first suspect, and the
fix is to let $F_a$ depend on $\{h_a(\tau)\}_{\tau\le t}$, which leaves section 2 intact and changes only the
single-site map.

## 5. Realizing $J_{ij}$ with controllable $g$ and $\gamma$

A broadcast board where everyone reads a shuffled list of all posts has no pair-specific coupling: $g=0$, the theory
reduces to Curie–Weiss with random fields, and $\gamma$ is undefined. To see anything in Claims 2–4 that is not
already in De Marzo et al. (2026), the pair-specific part must be built in. Three options:

**A. Displayed trust weights.** Each message is tagged with a number $w_{ij}\in[-1,1]$. This gives the finest
control: scale the weights to set $g$, shift them to set $\bar J$, correlate $w_{ij}$ with $w_{ji}$ to set $\gamma$.
The cost is that the agent's use of the number, $J_{ij}\propto f(w_{ij})$, is unknown and unlikely to be linear
(thresholding near zero, saturation). The realized reciprocity $\mathrm{Corr}(f(w_{ij}),f(w_{ji}))$ is then smaller
in magnitude than the nominal one, and both $g$ and $\gamma$ have to be calibrated for every model.

**B. Signed inboxes on a directed graph** (Physics of Agents' mechanism with direction added;
`envs/graphs.sparse_signed_couplings`). Agent $i$ reads $K$ peers, each in an "agree" or a "disagree" inbox, fixed for
the run. With channel gains $b_\pm$ the effective coupling is $a_{ij}\in\{b_+,-b_-,0\}$ and, for sign fractions
$p_\pm$, edge reciprocity $r=P(j\text{ reads }i\mid i\text{ reads }j)$, and reciprocated ties that carry equal signs,

$$\bar J=K\,(p_+b_+-p_-b_-),\qquad g^2=K\,(p_+b_+^2+p_-b_-^2),\qquad \gamma=r\quad(\text{up to }O(K/N)). \tag{13}$$

If reciprocated ties carry opposite signs, $\gamma=-r\cdot b_+b_-/(p_+b_+^2+p_-b_-^2)$, which reaches $-r$ only when
$b_+=b_-$. Reciprocity is exact and needs no calibration when ties are concordant. Negative $\gamma$ and a zero mean
both need $b_-\approx b_+>0$, an agent that moves away from distrusted sources as strongly as it moves toward trusted
ones. The Physics of Agents fits say the small models there do not do this, so $b_\pm$ must be measured for the
Claude models first; if $b_-\le0$, option B gives $\bar J/g$ of order $\sqrt K$, the mean dominates, and
only $\gamma\ge0$ is reachable.

**C. Sampled reading.** Agent $i$ reads $j$'s message with probability $p_{ij}$. If the draw is repeated every round,
the couplings are annealed: $J_{0j}(t)$ and $J_{j0}(t-1)$ are independent, the reaction term is zero for every
reciprocity, and the pair-specific noise is white in time. This is mean-field theory with extra noise; it cannot test
anything involving $\gamma$. If the draw is made once, it is option B without signs: $g/\bar J=K^{-1/2}$ and
$\gamma\ge0$ only.

**Recommendation: B, with three constraints.** (i) Fixed in-degree $K$ for every agent (`envs/graphs.sparse_signed_couplings`
draws Poisson in-degrees, and at $K\approx8$ a 35% spread in $K_i$ is a 35% spread in $g_i^2$, a quenched heterogeneity
that the block model (2) does not have). With a fixed in-degree and a nonzero mean coupling the noise covariance (5)
becomes $g^2C-(\bar J^2/K)\,m\,m$, a small correction derived in `claim5.md`. (ii) Vary $\gamma$ through the edge reciprocity $r$ at fixed sign agreement
$\pm1$, and if possible give every agent exactly $rK$ reciprocated ties. The per-agent reaction coefficient is
$\Gamma_i\propto\frac1K\sum_{j\,{\rm recip.}}\epsilon_{ij}\epsilon_{ji}$, whose spread across agents is
$\sqrt{r(1-r)/K}$ for binomial reciprocation (0.18 at $r=0.5$, $K=8$, against a mean of 0.5) and zero for an exact
count. (iii) $K$ between 8 and 12: below 8 the field is visibly non-Gaussian, above 12 equation (10) makes $g$ too
small unless the agents are very conformist. Option B keeps the couplings quenched, signed and pairwise, uses a label
that LLMs demonstrably act on, and makes $\gamma$ a property of the graph, not of the model. Its weakness is that
$g$ is set by the model's gains $b_\pm$ and by $K$, not freely by the experimenter; option A is the fallback if a wider
range of $g$ is needed at fixed $K$.

Whatever the option, log the realized block statistics of each graph (`envs/graphs.spectral_log`) and estimate the
effective couplings from single-agent calibration, not from the nominal design.

## 6. Numerical check

`theory/checks/check_claim1_dmft.py` (helpers in `theory/checks/toy.py`) simulates two populations with all the
structure of (1)–(2): $\rho=(0.6,0.4)$, $\bar J=\begin{psmallmatrix}0.5&-0.3\\0.8&0.2\end{psmallmatrix}$,
$g=\begin{psmallmatrix}1.0&0.6\\0.8&1.2\end{psmallmatrix}$, $\gamma=\begin{psmallmatrix}0.6&0.5\\0.5&-0.4\end{psmallmatrix}$,
private fields $\mu=(0.3,-0.2)$, $\sigma=(0.5,0.3)$, map $s(t+1)=\tanh(\beta_ah+h^{\rm priv}+\kappa_au)$ with an
exponentially weighted memory $u$ of the agent's own past states in population 0 ($\kappa=(0.4,0)$), $T=10$. It
compares $N=4000$ simulations (40 disorder samples) with the single-site process (4)–(8) solved by Monte Carlo over
$4\times10^5$ trajectories, in four versions. Maximum absolute error over $t\ge1$ and both populations (outputs in
`out_claim1_det.txt`, `out_claim1_stoch.txt`):

| single-site process | $m(t)$ | $C(t,t)$ | $C(t,t-2)$ |
|---|---|---|---|
| eqs. (4)–(8) | 0.004 | 0.003 | 0.002 |
| spec noise: connected $\langle\delta s\,\delta s\rangle$ in (5) | 0.034 | 0.028 | 0.026 |
| spec time index: $R(t,t'+1)s(t')$ in (4) | 0.039 | 0.062 | 0.048 |
| no reaction term | 0.050 | 0.159 | 0.154 |
| simulation standard error | 0.003 | 0.002 | 0.002 |

The correct equations agree with the simulation to within about 1.5 standard errors; a separate run at $N=500$ and
$2000$ showed no finite-size trend beyond that. Each of the spec's two errors alone moves the prediction by 10 to 40
standard errors. The stochastic variant ($s=\pm1$ with $P(+1)=(1+\tanh x)/2$, response computed by the score-function
estimator of `solver_spec.md`) gives errors of 0.007 in $m$ and 0.005 in $C(t,t-2)$ for the correct equations, against
0.027 and 0.163 for the spec's time index.

The same script measures the response in the simulation by kicking every agent's field at $t'=3$ with a random sign
and correlating the result with the sign:

| | $t'+1$ | $t'+2$ | $t'+3$ | $t'+4$ | $t'+5$ |
|---|---|---|---|---|---|
| population 1 (no memory), simulation | 0.713 | −0.002 | −0.050 | 0.005 | 0.019 |
| population 1, eq. (8) | 0.712 | 0 | −0.051 | 0.005 | 0.019 |
| population 0 (memory), simulation | 0.500 | 0.068 | 0.129 | 0.060 | 0.066 |
| population 0, eq. (8) | 0.502 | 0.065 | 0.129 | 0.059 | 0.066 |

The memoryless population has no response at lag 2 and its first echo at lag 3, which is the two-round echo of
equation (4). The echo is negative there because $\gamma_{11}=-0.4$.

**Not checked numerically:** the multi-channel equations (11)–(12), and anything about LLM agents. Sections 4 and 5
are assumptions and design arguments to be tested by the experiments.
