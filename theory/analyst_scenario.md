# Analyst scenario: frozen predictions for mutual, one-way and antagonistic trust at K = 16

Frozen 2026-11-25, before any analyst-scenario data exist. Numbers: `theory/checks/analyst_predictions.json`.
Code: `theory/checks/analyst_lib.py` (law, graph simulation, single-site solver) and `theory/checks/check_analyst.py`
(parts A to G, outputs `out_analyst_A.txt` … `out_analyst_G.txt`). The calibration plugs numbers into the mapping of
section 1; nothing else may change after the first swarm run.

**Verdict.**

1. *Committed BUY/SELL calls change the stability picture of `claim2.md`.* A call is a binary variable sampled from
   the belief, so every reader's field carries sampling noise of fixed variance $g^2$, and the mean slope of the
   response falls as the gain rises. In the one-way world this keeps the stability parameter below $2/\pi=0.64$ at any
   gain: there is no boundary. In the mutual world the reaction term returns each analyst's own call coherently, the
   static susceptibility of the mean call stays at 1, and $\Lambda=g^2$: the boundary is the spin-glass transition of
   the Sherrington–Kirkpatrick model. The antagonistic world is the mutual world seen through a sign pattern of period
   4, exactly so when the two gains are equal and there are no standing notes. So the factor between worlds is not
   $(1+|\gamma|)$: the reciprocal worlds have a boundary and the one-way world has none (sections 2.3, 3).
2. *Gain needed for $\Lambda=1$ at $K=16$* (root-mean-square gain $\bar a=\sqrt{(a_+^2+a_-^2)/2}$, in half log-odds per
   unit normalized field): mutual world $\bar a=4.23$ with no news and no standing notes (exact on a tree; $4.00$ at
   $K=\infty$), $5.0$ with news of logit s.d. 0.5, and about $5.9$ in the reference design (news 0.5, notes 0.3);
   antagonistic world about $5.1$ in the reference design; one-way world never. Memory moves the mutual boundary
   earlier and turns it static, and moves the antagonistic one away (section 3.4).
3. *The reciprocity signatures are large and need no boundary.* The echo function $\rho_2$ (how much an analyst's own
   call changes its belief two rounds later) is $+0.12$, $0$, $-0.12$ in the mutual, one-way and antagonistic worlds
   at a quiet gain ($\bar a=2$, $\Lambda_{\rm mutual}=0.16$) and $+0.26$, $0$, $-0.26$ near the boundary ($\bar a=4$,
   $\Lambda=0.57$), against a run-to-run s.d. of 0.01 at $N=256$, $T=20$. The spectrum of replica differences peaks
   at periods 2 and $\infty$ (mutual), period 4 (antagonistic) or nowhere (one-way). A kick twin returns an echo of
   $+0.25$, exactly 0, $-0.26$ per unit kick near the boundary, and 7 twins give 3 standard errors (sections 4, 5).
4. *The Gaussian closure holds at $K=16$.* The single-site solver ($K=\infty$) reproduces simulations of the same law
   on actual graphs to 0.002 in $\langle s^2\rangle$ and to 1 to 4% in the echo, at $N=256$, 1,024 and 4,096 alike; the
   offset in the echo goes away at $K=64$. Finite-$N$ offsets are below the errors already at $N=256$, so going to $N=1{,}024$ halves the
   errors and changes nothing else (section 2.4).
5. *Mean-field (Curie–Weiss) theory and the reciprocity-blind closure both predict three identical worlds.* They
   fail on one-time moments too: under a common BUY bias the mean belief is 0.108, 0.076, 0.047 in the three worlds
   (Curie–Weiss: one value for all), and in the tilted cell $\langle s^2\rangle$ is 0.50, 0.42, 0.47 against the
   Curie–Weiss 0.24 (sections 2.2, 2.4, 3.5).

Everything in sections 2 to 6 is derived for the law (A1)–(A3) and checked by simulation of that law. Whether Qwen
obeys the law is what the calibration and the swarm decide; section 8 lists what is not verified.

## 1. Law, parameters, and what the calibration must return

$\sigma_i(t)=\pm1$ is the call (BUY/SELL) that colleagues read, $s_i(t)=2P_i({\rm BUY})-1$ the belief read from the
decision token's logprobs, and $x_i$ the half log-odds, so that $s=\tanh x$ is a definition, not a model. The model
is that $x$ is linear in the inbox:

$$x_i(t)=\sum_jW_{ij}\,\sigma_j(t)+a_0h^0_i+\nu\,\xi_i(t+1)+c+\kappa\,\sigma_i(t),\qquad
W_{ij}=\begin{cases}+a_+/K&\text{ally}\\-a_-/K&\text{faded}\end{cases} \tag{A1}$$

$$s_i(t+1)=\tanh x_i(t) \tag{A2}$$

$$\sigma_i(t+1)=\pm1\ \text{ with probability }\ \tfrac12\big(1\pm s_i(t+1)\big). \tag{A3}$$

$h^0_i\sim\mathcal N(0,1)$ is the standing note and $\xi_i(t)\sim\mathcal N(0,1)$ the fresh news, both scaled so that
$a_0$ and $\nu$ are logit standard deviations; $c$ is a common bias toward BUY; $\kappa$ is the weight of the
analyst's own last call. Round 0 has no inbox. The graph is `envs/graphs.fixed_indegree_signed_couplings` at
$p_-=\tfrac12$: every analyst reads $K=16$ colleagues, all ties reciprocated and concordant (mutual), none
reciprocated (one-way), or all reciprocated and discordant (antagonistic).

| parameter | meaning | how the calibration measures it |
|---|---|---|
| $a_+$, $a_-$ | half log-odds per unit normalized field $f=\frac1K\sum\sigma_j$ of allies, of faded colleagues ($a_->0$: leans against them) | regression of $\operatorname{atanh}(2P-1)$ on $f_A$, $f_D$ with real notes. If the calibration reports the slope of $P({\rm BUY})$ at $P=\tfrac12$, $a=2\,dP/df$ |
| steepness | the shape of $P$ against the field | is $a_\pm$ itself when the log-odds are linear in the counts; the calibration must test that linearity (section 8) |
| $\nu$ | news gain times the s.d. of the news variable | same regression |
| $a_0$ | note gain times the s.d. of the note variable | same regression |
| $\kappa$ | memory | from the bare kernel: $R^0(2)/R^0(1)=\kappa\langle d\rangle$, $d=1-s^2$ |
| $c$ | BUY bias | intercept |

A full ally bloc (8 colleagues, all BUY) moves $x$ by $a_+/2$. The effective couplings are (`claim1.md` eq. (13) with
gains)

$$\bar J=\frac{a_+-a_-}2,\qquad g^2=\frac{a_+^2+a_-^2}{2K},\qquad
\Gamma=N\langle W_{ij}W_{ji}\rangle=\begin{cases}g^2&\text{mutual}\\0&\text{one-way}\\-a_+a_-/K&\text{antagonistic}\end{cases}
\qquad\gamma_{\rm eff}=\Gamma/g^2. \tag{A4}$$

The mutual world has $\gamma_{\rm eff}=1$ for any gains, because concordant ties make $W$ symmetric. The antagonistic
world has $\gamma_{\rm eff}=-2a_+a_-/(a_+^2+a_-^2)$: $-1$ for equal gains, $-0.96$ for $a_-/a_+=0.75$, $-0.8$ for
0.5, and positive if faded colleagues still attract ($a_-<0$), as they did for Haiku in the label wording. Mean trust
$\bar J$ is the same in the three worlds by construction.

**The running swarm measures the same parameters.** Beliefs carry no sampling noise, so $x_i(t)=\operatorname{atanh}
s_i(t+1)$ is observed for every analyst and round, and regressing it on $f_{A,i}(t)$, $f_{D,i}(t)$, $\xi_i(t+1)$,
$h^0_i$, $\sigma_i(t)$ returns $a_+,a_-,\nu,a_0,\kappa,c$ in the swarm with a residual that measures everything (A1)
leaves out. This is the check that the calibration transfers (the Haiku template calibration did not). It is a
planned analysis; it was not exercised here beyond being the definition of the simulated law.

**Reference design** for the tables: $K=16$, $\nu=0.5$, $a_0=0.3$, $c=0$, $\kappa=0$, sampled calls. Intensity grid
$(a_+,a_-)$: quiet (2,2), mid (3,3), near (4,4), close (5,5), tilted (4,3).

## 2. Single-site process

### 2.1 Equations

`claim1.md` eqs. (4)–(8) with the call as the transmitted state. For a representative analyst,

$$x(t)=\bar J\,m(t)+\eta(t)+\Gamma\sum_{t'<t}R(t,t')\,\sigma(t')+a_0h^0+\nu\,\xi(t+1)+c+\kappa\,\sigma(t), \tag{A5}$$

$$\langle\eta(t)\eta(t')\rangle=g^2\,C(t,t')-\frac{\bar J^2}{K}m(t)m(t'),\qquad C(t,t')=\langle\sigma(t)\sigma(t')\rangle,\quad C(t,t)=1, \tag{A6}$$

followed by (A2)–(A3). The order parameters are $m(t)=\langle\sigma(t)\rangle=\langle s(t)\rangle$, $C$, and the
response of the mean call to a field, $R(t,t')=\partial\langle\sigma(t)\rangle/\partial\theta(t')$. The second term
of (A6) is the fixed in-degree correction of `claim5.md`; it is below 1% of the noise variance in every case here
and the solver omits it. Three things differ from the coin game.

- *The noise variance is fixed.* $C(t,t)=1$, so $\eta$ has variance $g^2$ whatever the beliefs are. Beliefs enter
  through $q(t)=\langle s(t)^2\rangle$, which is measured but does not feed back into the noise.
- *The response kernel is a correlation of observables.* The call's innovation $\zeta_i(t)=\sigma_i(t)-s_i(t)$ is
  known exactly and has zero mean given the past. The score-function identity for (A3) gives

  $$R(t,t')=\big\langle s(t)\,\zeta(t'+1)\big\rangle\quad(t>t'+1),\qquad R(t,t-1)=\langle1-s(t)^2\rangle, \tag{A7}$$

  in any world and at any coupling. The swarm's own response kernel, the object inside the reaction term, is
  therefore read off base runs without an intervention. The **echo function** used below is its normalized form,

  $$\rho_\tau=\frac{\sum_{i,t}\zeta_i(t)\,s_i(t+\tau)}{\sum_{i,t}\zeta_i(t)^2}=\frac{R(\tau+1)}{1-q}. \tag{A8}$$

- *The private news is a white field inside the map.* It lowers the slope of the mean call,
  $\phi(u)=\langle\tanh(u+\nu z)\rangle_z$ in place of $\tanh u$, and it is shared by replicas of the same swarm, so
  it adds to the replica overlap.

To leading order in $g^2$ the first echo is $R(3)=d\,\Gamma\,d\,d$ with $d=1-q$, so

$$\rho_2=\Gamma\,(1-q)^2+O(g^4),\qquad\rho_1=\kappa\,(1-q)+O(g^2\kappa). \tag{A9}$$

At the quiet gain this gives $\pm0.115$ against $\pm0.119$ from the solver; near the boundary it is 20% low ($0.21$
against $0.26$) and the solver value is the prediction.

The solver is `theory/dmft_solver.forward_pass` with the map `analyst_lib.AnalystMap` (calls sampled, news drawn
inside the map and shared by the thermal copies of a site, response by (A7)); `analyst_lib.dmft(Agent(...), world)`
returns $m$, $q$, $C$, $R$, $\rho_\tau$, the replica overlap and $\Lambda(\omega)$.

### 2.2 What the two null theories say

*Curie–Weiss* (De Marzo et al. 2026; `docs/literature.md`) keeps only the first term of (A5): $x=\bar Jm+a_0h^0+\nu
\xi+c$. It has no disorder field, so $q=\langle\tanh^2\rangle$ of the private inputs alone (0.24 in the tilted cell
below, against 0.42 to 0.50 measured in simulation), no echo, white fluctuations, and one answer for the three worlds.

*The reciprocity-blind closure* keeps $\eta$ and drops the reaction term. It is exact for the one-way world and
predicts the one-way numbers for all three.

The test of the cavity theory is the difference between worlds; the test of the disorder field is the difference
between the one-way world and Curie–Weiss.

### 2.3 Exact structure of the three worlds

**One-way.** No reciprocated tie, so $\Gamma=0$. Without memory the calls are white in time apart from the frozen
part, $\rho_\tau=0$ for all $\tau$, and the bare slope is the dressed one.

**Mutual.** $W$ is symmetric, and for the logistic law (A2)–(A3) without news the synchronous dynamics has a known
stationary distribution, $P(\boldsymbol\sigma)\propto\prod_i2\cosh x_i(\boldsymbol\sigma)$ (Peretto 1984; cited from
memory): two successive rounds are the two layers of a bipartite Ising model at equilibrium. Three consequences,
each checked in part B1:

$$\langle\sigma_i(t)\,\sigma_i(t+2)\rangle=\langle s_i^2\rangle, \tag{A10}$$

on any graph and at any $N$ (rounds $t$ and $t+2$ are conditionally independent given round $t+1$, with the same
conditional mean);

$$\chi\equiv\sum_\tau R(\tau)=1-\langle m_i^2\rangle+c_1,\qquad c_1=\langle\sigma_i(t)\sigma_i(t+1)\rangle_{\rm conn}, \tag{A11}$$

by the fluctuation–dissipation relation of the bipartite model, where $c_1=0$ without memory and $m_i$ is the frozen
mean; and odd-lag correlations vanish without memory, so the swarm at even rounds and the swarm at odd rounds are two
independent copies of the same equilibrium system. That last statement is the period-2 flicker.

**Antagonistic, equal gains.** For antisymmetric $W$ and an odd response, define
$\tau_i(t)=\epsilon(t)\sigma_i(t)$ with $\epsilon=(+,+,-,-,+,+,\dots)$. Then $\tau(2k+1)$ is drawn from
$W\tau(2k)$ and $\tau(2k+2)$ from $-W\tau(2k+1)=W^{\!\top}\tau(2k+1)$, which is the bipartite equilibrium dynamics
again. So every correlation of the antagonistic world is the corresponding one of a mutual-type world times
$\epsilon(t)\epsilon(t')$: lag 2 inverted, lag 4 restored, odd lags zero. White symmetric news leaves the map intact.
Standing notes, a bias, memory and unequal gains break it, because a static field becomes a period-4 field after the
transformation. With $a_0=\nu=0$ at gains (4,4) the solver gives $\Lambda=0.83$, 0.82 at $\omega=0,\pi$ in the
mutual world and 0.81 at $\pi/2$ in the antagonistic one (part B3). In the reference design, which has standing
notes, the echoes still mirror ($\rho_2=+0.263$ against $-0.264$, part C) while $q$ differs (0.544 against 0.526).

### 2.4 The closure at K = 16

Part A compares the solver with simulations of (A1)–(A3) on actual graphs. Late-round averages, 4 replicas per
graph, s.e. over graphs in units of the last digit (8, 4 and 2 graphs at the three sizes, so the s.e. themselves are
rough; the run-to-run s.d. of $\rho_2$ at $N=1{,}024$ is 0.004). Near cell (4,4):

| | $q=\langle s^2\rangle$ | replica overlap | $\rho_2$ | $\rho_4$ |
|---|---|---|---|---|
| mutual, solver ($K=\infty$) | 0.5424 | 0.184 | +0.2566 | +0.1256 |
| mutual, $N=256$ | 0.5417 (24) | 0.186 (8) | +0.2604 (37) | +0.1291 (42) |
| mutual, $N=1{,}024$ | 0.5425 (15) | 0.190 (8) | +0.2605 (29) | +0.1291 (23) |
| mutual, $N=4{,}096$ | 0.5428 (2) | 0.190 (1) | +0.2637 (9) | +0.1296 (3) |
| mutual, $N=4{,}096$, $K=64$, same $g$ | 0.5405 (15) | 0.186 (11) | +0.2560 (2) | +0.1260 (19) |
| one-way, solver | 0.4476 | 0.150 | 0 | 0 |
| one-way, $N=256$ / 1,024 / 4,096 | 0.4494 (10) / 0.4492 (3) / 0.4493 (1) | 0.148 / 0.149 / 0.150 | +0.001 / +0.004 / +0.002 | 0.00 |
| antagonistic, solver | 0.5241 | 0.115 | −0.2580 | +0.1311 |
| antagonistic, $N=256$ / 1,024 / 4,096 | 0.5241 (14) / 0.5236 (5) / 0.5234 (2) | 0.121 / 0.114 / 0.121 | −0.2671 (19) / −0.2674 (13) / −0.2648 (1) | +0.136 |

The quiet cell agrees to 0.001 in $q$ and 0.003 in $\rho_2$ in all three worlds. In the tilted cell with a BUY bias
$c=0.15$ the mean belief is 0.180 (solver) against 0.184 (6), 0.164 (9), 0.165 (4) on graphs in the mutual world,
0.126 against 0.122 to 0.123 one-way, and 0.089 against 0.079 to 0.092 antagonistic; the transient $m(t)$, $q(t)$ over
rounds 0 to 8 agrees within the simulation s.e. (0.004 in $q$, 0.01 in $m$). The mean in the mutual world is the
weakest agreement: it sits close to herding, where sampling error is amplified.

Three statements follow.

- *$N$.* No statistic moves between $N=256$ and 4,096 by more than its error (0.002 in $q$, 0.004 in $\rho_2$). Going
  from 256 to 1,024 should halve the errors and leave the means where they are.
- *$K$.* The echo on $K=16$ graphs is 1 to 4% larger than the $K=\infty$ solver value, and the offset is gone at
  $K=64$. $q$ is unaffected. The reference for the experiment is the graph simulation; the solver is the theory with
  a known offset.
- *Shape of the field.* The excess kurtosis of $x$ is $-0.06$ in the one-way world, the binomial value
  $-(2/K)(g^2/\mathrm{Var}\,x)^2=-0.07$. In the reciprocal worlds it is $-0.37$ and $-0.31$, of which the solver itself
  gives $-0.28$ and $-0.19$: the reaction term makes the total field a mixture of two Gaussians displaced by
  $\pm\Gamma R(1)$ according to the analyst's own call two rounds earlier. A Gaussianity test must therefore be made
  on $x_i(t)$ conditional on $\sigma_i(t-1)$, or in the one-way world.

## 3. Stability boundary

### 3.1 Criterion

`claim2.md` eq. (8) holds for sampled calls with responses of mean calls:

$$\Lambda=\max_\omega\,g^2\big\langle|R_i(\omega)|^2\big\rangle,\qquad R_i(\omega)=\sum_\tau R_i(\tau)e^{-i\omega\tau}, \tag{A12}$$

with $R_i$ the dressed kernel of analyst $i$, averaged over its thermal noise at fixed notes and news. $\Lambda\to1$
is where the summed squared response of the swarm to one call diverges. The observable that carries it is the
spin-glass susceptibility of calls (section 4.4), which on a tree equals $1/(1-\Lambda)$.

What differs from `claim2.md` is the slope. There the slope $d$ was a property of the fixed point and the boundary
was $d\,g\,(1+|\gamma|)=1$. Here each field carries call noise of variance $g^2$, the thermal mean slope is
$d=\langle{\rm sech}^2x\rangle\approx0.80/g$ at large $g$, and the product $d\,g$ cannot grow.

### 3.2 Per world

**One-way.** $\Lambda=g^2\langle\bar d_i^{\,2}\rangle$ with $\bar d_i$ the thermal mean slope; without notes
$\Lambda=g^2\langle{\rm sech}^2(\sqrt{g^2+\nu^2}\,z)\rangle^2$, which is 0.17, 0.37, 0.53, 0.61, 0.635 at
$g=0.5,1,2,4,16$ and tends to $2/\pi$ (part B4). No gain reaches the boundary. On graphs the amplification rises from
1.45 to 2.34 between $\bar a=4$ and 12 and stays below $1/(1-2/\pi)=2.75$ (part G).

**Mutual.** By (A11), $\Lambda(0)=g^2\langle\chi_i^2\rangle$, and with no notes, news or memory $\chi=1$:

$$\Lambda_{\rm mutual}=g^2=\frac{a_+^2+a_-^2}{2K}\quad(K=\infty),\qquad
\Lambda_{\rm mutual}=\frac{K-1}2\Big[\tanh^2\frac{a_+}K+\tanh^2\frac{a_-}K\Big]\quad(\text{tree of degree }K). \tag{A13}$$

The solver gives $\chi=0.993\pm0.004$ and $0.978\pm0.005$ at $g=0.5$, 0.75 (sum over 25 lags; the remainder is the
tail of the kernel). The tree form is checked against a long simulation at $N=2{,}048$ (part B2): summed squared
correlations of calls at equal times 1.061, 1.387, 2.237 against 1.061, 1.385, 2.249 predicted, and at lag 1 0.262,
0.750, 1.689 against 0.261, 0.748, 1.701, for $a=2,3,3.6$. The boundary is at $\bar a=4.23$ for equal gains (4.79 in
$a_+$ for $a_-=0.75a_+$, 5.39 for $a_-=0.5a_+$).

**News** lowers the slope of the mean call. The solver's $\chi$ is 0.94, 0.84, 0.73, 0.62 at $\nu=0.25,0.5,0.75,1$,
within 2% of $(1+\pi\nu^2/2)^{-1/2}$, so

$$\bar a_c(\nu)\simeq4.23\,\sqrt{1+\pi\nu^2/2}\qquad(K=16,\ \text{no notes}): \quad4.4,\ 5.0,\ 5.8,\ 6.8. \tag{A14}$$

Long simulations place it at $5.03$ for $\nu=0.5$ (part G; predicted 4.99) and at $4.40$ for $\nu=0$ (4.23; the
linear fit of $1/A$ overshoots by a few percent).

**Standing notes** are static random fields, and the mutual boundary is its de Almeida–Thouless line
$g^2\langle(1-m_i^2)^2\rangle=1$: frozen means, themselves amplified near the boundary, lower the susceptibility. With
$a_0=0.3$, $\nu=0.5$ the simulated boundary is at $\bar a\approx5.9$, against 5.0 without notes. Notes of 0.15 with
news of 0.25 give 4.9.

**Antagonistic.** Equal gains and no notes: the same boundary as the mutual world, at $\omega=\pi/2$. Standing notes
are off resonance for the period-4 mode and cost less: the simulated boundary of the reference design is at
$\bar a\approx5.1$, below the mutual one. Unequal gains reduce $|\gamma_{\rm eff}|$ and $\Lambda(\pi/2)$: at $g^2=1$,
$\nu=0.5$ the solver gives 0.71, 0.68, 0.59, 0.44, 0.33 for $a_-/a_+=1,0.75,0.5,0.25,0$ (mean coupling set to zero).
Whether $\Lambda$ reaches 1 at all for $|\gamma_{\rm eff}|<1$ is not settled here; the literature on asymmetric spin
glasses (Crisanti & Sompolinsky 1987, 1988, as cited in `claim2.md`) says a noisy system does not freeze for
$|\gamma|<1$, and the solver's truncated values stay below 0.92 up to
$g^2=4$.

### 3.3 The numbers in the reference design

Solver $\Lambda$ (3,000 sites × 64 copies, 40 lags, Monte Carlo error 0.02) and the same quantity read from the
graph simulation as $1-1/\mathcal A$ at $N=1{,}024$, $T=40$ (part C) and from 1,500 rounds at $N=2{,}048$ (part G):

| gains $(a_+,a_-)$ | $g^2$ | mutual: solver, $1-1/\mathcal A$, long | one-way: solver, $1-1/\mathcal A$ | antagonistic: solver, $1-1/\mathcal A$, long |
|---|---|---|---|---|
| (2,2) | 0.25 | 0.16, 0.17, – | 0.13, 0.13 | 0.18, 0.16, – |
| (3,3) | 0.56 | 0.37, 0.36, 0.34 | 0.22, 0.21 | 0.40, 0.35, 0.37 |
| (4,4) | 1.00 | 0.60, 0.58, 0.57 | 0.31, 0.28 | 0.69, 0.54, 0.65 |
| (5,5) | 1.56 | 0.79, 0.74, 0.81 | 0.38, 0.38 | 0.86, 0.73, 0.91 |
| (6,6) | 2.25 | 0.87, –, 0.98 | 0.41, 0.43 | 0.89, –, 0.99 |
| (4,3) | 0.78 | 0.49, 0.48, – | 0.27, 0.28 | 0.55, 0.46, – |

The solver's kernel is truncated at 40 lags and its $\Lambda$ saturates below 1; near and beyond the boundary the
long simulation is the reference. In the reciprocal worlds the $K=\infty$ solver value should also be multiplied by
the tree factor of (A13), $(15/16)\,[\tanh(a/16)/(a/16)]^2=0.90$ at $a=4$, before comparison with graphs. The one-way
entry at (6,6) is from the long simulation.

### 3.4 Memory

An own-call weight $\kappa$ keeps $W+\kappa\,\mathbb 1$ symmetric, so the mutual world stays an equilibrium system and
(A11) gives, without news or notes,

$$\Lambda_{\rm mutual}(0)=g^2(1+c_1)^2,\qquad\Lambda_{\rm mutual}(\pi)=g^2(1-c_1)^2. \tag{A15}$$

Solver, $\kappa=0.5$: $\chi=1.53\pm0.02$ against $1+c_1=1.52$ at $a=1.5$, and $1.54\pm0.02$ against 1.57 at $a=2$,
where $\Lambda(0)=0.60$ and $\Lambda(\pi)=0.05$ although $g^2$ is only 0.25. So anchoring brings the mutual boundary
forward by the factor $1+c_1$ in $g$ and makes it static: the flicker mode is suppressed by $(1-c_1)^2/(1+c_1)^2$.
This is the direction of `quiet_regime.md` section 2.2 (anchoring is a low-pass filter). The form differs because the
memory there was a convex combination with the target, which left the static boundary in place, and here it is added
to the log-odds, which raises the static susceptibility.

The period-4 mode loses. In the reference design at gains (4,4) the solver gives $\Lambda_{\rm antag}(\pi/2)=0.69$,
0.30, 0.16 for $\kappa=0,0.5,1$, while $\Lambda_{\rm mutual}(0)$ goes from 0.60 to 0.86 and the one-way static value
from 0.31 to 0.47 and 0.54. With memory the one-way world acquires a static boundary of its own, which is every
analyst repeating its call and nothing collective. $\kappa=0.5$ corresponds to $\rho_1=0.24$ to 0.34, a modest
own-call weight. If the calibration finds more than that, the oscillatory signatures of section 4.2 are not
available and the echo contrast of section 4.1 carries the test.

### 3.5 Herding, the other boundary

The mean obeys $\delta m=\chi\,(\bar J\,\delta m+\delta c)$, so a common bias is amplified to $\chi\,\delta
c/(1-\bar J\chi)$ and the swarm herds when $\bar J\chi=1$. This is the Curie–Weiss instability, with one
difference: $\chi$ is the dressed susceptibility, which depends on the world. Reference design (part B5):

| $g^2$ | $\chi$ mutual | one-way | antagonistic | Curie–Weiss |
|---|---|---|---|---|
| 0.25 | 0.78 | 0.70 | 0.62 | 0.785 |
| 0.56 | 0.78 | 0.62 | 0.50 | 0.785 |
| 1.00 | 0.73 | 0.56 | 0.39 | 0.785 |
| 1.56 | 0.64 | 0.49 | 0.30 | 0.785 |

Curie–Weiss is about right in the mutual world, where the fluctuation–dissipation relation fixes $\chi$, and wrong in
the other two by up to a factor 2.6. On graphs at gains (4.4, 3.6) with a bias $c=0.1$ the mean belief is
0.108 (9), 0.076 (2), 0.047 (2) in the three worlds, against 0.107, 0.072, 0.047 from the solver.

Herding competes with the boundary of section 3.2. In the mutual world the tilt at which herding sets in is
$\bar J_c=1/\chi\approx1.3$ to 1.6, so the glassy boundary comes first only if $(a_+-a_-)/2<1.3$ at the gain where
$\Lambda\to1$. At $\bar a\approx5$ that requires $a_-/a_+\gtrsim0.6$.

## 4. Signatures per world

All numbers: reference design on $K=16$ graphs, $N=1{,}024$, 6 graphs × 4 replicas, $T=40$, rounds 5 to 40 (part C;
part E gives the same statistics at $T=20$ with their run-to-run spread). Replicas share the graph, the notes and
the news, and differ in the sampled calls.

### 4.1 The two-round echo

$\rho_\tau$ of (A8), lags 1 to 4. No replicas are needed.

| gains | mutual | one-way | antagonistic |
|---|---|---|---|
| quiet (2,2) | 0, **+0.123**, 0, +0.029 | 0, **+0.001**, 0, 0 | 0, **−0.121**, 0, +0.029 |
| mid (3,3) | 0, **+0.208**, 0, +0.082 | 0, **−0.001**, 0, 0 | 0, **−0.207**, 0, +0.085 |
| near (4,4) | 0, **+0.263**, 0, +0.130 | 0, **0.000**, 0, −0.001 | 0, **−0.264**, 0, +0.136 |
| close (5,5) | 0, **+0.296**, 0, +0.161 | 0, **0.000**, 0, −0.001 | 0, **−0.295**, 0, +0.167 |
| tilted (4,3) | 0, **+0.241**, 0, +0.109 | 0, **−0.001**, 0, −0.001 | 0, **−0.235**, 0, +0.108 |

The solver gives 0.119, 0.200, 0.255, 0.286, 0.233 for the mutual column (the 3% of section 2.4). The fourth-lag
echo is positive in both reciprocal worlds: the sign pattern is $(+,+)$ against $(-,+)$.

With memory $\kappa=0.5$ the lag-1 echo is nearly the same in the three worlds (0.34 at the quiet gain; 0.27, 0.28,
0.24 near the boundary), and the lag-2 echo becomes
0.198, 0.118, 0.022 at the quiet gain and 0.239, 0.078, −0.170 near the boundary. Two statistics remove the memory:
the contrast with the one-way world, and

$$X_2=\rho_2-\rho_1^2, \tag{A16}$$

which is 0.000 in the one-way world at every gain (an own-call chain gives $\rho_2=\rho_1^2$) and $+0.085$, $-0.093$
(quiet), $+0.167$, $-0.226$ (near) in the mutual and antagonistic worlds.

If the note transmits part of the sender's belief and not only its call (fraction "leak"), the echo shrinks, to
$\pm0.21$ at leak 0.25 and $\pm0.14$ at 0.5 near the boundary, and the lag-2 correlation of replica differences grows
(0.60, 0.67, 0.74), because less call noise is injected. The signs do not change.

### 4.2 Period 2, period 4, or neither

Replica differences of beliefs, $\delta_i(t)=s_i^\alpha(t)-s_i^\beta(t)$, remove the notes and the news. $c_\tau$ is
their lag correlation and $S(\omega)=1+2\sum_{\tau\le4}c_\tau\cos\omega\tau$ a four-lag spectrum, quoted at
$\omega=0,\pi/2,\pi$ (periods $\infty$, 4, 2).

| gains | mutual: $c_2$, $c_4$; $S$ | one-way: $c_2$; $S$ | antagonistic: $c_2$, $c_4$; $S$ |
|---|---|---|---|
| quiet (2,2) | +0.24, +0.08; 1.61, 0.68, 1.67 | 0.00; 0.99, 0.99, 1.00 | −0.24, +0.08; 0.66, 1.64, 0.68 |
| mid (3,3) | +0.44, +0.24; 2.33, 0.59, 2.38 | 0.00; 1.00, 1.00, 0.99 | −0.45, +0.25; 0.60, 2.40, 0.60 |
| near (4,4) | +0.60, +0.42; 3.03, 0.63, 3.04 | 0.00; 1.01, 1.00, 1.01 | −0.62, +0.45; 0.65, 3.12, 0.68 |
| close (5,5) | +0.72, +0.57; 3.68, 0.71, 3.49 | 0.00; 1.00, 1.00, 0.99 | −0.73, +0.61; 0.75, 3.68, 0.76 |

Odd lags are zero ($|c_1|,|c_3|\le0.02$). The mutual world has equal peaks at periods $\infty$ and 2 and a dip at
period 4; the antagonistic world has the reverse; the one-way world is flat. The mutual peak at $\omega=\pi$ is the
flicker: the belief pattern at round $t$ is correlated with $t-2$ and not with $t-1$.

These are the period measures that survive memory. `quiet_regime.md` section 2.3 showed that
$P_\delta=(1-c_2)/(1-c_1)$ is a function of the anchoring alone. The discriminating quantity is the excess of the
lag-2 correlation over what lag 1 implies, $c_2-c_1^2$ or $X_2$, and the sign of $c_2$ relative to the one-way world.
With $\kappa=0.5$ near the boundary, $c_1,c_2$ are 0.84, 0.78 (mutual), 0.52, 0.23 (one-way), 0.23, −0.43
(antagonistic), and $S$ is 7.0, 0.8, 0.8; 2.8, 0.6, 0.3; 0.2, 2.1, 0.6: memory removes the mutual peak at period 2
and leaves the antagonistic peak at period 4 in a weakened form, as section 3.4 says it should.

### 4.3 The asynchronous control

One analyst updates at a time in a fresh random order per round and reads the latest calls (`envs/swarm.py`). The
partner's reply to an analyst's call can then arrive within the same sweep, and the echo moves to lag 1:

| gains | mutual $\rho_{1..4}$ | one-way | antagonistic |
|---|---|---|---|
| quiet (2,2) | +0.095, +0.044, +0.013, +0.007 | 0, 0, 0, 0 | −0.092, −0.016, +0.008, 0.000 |
| near (4,4) | +0.212, +0.139, +0.081, +0.058 | 0, 0, 0, 0 | −0.215, +0.002, +0.028, −0.015 |

The echo keeps the sign of $\Gamma$ and loses its period: the mutual world relaxes monotonically and the
antagonistic world shows one negative lag. Near the boundary the spectral peaks go from $S(\pi)=3.04$ to 0.47
(mutual) and from $S(\pi/2)=3.12$ to 1.15 (antagonistic), where the one-way world under the same schedule has 0.52
and 0.97. The replica-difference correlation has a schedule artifact under asynchronous updates ($c_1=0.26$ in the
one-way world, because consecutive beliefs of one analyst share part of an inbox), so asynchronous runs must be
compared with the asynchronous one-way world and not with 1. The echo function has no such artifact. There is no
single-site theory of the asynchronous schedule here; these numbers are simulations of the law.

### 4.4 Amplification and correlation time

*Amplification.* With $c_{ij}(k)$ the thermal covariance of $\sigma_i(t+k)$ and $\sigma_j(t)$,

$$\mathcal A=\frac{\sum_{i,j}c_{ij}(0)^2+c_{ij}(1)^2}{\sum_ic_{ii}(0)^2}=\frac1{1-\Lambda}\ \text{ on a tree}, \tag{A17}$$

estimated from four replicas: one replica pair gives $c_{ij}$ from half-differences of calls, and the square is the
product of the estimates of two disjoint pairs, so sampling noise adds no bias (`analyst_lib.chi_sg`). Equal-time
terms come from even graph distances and lag-1 terms from odd ones.

*Correlation time.* $\tau_c=-2/\ln|c_4/c_2|$ rounds, from the even lags of section 4.2.

| gains | $\mathcal A$: mutual, one-way, antagonistic ($T=40$) | $\tau_c$: mutual, antagonistic | $\mathcal A$ from 1,500 rounds: mutual, antagonistic | $c_{20}$, $c_{40}$ mutual |
|---|---|---|---|---|
| quiet (2,2) | 1.20, 1.15, 1.19 | 1.8, 1.7 | – | – |
| mid (3,3) | 1.55, 1.27, 1.53 | 3.2, 3.4 | 1.51, 1.58 | 0.005, 0.001 |
| near (4,4) | 2.4, 1.39, 2.2 | 5.5, 6.3 | 2.30, 2.84 | 0.08, 0.02 |
| close (5,5) | 3.9, 1.60, 3.7 | 8.9, 10.8 | 5.4, 10.6 | 0.27, 0.16 |
| (6,6) | – | – | 43, 122 | 0.43, 0.34 |
| (7,7) | – | – | 108, 13 | 0.61, 0.53 |

The one-way world has no correlation time (its $c_\tau$ vanish) and its amplification follows $1/(1-g^2\bar d^2)$:
1.45, 1.75, 2.01, 2.34 at $\bar a=4,6,8,12$. In the reciprocal worlds $\mathcal A$ and $\tau_c$ grow together:
$\tau_c$ is 2, 3, 6, 9 rounds as $\Lambda$ goes through 0.16, 0.36, 0.58, 0.78. With $T=40$ the correlation time is
resolved up to the "close" cell; past the boundary the lag-40 correlation is 0.3 to 0.6 and a 40-round run sees a
pattern that does not decay. At the boundary $\mathcal A$ is limited by $N$ (of order $N^{1/3}$ for the
Sherrington–Kirkpatrick class, cited from memory: 6 at $N=256$ and 10 at 1,024); the simulations here are too noisy near the boundary to
confirm that exponent, and single values of $\mathcal A$ above 10 in the table scatter by a factor of 2 or more
between graphs.

A second, cheaper estimate of $\Lambda$ uses (A7): $\hat\Lambda(\omega)=\hat g^2(1-q)^2\,|1+\sum_\tau\rho_\tau
e^{-i\omega\tau}|^2$ with $\hat g$ from the regression of section 1. It uses the pooled kernel and so is a lower
bound when analysts differ, since $\langle|R_i|^2\rangle\ge|\langle R_i\rangle|^2$. It was not evaluated on the
simulations here.

### 4.5 Zero-shot curves

$m(t)$, $q(t)$, the two-time correlation $C(t,t')$ and the replica overlap come from
`analyst_lib.dmft(Agent(...), world)` and, as the reference at $K=16$, from `check_analyst.sim_runs`. The world
dependence of the one-time quantities at gains (4,4): $q$ = 0.542, 0.448, 0.524; replica overlap 0.19, 0.15, 0.12;
and under a bias the means of section 3.5. In the mutual world without news, (A10) is a parameter-free check: on
graphs $\langle\sigma_i(t)\sigma_i(t+2)\rangle$ is 0.432 (6), 0.574 (6), 0.648 (8) against $\langle s^2\rangle$ =
0.433, 0.574, 0.650 for three settings including memory. News of 0.5 breaks it visibly (0.455 against 0.532), so it
tests the solver and a no-news arm, not the reference design.

## 5. Kick and echo

Protocol as in `envs/kick.py` and `quiet_regime.md` section 3.1: after round $t_0$, the readers of analyst $i$ see its
call flipped, with the note that goes with the flipped call; $i$'s own history stays true. The kick is
$D_i=-2\sigma_i(t_0)$, so $D^2=4$ for every twin. Two properties are new here.

- *First-order responses carry no noise.* A reader's belief at $t_0+1$ is a deterministic function of its prompt,
  so $\Delta s_j(t_0+1)=\tanh(x_j+W_{ji}D_i)-\tanh x_j$ exactly. In logit, $\Delta x_j=W_{ji}D_i$: one twin measures
  $a_+/K$ or $a_-/K$ for each of its 16 readers.
- *Calls should be drawn with common random numbers.* If the call is sampled by the experimenter as
  $\sigma=+1$ iff $u<P({\rm BUY})$ with a stored uniform $u$ per (analyst, round), a twin flips a reader's call with
  probability $|\Delta s_j|/2$ and otherwise repeats it. Redrawing every changed analyst independently gives the
  same mean echo with 1.2 to 1.9 times the noise.

Per unit kick, with $r$ the first-order belief response of a reader and $E$ the belief change of the kicked analyst
at $t_0+2$:

$$r_A=\frac{a_+}K\langle d_j\rangle,\quad r_D=-\frac{a_-}K\langle d_j\rangle,\qquad
E=\Gamma\,\langle d_id_j\rangle=\begin{cases}\tfrac K2\,(r_A^2+r_D^2)\,\theta&\text{mutual}\\0\ \text{exactly}&\text{one-way}\\K\,r_Ar_D\,\theta&\text{antagonistic}\end{cases} \tag{A18}$$

to linear order, where $\theta=\langle d_id_j\rangle/\langle d\rangle^2$. These are (K1)–(K4) of `quiet_regime.md`
with $n_r=K$. A flipped call is not a small kick (it moves a reader's log-odds by $2a/K$, 0.5 at $a=4$), so the
frozen values are the simulated ones (part D: $N=256$, 4,096 twins per cell):

| gains | world | $r_A$ | $r_D$ | $E$ (s.e.) | linear | $E$ in logit | s.d. per twin (common, independent) | twins for 3 s.e. | $E$ over the $r$ form of (A18) with $\theta=1$ |
|---|---|---|---|---|---|---|---|---|---|
| quiet (2,2) | mutual | +0.085 | −0.085 | +0.123 (2) | +0.115 | +0.177 | 0.11, 0.21 | 7, 31 | 1.07 |
| | one-way | +0.088 | −0.088 | 0 | 0 | 0 | 0 | – | – |
| | antagonistic | +0.087 | −0.086 | −0.118 (2) | −0.121 | −0.168 | 0.11, 0.21 | 8, 27 | 0.98 |
| near (4,4) | mutual | +0.116 | −0.117 | +0.252 (3) | +0.209 | +0.478 | 0.22, 0.29 | 7, 13 | 1.17 |
| | one-way | +0.140 | −0.140 | 0 | 0 | 0 | 0 | – | – |
| | antagonistic | +0.120 | −0.119 | −0.260 (4) | −0.223 | −0.483 | 0.22, 0.29 | 7, 12 | 1.13 |
| close (5,5) | mutual | +0.118 | −0.118 | +0.287 (4) | +0.215 | +0.615 | 0.26, 0.31 | 7, 12 | 1.29 |
| | antagonistic | +0.120 | −0.120 | −0.264 (4) | −0.215 | −0.583 | 0.25, 0.31 | 8, 12 | 1.15 |
| tilted (4,3) | mutual | +0.130 | −0.098 | +0.256 (3) | +0.204 | +0.447 | 0.21, 0.29 | 6, 14 | 1.21 |
| | antagonistic | +0.133 | −0.100 | −0.236 (3) | −0.213 | −0.408 | 0.20, 0.29 | 6, 14 | 1.11 |

The first-order responses match the linear values within 4%. The echo is between 2% below and 33% above linear
response, with the excess growing toward the boundary; a flipped call is a finite kick, and the slopes of the two
ends of a reciprocated tie are correlated. In
the one-way world the largest echo over 4,096 twins is $10^{-16}$: the prompt of the kicked analyst at $t_0+2$ is
unchanged. Kick echo and echo function measure the same kernel, one by intervention and one from spontaneous calls:
$E$ and $\rho_2$ agree within 5% in nine of the ten reciprocal cells (0.252 against 0.263 near the boundary, mutual);
the exception is the close antagonistic cell ($-0.264$ against $-0.295$).

**Factorization.** In logit the law is linear, so the echo of each twin is predicted with no slope factors:

$$\Delta x_i(t_0+1)=\sum_{j\ {\rm recip.}}\hat w(J_{ij})\,\Delta\sigma_j(t_0+1),\qquad\hat w({\rm type})=\frac{\langle\Delta x_j\,D_i\rangle_{\rm type}}{4}, \tag{A19}$$

with $\Delta\sigma_j\in\{0,\pm2\}$ the realized call changes of the partners and $\hat w$ the first-order logit
response of the other twins. The slope $F$ of the measured logit echo on this prediction is 1 under (A1), and the
same regression for analysts that read a reader of $i$ without being read by $i$ gives $F_{\rm open}=1$. This is the
test (K5) with the slope-factor problem removed. It is an identity of the law and was not simulated. A value
$F<F_{\rm open}$ means analysts discount the return of their own influence; $F=F_{\rm open}<1$ means the log-odds
are not additive over several simultaneous changes; residual scatter around the prediction at $F=1$ measures what the
notes carry beyond the call.

**Cost of a twin.** Only the readers at $t_0+1$ (16 in the reciprocal worlds, 16 on average one-way) and the kicked
analyst at $t_0+2$ need new calls, plus any
open-path analysts wanted for $F_{\rm open}$: 17 calls for the echo. The two-step neighbourhood of a kick is the whole
swarm at $N=256$ ($K^2=N$), so twins cannot be packed there and each is evaluated only on the analysts it needs.

## 6. Sizing

Run-to-run s.d. of each statistic for one run = one graph, 2 replicas, $T=20$ (rounds 5 to 20), from 24 runs at
$N=256$ and 12 at $N=1{,}024$ (part E). Number of runs per world for a difference of 3 s.e. between two worlds is
$9(\sigma_1^2+\sigma_2^2)/\Delta^2$.

| signature | effect $\Delta$ (quiet; near) | s.d. per run, $N=256$ | $N=1{,}024$ | runs per world for 3 s.e. |
|---|---|---|---|---|
| echo $\rho_2$, sign against one-way | 0.12; 0.26 | 0.008; 0.012 | 0.005; 0.008 | 1 at any gain with $\bar a\ge1$ |
| $X_2$ (memory-robust echo) | 0.12; 0.26 | 0.008; 0.012 | 0.005; 0.008 | 1 |
| $c_2$ of replica differences | 0.24; 0.59 | 0.036; 0.038 | 0.014; 0.024 | 1 |
| period-4 against period-2 weight, $S(\pi/2)$ | 0.32, 0.68; 0.39, 2.07 | 0.04–0.12; 0.05–0.17 | 0.03–0.06; 0.02–0.07 | 1 to 2 |
| $q$ mutual against one-way | 0.024; 0.092 | 0.006; 0.008 | 0.003; 0.003 | 1 |
| $q$ against the zero-shot value to 1% | 0.003; 0.005 | 0.006; 0.008 | 0.003; 0.003 | 36, 23 at $N=256$; 9, 4 at $N=1{,}024$ |
| $\rho_2$ against the zero-shot value to 4% | 0.005; 0.010 | 0.008; 0.012 | 0.005; 0.008 | 23, 12 at $N=256$; 9, 6 at $N=1{,}024$ |
| mean belief under a bias $c=0.1$, between worlds (s.d. from the tilted cell) | 0.03 | 0.01–0.04 | 0.008–0.017 | about 16 at $N=256$, 4 at $N=1{,}024$ |
| amplification $\mathcal A$, mutual against one-way (4 replicas) | 0.02; 0.35–0.74 at $T=20$ | 0.16; 0.45 | 0.16; 0.82 | not resolvable when quiet; 16 near the boundary at $T=20$, 3 at $T=40$ |
| correlation time | 2 to 9 rounds | needs $c_4$: $T\ge40$ near the boundary | | 2 to 4 graphs at $T=40$ |
| kick echo $E$ | 0.12; 0.25 | 0.11; 0.22 per twin | same | 7 twins; 250 twins give 6% |
| first-order kick responses | 0.085; 0.116 | 0 | 0 | one twin |

Reading it as a design:

- *Signs and period structure in all three worlds* need one run per (world, intensity) at $N=256$, $T=20$, 2
  replicas. Two graphs give the graph-to-graph check. The smallest echo that 4 runs per world resolve is 0.016, which
  is $\bar a\approx0.7$: a swarm far below the boundary still shows the reciprocity signature.
- *Quantitative zero-shot comparison* is where $N=1{,}024$ earns its cost: 4 to 9 runs at $N=1{,}024$ against 12
  to 36 at $N=256$ for $q$ to 1% and $\rho_2$ to 4%, and 4 against 16 for the mean belief. What should improve with
  $N$ is the error, by a factor 2 from 256 to 1,024 (observed 1.5 to 2.6 across statistics). The means should not
  move: the simulated offsets between $N=256$ and 4,096 are below 0.002 in $q$ and 0.004 in $\rho_2$.
- *Amplification and correlation time* need $T=40$ and 4 replicas, at 3 or more graphs for the top two
  intensities. More rounds help more than more analysts: $\mathcal A$'s error does not fall with $N$, and near the
  boundary it grows.
- *Kicks*: 250 twins per (world, intensity), on existing base runs, for the echo to 6%; 7 for its sign.
- *Asynchronous control*: one run per world at two intensities; its echo has the errors of the first row.

## 7. Falsifiers and the go/no-go rule

Stated against the frozen numbers, with the calibrated parameters substituted and errors as in section 6.

1. **Worlds indistinguishable at equal mean trust.** $\rho_2$(mutual) $-$ $\rho_2$(one-way) and $\rho_2$(antagonistic)
   $-$ $\rho_2$(one-way) both within 3 s.e. of zero while the in-swarm gains predict $|\Gamma|(1-q)^2>0.03$. This is
   the mean-field outcome and refutes the reaction term.
2. **Echo signs not following $\Gamma$.** The mutual echo must be positive whatever the gains. The antagonistic echo
   must have the sign of $-a_+a_-$ with the gains measured in the swarm (first-order kick responses or the logit
   regression). A negative mutual echo, or an antagonistic echo with the sign of $+a_+a_-$, at 3 s.e. in either the
   echo function or the kick, falsifies the effective-coupling picture. $\rho_4$ must be positive in both.
3. **A nonzero echo in the one-way world**, beyond what memory gives: $X_2\ne0$ at 3 s.e. The kick echo there is
   zero by construction and checks the protocol, not the theory.
4. **Zero-shot curves off and not improving.** With gains from the single-agent calibration, $q$, $\rho_2$, $c_2$ and
   $m(t)$ from the graph simulation of the law are the prediction. A discrepancy larger than 3 s.e. plus the
   calibration's propagated error, of the same size at $N=256$ and 1,024, is a failure of the law, not a finite-size
   effect. Two outcomes must be told apart with the in-swarm regression of section 1: if the in-swarm gains differ
   from the calibration by more than their errors, the calibration did not transfer, and the theory is then tested
   with the in-swarm gains (no longer zero-shot, and to be reported as such); if the in-swarm gains agree and the
   curves still miss, the single-site reduction fails.
5. **Instability unrelated to the single-agent prediction.** A correlation time above 10 rounds or $\mathcal A>4$
   where the calibrated $\Lambda$ is below 0.5; or $\tau_c<3$ and $\mathcal A<1.5$ in a reciprocal world where it is
   above 0.8; or any persistent pattern in the one-way world without memory ($c_2$ of replica differences above
   0.1).
6. **Oscillations that survive the asynchronous control.** $S(\pi)$ in the mutual world or $S(\pi/2)$ in the
   antagonistic world above the asynchronous one-way value by more than 0.5 under asynchronous updates.
7. **Period structure wrong under synchronous updates without memory**: $c_1$ of replica differences above 0.1 in a
   reciprocal world when $\rho_1<0.05$, or a mutual peak at period 4.
8. **Composition fails.** $F<F_{\rm open}-0.2$ in (A19): analysts discount their own returning influence, and the
   reaction term overstates the echo. $F=F_{\rm open}$ well below 1: the log-odds are not additive, and the gains
   are functions of the inbox, to be recalibrated.
9. **Response to a common bias the same in the three worlds**, when $\bar J\ne0$ or a public signal is used: the
   ordering mutual > one-way > antagonistic of section 3.5 is predicted at every gain.
10. **Field not of the predicted shape**: excess kurtosis of $x_i(t)$ conditional on $\sigma_i(t-1)$ below $-0.3$, or
    bimodal, in the one-way world.

**Go/no-go for the calibration.** Let $\bar a=\sqrt{(a_+^2+a_-^2)/2}$, $r=a_-/a_+$, and $\rho_1=\kappa\langle d\rangle$.

- *The reciprocity test (echo, period structure, kicks) is a go* if any wording gives $\bar a\ge1$ with $r>0.3$. The
  echo is then at least 0.035 in the mutual world and 0.02 in the antagonistic one, and one or two runs resolve it. If $0<r<0.3$ the mutual-against-one-way
  half is still a go and the antagonistic world sits near the one-way one. If $r\le0$ (faded colleagues ignored or
  followed) the three-world contrast is a no-go until a wording fixes the fade channel, as the inverted wording did
  for Haiku.
- *The boundary sweep is a go* if the strongest wording gives $\Lambda_{\rm mutual}\ge0.7$, which in the reference
  design is $\bar a\ge4.6$ and without standing notes $\bar a\ge3.5\sqrt{1+\pi\nu^2/2}$ (4.1 at $\nu=0.5$, 3.5 with
  negligible news). If no wording reaches $\bar a=3.5$, the sweep cannot reach the boundary at $K=16$ and the
  amplification and correlation-time predictions drop out, leaving the first bullet.
- *If the gain is far above the boundary* ($\bar a>7$), a trust-intensity sweep is not needed: the news amplitude is
  the control parameter. $\nu$ is set by the experimenter, and (A14) puts the boundary at
  $\nu_c=\sqrt{(2/\pi)\,[(\bar a/4.23)^2-1]}$, 1.5 at $\bar a=9$. A sweep of news strength at fixed wording also
  leaves $a_\pm$ untouched, which a sweep of wordings does not.
- *Herding first*: if $(a_+-a_-)/2>1.3$ at the top intensity, the mutual world herds before it reaches the boundary.
  Then either a wording with $r\ge0.6$ is needed, or the sweep becomes a test of section 3.5 (world-dependent
  herding threshold), which is also a cavity prediction.
- *Memory*: if $\rho_1>0.3$, expect no period-2 peak and a weak period-4 peak; plan on $X_2$ and the kick echo, and
  consider removing the analyst's own last call from its context.
- *Standing notes*: they cost about 20% in boundary gain at $a_0=0.3$. If the calibration finds $a_0>0.3$ for the
  intended note, weaken the note.

## 8. What the coin-game calibration would change, and what is not verified

The Qwen coin-game calibration (`experiments/06-qwen-calibration`) measures stated probabilities, not a committed
call, so its gains do not map onto $a_\pm$. Two of its outputs would inform this note: the ratio of the fade gain to
the ally gain, which sets $r$, $\gamma_{\rm eff}$ and the herding margin, and the anchoring weight, which sets
expectations for $\rho_1$. Neither is assumed here.

**Not verified.**

- That Qwen's log-odds are linear in the counts of BUY and SELL among allies and faded colleagues. A majority rule
  or a saturating response would make $a_\pm$ depend on the inbox. The graph simulation accepts any measured
  response; the solver and the closed forms then need local slopes at the operating point. The calibration should
  include inboxes with the same $f_A,f_D$ and different compositions, as in `claim1.md` section 4.1.
- That readers respond to the call and not to the sender's belief through the note. The leak sensitivity of section
  4.1 is from 3 graphs and one gain, and the solver does not implement it.
- The form of the memory. (A1) uses the last call only; a dependence on older calls or on the analyst's own notes
  changes $R^0$ and section 3.4. The anchoring law of `quiet_regime.md` (weight on the own answer that depends on how
  much the inbox changed) was not carried over, on the expectation that fresh news each round keeps the inbox
  changing; this is an assumption.
- Greedy calls ($\sigma={\rm sign}\,s$). The map is in `AnalystMap` (the news is then the only noise, the mean call is
  $\mathrm{erf}(u/\nu\sqrt2)$ and the news strength alone sets the temperature), and an exploratory run matched graphs in
  $q$, but no committed check covers it and the echo function (A8) does not exist for it.
- Boundary values with standing notes (5.9 and 5.1) come from one graph per world and a linear fit of
  $1/\mathcal A$ over 3 or 4 gains; they are uncertain by about 0.3. The no-notes values rest on (A13)–(A14).
- Whether the antagonistic world has a boundary at all when $a_-\ne a_+$.
- The solver's $\Lambda$ near the boundary: the kernel sum is truncated (25 to 80 lags) and underestimates.
- Finite-size scaling of $\mathcal A$ at the boundary, and everything beyond the boundary, where the numbers are
  simulations at $T\le1{,}500$ of a state that keeps aging.
- Asynchronous updates: simulation only, 3 graphs, and its replica-difference statistics carry the schedule artifact
  of section 4.3.
- The factorization slope (A19) and the in-swarm regression of section 1 are identities of the law and were not run
  on simulated data.
- Per-agent variation of the gains. With analysts that differ in $a_\pm$ the block parameters become averages and
  (A12) takes the per-agent form; not worked out.

## 9. Summary of what is established

| statement | status |
|---|---|
| single-site process (A5)–(A7) for sampled calls with news | `claim1.md` adapted; matches graphs at $K=16$ to 0.002 in $q$, 3% in $\rho_2$ (part A) |
| response kernel from innovations, (A7)–(A8) | identity; solver and graphs agree through $\rho_\tau$ (parts A, C) |
| mutual world: (A10), $\chi=1+c_1$, $\Lambda=g^2(1\pm c_1)^2$ | exact for the logistic law; checked in solver and on graphs (part B1) |
| tree form (A13) and boundary $\bar a=4.23$ at $K=16$ | exact on a tree; susceptibility checked at $N=2{,}048$ to 1% (part B2), boundary by long simulation (part G) |
| news factor (A14) | solver within 2%; boundary 5.03 against 4.99 simulated (parts B4, G) |
| one-way world: $\Lambda\le2/\pi$ | closed form; amplification on graphs stays below 2.75 (parts B4, G) |
| antagonistic = mutual under the period-4 sign pattern | exact for equal gains without static fields; numbers mirror (parts B3, C) |
| memory: mutual boundary earlier and static, period-4 mode weakened | solver (parts B1, B3) and graphs (part C) |
| world-dependent susceptibility and herding threshold | solver; one check on graphs (part B5) |
| echo, spectra, amplification, correlation time per world | simulations of the law on $K=16$ graphs (parts C, G) |
| kick echo, noise per twin | simulations (part D) |
| sizing | simulations (part E) |
| the law itself for Qwen | not tested; calibration and section 7 |
