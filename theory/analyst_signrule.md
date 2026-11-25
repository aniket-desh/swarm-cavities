# De-biased sign-rule test: frozen predictions for Qwen analysts at K = 16

Frozen 2026-11-25, before any de-biased analyst swarm data exist. It replaces sections 4 to 7 of
`analyst_scenario.md` for this design; that note's law, notation and solver stand. Numbers:
`theory/checks/analyst_signrule_predictions.json`. Code: `theory/checks/analyst_signrule_check.py` (parts A to F,
outputs `analyst_signrule_out_<part>.txt`; calibration fits and bootstrap draws in `analyst_signrule_calib.json`).
Amendment 1 at the end adds the asynchronous control and restates falsifier 7; it was frozen before any asynchronous
run.

**Design.** Qwen3.5-9B analysts, $N=256$, $K=16$, $T=20$, 2 replicas per graph, about 12 runs over the mutual
($\gamma=+1$), one-way ($\gamma=0$) and "antagonistic" ($\gamma=-1$) graphs at $p_-=\tfrac12$. The model returns the
half log-odds $x$ of BUY. The experimenter samples the call from $\tanh(x-\hat c)$ with a stored uniform per
(replica, round, analyst), and the model then writes its note after that call, as in `envs/analyst.py`.

**Verdict.**

1. *Cell.* Use 08b's P0: the "max" wording with the default prompt (question "close up", BUY listed first,
   ordinary fading), and $\hat c=-0.637$. It has the larger reciprocity echo of the two candidates once memory is
   removed ($X_2=+0.057$ against $+0.048$ for the SELL-first cell P1 in the mutual world, $+0.010$ against $+0.007$
   in the antagonistic one), and it is the cell with a reproduction, a template arm and the premise fits. P1's
   larger $\rho_2$ in 08b's table was its anti-persistent own call, which the one-way world shows equally.
2. *Sign rule.* With the measured couplings every reciprocal echo is positive: mutual $X_2=+0.057$
   (calibration s.e. 0.002; $+0.03$ to $+0.07$ over the law variants), one-way $0$, "antagonistic" $+0.010$ (s.e.
   0.002; $+0.005$ to $+0.021$ over the variants). The antagonistic sign is that of $-a_+a_-$ and $a_-=-0.146$
   (s.e. 0.037): faded colleagues attract, so discordant ties return an analyst's call with a positive sign.
3. *What 12 runs resolve.* Mutual against one-way is 10 standard errors with 2 and 5 runs. Antagonistic against
   one-way is 2.6 standard errors with 5 runs each under the calls-only law, and 4.4 if the note carries the
   sender's belief as the calibration's leak fit says. So the echo function can refute a negative antagonistic
   echo, and it confirms a positive one only in the second case.
4. *Kicks decide the antagonistic sign.* A kick twin costs 34 generations against 21,504 for a run, and with
   common random numbers 48 twins give 3 standard errors on the antagonistic echo ($E=+0.009$ per unit kick), 209 if
   rewritten notes add 0.1 of logit noise. Recommended: 2 mutual, 5 one-way and 5 antagonistic runs, plus about 570
   twins (0.9 of a run).
5. *The note channel changes the mean, not the signs.* If readers respond to the sender's belief through its note
   as the leak fit says, the de-biased swarm drifts to BUY: mean belief $+0.35$ (s.e. 0.14) instead of $0.00$
   (s.e. 0.03). The echoes move by less than their calibration error in the mutual world and rise to $+0.017$ in
   the antagonistic one. The mean response to a common bias differs by 5% between worlds and is not resolvable.
6. *Scope.* This tests the sign rule of the reaction term given the couplings Qwen actually has. It does not
   test whether agents act on distrust: they do not, in any wording tried.

All statistics below are simulations of the stated law on $K=16$ graphs at the design size. Whether Qwen obeys the
law in a de-biased swarm is what the experiment decides; section 9 lists what is not verified.

## 1. The calibration cell

Both candidates are the "max" wording with ordinary fading; they differ in the order of the options. M1 is the law
(A1) by least squares on $x$; 1,000 base probes per cell, bootstrap s.e. in brackets (part A, which reproduces
08b's `summary.json` to machine precision).

| cell | $a_+$ | $a_-$ | $\nu$ | $a_0$ | $\kappa$ | $c$ | residual s.d. |
|---|---|---|---|---|---|---|---|
| P0, BUY first | 1.735 (0.040) | −0.146 (0.037) | 0.299 (0.022) | 0.090 (0.012) | −0.095 (0.012) | −0.637 (0.012) | 0.396 (0.010) |
| P1, SELL first | 1.583 (0.040) | −0.109 (0.033) | 0.269 (0.021) | 0.093 (0.012) | −0.167 (0.012) | −0.382 (0.011) | 0.385 (0.011) |

P0 is the choice, for four reasons.

- *Echo size.* Simulated at the design size with each cell's own $\hat c$ (part C, 300 runs per entry), the
  memory-robust echo $X_2=\rho_2-\rho_1^2$ is $+0.0571$, $-0.0005$, $+0.0096$ (mutual, one-way, antagonistic) for
  P0 and $+0.0484$, $-0.0004$, $+0.0069$ for P1.
- *Memory.* 08b's solver table had P1 ahead ($\rho_2=+0.079$ against $+0.075$, antagonistic $+0.029$ against
  $+0.018$). That lead is $\rho_1^2$: P1's own-call weight $\kappa=-0.167$ gives $\rho_1=-0.132$ and
  $\rho_2=+0.017$ in the one-way world, where there is no reciprocity at all, against $+0.005$ for P0. In P1 the
  memory term is 2.5 times the antagonistic reaction echo it has to be separated from.
- *Support.* P0 was measured in 08 and repeated in 08b with the same coefficients to the second decimal. It is the
  only cell with a template arm (T0) and with the leak, linearity and averaging fits, and its prompt is the default
  whose hash the tests freeze.
- *The smaller bias of P1 buys nothing.* Once calls are de-biased, $q=\langle s^2\rangle$ is 0.226 (P0) and 0.217
  (P1), and under the leak law both cells drift alike (one-way mean belief $+0.35$ and $+0.40$; section 5, part C).

The note pool is also close to what P0's swarm will write. The pool's BUY notes were written at mean raw belief
$-0.52$ and its SELL notes at $-0.78$, because the pilot was bearish. A de-biased P0 swarm writes its BUY notes at
$-0.28$ and its SELL notes at $-0.64$, and 75% of its BUY notes come from analysts whose own probability of BUY is
below one half (part B4). So the gains were measured on BUY notes written against the writer's belief, which is
the kind this swarm produces.

**The swarm design must use exactly this cell:** wording "max", `PromptVariant()` default, news amplitude
$\nu_{\rm nom}=0.5$, note amplitude $a_{0,\rm nom}=0.3$, and $\hat c=-0.637$ (the calibration s.e. of 0.012 moves
the mean belief by 0.03, section 5).

## 2. The law under de-biased sampling

$X_i(t)$ is the half log-odds the model returns for its round-$t$ decision. Two beliefs follow from it,

$$s^r_i(t)=\tanh X_i(t)\quad\text{(raw: what the model holds and writes its note from)},\qquad
s_i(t)=\tanh\big(X_i(t)-\hat c\big)\quad\text{(sampling: what the call is drawn from)}, \tag{S1}$$

$$\sigma_i(t)=\pm1\ \text{ with probability }\ \tfrac12\big(1\pm s_i(t)\big), \tag{S2}$$

and the law for the next logit keeps the form (A1) with one addition, a pull toward the sender's raw belief:

$$X_i(t+1)=\frac1K\sum_{j\in\text{allies}}\big[u_A\sigma_j(t)+v_As^r_j(t)\big]
+\frac1K\sum_{j\in\text{faded}}\big[u_D\sigma_j(t)+v_Ds^r_j(t)\big]
+a_0h^0_i+\nu\xi_i(t+1)+\kappa\sigma_i(t)+c+\varepsilon_i(t+1). \tag{S3}$$

$u_A,u_D$ are pulls toward a colleague's call and $v_A,v_D$ pulls toward the belief its note carries, all per unit
normalized field. $\varepsilon$ is the residual of the calibration fit, taken white with the measured s.d.: in the
pilot's in-swarm regression 3 to 5% of its variance is a per-analyst constant and its lag-1 autocorrelation is
$+0.03$ to $+0.09$ (part A). Two laws are carried through every prediction.

| law | $u_A$ | $v_A$ | $u_D$ | $v_D$ | $c$ | source |
|---|---|---|---|---|---|---|
| M1, calls only (central) | 1.735 | 0 | 0.146 | 0 | −0.637 | (A1): $u_A=a_+$, $u_D=-a_-$ |
| M2, note channel | 1.663 | 0.426 (0.098) | 0.062 | 0.468 (0.100) | −0.371 (0.046) | 08's leak model: $u=a-b_L$, $v=b_L$ |

Three statements about this law, each checked in part B.

**De-biasing is exact for the calls-only law.** With $v=0$, (S1)–(S3) is (A1)–(A3) for the variable $X-\hat c$
with bias $c-\hat c$. On the same graph, notes, news and uniforms the two simulations give identical calls (0 of
10,752 differ in each world, beliefs equal to $4\times10^{-16}$; part B1). So everything `analyst_scenario.md`
derives holds with $c\to c-\hat c$, and 08b's "$c=0$" columns are this design to the extent that the measured $c$ is
the swarm's.

**Every statistic uses the sampling belief.** The innovation $\zeta_i(t)=\sigma_i(t)-s_i(t)$ has zero mean given
the past only for $s$ of (S1), and the score-function identity (A7) holds for it exactly because the experimenter
does the sampling. The echo function, $X_2$, $q$, $m$ and the replica differences are all defined on
$s=\tanh(x-\hat c)$. Feeding the raw belief to the estimator gives $X_2=+0.040$ instead of $+0.058$ in the mutual
world and $+0.008$ instead of $+0.010$ in the antagonistic one (part B3).

**What the note does.** The note is written from $s^r$, which is biased, after a call drawn from $s$. In the
calibration pool the sender's belief hardly depended on its call (slope of $s^r$ on $\sigma$: $\beta=0.13$), so the
M1 gains are $u+v\beta$. In the de-biased swarm $\beta=0.18$ to 0.22, and an in-swarm regression on calls alone
returns $a_+=1.76$, $a_-=-0.19$ under M2 (part B4): the call gains barely move. The note channel shows up in two
other places. It adds a route to the reaction term that carries no sampling noise (section 3), and it couples the
mean belief to the mean raw belief, which is negative (section 5). The M2 fit is weakly identified: in the probe
design $L$ and $f$ correlate at $-0.9$, the placebo cells give $b_{LD}=0.22$ (0.09) with template notes and 0.24
(0.10) with no faded notes shown, and with group size as a regressor P0 gives $b_{LA}=0.53$ (0.11), $b_{LD}=0.37$
(0.12) (part A). I therefore treat M1 as the central law and M2 as the outer bracket of the note channel. A fit with the
ally leak alone (M2A: $\lambda_A=0.314/1.995=0.16$) stays close to M1: mutual $X_2=+0.061$, antagonistic $+0.011$,
mean belief $+0.06$ (part D).

## 3. The reaction term and the sign rule

An innovation $\zeta_i(t)$ in analyst $i$'s call shifts the logit of each reader $j$ by $u_{g(j\leftarrow
i)}\zeta/K$, where $g(j\leftarrow i)$ is the group $j$ puts $i$ in. Only the call pull acts, because $i$'s raw belief
is what it was. Reader $j$'s next message to $i$ then shifts in expectation by two routes: its call, with slope
$d_j=1-s_j^2$, and the belief its note carries, with slope $d^r_j=1-(s^r_j)^2$ and no sampling. Summing over the $K$
reciprocated partners, half in each group,

$$\tau_g=u_g\,d+v_g\,d^r,\qquad
G=\begin{cases}\dfrac{u_A\tau_A+u_D\tau_D}{2K}&\text{mutual (ally–ally and faded–faded ties)}\\[2mm]
0&\text{one-way}\\[1mm]
\dfrac{u_A\tau_D+u_D\tau_A}{2K}&\text{antagonistic graph (ally–faded ties)}\end{cases} \tag{S4}$$

$$X_2\equiv\rho_2-\rho_1^2=G\,d+O(G^2),\qquad\rho_1=\kappa\,d, \tag{S5}$$

with $d=\langle1-s^2\rangle$, $d^r=\langle1-(s^r)^2\rangle$. For $v=0$ this is $\Gamma d^2$ with $\Gamma$ of (A4):
$(a_+^2+a_-^2)/2K$ and $-a_+a_-/K$. (S5) against the graph simulation (part C): $+0.0567$ against $+0.0571$ (M1,
mutual), $+0.0096$ against $+0.0096$ (M1, antagonistic), $+0.0517$ against $+0.0539$ and $+0.0167$ against $+0.0170$
(M2). The $K=\infty$ solver gives $+0.0568$ and $+0.0092$ for M1.

**The rule, per world, with the measured couplings.**

| world | sign of $X_2$ and of the kick echo | why |
|---|---|---|
| mutual | $+$ | calls only: a sum of squares, positive whatever the gains. With the note channel: $u_A\tau_A+u_D\tau_D=2.42+0.02>0$ |
| one-way | exactly 0 | no reciprocated tie, under any law |
| "antagonistic" | $+$, weak | calls only: sign of $-a_+a_-$, and $a_-=-0.146$ is 3.9 s.e. below zero. With the note channel: $u_A\tau_D+u_D\tau_A=0.66+0.09>0$ |

The antagonistic graph has discordant ties: where $i$ counts $j$ as an ally, $j$ counts $i$ as faded. If faded
colleagues repelled, $i$'s call would push $j$ away and $j$'s reply would pull $i$ back toward the opposite call, a
negative echo. They attract, so both steps have the same sign and the echo is positive, smaller than the mutual one
by the factor $2a_+|a_-|/(a_+^2+a_-^2)=0.17$. The simulation shows the rule turning with $a_-$ at fixed $a_+$ (part
D, 100 runs each): $X_2=+0.0104$ at the measured $a_-=-0.146$, $+0.0010$ (0.0007) at $a_-=0$, and $-0.0090$ at
$a_-=+0.15$.

## 4. Predicted statistics per world

Design size, rounds 5 to 20, sampling belief. Central value from 300 runs (Monte Carlo s.e. 0.0004 for the echoes,
0.001 to 0.002 for $c_2$); the bracket is the calibration s.e. propagated through the simulation by 200 bootstrap
draws on common graphs and random numbers (part E). $c_\tau$ are lag correlations of replica differences of $s$,
$X^c_2=c_2-c_1^2$.

| statistic | law | mutual | one-way | "antagonistic" |
|---|---|---|---|---|
| $\rho_1$ | M1 | −0.074 (0.010) | −0.075 (0.010) | −0.074 (0.009) |
| $\rho_2$ | M1 | +0.0626 (0.0025) | +0.0051 (0.0014) | +0.0151 (0.0027) |
| | M2 | +0.058 (0.010) | +0.0043 (0.0012) | +0.0215 (0.0031) |
| $X_2$ | M1 | **+0.0571 (0.0024)** | **−0.0005 (0.0003)** | **+0.0096 (0.0024)** |
| | M2 | +0.0539 (0.0085) | −0.0002 (0.0008) | +0.0170 (0.0023) |
| $c_2$ | M1 | +0.097 (0.006) | +0.014 (0.002) | +0.029 (0.005) |
| | M2 | +0.131 (0.016) | +0.032 (0.006) | +0.078 (0.010) |
| $X^c_2$ | M1 | +0.082 (0.005) | +0.006 (0.002) | +0.020 (0.005) |
| $q$ | M1 | 0.226 (0.007) | 0.223 (0.007) | 0.223 (0.007) |
| $m$ | M1 | 0.00 (0.03) | 0.00 (0.03) | 0.00 (0.02) |
| | M2 | +0.37 (0.15) | +0.35 (0.14) | +0.32 (0.14) |

- *$\rho_2$ is not zero in the one-way world.* The own-call weight $\kappa<0$ gives $\rho_2=\rho_1^2=+0.005$ there.
  Comparisons use $X_2$, or the difference from the one-way world.
- *Range over law variants* (part D, 100 runs each; mutual, antagonistic $X_2$): gains shifted as the pilot's
  in-swarm fit was from its probes ($a_+\times0.76$, $a_-\times1.85$) $+0.036$, $+0.015$; averaging within groups
  (M4 gains, per-reader pull $\propto1/n_{\rm group}$) $+0.062$, $+0.011$; residual 0.33 as in the pilot $+0.060$,
  $+0.011$; news 1.5 times stronger $+0.051$, $+0.009$; ally leak only (M2A) $+0.061$, $+0.011$; M2 de-biased at
  its own fixed point $+0.063$, $+0.018$. With no residual at all, the basis of 08b's numbers, $+0.068$, $+0.012$.
  The zero-shot intervals are therefore **mutual $X_2\in[+0.03,+0.07]$, antagonistic $X_2\in[+0.005,+0.021]$**, and
  the one-way value is zero in every variant ($|X_2|\le0.0009$).
- *A residual bias costs little.* If the swarm's bias differs from $\hat c$ by $\delta c$, the mean belief is
  $2.2\,\delta c$ and the echoes fall with $(1-q)^2$: at $\delta c=\pm0.10$ ($m=\pm0.22$) the mutual $X_2$ is
  $+0.052$ to $+0.053$ and the antagonistic one $+0.009$ to $+0.010$; at $\pm0.20$ ($m=\pm0.41$) they are $+0.043$
  and $+0.0075$ (parts D, M). No sign changes.
- *$c_2$ depends on what the residual is.* The table takes $\varepsilon$ independent between replicas. If replicas
  share it (it is a function of the prompt, and replicas share notes and news), $c_2$ is $+0.149$, $+0.032$,
  $+0.046$ under M1. Under M2 the slow mean mode adds to $c_2$ in every world, the one-way world included, so
  $c_2$ is read as a difference from the one-way world: $+0.08$ to $+0.13$ for the mutual world, $+0.015$ to
  $+0.045$ for the antagonistic one.
- *$\rho_4$* is $+0.009$ in the mutual world and $+0.001$ elsewhere, against a run-to-run s.d. of 0.006 to 0.007 for
  $\rho_2$, so it is not a planned test.

## 5. Mean belief and the response to a common bias

For the calls-only law the mean obeys $\delta m=\chi(\bar J\,\delta m+\delta c)$ with $\bar J=(u_A+u_D)/2=0.941$ and,
in the one-way world, $\chi=d/(1-\kappa d)=0.723$, so $m=2.27\,\delta c$. Shifting the de-bias constant by $\pm b$
in the simulation gives slopes $dm/db=2.31$, $2.21$, $2.11$ in the mutual, one-way and antagonistic worlds (part M).
The ordering is that of `analyst_scenario.md` section 3.5, but the worlds are 5% apart. One pair of runs at
$b=\pm0.1$ on different graphs measures the slope to 0.15, so separating the mutual from the antagonistic world at 3
s.e. takes about 13 pairs per world. **Not resolvable in this budget, and not recommended.**

The mean belief itself is informative in a different way: it reads the bias left after de-biasing, and it
separates the two laws. Under M1, $m=0.00$ with calibration s.e. 0.026 and run-to-run s.d. 0.025. Under M2 the
intercept of $X-\hat c$ at $m=0$ is $c-\hat c+\tfrac12(v_A+v_D)\langle s^r\rangle=-0.371+0.637-0.447\times0.46=
+0.06$: the pool's senders were more bearish than a de-biased swarm's raw beliefs, so $\hat c$ over-corrects. The
belief route also strengthens the mean loop, and the swarm settles at $m=+0.37$, $+0.35$, $+0.32$ (calibration s.e.
0.14, 95% of draws between $+0.11$ and $+0.69$; run-to-run s.d. 0.05). The ally leak alone gives $m=+0.06$. Under
M2 the constant that would leave $m=0$ is $-0.544$ (part D). I do not recommend changing $\hat c$ on the strength
of the M2 fit; the first run's mean belief says which law holds, at 7 run s.d. between them.

## 6. Kick twins

Protocol of `analyst_scenario.md` section 5: after round $t_0$ the readers of analyst $i$ see its call flipped, with
the note the model writes for the flipped call; $i$'s own history stays true; rounds $t_0+1$ and $t_0+2$ are
replayed with the base run's uniforms. $D=-2\sigma_i(t_0)$. Per unit kick, on the sampling belief (part K, 8,192
twins per entry; the replay reproduces `analyst_lib.kick_twins` to $3\times10^{-16}$ for the calls-only law):

| world | law | $r_A$ | $r_D$ | first-order logit, ally / faded reader | $E$ (s.e.) | $E$ in logit | s.d. per twin | twins for 3 s.e. at note noise 0 / 0.1 / 0.2 |
|---|---|---|---|---|---|---|---|---|
| mutual | M1 | +0.084 | +0.0071 | +0.1085 / +0.0091 | +0.0553 (0.0008) | +0.072 | 0.069 | 14 / 19 / 32 |
| | M2 | +0.072 | +0.0027 | +0.1039 / +0.0039 | +0.0545 (0.0007) | +0.077 | 0.063 | 12 / 16 / 26 |
| one-way | both | +0.084 | +0.0071 | as above | 0 exactly ($<3\times10^{-16}$) | 0 | 0 | – |
| "antagonistic" | M1 | +0.085 | +0.0071 | +0.1085 / +0.0091 | **+0.0092 (0.0002)** | +0.012 | 0.021 | 48 / 209 / 689 |
| | M2 | +0.075 | +0.0028 | +0.1039 / +0.0039 | **+0.0170 (0.0002)** | +0.024 | 0.015 | 7 / 46 / 164 |

In plain units: flipping one call moves the flipped analyst's own log-odds two rounds later by $2E_x=0.14$ toward
the flipped call in the mutual world (0.15 under M2), by 0.02 to 0.05 in the antagonistic world, and by nothing in the one-way
world. The calibration s.e. of $E$ is that of $X_2$ (4% mutual, 25% antagonistic under M1). $E$ agrees with $X_2$ to
4%: they are the same kernel.

- *First-order responses carry no sampling noise and measure the call pulls in the swarm:* $\Delta x_j=u_g
  D/K$ for every reader, so each twin returns $a_+$ from its ally-tie readers and $a_-$ from its faded-tie readers.
  A faded-tie reader should move **toward** the flipped call by $2\times0.0091=0.018$ in logit (M1).
- *Composition, (A18).* Under the calls-only law $E=K\,r_Ar_D\,\theta$ in the antagonistic world and
  $\tfrac K2(r_A^2+r_D^2)\theta$ in the mutual one: the simulated ratios are 0.96 and 0.98. Under M2 they are 5.1
  and 1.3, because the note returns the reader's shifted belief without a call flip. The ratio is the cleanest
  measure of the note route.
- *Two ways to replay $t_0+2$.* If a reader's note is rewritten only when its call flips ("notes kept"), the belief
  route is closed for the others: $E$ stays $+0.0092$ under M1 and falls to $+0.0043$ (antagonistic) and $+0.043$
  (mutual) under M2. Both replays share the $t_0+1$ step and differ in one decision call per twin.
- *Note noise.* All 16 partners' prompts change at $t_0+1$, so their rewritten notes differ in wording even when
  the call does not, and the kicked analyst reads 16 new texts. The law has no term for this. The last column adds
  a logit noise of the stated s.d. at the kicked analyst. It has to be measured with null twins: rewrite the 16
  partners' notes with a new seed, no kick, and record the change of $i$'s logit.

## 7. Sizing and the split of runs

Run-to-run s.d. of one run (one graph, 2 replicas, $T=20$, rounds 5 to 20), from 300 runs per world (part S):

| statistic | mutual | one-way | "antagonistic" | with 4 replicas |
|---|---|---|---|---|
| $X_2$ (and $\rho_2$) | 0.0066 | 0.0059 | 0.0066 | 0.0042, 0.0044, 0.0043 |
| $X^c_2$ | 0.021 | 0.022 | 0.020 | 0.015, 0.014, 0.014 |
| $m$ | 0.027 | 0.024 | 0.025 | 0.020 to 0.023 |
| $q$ | 0.003 | 0.003 | 0.003 | 0.002 |

The echo's noise is call sampling, so it falls as $1/\sqrt{R}$ and a replica is worth as much as a new graph.
Comparisons in standard errors, law M1 / law M2:

| split (mutual, one-way, antagonistic) | $X_2$: mutual − one-way | $X_2$: antagonistic − one-way | $X_2$: mutual − antagonistic | $X^c_2$: mutual − one-way | s.e. of one-way $X_2$ |
|---|---|---|---|---|---|
| (4, 4, 4) | 12.9 / 11.7 | 2.3 / 3.9 | 10.1 / 7.8 | 5.1 / 3.8 | 0.0030 |
| (3, 4, 5) | 11.9 / 10.7 | 2.4 / 4.2 | 9.8 / 7.5 | 4.7 / 3.5 | 0.0030 |
| **(2, 5, 5)** | **10.7 / 9.6** | **2.6 / 4.4** | **8.6 / 6.5** | **4.3 / 3.3** | **0.0027** |
| (1, 5, 6) | 8.1 / 7.2 | 2.7 / 4.6 | 6.6 / 4.9 | 3.3 / 2.5 | 0.0027 |
| (2, 5, 5) with 4 replicas | 16.1 / 13.7 | 3.6 / 6.4 | – | 6.1 / 4.2 | 0.0020 |

Which comparisons are resolvable:

- **Mutual against one-way: yes**, at 10 s.e. with 2 mutual runs, in the echo and at 3 to 4 s.e. in the replica
  differences. One run per world already gives 6 s.e.
- **Mutual against antagonistic (ordering): yes**, 6 to 9 s.e.
- **One-way null: yes**, $X_2=0$ to 0.0027, a twentieth of the mutual echo.
- **Antagonistic against one-way: marginal in the echo function.** 2.6 s.e. expected under the calls-only law, so
  about one experiment in three would reach 3 s.e.; 4.4 s.e. if the note channel is as fitted; 3.7 with the pilot's
  in-swarm gains. Reaching 3 s.e. under M1 takes 7 runs in each of the two worlds at 2 replicas. What the echo
  function does for certain is exclude an antagonistic echo below $-0.012$.
- **Antagonistic sign by kicks: yes.** 400 twins give 3 s.e. under M1 as long as the note noise is below 0.15 in
  logit, and 9 s.e. without it.
- **$c_2$ in the antagonistic world, mean response per world, $q$ between worlds, $\rho_4$: no.**

Recommended plan, all at $\hat c=-0.637$, 2 replicas, $T=20$:

| | mutual | one-way | "antagonistic" |
|---|---|---|---|
| base runs (graphs) | 2 | 5 | 5 |
| kick twins, both replays of $t_0+2$ | 100 | 20 (protocol check: exactly 0) | 400 |
| null twins (note noise) | 25 | – | 25 |

The twins cost $570\times34=19{,}400$ generations, 0.9 of a run. If the budget is 12 run-equivalents in total, drop
one one-way run. Twins are spread over graphs, both replicas and $t_0\in\{8,12\}$, one kick per analyst. The first
run should be a mutual one: it gives $m$ (which law), $q$, the mutual echo at 6 s.e., and the in-swarm regression.
An asynchronous control would cost 2 runs (mutual and one-way) and is left out; if it is added, the prediction is
that the lag-2 contrast falls from $+0.057$ to $+0.009$ and a lag-1 contrast of $+0.043$ appears ($\rho_1=-0.030$
against $-0.073$, run s.d. 0.005; part S).

## 8. Falsifiers

"s.e." is the standard error of the stated comparison from the run-to-run or twin-to-twin spread in the data. The
split (2, 5, 5) gives 0.0054 (mutual − one-way), 0.0040 (antagonistic − one-way) and 0.0027 (one-way) if the
law's noise is right.

1. **Mutual echo not positive.** $X_2({\rm mutual})-X_2({\rm one\text{-}way})$ below 3 s.e. (about $+0.016$), or the
   mutual kick echo below 3 s.e. with 100 twins, while the in-swarm gains predict $Gd>0.03$. A negative value at 3
   s.e. falsifies the reaction term whatever the gains.
2. **One-way echo not zero.** $|X_2|>3$ s.e. (0.008) in the one-way world. The kick echo there is zero by
   construction and checks the replay, not the theory.
3. **Antagonistic sign against the rule, conditional on the swarm's own couplings.** The predicted sign is that of
   $\hat G_A=(\hat u_A\hat\tau_D+\hat u_D\hat\tau_A)/2K$ of (S4), computed from the swarm: $\hat u$ from the
   first-order kick responses, $\hat v$ from the in-swarm regression with the leak terms $L_A,L_D$ (part B4
   recovers them from a simulated swarm), $d$ and $d^r$ from the beliefs. In the calls-only limit this is the sign
   of $-\hat a_+\hat a_-$.
   - Zero-shot, $\hat G_A>0$: a **negative** antagonistic echo at 3 s.e., in $X_2$ against the one-way world or in
     the kick echo, falsifies the rule.
   - If the swarm shows $\hat a_->0$ (faded-tie readers move away from the flipped call, and the regression
     agrees), the rule predicts a **negative** echo, and a positive one at 3 s.e. falsifies it.
   - With the notes kept, the test needs no regression: $E$ must have the sign of $r_Ar_D$ from the same twins.
   - If $|\hat G_A d|<0.004$, no sign is predicted and the antagonistic world is a second null.
4. **Ordering.** Mutual $>$ antagonistic $\ge$ one-way $=0$ in $X_2$ and in $E$. An antagonistic echo not below
   the mutual one (difference under 3 s.e., predicted 6 to 9) contradicts $|\Gamma_A|/\Gamma_M=2a_+|a_-|/(a_+^2+
   a_-^2)$, which cannot exceed 1.
5. **Magnitudes, zero-shot.** Mutual $X_2$ outside $[0.03,0.07]$ or antagonistic $X_2$ outside $[0.005,0.021]$ by
   more than 3 s.e. means the calibration did not transfer. The theory is then tested with in-swarm numbers and
   reported as such: $X_2$ against $\hat G d$ of (S4). A mismatch above 25% plus 3 s.e. there is a failure of the
   single-site reduction.
6. **Kick and echo function disagree.** $E$ and $X_2$ of the same world differing by more than 3 s.e. plus 15%:
   the response kernel read from spontaneous calls is not the one an intervention measures.
7. **Asynchronous control, if run.** A lag-2 contrast above half the synchronous one under asynchronous updates.

Not falsifiers, but recorded: the mean belief (M1 $0.00\pm0.04$, M2 $+0.35\pm0.15$; above $|m|=0.5$ the echoes are
down by about 30% and the constant should be reconsidered before the remaining runs); the composition ratio
$E/(Kr_Ar_D)$ (1 for calls only, about 5 with the fitted note channel); and the refusal and missing rates, since a
missing decision is a missing belief.

## 9. What this tests, and what is not verified

**What it tests.** The reaction term's sign rule given the measured couplings: an analyst's own call comes back
through reciprocated ties with the sign of the product of the two couplings, and not at all without reciprocity.
With these couplings all three worlds have non-negative echoes, so the test is "large, zero, small" and not
"plus, zero, minus". It does not test whether agents act on distrust. Qwen follows its faded colleagues in every
wording, and the "antagonistic" graph is antagonistic in its labels only. The notes are still written from the
biased belief: under the calls-only law three quarters of the BUY notes come from analysts whose own probability
of BUY is below one half, and a reader may respond to that. Section 5 and the M2 columns give the size of that
effect if the calibration's leak fit is taken at face value.

**Not verified.**

- That the bias is the additive constant the probes measured once the swarm is balanced. The probes' inboxes were
  balanced by design, which is where the de-biased swarm sits and the pilot did not; no swarm has run there.
- The note channel. M2 is weakly identified (section 2), and the pilot cannot settle it. Its consequences for the
  mean (section 5) are an extrapolation from a pool whose writers were bearish for both calls (mean raw
  belief $-0.52$ for BUY notes, $-0.78$ for SELL notes).
- That a note written for a flipped call carries the same raw belief as the original note. The kick replay
  assumes it.
- Note noise in twins (section 6): no number exists; the null twins measure it.
- Transfer of the gains to the swarm. The pilot's in-swarm fit, at saturated beliefs, had $a_+$ 24% below and
  $a_-$ 1.9 times its probe values; that case is the lower end of the mutual interval.
- The residual. Its whiteness is from the pilot, in the mild wording; what share replicas have in common is
  unknown and moves $c_2$ by a factor 1.5.
- Memory beyond the last call. $X_2=0$ in the one-way world holds for an own-call weight; a dependence on the
  analyst's own earlier notes would show as falsifier 2.
- The single-site solver has no note channel. For M2 the checks are the leading order (S5) and graph simulations.
- Averaging within groups enters only as per-reader gains (one variant); the theory of per-agent gain spread is
  not worked out.
- The asynchronous numbers are simulations of the calls-only law on 24 runs per world.

## Amendment 1 (2026-11-25): the asynchronous control

Added at the user's request after the synchronous runs of `experiments/09-analyst-signrule` had started, and frozen
before any asynchronous run. I did not read the synchronous data: every number below comes from the calibration
cell, as in sections 1 to 8. It replaces the last two sentences of section 7 and falsifier 7 of section 8; nothing
else in the note changes. Numbers: `theory/checks/analyst_signrule_async_predictions.json`. Code:
`theory/checks/analyst_signrule_async_check.py` (output `analyst_signrule_async_out.txt`).

**Verdict.**

1. *Schedule.* Block-random updates with 16 blocks of 16 analysts, a fresh random partition every round. It keeps
   the engine batched (16 batches of 32 prompts per round at 2 replicas) and differs from one-at-a-time updating by
   3 points in where the echo goes.
2. *Prediction.* The echo moves to lag 1. Of the synchronous lag-2 echo, 72% reappears at lag 1 and 28% stays at
   lag 2. Mutual minus one-way: lag-1 contrast $D_1=+0.041$ (calls-only) or $+0.037$ (note channel), lag-2 contrast
   $D_2=+0.012$ or $+0.010$, against a synchronous $D_2$ of $+0.0575$ or $+0.054$. The predicted ratio of the
   asynchronous to the synchronous $D_2$ is 0.21 (calibration s.e. 0.02) for the calls-only law and 0.19 for the
   note channel, against the falsifier's threshold of 0.5.
3. *Sizing.* 2 asynchronous mutual and 2 asynchronous one-way runs put falsifier 7 at 2.5 standard errors and the
   lag-1 check at 6.7. Reaching 3 takes 3.3 runs per world, so 4 and 4, or 2 and 2 with 4 replicas each, which
   is the same number of calls. The run-to-run s.d. of 0.006 is right; the comparison is a difference of two worlds
   set against a threshold that has its own error.
4. *An asynchronous antagonistic run adds nothing resolvable*: its lag-1 contrast is $+0.007$ to $+0.010$, 1.2 to 1.6
   standard errors with 2 runs per world.

### A1. Schedule

Round 0 is synchronous, since there is no inbox. In every round $t\ge1$:

1. A fresh random permutation of the 256 analysts is cut into $B=16$ blocks of 16. The permutation is drawn from a
   stored seed per (graph, round) and shared by the replicas of a graph, as the notes and the news are.
2. The blocks update one after another. All analysts of a block decide in one engine batch, their calls are drawn
   from $\tanh(x-\hat c)$ with the stored uniforms $U(\text{replica},t,i)$, and they write their notes, before the
   next block starts.
3. An analyst reads each colleague's **latest** call and note: the round-$t$ ones if that colleague's block came
   earlier in this round, the round-$(t-1)$ ones otherwise. Its own previous call and note are its round-$(t-1)$
   ones and its news item is $\xi_i(t)$, as in the synchronous design.
4. Every analyst updates exactly once per round, so rounds, news, uniforms and the echo function's lags keep their
   meaning. The block index of every (round, analyst) is stored with the record.

*Feasibility.* `envs/analyst.AnalystSwarm._round` builds the contexts of all analysts from the previous round's
calls and notes and sends one engine batch. The block schedule calls the same code on 16 analysts at a time and
writes their calls and notes into that state before the next block; an analyst's own entry in the state is still
its previous-round one when it updates, so the context code is unchanged. A round is then 16 batches of 32 prompts
(2 replicas in lockstep) for decisions and 16 for notes, in place of one batch of 512 each. The number of prompts
is the same, so the cost should be close to a synchronous round's, which was 50 s per replica in the pilot; I have
not timed it. Checkpointing has to treat a round as complete only when its 16 blocks are. One analyst at a time
($B=256$) would be 256 batches of 2 per round and would give up the batching the pilot's throughput came from; it
buys a lag-2 share of 0.25 in place of 0.28, which is not worth it.

Statistics are those of section 2 and nothing else: the echo function on the sampling belief, rounds 5 to 20, lags
counted in rounds. Replica-difference correlations are not used for this control, because sequential updates give
them a schedule artifact (`analyst_scenario.md` section 4.3).

### A2. Where the echo goes

Let $p=(N/B-1)/(N-1)$ be the probability that two given analysts share a block; $p=15/255=0.059$ here, 0 for one
at a time, 1 for synchronous rounds. Take a reciprocated pair $i,j$ and the call $\sigma_i(t)$.

- $j$ reads $\sigma_i(t)$ in round $t$ if its block comes after $i$'s, and again or for the first time in round
  $t+1$ if its block does not come after $i$'s there.
- $i$ reads $j$'s answer at round $t+1$ in two ways: $j$ answered in round $t$ and $i$ is not after $j$ in round
  $t+1$, or $j$ is strictly before $i$ in round $t+1$. These exclude each other, and their probabilities add to
  $w_1$.
- $i$ reads $j$'s round-$(t+1)$ answer a second time in round $t+2$ if $j$ was not after $i$ in round $t+1$ and
  $i$ is not after $j$ in round $t+2$, with probability $w_2$.

$$w_1=\frac{(1-p)(3+p)}4,\qquad w_2=\Big(\frac{1+p}2\Big)^2,\qquad w_1+w_2=1, \tag{S6}$$

$$\rho_1=\kappa d+w_1Gd,\qquad X_2=w_2\,Gd+O(G^2),\qquad
D_1=w_1Gd,\qquad D_2\simeq w_2Gd+w_1Gd\,(2\kappa d+w_1Gd), \tag{S7}$$

where $D_\tau=\rho_\tau({\rm mutual})-\rho_\tau(\text{one-way})$ under the same schedule and $G$ is that of (S4).
The last term of $D_2$ is the difference of $\rho_1^2$ between the worlds: the lag-1 echo cancels part of the
anti-persistent own call ($\kappa<0$) in the mutual world, which lowers its $\rho_2$. For $B=16$: $w_1=0.720$,
$w_2=0.280$; for one at a time $0.75$ and $0.25$; for synchronous rounds 0 and 1.

Checks (parts 1, 2). The block simulation with $B=1$ equals the synchronous one exactly and with $B=N$ equals
`analyst_lib.simulate(mode="async")` on the same permutations (0 calls differ). Without memory, the simulated $D_1$
follows $w_1Gd$ within 1 s.e. for $B=1,2,4,8,16,32,256$ ($+0.0417$ against $+0.0419$ at $B=16$), and $D_2$ exceeds
$w_2Gd$ by 0.002 to 0.004 at every $B>1$ ($+0.0203$ against $+0.0163$ at $B=16$), of which $(w_1Gd)^2=0.0018$ is
the lag-1 echo taken twice. With the cell's memory, (S7) gives $D_1=+0.0415$ and $D_2=+0.0117$ at $B=16$ against
$+0.0407$ and $+0.0120$ simulated.

### A3. Predictions for B = 16

Design size, rounds 5 to 20, 300 runs per entry on the graphs, notes, news and uniforms of section 4 (Monte Carlo
s.e. 0.0004 per world, 0.0005 per contrast). Brackets: calibration s.e. from 200 bootstrap draws (part 4).

| | law | mutual | one-way | "antagonistic" |
|---|---|---|---|---|
| $\rho_1$ | M1 | −0.0330 | −0.0737 | −0.0667 |
| | M2 | −0.0272 | −0.0643 | −0.0547 |
| $\rho_2$ | M1 | +0.0179 | +0.0059 | +0.0080 |
| | M2 | +0.0166 | +0.0061 | +0.0084 |
| $X_2$ | M1 | +0.0168 | +0.0004 | +0.0036 |
| | M2 | +0.0158 | +0.0019 | +0.0054 |
| $\rho_3$ | M1 | −0.0001 | +0.0003 | −0.0003 |
| **lag-1 contrast $D_1$** | M1 | **+0.0407 (0.0019)** | – | +0.0070 (0.0020) |
| | M2 | +0.0370 (0.0065) | – | +0.0096 (0.0017) |
| **lag-2 contrast $D_2$** | M1 | **+0.0120 (0.0012)** | – | +0.0022 (0.0006) |
| | M2 | +0.0104 (0.0021) | – | +0.0023 (0.0009) |
| contrast in $X_2$ | M1 | +0.0164 (0.0008) | – | +0.0032 (0.0007) |
| | M2 | +0.0138 (0.0029) | – | +0.0034 (0.0010) |
| $D_2$ asynchronous / synchronous | M1 | 0.21 (0.02; 95% of draws 0.19 to 0.25) | | |
| | M2 | 0.19 (0.02; 95% of draws 0.16 to 0.23) | | |

- The synchronous contrasts on the same graphs are $D_2=+0.0575$ (M1) and $+0.0538$ (M2) with $D_1=+0.001$ and
  $+0.002$: the sign of $D_1-D_2$ turns from $-0.057$ to $+0.029$.
- One analyst at a time gives $D_1=+0.0420$, $D_2=+0.0099$ (M1) and $+0.0386$, $+0.0091$ (M2), so the block
  schedule is within 0.002 of it.
- The synchronous $\rho_3=-0.012$ of the mutual world, a product of memory and echo, goes to zero.
- Under M2 the mean belief is $+0.41$, $+0.37$, $+0.34$, a little above the synchronous $+0.37$, $+0.35$, $+0.32$,
  and has the same calibration uncertainty. Under M1 it stays at 0.
- The law variants of section 4 scale $G$, so they move $D_1$ and $D_2$ in proportion to the synchronous contrast
  and leave the ratio near its value here; that is why the falsifier is a ratio.

### A4. Sizing

Run-to-run s.d. of one asynchronous run (2 replicas): $\rho_1$ 0.0060 (mutual), 0.0061 (one-way); $\rho_2$ 0.0061,
0.0064. With 4 replicas they are 0.0041 to 0.0045, so a replica is again worth a graph. The test statistic of
falsifier 7 is

$$T_7=\tfrac12D_2^{\rm sync}-D_2^{\rm async},\qquad
{\rm s.e.}(T_7)^2=\tfrac14\,{\rm s.e.}(D_2^{\rm sync})^2+\frac{\sigma_{m}^2+\sigma_{o}^2}{n},$$

with $D_2^{\rm sync}$ from the experiment's own synchronous runs (2 mutual, 5 one-way: s.e. 0.0054), $\sigma_m$,
$\sigma_o$ the asynchronous run s.d. of $\rho_2$ in the two worlds and $n$ asynchronous runs per world. Predicted
$T_7=+0.0167$ (M1), $+0.0164$ (M2).

| asynchronous runs per world (2 replicas) | $T_7$ in s.e., M1 / M2 | lag-1 check $D_1$ in s.e., M1 / M2 | an echo that did not move: s.e. past the threshold |
|---|---|---|---|
| 1 | 1.8 / 1.9 | 4.8 / 4.5 | 3.1 / 3.0 |
| 2 | 2.5 / 2.5 | 6.7 / 6.3 | 4.2 / 4.1 |
| 3 | 2.9 / 2.9 | 8.2 / 7.8 | 5.0 / 4.8 |
| **4** | **3.2 / 3.2** | **9.5 / 9.0** | **5.5 / 5.3** |

- **The coordinator's 2 + 2 gives 2.5 s.e., not 3.** The s.d. of 0.006 is right for one world. The contrast of two
  worlds has s.d. 0.0088 per pair of runs, half the synchronous contrast carries an error of 0.0027 of its own, and
  the gap to resolve is 0.017, not 0.029. Three standard errors need $n=3.3$: **4 mutual and 4 one-way asynchronous
  runs at 2 replicas, or 2 and 2 at 4 replicas.**
- With 2 + 2 the control still does most of its work: the lag-1 contrast is at 6.7 s.e., and a lag-2 echo that had
  not moved at all would be 4.2 s.e. on the wrong side of the threshold. What 2 + 2 cannot do is separate the
  predicted ratio of 0.21 from 0.5 at 3 s.e.
- The requirement depends on the synchronous contrast actually measured, because the gap is 0.29 of it. Runs per
  world for 3 s.e.: 2.0 at $D_2^{\rm sync}=0.070$, 3.3 at 0.0575, 4.8 at 0.050, 10 at 0.040, 68 at 0.030 (M1; 1.6,
  2.7, 4.0, 8.4, 66 under M2). Below about 0.04 the comparison is out of reach in practice, and the lag-1 check is
  then the test.
- Within the asynchronous runs alone, $D_1-D_2=+0.029$ against $-0.057$ synchronous, with s.d. 0.0117 per pair of
  runs: 2.4 s.e. with 1 run per world and 3.5 with 2 (3.3 under M2). It needs no synchronous threshold and is
  recorded as a secondary statistic.
- **Antagonistic.** $D_1=+0.0070$ (M1) to $+0.0096$ (M2) and $D_2=+0.002$, against 0.0084 per pair of runs: 1.2 to 1.6
  s.e. with 2 runs per world and 1.9 to 2.6 with 5. Not included.

### A5. Falsifier 7, restated

Asynchronous runs: block-random schedule of A1, $\hat c=-0.637$, cell and statistics as in the synchronous runs,
$n$ mutual and $n$ one-way graphs. $D_\tau=\rho_\tau({\rm mutual})-\rho_\tau(\text{one-way})$, world means over
runs, s.e. from the run-to-run spread.

7. **The lag-2 echo survives asynchronous updating.** $D_2^{\rm async}>\tfrac12D_2^{\rm sync}$, that is $T_7<0$, by
   more than 3 s.e.$(T_7)$. Predicted: $D_2^{\rm async}/D_2^{\rm sync}=0.21$ (M1), 0.19 (M2), $T_7=+0.017$.
   - *Confirmed* if $T_7>3$ s.e.; *falsified* if $T_7<-3$ s.e.; otherwise *not resolved*, to be reported as such
     with the ratio and its error.
   - *Positive check, same runs:* $D_1^{\rm async}>0$ at 3 s.e., predicted $+0.041$ (M1), $+0.037$ (M2), which is
     0.7 of the synchronous $D_2$. $D_1^{\rm async}$ within 3 s.e. of zero while the synchronous $D_2$ exceeds
     $0.03$ falsifies the reaction term's account of where the echo goes, whatever $T_7$ is.
   - *Magnitude, reported and not tested:* $D_1^{\rm async}/D_2^{\rm sync}$, predicted 0.71 (M1) and 0.69 (M2)
     against $w_1=0.72$ at leading order, whatever the gains. Four runs per world measure it to about 15%.

**Not verified in this amendment.** The GPU cost of the block schedule (estimated, not timed). That Qwen's response
to an inbox that mixes this round's and last round's notes is the one the probes measured; the prompt says "latest
calls" and does not mark which are new. The single-site theory of the block schedule is (S6)–(S7) at leading order
plus simulation; the solver has no asynchronous mode. The asynchronous numbers under M2 rest on the same weakly
identified note channel as section 5.
