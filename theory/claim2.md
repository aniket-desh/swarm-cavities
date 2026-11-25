# Claim 2: stability boundary, and what lies beyond it

**Verdict.** The spec's "sharp $\gamma_c$ above which a truth-tracking swarm fractures into frozen spin-glass
polarization" is false as stated, for four separate reasons.

1. The sharp boundary that exists is the linear instability of the quiet state, already known. Under synchronous
   rounds it depends on $|\gamma|$, not $\gamma$: negative reciprocity destabilizes as much as positive reciprocity
   does. The spec's and `docs/literature.md`'s $g(1+\gamma)\chi=1$ is the continuous-time (or asynchronous) result.
2. With heterogeneous agents the boundary is set by the mean *square* of the in-swarm susceptibility,
   $g^2\langle\chi_i^2\rangle=1$, not by the mean susceptibility. Using $\langle\chi\rangle$ misplaces it by 7–27% in
   the check below.
3. Beyond the boundary the state for $0<\gamma<1$ is not frozen. In the noiseless toy model it is chaotic with a
   strong period-2 component, and its correlation time grows smoothly and steeply with $\gamma$ (from 14 rounds at
   $\gamma=0.3$ to about 570 at $\gamma=0.8$). Exact freezing appears at $\gamma=1$, and at finite $N$ for
   $\gamma\gtrsim0.9$, and what freezes is a 2-cycle in which most agents flicker, not a static polarized pattern.
4. None of the freezing time scales (hundreds of rounds at $N\approx125$) is reachable in $T\approx10$ rounds. At
   $T=10$ and $N=64$ every observable we tried is a smooth function of $\gamma$.

The defensible claim is: reciprocity moves the instability boundary by a factor $1+|\gamma|$ in $g\chi$, selects the
kind of oscillation beyond it (period 2 for $\gamma>0$, period 4 for $\gamma<0$), and lengthens the correlation time.
Sections 1–3 are derived and checked; section 4 is numerical observation on the toy model; section 5 proposes
observables.

## 1. Linear stability in discrete time

Take one population, zero mean coupling, and a stationary state of the swarm. A small perturbation obeys

$$\delta s_i(\omega)=R^0_i(\omega)\sum_jJ_{ij}\,\delta s_j(\omega),\qquad R^0_i(\omega)=\sum_{\tau\ge1}R^0_i(\tau)\,e^{-i\omega\tau}, \tag{1}$$

where $R^0_i(\tau)$ is the *bare* response kernel of agent $i$: its response $\tau$ rounds after a kick to its field,
with its peers held fixed. It includes the agent's own memory and excludes the echo through the network. For the mock
agent $s(t+1)=\tanh(\beta h(t)+h^{\rm priv})$ at a fixed point $s^*$, $R^0_i(\omega)=d_ie^{-i\omega}$ with
$d_i=\beta(1-s_i^{*2})$, and (1) is $\delta s(t+1)=DJ\,\delta s(t)$, $D=\mathrm{diag}(d_i)$.

**Homogeneous agents.** If $R^0_i=R^0$ for all $i$, (1) has a growing solution exactly when $1/R^0(z)$ equals an
eigenvalue of $J$ for some $|z|>1$. The eigenvalues of $J$ fill the ellipse of Sommers et al. (1988),
$\{g[(1+\gamma)\cos\theta+i(1-\gamma)\sin\theta]\}$, so

$$\text{the state is stable iff the curve }\ \omega\mapsto 1/R^0(\omega)\ \text{ stays outside the ellipse with semi-axes } g(1+\gamma),\ g(1-\gamma). \tag{2}$$

This is the form to use with a measured kernel: compute $R^0(\omega)$ from $R^0(\tau)$, $\tau=1,\dots,\tau_{\max}$, and
plot its inverse against the ellipse. For a memoryless agent the curve is the circle of radius $1/d$, and it touches
the ellipse at its long axis:

$$\boxed{\;d\,g\,(1+|\gamma|)=1\;} \qquad\text{touching at }z=\pm1\ (\gamma>0),\quad z=\pm i\ (\gamma<0),\quad\text{everywhere }(\gamma=0). \tag{3}$$

Three differences from the continuous-time rule $g(1+\gamma)\chi=1$ follow.

- *The sign of $\gamma$ does not matter for the threshold.* In continuous time or under asynchronous updates only the
  real part of an eigenvalue matters, so $\gamma<0$ stabilizes. Under synchronous rounds the modulus matters, and the
  ellipse is as long along the imaginary axis for $-\gamma$ as along the real axis for $+\gamma$.
- *For $\gamma>0$ two modes go unstable together*: $z=+1$ (a static pattern grows) and $z=-1$ (a pattern that
  alternates sign every round grows). For $\gamma<0$ the unstable modes are at $z=\pm i$, period 4.
- *Own-history memory splits the two modes.* For a linearized agent $s(t+1)=\kappa s(t)+d\,h(t)$,
  $R^0(\omega)=d/(e^{i\omega}-\kappa)$, and (2) gives the static threshold $g(1+\gamma)\,d/(1-\kappa)=1$ and the period-2
  threshold $g(1+\gamma)\,d/(1+\kappa)=1$. An agent that anchors on its own last answer ($\kappa>0$) suppresses the
  period-2 mode by the factor $(1-\kappa)/(1+\kappa)$ relative to the static one. In terms of the measured kernel the
  two numbers are the static susceptibility $\chi=\sum_\tau R^0(\tau)$ and the staggered susceptibility
  $\chi_\pi=\sum_\tau(-1)^\tau R^0(\tau)$:
  $$g(1+\gamma)\max\big(|\chi|,|\chi_\pi|\big)=1\qquad(\gamma>0,\ \text{homogeneous agents}). \tag{4}$$
  `theory/checks/check_claim2_memory.py` confirms this at $N=1500$, $\gamma=0.5$ (`out_claim2_memory.txt`): with
  $\kappa=0.3$ a small perturbation decays at $\beta=0.43$ (growth factor 0.939 per round, predicted 0.945) and grows
  as a static pattern at $\beta=0.50$ (1.048, predicted 1.050), although the period-2 criterion is only 0.58 there;
  with $\kappa=-0.3$ the same two values of $\beta$ give decay and growth of an alternating pattern.

## 2. Heterogeneous agents and quenched private fields

With private fields the agents sit at different points of their response curves and $d_i$ varies. The spectrum of
$DJ$ is no longer an ellipse, and the fixed point itself depends on $\gamma$ through the reaction term. Both effects
are captured by the cavity method.

*Fixed point.* From `claim1.md` eq. (4) with time-independent kernels, the static single-site problem is

$$s=\tanh\!\big(\beta\,[\,g\sqrt q\,z+\Gamma\chi\,s\,]+h^{\rm priv}\big),\quad z\sim\mathcal N(0,1),\qquad q=\langle s^2\rangle,\quad\Gamma=\gamma g^2, \tag{5}$$

$$\chi_i=\frac{d_i}{1-d_i\,\Gamma\,\chi},\qquad\chi=\langle\chi_i\rangle,\qquad d_i=\beta(1-s_i^2). \tag{6}$$

$\chi_i=\sum_\tau R_i(\tau)$ is the *dressed* static susceptibility of agent $i$: its response inside the swarm,
including the echo. (With a mean coupling and zealots the argument of the tanh gains a mean-field term; see
`claim4.md`.)

*Spectral edge.* For a spectral parameter $\lambda$ outside the spectrum of $DJ$, the resolvent
$G=(\lambda D^{-1}-J)^{-1}$ has diagonal $G_{ii}=1/(\lambda/d_i-\gamma g^2\bar G)$ with $\bar G=\langle G_{ii}\rangle$, and
the off-diagonal elements satisfy $N\langle|G_{ij}|^2\rangle=g^2\langle|G_{ii}|^2\rangle^2/(1-g^2\langle|G_{ii}|^2\rangle)$
(derived in `claim3.md`, section 3, by the two-site cavity). The spectrum ends where this diverges:

$$g^2\Big\langle\Big|\frac{d_i}{\lambda-\gamma g^2\,\bar G(\lambda)\,d_i}\Big|^2\Big\rangle=1,\qquad
\bar G(\lambda)=\Big\langle\frac{d_i}{\lambda-\gamma g^2\,\bar G(\lambda)\,d_i}\Big\rangle. \tag{7}$$

On the imaginary axis with $\gamma<0$ these equations are the same as on the real axis with $|\gamma|$, so the
spectral radius is reached at $\pm r^*$ for $\gamma>0$ and at $\pm ir^*$ for $\gamma<0$, with $r^*$ the largest real
solution of (7) with $\gamma\to|\gamma|$.

*Criterion.* Setting $\lambda=e^{i\omega}$, the quantity in (7) is the dressed response at frequency $\omega$, and
the boundary $r^*=1$ reads

$$\boxed{\;\Lambda\equiv\max_\omega\ g^2\big\langle|R_i(\omega)|^2\big\rangle=1\;},\qquad
R_i(\omega)=\frac{R^0_i(\omega)}{1-\gamma g^2\,\bar R(\omega)\,R^0_i(\omega)}, \tag{8}$$

with the maximum at $\omega=0,\pi$ for $\gamma>0$ (where $R_i(0)=\chi_i$ of eq. (6)) and at $\omega=\pm\pi/2$ for
$\gamma<0$. For several populations, by the same steps, the condition is that the matrix
$M_{ab}(\omega)=\langle|R_a(\omega)|^2\rangle\,\rho_b\,g_{ab}^2$ reaches spectral radius 1 (for $\gamma=0$ this is
Aljadeff et al. 2015; the multi-population form is derived, not simulated here).

Three remarks on (8).

- For homogeneous agents it reduces to (3): $g\chi=1$ with $\chi=d/(1-\gamma g^2d\chi)$ gives $g\,d\,(1+\gamma)=1$. So
  the spec's formula is right only if "$\chi$" means the bare susceptibility of an isolated agent. In terms of the
  in-swarm susceptibility the condition is $g\chi=1$ with no $\gamma$ at all.
- With heterogeneity the average is of the square. At $\gamma=0$, $\Lambda=g^2\langle d_i^2\rangle$, the classical
  result; $g(1+|\gamma|)\sqrt{\langle d^2\rangle}$ is a usable approximation for $\gamma\neq0$ (within 7% below) and
  $g(1+|\gamma|)\langle d\rangle$ is not.
- $\Lambda<1$ must be evaluated on the solution branch continuous with weak coupling. Past the boundary equations
  (5)–(6) still have solutions, with $\Lambda>1$, which describe nothing.

**Check.** `theory/checks/check_claim2_boundary.py` (solver in `static_cavity.py`) finds the fixed point of
$s(t+1)=\tanh(\beta Js(t)+h^{\rm priv})$ for $N=1200$, $g=1$, $\sigma_{\rm priv}=0.7$, three samples per cell, and
computes the eigenvalues of $DJ$. Output in `out_claim2_boundary.txt`; a selection:

| $\gamma$ | $\beta$ | radius, simulation | radius, eq. (7) | angle of top eigenvalue | $\Lambda$ | $g(1+\lvert\gamma\rvert)\langle d\rangle$ | $g(1+\lvert\gamma\rvert)\sqrt{\langle d^2\rangle}$ |
|---|---|---|---|---|---|---|---|
| −0.6 | 0.70 | 0.885 | 0.890 | 87–90° | 0.53 | 0.83 | 0.87 |
| −0.6 | 0.746 | 0.933 | 0.949 | 87–89° | 0.72 | 0.88 | 0.93 |
| 0 | 1.43 | 0.913 | 0.907 | anywhere | 0.82 | 0.77 | 0.91 |
| 0 | 1.60 | 0.988 | 0.971 | anywhere | 0.94 | 0.80 | 0.97 |
| 0.5 | 1.02 | 0.928 | 0.939 | 0° or 180° | 0.71 | 0.76 | 0.90 |
| 0.5 | 1.135 | 0.980 | 0.983 | 0° or 180° | 0.90 | 0.76 | 0.94 |
| 1 | 0.80 | 0.960 | 0.969 | 0° or 180° | 0.58 | 0.77 | 0.92 |
| 1 | 0.89 | 0.987 | 0.995 | 0° or 180° | 0.81 | 0.75 | 0.94 |

The predicted radius agrees with the simulated one to 1–2%, which is the expected size of the edge rounding at
$N=1200$. The boundary values of $\beta$ from $\Lambda=1$ are $0.786$ ($\gamma=-0.6$; the simulated radius crosses 1 at
$0.791$), $1.68$ ($\gamma=0$), $1.195$ ($\gamma=0.5$) and $0.937$ ($\gamma=1$). At those points the mean-susceptibility
formula $g(1+|\gamma|)\langle d\rangle$ evaluates to 0.93, 0.82, 0.75 and 0.73 where it should be 1. With private
fields the boundary is not symmetric in $\gamma$, because positive reciprocity pushes agents toward saturation and
lowers $d_i$ ($q=0.50$ at $\gamma=0.5$ against $0.26$ at $\gamma=-0.6$): negative reciprocity is the more
destabilizing sign here.

## 3. Using a measured kernel

For LLM agents the inputs to (8) come from the cavity probe: per-agent (or per-evidence-level) bare kernels
$R^0_i(\tau)$, $\tau=1,\dots,5$, measured by field perturbation at fixed peers. Then

1. compute $R^0_i(\omega)$ at $\omega\in\{0,\pi/2,\pi\}$;
2. solve $\bar R(\omega)=\langle R^0_i/(1-\gamma g^2\bar RR^0_i)\rangle$ by iteration from $\bar R=\langle R^0_i\rangle$;
3. report $\Lambda(\omega)=g^2\langle|R_i(\omega)|^2\rangle$ at the three frequencies.

The prediction is that the swarm leaves its quiet state when the largest of the three passes 1, into a static or
period-2 pattern if the maximum is at $0$ or $\pi$, and into a period-4 oscillation if it is at $\pi/2$. Note
that $\langle|R_i|^2\rangle$ needs per-agent responses; a population-mean kernel gives $\langle R\rangle^2$, which
underestimates $\Lambda$. The kernel must also be measured at the operating point: an agent whose belief is saturated
by strong private evidence has a small $d_i$, so the same model can be far below the boundary on an easy task and
above it on an ambiguous one.

**Asynchronous control.** With one agent updating at a time the modes at $z=-1$ and $z=\pm i$ are stable and the
boundary is the continuous-time one, $\mathrm{Re}\,\lambda_{\max}=1$, i.e. $g(1+\gamma)\chi=1$ for homogeneous agents
(Martí et al. 2018; not re-derived or simulated here). A swarm with $\gamma<0$ that oscillates under synchronous
rounds should therefore be quiet under asynchronous ones. This is a cheap discriminating test and it separates
update-schedule artifacts from anything attributable to reciprocity.

## 4. Beyond the boundary: $\gamma<1$ against $\gamma=1$

**$\gamma=1$.** For symmetric $J$ and a monotone single-agent map, synchronous dynamics has a Lyapunov function of two
consecutive states, and every attractor is a fixed point or a 2-cycle (Marcus & Westervelt 1989; Goles & Olivos 1981
for binary units; cited from memory). Chaos is impossible. Without own-history memory the 2-cycle is the generic
outcome and the fixed point the exception: the states at even and odd rounds form two populations coupled only to
each other, $a=\tanh(\beta Jb)$, $b=\tanh(\beta Ja)$, and nothing forces $a=b$.

**$\gamma<1$.** Results of `check_claim2_regimes.py`, parts A and C, for the noiseless map $s(t+1)=\tanh(\beta Js(t))$,
$\beta=2$, $g=1$, no private fields, $N=2000$, after 3000 rounds of burn-in (`out_claim2_regimes_A.txt`). $q$ is the
mean square activity, $c(\tau)$ the autocorrelation normalized by $q$, $\tau_{\rm even}$ the first even lag with
$c<1/e$:

| $\gamma$ | $c(1)$ | $c(2)$ | $c(100)$ | Lyapunov exponent | $\tau_{\rm even}$ (rounds) | attractor |
|---|---|---|---|---|---|---|
| −1 | 0.00 | −1.00 | 1.00 | −0.02 | – | exact 4-cycle, $s(t+2)=-s(t)$ |
| −0.5 | 0.00 | −0.98 | 0.07 | +0.10 | – | chaos with a period-4 component |
| 0 | 0.00 | −0.01 | 0.00 | +0.15 | – | chaos, white in time |
| 0.3 | 0.00 | 0.92 | 0.01 | +0.12 | 14 | chaos with a period-2 component |
| 0.4 | −0.02 | 0.96 | 0.02 | +0.11 | 20 | 〃 |
| 0.5 | −0.04 | 0.98 | 0.05 | +0.10 | 31 | 〃 |
| 0.6 | −0.05 | 0.99 | 0.20 | +0.09 | 57 | 〃 |
| 0.7 | −0.06 | 1.00 | 0.46 | +0.07 | 120–170 | 〃 |
| 0.8 | −0.08 | 1.00 | 0.75 | +0.04 | 540–590 | 〃 |
| 0.9 | −0.07 | 1.00 | 1.00 | −0.01 | none | exact 2-cycle reached |
| 1 | 0.04 | 1.00 | 1.00 | −0.02 | none | exact 2-cycle |

($c(1)=1-A_1/2q$ and $c(2)=1-A_2/2q$ from the printed lag displacements.) Four observations:

1. *Positive reciprocity produces flicker, not stasis.* For every $\gamma\ge0.3$ the lag-2 correlation is above 0.9
   while the lag-1 correlation is essentially zero: the swarm alternates between two nearly unrelated configurations.
   An analysis that looks only at $C(t+\tau,t)$ averaged over lags, or at $|s_i(t+1)-s_i(t)|$, would call this churn.
2. *The correlation time of the slow drift grows steeply and smoothly with $\gamma$.* Against $1-\gamma$ the local
   logarithmic slope of $\tau_{\rm even}$ rises from about 2.4 (between $\gamma=0.3$ and $0.5$) to about 3.4 (between
   $0.7$ and $0.8$). It is not a clean power law over this range and we have no derivation of its form. This is a
   numerical observation.
3. *The largest Lyapunov exponent falls steadily with $\gamma$* and changes sign between $\gamma=0.8$ and $0.9$ at $N=2000$.
4. *Exact freezing at $\gamma<1$ is a finite-size effect whose time scale is far beyond any experiment.* Part C
   (`out_claim2_regimes_C.txt`) records the number of rounds until $\max_i|s_i(t+2)-s_i(t)|<10^{-6}$ (median of 4–8
   samples; "–" means not reached in $2\times10^4$ rounds, or 8000 at $N=2000$):

   | $\gamma$ | $N=125$ | 250 | 500 | 1000 | 2000 |
   |---|---|---|---|---|---|
   | 0.7 | 400 | 1265 | – | – | – |
   | 0.8 | 340 | 795 | – | 17305 | – |
   | 0.9 | 245 | 365 | 910 | 2130 | 6420 |
   | 0.95 | 280 | 580 | 660 | 1585 | 1885 |
   | 1.0 | 250 | 340 | 610 | 780 | 2320 |

   For $\gamma\le0.8$ the freezing time blows up with $N$, consistent with chaos at $N=\infty$. For $\gamma\ge0.9$ it
   grows roughly as a power of $N$ (exponent about 1.2 at $\gamma=0.9$, about 0.8 at $\gamma=1$) and we cannot tell
   from these sizes whether a noiseless $N=\infty$ swarm at $\gamma=0.9$ freezes. The zero-temperature literature on
   asymmetric spin glasses reports a change of attractor type at intermediate symmetry for binary units (Gutfreund,
   Reger & Young 1988; Bastolla & Parisi 1998; cited from memory, not checked), so a genuine transition of the
   *noiseless* map somewhere in $0.8<\gamma<1$ is possible. **Conjecture, not established.**

**With sampling noise**, which every LLM agent has, the classical result applies: for any $\gamma<1$ there is no
frozen phase at $N=\infty$ (Crisanti & Sompolinsky 1987, 1988), and replicas of the same swarm decorrelate on the
correlation time above. So the honest statement of what happens as $\gamma\to1$ is a rapidly growing correlation
time of a flickering state, with exact 2-cycles at $\gamma=1$, and at finite $N$ a band below $\gamma=1$ where
noiseless swarms also lock. No experiment at $T\approx10$ distinguishes these cases, as section 5 shows.

**Truth-tracking and polarization.** The spec's first phase (consensus on the truth) needs a positive mean coupling
or aligned private fields, and belongs to the mean-field sector treated in `claim4.md`. Its third phase requires
$m\approx0$ with a frozen pattern. With quenched private fields $C(t+\tau,t)$ never decays to zero in any phase, and
with a positive mean coupling the instability at $\Lambda=1$ competes with the ferromagnetic one. "Polarization" in
the cited LLM papers comes from opposed evidence or zealots, which is a random-field effect, not this instability.

## 5. Observables for $T\approx10$, $N\le128$

All use replicas: runs that share the graph, the private evidence and the initial answers, and differ in the
agents' sampling. LLM sampling cannot be seeded, so "identical initial conditions plus a small perturbation" is not
available; the agents' own noise is the perturbation.

| observable | definition | what it measures |
|---|---|---|
| noise floor $D_{\rm floor}(t)$ | mean squared difference between two re-queries of the same agent on the same inbox and history | single-agent sampling variance, $2\langle v^{(1)}\rangle$ |
| amplification $\mathcal A(t)$ | $D_{\alpha\beta}(t)/D_{\rm floor}(t)$, $D_{\alpha\beta}=\frac1N\sum_i(s^\alpha_i-s^\beta_i)^2$ | how much the network amplifies sampling noise |
| period index $\mathcal P(t)$ | $A_2/A_1$, $A_k=\frac1N\sum_i(s_i(t)-s_i(t-k))^2$ | $\to0$: period 2; $\approx1$: no temporal structure; $\to2$: period 4 |
| overlap excess $\Delta q(t)$ | $q_{\rm same\ graph}-q_{\rm other\ graph}$, both with the same private evidence | the part of each agent's state fixed by its couplings, beyond its own evidence and the mean field |

$\Delta q$ is the "replica overlap minus the field-induced part". In the two-replica theory (`claim1.md` eq. (9)) it
is $C^{\alpha\beta}(t,t)$ with shared couplings minus the same with independent couplings.
`analysis.metrics.overlap_excess` computes it when given the other-graph ensemble as baseline; the $g=0$ control is
the wrong baseline when $\bar J\neq0$, because it also removes the mean field.

**Amplification gives $\Lambda$ directly.** For $\gamma=0$ and small noise, the replica distance obeys
$D(t+1)=\Lambda\,D(t)+D_{\rm floor}$, so

$$\mathcal A(t)=\frac{1-\Lambda^{t}}{1-\Lambda},\qquad\hat\Lambda(t)=\frac{D_{\alpha\beta}(t+1)-D_{\rm floor}(t+1)}{D_{\alpha\beta}(t)}. \tag{9}$$

Below the boundary $\mathcal A$ saturates at $1/(1-\Lambda)$; above it $\mathcal A$ grows geometrically until the
nonlinearity stops it. For $\gamma\neq0$ the growth is not a single geometric series (the echo returns two rounds
later) and the prediction comes from the two-replica solver, but $\mathcal A(T)\gg1$ against $\mathcal A(T)\approx1$
remains the discriminator. Ten rounds resolve $\Lambda$ from 1 if $|1-\Lambda|\gtrsim0.15$. This is the relaxation-time
measurement in the form that needs no seed control.

**What the toy model gives at experiment scale.** `check_claim2_regimes.py` part B: $N=64$, $T=10$,
$s(t+1)=\tanh(\beta h+h^{\rm priv}+0.3\,\xi)$, $\sigma_{\rm priv}=0.5$, $g=1$, 400 graphs per cell
(`out_claim2_regimes_B.txt`):

| $\beta$ | $\gamma$ | $\Delta q$ (s.e. 0.001–0.009) | $\mathcal A(10)$ | $\mathcal P(10)$ |
|---|---|---|---|---|
| 0.4 | −1 … 1 | 0.013 … 0.023 | 1.11–1.13 | 1.11 … 0.89 |
| 2.0 | −1 | 0.55 | 19 | 1.95 |
| 2.0 | −0.5 | 0.40 | 15 | 1.85 |
| 2.0 | 0 | 0.24 | 10 | 1.06 |
| 2.0 | 0.5 | 0.50 | 11 | 0.13 |
| 2.0 | 0.9 | 0.51 | 15 | 0.05 |
| 2.0 | 1 | 0.51 | 16 | 0.04 |

Reading the table:

- *Below against above the boundary* is unambiguous: $\mathcal A$ is 1.1 against 10–19, $\Delta q$ is 0.02 against
  0.2–0.5. With 50 graphs per cell the standard error of $\Delta q$ is about 0.025, so this needs few graphs.
- *The sign and size of $\gamma$* show in $\mathcal P$: 1.95, 1.85, 1.06, 0.13, 0.05, 0.04. This is the cleanest
  signature of reciprocity in ten rounds.
- *$\gamma=0.5$, $0.9$ and $1$ are nearly indistinguishable.* At $N=\infty$ the first is chaotic with a 31-round
  correlation time, the last is frozen, and at $T=10$ they differ by 0.09 in $\mathcal P$ and by 0.01 in $\Delta q$.
  There is no feature in any column that could be read as a critical $\gamma_c$.
- $\Delta q>0$ on both sides of the boundary. It is not an order parameter of a phase; it measures how strongly the
  couplings pin each agent, and it grows smoothly with $g$.

So the experiment should report $\mathcal A(t)$, $\hat\Lambda$, $\mathcal P(t)$ and $\Delta q(t)$ as functions of
$\gamma$ and compare the curves with the two-replica DMFT, and should not report a phase diagram with a frozen phase.

**Existing data.** The Physics of Agents release is a $\gamma=1$ synchronous swarm without own-history memory, and it
shows the period-2 signature: the two-round change of beliefs is 0.2–0.7 of the one-round change on bipartite and
random signed graphs and about equal to it on non-bipartite lattices, and $\Delta q$ is zero at round 0 and positive
afterwards (`claim5.md` section 4, `check_poa_zero_cost.py`).

## 6. Summary of what is established

| statement | status |
|---|---|
| homogeneous boundary $d\,g(1+\lvert\gamma\rvert)=1$, modes at $\pm1$ or $\pm i$ (eq. 3) | derived; eigenvalue angles confirmed numerically |
| heterogeneous boundary $\Lambda=g^2\langle\lvert R_i\rvert^2\rangle=1$ (eqs. 7–8) | derived; radius confirmed to 1–2% at $N=1200$ |
| ellipse criterion for a measured kernel with memory (eqs. 2, 4) | derived; growth rates confirmed for a one-round memory |
| multi-population criterion via $M_{ab}(\omega)$ | derived by the same steps; not simulated |
| 2-cycles or fixed points only at $\gamma=1$ | theorem from the literature (memory) |
| correlation time growing steeply with $\gamma$; local exponent 2.4–3.4 in $1-\gamma$ | numerical observation, $N=2000$, two samples |
| finite-$N$ locking into 2-cycles for $\gamma\gtrsim0.9$ | numerical observation; $N\to\infty$ limit open |
| asynchronous boundary $g(1+\gamma)\chi=1$ | literature; not checked here |
| "sharp $\gamma_c$ into frozen spin-glass polarization" | false as stated |
