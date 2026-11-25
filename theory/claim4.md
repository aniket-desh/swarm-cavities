# Claim 4: zealots, the saddle-node, and what reciprocity does to it

**Verdict.** The existence of a critical zealot fraction with hysteresis is already published for LLM populations on
a fully connected board (De Marzo et al. 2026), and it requires a sigmoidal single-agent conformity curve; a
voter-like agent has no threshold. The new content is the dependence on reciprocity, and here the spec has the sign
of the main effect wrong. Zealots do not respond, so the zealot–benign links carry no echo. The echo that exists is
between benign agents, and it acts on each of them as a positive self-coupling that reinforces whatever the agent
currently believes. Consequently:

- $\gamma>0$ **raises** the forward threshold $\rho_c^{\uparrow}$ (capture is slightly harder), by a small amount;
- $\gamma>0$ **lowers** the backward threshold $\rho_c^{\downarrow}$ (recovery after capture is harder), by a larger
  amount, so the hysteresis loop widens substantially;
- coupling disorder $g$ by itself, at $\gamma\le0$, lowers $\rho_c^\uparrow$ and narrows the loop, and enough of it
  removes the saddle-node (no bistable window at $g=0.45$, $\gamma=-0.5$).

In the checked example ($g=0.3$), going from $\gamma=-0.5$ to $\gamma=1$ moves $\rho_c^\uparrow$ from 0.204 to 0.228
and widens the loop from 0.010 to 0.103. The spec's "the echo amplifies the zealot field so that $\rho_c$ drops
dramatically" is false; its hysteresis statement (the benign cluster stays trapped after the zealots are removed) is
the part that survives, and reciprocity strengthens it. All of this needs pair-specific couplings: on a plain
broadcast board $g=0$ and $\gamma$ has no effect at all.

Sections 1–3 are derived and checked against simulation; section 4 is derived with one part unchecked; section 5 is
checked.

## 1. Model

Benign agents (fraction $1-\rho$) and zealots (fraction $\rho$) frozen at $s_Z=-1$, so $R_Z=0$ and
$C_Z(t,t')=1$. Couplings as in `claim1.md` eq. (2) with the same $\bar J$, $g$, $\gamma$ in every block. Benign
agents follow $s(t+1)=\phi(\beta h(t)+h^{\rm priv})$ with a conformity curve $\phi$ ($\tanh$ in the checks) and
$h^{\rm priv}\sim\mathcal N(\mu,\sigma^2)$, $\mu>0$ pointing to the truth. From `claim1.md` eqs. (4)–(6) the field on
a benign agent is

$$h(t)=\bar J\big[(1-\rho)\,m(t)-\rho\big]+\eta(t)+\Gamma\sum_{t'<t}R(t,t')\,s(t'),\qquad
\langle\eta(t)\eta(t')\rangle=g^2\big[(1-\rho)\,C(t,t')+\rho\big],\qquad\Gamma=(1-\rho)\,\gamma\,g^2. \tag{1}$$

Three features are worth stating in words. The zealots contribute a constant mean field $-\bar J\rho$ and a
*quenched* random field of variance $\rho g^2$ (this is where the full correlation in the noise matters: the
connected correlation of a zealot is zero). They contribute nothing to the reaction term, whose coefficient carries
the factor $1-\rho$. And $m$, $C$, $R$ are those of the benign population alone.

## 2. Fixed points and the saddle-node condition

At a stable fixed point the kernels are time-independent and (1) becomes a static problem for one benign agent:

$$s=\phi\Big(\beta\big[\bar J((1-\rho)m-\rho)+g\sqrt Q\,z+\Gamma\chi\,s\big]+h^{\rm priv}\Big),\qquad z\sim\mathcal N(0,1), \tag{2}$$

$$m=\langle s\rangle,\qquad q=\langle s^2\rangle,\qquad Q=(1-\rho)\,q+\rho, \tag{3}$$

$$\chi=\Big\langle\frac{d}{1-d\,\Gamma\chi}\Big\rangle,\qquad d=\beta\,\phi'(\cdot), \tag{4}$$

with averages over $z$ and $h^{\rm priv}$. The term $\Gamma\chi\,s$ in (2) is the reaction term at a fixed point: the
agent's own state, fed back through its benign neighbours with their static susceptibility $\chi$. Equation (2) is
implicit in $s$; it has a unique solution per agent as long as $\beta\Gamma\chi\,\phi'<1$, which holds throughout the
checks. The solution is valid where the fixed point is stable against the bulk instability of `claim2.md`,
$\Lambda=(1-\rho)g^2\langle\chi_i^2\rangle<1$ ($\Lambda\le0.30$ in the checks).

For fixed parameters, (2)–(4) have one or three solutions $m$. Holding $m$ fixed inside the field, solving (3)–(4)
for $(q,\chi)$, and writing the result as $\langle s\rangle=f(m;\rho)$, the solutions are the roots of $f(m;\rho)=m$ and
the two critical fractions are where a pair of roots merges:

$$f(m_c;\rho_c)=m_c,\qquad\frac{\partial f}{\partial m}(m_c;\rho_c)=1. \tag{5}$$

$\rho_c^\uparrow$ is where the benign-majority root ($m>0$) disappears as $\rho$ is raised, and $\rho_c^\downarrow$
where the captured root ($m<0$) disappears as $\rho$ is lowered; $\rho_c^\downarrow=0$ means the captured state
survives the removal of every zealot. The hysteresis width is $\rho_c^\uparrow-\rho_c^\downarrow$.

**Without pair-specific couplings** ($g=0$) this is a random-field Curie–Weiss model,
$m=\langle\phi(\beta\bar J((1-\rho)m-\rho)+h^{\rm priv})\rangle$, and (5) reads

$$\beta\bar J\,(1-\rho_c)\,\langle\phi'\rangle=1\quad\Longleftrightarrow\quad\frac{\rho_c}{1-\rho_c}=\beta\bar J\langle\phi'\rangle-1. \tag{6}$$

For $\phi=\tanh$, $\bar J=1$ and no heterogeneity this is De Marzo et al.'s
$z_c=\beta\,\mathrm{sech}^2[\beta(m_c+h)]-1$, with $z=\rho/(1-\rho)$ the ratio of stubborn to ordinary agents. So their
result is the $g=0$ limit of (2)–(5).

## 3. The effect of $g$ and $\gamma$

**First order in $g^2$.** Expanding (2) around the $g=0$ solution with argument $x$,
$\langle s\rangle=\langle\phi(x)\rangle+g^2\,\delta f$, with

$$\delta f=\beta^2\Big[\tfrac12Q\,\langle\phi''\rangle+(1-\rho)\,\gamma\,\langle\phi'\rangle\,\langle\phi\,\phi'\rangle\Big],
\qquad\delta\rho_c=g^2\,\frac{(1-\rho_c)\,\delta f}{1+m_c}, \tag{7}$$

both evaluated at the $g=0$ saddle-node (the second formula follows from (5) and
$\partial_\rho f=-(1+m_c)/(1-\rho_c)$ at that point). The two terms are the two things pair-specific couplings do:

- *Noise term.* The random field $g\sqrt Qz$ smears the argument of the conformity curve. On the benign-majority
  branch the curve is concave ($\langle\phi''\rangle<0$), so the mean drops and $\rho_c^\uparrow$ decreases. On the
  captured branch it is convex and $\rho_c^\downarrow$ increases. Disorder alone narrows the loop from both sides,
  and enough of it removes the saddle-node altogether.
- *Reaction term.* Its sign is that of $\gamma\langle\phi\phi'\rangle$, i.e. of $\gamma\,m$. For $\gamma>0$ it pushes
  each branch further from zero: $\rho_c^\uparrow$ increases, $\rho_c^\downarrow$ decreases, and the loop widens from
  both sides. For $\gamma<0$ it does the opposite.

So $\partial\rho_c^\uparrow/\partial\gamma>0$ and $\partial(\text{width})/\partial\gamma>0$ whenever a saddle-node exists,
which confirms the inference in `docs/literature.md` (d). The net effect of $g$ at fixed $\gamma$ has no fixed sign: for
$\phi=\tanh$ and no heterogeneity, $\delta f\propto\phi\phi'\,[(1-\rho)\gamma(1-\phi^2)-Q]$, which at $\gamma=1$ changes
sign near $\phi^2\approx\tfrac12$.

**Check.** `theory/checks/check_claim4_zealots.py` (solver: `static_cavity.saddle_nodes`), $\beta=1.5$, $\bar J=1$,
$\mu=0.3$, $\sigma=0.2$. The simulation sweeps $\rho$ up and then down in steps of 0.005 at $N=3000$, 150 rounds
per step, two samples (`out_claim4_zealots.txt`):

| $g$ | $\gamma$ | $\rho_c^\uparrow$ cavity | first order (7) | simulation | $\rho_c^\downarrow$ cavity | simulation | width cavity | simulation |
|---|---|---|---|---|---|---|---|---|
| 0 | – | 0.2202 | 0.2202 | 0.2200 | 0.1539 | 0.1525 | 0.066 | 0.068 |
| 0.3 | −0.5 | 0.2039 | 0.1979 | 0.2050 | 0.1938 | 0.1900 | 0.010 | 0.015 |
| 0.3 | 0 | 0.2108 | 0.2087 | 0.2100 | 0.1792 | 0.1725 | 0.032 | 0.038 |
| 0.3 | 0.5 | 0.2189 | 0.2196 | 0.2200 | 0.1567 | 0.1475 | 0.062 | 0.073 |
| 0.3 | 1.0 | 0.2278 | 0.2304 | 0.2300 | 0.1244 | 0.1125 | 0.103 | 0.118 |
| 0.45 | −0.5 | no bistable window | 0.1700 | crossover at 0.200 | – | – | 0 | 0 |
| 0.45 | 0 | 0.2023 | 0.1944 | 0.2025 | 0.1966 | 0.1925 | 0.006 | 0.010 |
| 0.45 | 0.5 | 0.2134 | 0.2188 | 0.2150 | 0.1708 | 0.1625 | 0.043 | 0.053 |
| 0.45 | 1.0 | 0.2261 | 0.2433 | 0.2275 | 0.1220 | 0.1050 | 0.104 | 0.123 |

The forward thresholds agree with the simulation to within the sweep step. The simulated backward thresholds are
lower than the cavity values by 0.004–0.012, the direction expected from critical slowing down at 150 rounds per
step (a sweep overshoots a saddle-node it approaches), so the simulated widths are upper bounds. Reading the table
at $g=0.3$: reciprocity changes the forward threshold by about $+0.016$ per unit of $\gamma$ (a 7% relative change
across the whole range), and the backward threshold by about $-0.046$ per unit, three times as much. The loop at
$\gamma=-0.5$ has almost closed, and at $\gamma=1$ it is 60% wider than without any pair-specific coupling. At
$g=0.45$ the same pattern is stronger: at $\gamma=-0.5$ the saddle-node is gone and the benign mean crosses zero
continuously (the simulation finds no hysteresis), at $\gamma=0$ the loop is 0.006 wide, and at $\gamma=1$ it is 0.104.
The first-order formula (7) is accurate to 0.006 at $g=0.3$ and overshoots the $\gamma$ dependence at $g=0.45$, as
expected of a $g^2$ expansion. In units of $g^2$ the slopes per unit $\gamma$ are $+0.12$ to $+0.18$ for the forward
threshold and $-0.37$ to $-0.51$ for the backward one.

The effect scales as $g^2$. For LLM agents reading $K$ messages, `claim1.md` eq. (10) gives
$g^2\approx\beta_{\rm resp}^2\langle w^2\rangle/K$, so the shift of the thresholds with $\gamma$ is of relative order
$\beta_{\rm resp}^2/K$ and will be visible only for conformist models on small inboxes.

## 4. Dependence on the shape of the conformity curve

A saddle-node needs the mean-field map to have slope above one somewhere,
$\max_m\partial f/\partial m>1$; at $g=0$ this is $\beta\bar J(1-\rho)\max\langle\phi'\rangle>1$. Three cases:

- **Sigmoidal, steep** ($\beta\bar J\,\phi'_{\max}>1$ after averaging over private evidence): threshold and hysteresis
  exist, eq. (6), and reciprocity moves them as in section 3. Heterogeneous private evidence lowers
  $\langle\phi'\rangle_{\max}$ and can remove the saddle-node, which is Okawa's (2026) rounding of the transition.
- **Sigmoidal, shallow** ($\beta\bar J\,\phi'_{\max}<1$): one solution for every $\rho$; the benign mean decreases
  continuously through zero. There is no $\rho_c$ and nothing for $\gamma$ to shift. A crossover fraction
  $\rho_{1/2}$ where $m=0$ can still be defined, and by the argument of section 3 a positive $\gamma$ steepens the
  crossover around it.
- **Voter-like** (linear copying, $\phi(x)=x$ clipped to $[-1,1]$, $\beta\bar J\le1$): the fixed point is
  $m=(\mu-\beta\bar J\rho)/(1-\beta\bar J(1-\rho))$, continuous in $\rho$, with no threshold and no hysteresis (Mobilia
  2003). The noise term in (7) vanishes because $\phi''=0$. The reaction term does not: in the linear regime (2)
  gives $s=x/(1-\beta\Gamma\chi)$ and
  $$\chi=\frac{\beta}{1-\beta\Gamma\chi}=\frac{1-\sqrt{1-4\beta^2\Gamma}}{2\beta\Gamma}, \tag{8}$$
  an enhanced gain, so the mean-field slope becomes $\bar J(1-\rho)\chi$ and exceeds one when $\gamma g^2$ is large
  enough. A sufficiently reciprocal graph can therefore *create* a threshold and hysteresis for agents that have
  none on a broadcast board. (Derived from (2)–(4); not simulated. It requires $g$ of order one.)

The practical order of work follows: measure each model's conformity curve $\langle s'\rangle(h)$ first. If its
maximal slope against the normalized mean field is below one there is no saddle-node to study on a broadcast board,
and the zealot experiment for that model should be framed as a crossover.

## 5. Finite horizon

The saddle-node is a statement about $t\to\infty$. In $T$ rounds from the initial answers, the benign mean changes
sign only if $\rho$ exceeds $\rho_c^\uparrow$ by enough to pass the bottleneck in time; the passage takes of order
$(\rho-\rho_c)^{-1/2}$ rounds. The quantity an experiment at $T=10$ measures is therefore a horizon-dependent
threshold $\rho_c(T)>\rho_c^\uparrow$, and the prediction to compare with is the trajectory $m_B(t)$ from the dynamic
two-population solver, not the static threshold.

**Check.** `theory/checks/check_claim4_transient.py`: $g=0.3$, $\gamma=0.5$ (static $\rho_c^\uparrow=0.219$), single-site
process of `claim1.md` with a frozen zealot population against simulation at $N=3000$, 6 samples
(`out_claim4_transient.txt`):

| $\rho$ | $m_B(4)$ DMFT / sim | $m_B(10)$ DMFT / sim | max. difference over $t\le10$ |
|---|---|---|---|
| 0.10 | 0.774 / 0.775 | 0.809 / 0.810 | 0.003 |
| 0.20 | 0.393 / 0.394 | 0.505 / 0.504 | 0.002 |
| 0.24 | 0.133 / 0.139 | −0.151 / −0.138 | 0.014 |
| 0.28 | −0.128 / −0.130 | −0.556 / −0.560 | 0.005 |
| 0.32 | −0.334 / −0.336 | −0.624 / −0.626 | 0.003 |

The largest error is just above the threshold, where the trajectory spends longest in the bottleneck and
finite-size fluctuations are amplified most. At $T=10$ the benign mean has turned negative for $\rho\ge0.24$ and not
for $\rho\le0.20$: the horizon-dependent threshold is about 0.23 against the static 0.219. At $N=32$ the flip is
stochastic and spread over a range of $\rho$ (the flipped fractions in `claim3.md` section 5 run from 3% at
$\rho=0.125$ to 43% at $\rho=0.22$ for $g=0$), so the measured object is a flip-probability curve, and its width in
$\rho$ is a finite-size effect that should not be read as the absence of a threshold.

**Hysteresis protocol.** The spec's "mute the zealots after round 4" is a test of whether the captured state exists
at $\rho=0$, i.e. of $\rho_c^\downarrow=0$. In the example above $\rho_c^\downarrow>0$ for every $\gamma$, so muting all
zealots would always restore the benign majority given enough rounds, while removing only part of them (leaving
$\rho>\rho_c^\downarrow$) would not. The informative protocol is a partial removal to several levels of $\rho$, with
the benign agents' context retained, compared across $\gamma$. Note that own-history memory in $F$ (a benign agent
that sees its own captured answers) acts as an additional self-coupling and lowers $\rho_c^\downarrow$ on its own;
the $\gamma$ effect has to be read as a difference between graphs at fixed memory.

## 6. Summary of what is established

| statement | status |
|---|---|
| fixed-point equations (2)–(4) and saddle-node condition (5) | derived; thresholds agree with simulation to the sweep step |
| De Marzo et al.'s $z_c$ is the $g=0$ limit (6) | derived |
| $\gamma>0$ raises $\rho_c^\uparrow$, lowers $\rho_c^\downarrow$, widens the loop; first-order formula (7) | derived; confirmed by cavity solution and simulation |
| disorder alone ($\gamma\le0$) lowers $\rho_c^\uparrow$, narrows the loop, and can remove it | derived; confirmed |
| voter-like agents: no threshold at $g=0$; reciprocity can create one (8) | derived; not simulated |
| dynamic two-population solver reproduces $m_B(t)$ with zealots | checked, errors 0.002–0.014 |
| spec: "echo amplifies the zealot field, $\rho_c$ drops dramatically at $\gamma\approx1$" | false: wrong mechanism, wrong sign, and the effect on $\rho_c^\uparrow$ is small |
| spec: hysteresis after zealot removal | holds for steep conformity curves; already known at $g=0$; strengthened by $\gamma>0$ |
