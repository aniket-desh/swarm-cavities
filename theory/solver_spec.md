# Specification of `theory/dmft_solver.py`

This file states the algorithms the solver must implement and the numbers it must reproduce. The equations are in
`claim1.md` (single-site process), `claim2.md` (stability), `claim3.md` (two-site, PR) and `claim4.md` (zealots).
Working reference implementations exist for every mode in `theory/checks/` (`toy.py::dmft_mc`, `static_cavity.py`,
`two_site.py`); they are prototypes written to check formulas, so port the algorithm and the tests, not the code
style. Every reference number below is printed by `theory/checks/reference_values.py` (`out_reference_values.txt`)
or by the check script named next to it.

## 0. Conventions

- Rounds $t=0,\dots,T$; arrays over time have length $n=T+1$. Update convention $s(t+1)=F(\dots,h(t),\dots)$.
- $J_{ij}$ = influence of $j$ on $i$, $h=Js$; populations as in `envs/graphs.py`. Block parameters `rho[a]`,
  `Jbar[a,b]`, `g[a,b]`, `gamma[a,b]` (symmetric), meaning mean $\bar J_{ab}/N$, variance $g_{ab}^2/N$,
  $\mathrm{Corr}(J_{ij},J_{ji})=\gamma_{ab}$.
- $C_a(t,t')=\langle s_a(t)s_a(t')\rangle$ is the full second moment. $R_a(t,t')=\langle\partial s_a(t)/\partial h_a(t')\rangle$
  is zero for $t'\ge t$. $\Gamma_{ab}=\rho_b\gamma_{ab}g_{ab}g_{ba}$.
- A single-site map is any callable `step(hpriv, own_history, h_t, rng) -> s_next`, vectorized over trajectories.
  The mock agent is `tanh(beta * h_t + hpriv)`; its stochastic variant returns $\pm1$ with
  $P(+1)=(1+\tanh(\beta h_t+h^{\rm priv}))/2$. Initial state $s(0)=\tanh(h^{\rm priv})$ (or its $\pm1$ sample).

## 1. Mode (a): known single-site map

### 1.1 Monte Carlo over trajectories, one causal forward pass

Keep $M$ trajectories per population. No outer fixed-point loop is needed, because everything required at round
$t$ depends on rounds $\le t$.

```
draw hpriv[a, m], s[a, 0, m];  set m_a(0), C_a(0,0)
for t = 0 .. T-1:
    for each population a:
        K_a(t, t')  = sum_b rho_b g_ab^2 C_b(t, t'),  t' <= t                      # claim1 eq. (5)
        eta_a(t)    = E[eta(t) | eta(<t)] + sqrt(Var[eta(t) | eta(<t)]) * z        # conditional Gaussian, per trajectory
        h_a(t)      = sum_b rho_b Jbar_ab m_b(t) + eta_a(t)
                      + sum_b Gamma_ab sum_{t' < t} R_b(t, t') s_a(t')             # claim1 eqs. (4), (6)
        s_a(t+1)    = step(hpriv, s_a(<= t), h_a(t), rng)                          # claim1 eq. (7)
    for each population a:
        m_a(t+1), C_a(t+1, t' <= t+1), R_a(t+1, t' <= t)                           # claim1 eq. (8)
```

Details that matter:

1. **Noise.** The conditional mean is $w^\top\eta(<t)$ with $w=K_{<t,<t}^{+}K_{<t,t}$ and the conditional variance
   $K_{tt}-K_{t,<t}w$, clipped at zero. Use a pseudo-inverse: $K$ becomes singular whenever the agents stop moving
   (frozen states, zealots), and a Cholesky update fails there.
2. **Frozen populations** (zealots): never updated, $R=0$; they still contribute to the mean field and to $K_a$
   through $C_Z=1$.
3. **Response, three estimators.** Use the first that applies.
   - *Tangent* (differentiable deterministic map): propagate $\partial s(t)/\partial h(t')$ along each trajectory,
     $$\frac{\partial s(t+1)}{\partial h(t')}=\partial_hF\cdot\Big[\delta_{tt'}+\sum_b\Gamma_{ab}\sum_{u<t}R_b(t,u)\frac{\partial s(u)}{\partial h(t')}\Big]+\sum_{u\le t}\partial_{s(u)}F\cdot\frac{\partial s(u)}{\partial h(t')},$$
     and average. Exact; no extra noise. The population kernels $R_b$ inside the bracket are held fixed.
   - *Score function* (stochastic map with a known conditional law): $R(t,t')=\langle s(t)\,\partial_{h(t')}\log P(s(t'+1)\mid\cdot)\rangle$;
     for the stochastic mock agent $R(t,t')=\beta\langle s(t)\,[s(t'+1)-\tanh x(t')]\rangle$, and the lag-1 entry is
     better taken as $\beta\langle1-\tanh^2x(t)\rangle$.
   - *White probe* (black box; section 2).

   Do **not** use the Gaussian integration-by-parts estimator $R(t,\cdot)=K^{-1}\langle\eta(\cdot)s(t)\rangle$. It is
   exact in principle, but $K$ is nearly singular whenever private fields make the states persistent: in test A3
   below its condition number is $2.5\times10^4$ and the estimate is off by up to 0.066 with $4\times10^5$
   trajectories.
4. **Two replicas / nested sampling** (needed for $\Delta q$, $\mathcal A$, and the per-agent thermal quantities
   of mode (c)). Give each of $M_{\rm out}$ "sites" $M_{\rm in}$ thermal copies that share $h^{\rm priv}$ and the
   frozen part of the noise. With $q_b(t,t')=\langle\mu_i(t)\mu_i(t')\rangle$ the cross-copy correlation
   (`claim1.md` eq. (9)), draw $\eta=\eta^{f}+\eta^{th}$ with $\eta^f$ shared across copies, covariance
   $\sum_b\rho_bg_{ab}^2q_b$, and $\eta^{th}$ independent per copy, covariance $\sum_b\rho_bg_{ab}^2(C_b-q_b)$, both
   sampled causally as in item 1. The reaction term is applied per copy. Outputs per site:
   $\mu_i(t)$, $v_i(t,t')$, $R_i(t,t')$ (averages over the copies).

### 1.2 Exact Gaussian integration (use for the gate tests)

For $\gamma=0$ and a memoryless deterministic map, $s(t+1)$ depends only on $(\eta(t),h^{\rm priv})$, so the
recursion closes on $m$ and the matrix $C$ with integrals of dimension at most three:

$$m(t+1)=\int\!Dz\,Dh\;\phi_t(z,h),\qquad C(t+1,t'+1)=\int\!Dz\,Du\,Dh\;\phi_t(z,h)\,\phi_{t'}\big(cz+\sqrt{1-c^2}\,u,\,h\big),$$

with $\phi_t(z,h)=F\big(\bar Jm(t)+g\sqrt{C(t,t)}\,z,\ h\big)$, $c=C(t,t')/\sqrt{C(t,t)C(t',t')}$, and
$C(t+1,0)=\int Dz\,Dh\,\phi_t(z,h)\,s_0(h)$. Gauss–Hermite quadrature with 60 nodes per dimension gives six digits.
The static fixed point (`claim2.md` eqs. (5)–(6), `claim4.md` eqs. (2)–(4)) is likewise a two-dimensional
quadrature with a scalar root solve for the implicit self-coupling.

### 1.3 Tests for mode (a)

**A1. Exact, $\gamma=0$, memoryless mock agent** ($\beta=1.2$, $g=1$, $\bar J=0.5$, $\mu=0.3$, $\sigma=0.5$, one
population). From section 1.2:

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $m(t)$ | 0.24353 | 0.30679 | 0.30917 | 0.30332 | 0.29962 | 0.29802 | 0.29744 |
| $C(t,t)$ | 0.21708 | 0.34969 | 0.39606 | 0.40780 | 0.40992 | 0.41001 | 0.40984 |
| $C(t,t-1)$ | – | 0.20523 | 0.32231 | 0.36965 | 0.38829 | 0.39681 | 0.40148 |
| $C(t,0)$ | 0.21708 | 0.20523 | 0.19619 | 0.19229 | 0.19093 | 0.19051 | 0.19040 |

The quadrature mode must match to $10^{-4}$. The Monte Carlo mode at $M=4\times10^5$ must match to 0.003 (the
prototype's maximum deviations are 0.0009 in $m$ and 0.0014 in $C$).

**A2. Linear map, closed forms** ($x(t+1)=a\,h(t)+\xi(t)$, unit-variance white $\xi$, $x(0)=0$, one population,
$\bar J=0$). With $x_g=(ag)^2$:

- Response: $R(t_0+2k+1,t_0)=a^{2k+1}(\gamma g^2)^k\,\mathrm{Cat}_k$ and zero at even lags. For $a=0.55$, $g=1$,
  $\gamma=0.6$: lags 1, 3, 5, 7 give 0.55000, 0.09983, 0.03624, 0.01644.
- Variance: $v(1)=1$, $v(2)=1+x_g$, $v(3)=1+x_g+x_g^2(1+\gamma^2)$. For the same parameters
  $v(1..5)=1,\ 1.30250,\ 1.42695,\ 1.49449,\ 1.53433$.
- Stationary variance: $1/(1-x_g)$ at $\gamma=0$ ($a=0.8$: 2.77778); $[1-\sqrt{1-4x_g}]/(2x_g)$ at $\gamma=1$
  ($a=0.45$: 1.39286).

Tolerance 1% for the Monte Carlo mode, $10^{-6}$ for a deterministic kernel recursion.

**A3. Response estimators** (one population with memory: $\beta=1$, $g=1$, $\gamma=0.5$, $\bar J=0.3$, $\mu=0.2$,
$\sigma=0.5$, own-history weight $\kappa=0.3$ on an exponential average with decay 0.5, $T=8$). Tangent reference:
$R(8,t')$ for $t'=3,\dots,7$ is $0.0509,\ 0.0399,\ 0.0924,\ 0.0440,\ 0.4284$. The white-probe estimator of section 2
with $\epsilon=0.3$ must reproduce these with r.m.s. error per entry of about 0.009 at $M=4\times10^5$ (dominated by
the $O(\epsilon^2)$ shift of the operating point: $R(8,7)$ reads 0.4395), 0.04 at $M=2000$, 0.08 at $M=500$.

**A4. Static fixed point at small $g$** ($\beta=1$, $g=0.2$, $\gamma=0.5$, $\mu=0.5$, $\sigma=0.3$, $\bar J=0$):
$m=0.43441$, $q=0.24839$, $\chi=0.76088$. The second-order expansion (`claim4.md` eq. (7) with $\rho=0$) gives
$m=0.43453$ and the uncoupled value is $0.43292$, so the test resolves the $g^2$ term, including its reaction part.
Tolerance $2\times10^{-4}$ on $m$. The long-time limit of the dynamic solver must agree with the static one to 1%.

**A5. Large-$N$ simulation reference, everything switched on** (two populations, asymmetric $g$, mixed-sign
$\gamma$, mean couplings, memory; parameters in `claim1.md` section 6 and `check_claim1_dmft.py`). Reference: direct
simulation, $N=4000$, 40 disorder samples, standard errors 0.0028 ($m$), 0.0016 ($C$).

| | $t=2$ | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| $m_0$ simulation | 0.3679 | 0.3743 | 0.3677 | 0.3603 | 0.3541 |
| $m_0$ prototype | 0.3640 | 0.3704 | 0.3642 | 0.3573 | 0.3518 |
| $C_0(t,t)$ simulation | 0.4763 | 0.5831 | 0.6286 | 0.6515 | 0.6645 |
| $C_0(t,t)$ prototype | 0.4737 | 0.5819 | 0.6278 | 0.6508 | 0.6640 |
| $m_1$ simulation | −0.0456 | −0.0363 | −0.0336 | −0.0358 | −0.0375 |
| $m_1$ prototype | −0.0477 | −0.0379 | −0.0351 | −0.0369 | −0.0382 |
| $C_1(t,t)$ simulation | 0.1808 | 0.2077 | 0.2203 | 0.2266 | 0.2301 |
| $C_1(t,t)$ prototype | 0.1817 | 0.2086 | 0.2213 | 0.2273 | 0.2309 |

Also required: the response to a kick at $t'=3$ in population 1 is $0.712,\ 0,\ -0.051,\ 0.005,\ 0.019$ at lags 1–5
(zero at lag 2 is the time-index test), and switching the noise covariance to the connected correlation or shifting
the reaction index must *fail* this test (errors above 0.03). The stochastic variant
(`check_claim1_dmft.py --stochastic`) has standard errors 0.005 and the prototype matches to 0.007 ($m$), 0.005
($C(t,t-2)$).

## 2. Mode (b): black-box single-site map (the hybrid LLM solver)

The forward pass of section 1.1 is unchanged; what changes is how the field reaches the agent and how $R$ is
estimated. Budget is counted in agent calls: one pass costs $M\times T$ calls per population.

**When $R$ is needed.** For $\gamma=0$ the single-site process contains no $R$: $m$ and $C$ need no response
measurement. $R$ is required for the reaction term ($\gamma\ne0$), for the stability parameter $\Lambda$, and for
mode (c).

**White-probe estimator.** Add to every trajectory's field an independent probe
$\theta(t)\sim\mathcal N(0,\epsilon^2)$, known to the solver, and estimate all entries at once from the same pass:

$$\hat R(t,t')=\frac{\langle s(t)\,\theta(t')\rangle}{\epsilon^2},\qquad\mathrm{s.d.}\big[\hat R(t,t')\big]\le\frac{\sqrt{C(t,t)}}{\epsilon\sqrt M}. \tag{S1}$$

It is unbiased to first order in $\epsilon$, needs no matrix inverse and no seed control, and its bias is the
$O(\epsilon^2)$ change of the operating point caused by the extra field variance (test A3). Choose $\epsilon$ at
about a third of the field's standard deviation. With $M=500$ each entry has s.d. about 0.1, which is too coarse
entry by entry; average entries of equal lag over $t$ (valid once the kernel is near-stationary) or fit a
two-parameter kernel (lag-1 gain and one memory decay) to bring this below 0.04. A paired finite difference
(same trajectory with and without a kick) is not available for API models, because the sampling cannot be seeded.
Do not use the regression on $\eta$ (section 1.1, item 3).

**Delivering the field.** Two routes; implement the first, keep the second for validation.

1. *Surrogate route (recommended).* Use single-agent probes to fit a parametric response surface
   $\hat F(h^{\rm priv},\text{own-history summary},h)$, including its noise law, and run mode (a) with $\hat F$. The
   probes are the calibration set of the sufficiency test (`claim1.md` section 4.1), so the two share their cost.
   The surrogate makes the solver deterministic and replayable and gives $R$ by the tangent or score estimator. Its
   error is the misfit of $\hat F$, which is measured on held-out probes. Physics of Agents' fitted logistic rule is
   the baseline form of $\hat F$.
2. *Live route.* Query the model at every step on an inbox synthesized to carry the target field. For $\gamma=0$ on
   a graph with in-degree $K$ this needs no synthesis: let each trajectory read $K$ randomly chosen other
   trajectories of the pool with random signs, fixed for the run. That is population dynamics on a random directed
   graph of $M$ agents (`claim5.md` section 1), it makes no Gaussian or scalar-field assumption, and it costs
   $M\times T$ calls. Shared ancestry in the pool is not controlled once $K^T>M$; check by doubling $M$. For
   $\gamma\ne0$ the reaction term must be added to the field, which requires translating a scalar shift into an
   inbox; that presupposes scalar sufficiency and a calibrated influence function, and is therefore no more
   assumption-free than the surrogate route.

**Tests for mode (b).** Wrap the mock agent as a black box (no access to its formula). The live route with Gaussian
field delivery must reproduce A1 to within the Monte Carlo error at the $M$ used ($\lesssim0.02$ at $M=2000$), and
the white-probe kernel must match A3 at the stated accuracy. The surrogate route, fitted on $\le600$ probes of the
mock agent with a logistic family in $(h,h^{\rm priv})$, must then reproduce A1 to 1%, which tests the whole
calibration path.

## 3. Mode (c): two-site PR predictor

Inputs: per-agent thermal responses $R_i(t,t')$ and covariances $v_i(t,t')$ for a sample of agents $i=1..M_{\rm out}$
(section 1.1, item 4), with weights; $g$, $\gamma$, and the updating fraction $\rho_B$.

1. Form the site moments $\langle R\otimes R\rangle$, $\langle R\otimes v\rangle$, $\langle v\otimes v\rangle$ as arrays with four
   time indices. When $R_i$, $v_i$ are themselves Monte Carlo estimates, compute each moment from two disjoint halves
   of the inner copies ($\hat R_i^{(1)}\otimes\hat R_i^{(2)}$, symmetrized), otherwise the estimation noise of
   $\hat R_i$ inflates the moments.
2. Iterate the linear system (`claim3.md` eqs. (5)–(12); exact index placement in
   `theory/checks/two_site.py::four_point`) from zero until unchanged; at most $n$ sweeps.
3. $\mathrm{PR}(t)/N_B=\bar v(t)^2/[\langle v_i(t)^2\rangle+\psi_B(t)+\Lambda_1(t)^2/N_B]$ with
   $\psi_B(t)=\Psi^{\Sigma\Sigma}(t,t;t,t)$ from the system solved with $G_2=\rho_Bg^2$, $\tau=\rho_B\gamma g^2$; this is
   $N_B\langle\Sigma_{ij}^2\rangle=\rho_B\psi$ in the notation of `claim3.md`, so no further factor of $\rho_B$ is applied. The last term is the mean-mode contribution (`claim3.md` eq. (16)); for a
   transient, obtain $N_B\mathrm{Var}(m_B(t))$ from the linear recursion
   $\mathrm{Var}\,\delta m(t+1)=\lambda(t)^2\mathrm{Var}\,\delta m(t)+\bar v(t+1)/N_B$ and
   $\Lambda_1(t+1)=\kappa(t)^2N_B\mathrm{Var}\,\delta m(t)$. (The stationary limit of this
   recursion is checked in C4; the transient form is derived only.)
4. For $\gamma=0$ and a memoryless map use the equal-time recursion `claim3.md` eq. (4) directly; it must agree
   with the general system.

Memory: the arrays have $n^4$ entries, 15 thousand at $T=10$. Scope: one updating population plus frozen ones. The
general multi-population system is specified in `claim3.md` section 3 but has no reference implementation.

**Tests for mode (c).**

- **C1, exact closed form.** Linear map, $\gamma=0$, $a=0.8$, $g=1$: stationary $\mathrm{PR}/N=1-(ag)^4=0.59040$,
  $\bar v=2.77778$.
- **C2, transient with reciprocity.** Linear map, $a=0.55$, $g=1$, $\gamma=0.6$, unit noise, $x(0)=0$:
  $\mathrm{PR}(t)/N$ for $t=1,\dots,5$ is $1,\ 0.94882,\ 0.90100,\ 0.86145,\ 0.83289$, and $0.76436$ at $t=11$. The same
  values must result for $\gamma=-0.6$. With 40% of the agents frozen ($\rho_B=0.6$, $a=0.75$, $\gamma=0.5$):
  $\bar v(11)=1.6811$, $\mathrm{PR}(11)/N_B=0.7429$ (exact matrix recursion: 1.6825, 0.7409). Tolerance $10^{-4}$
  (deterministic). `check_claim3_linear.py` compares these
  with exact matrix recursions at $N=1000$ (agreement 0.1–0.25% in PR/N).
- **C3, nonlinear stochastic agents, $\gamma=0$** ($\pm1$ agents, $\beta=1.2$, $g=1$, $\sigma=0.5$, no mean
  coupling): $\psi(t)$ for $t=1,2,4,8$ is $0.1053,\ 0.1286,\ 0.1370,\ 0.1382$ and $\mathrm{PR}/N$ is
  $0.8544,\ 0.8419,\ 0.8390,\ 0.8387$. Simulation reference ($N=1000$, 600 replicas, 4 samples): PR/N within 0.4%.
  Tolerance 1%.
- **C4, mean mode.** $\pm1$ agents with zealots, $g=0$, $\beta=1.5$, $\bar J=1$, $\mu=0.3$, $\sigma=0.2$, $N=128$:
  stationary $\mathrm{PR}/N_B=0.875,\ 0.897,\ 0.905$ and $N_B\mathrm{Var}(m_B)=0.14,\ 0.31,\ 0.39$ at
  $\rho=0,\ 0.102,\ 0.125$ (`check_claim3_flip.py`; simulation 0.873, 0.895, 0.903). Tolerance 1% on PR/$N_B$.

Empirical PR must be computed with `analysis.metrics.participation_ratio_corrected` and intervals with
`analysis.metrics.subsample_ci`.

## 4. Mode (d): zealot fixed points and saddle-nodes

**Static solver** (deterministic maps). Unknowns $(m,q,\chi)$ for the benign population at zealot fraction $\rho$.

1. For given $(m,q,\chi)$ evaluate the single-agent solution of `claim4.md` eq. (2) on a quadrature grid in
   $(z,h^{\rm priv})$, solving the implicit equation in $s$ by Newton from the previous iterate (this selects the
   branch continuously), then update $q$ and $\chi$ by eqs. (3)–(4) with damping 0.5 until the change is below
   $10^{-10}$.
2. *All branches at fixed $\rho$:* hold $m$ fixed in the field, solve step 1 for $(q,\chi)$, and find the roots of
   $\langle s\rangle(m)-m$ on a grid of $m\in(-1,1)$ refined by bisection. One root: no bistability. Three: the outer
   two are the stable branches.
3. *Saddle-nodes:* bisect in $\rho$ on the number of roots. $\rho_c^\uparrow$ is the upper end of the three-root
   window, $\rho_c^\downarrow$ its lower end (zero if the window includes $\rho=0$). Return `None` if there is no
   window.
4. Report $\Lambda=(1-\rho)g^2\langle\chi_i^2\rangle$ at each solution and refuse to return a fixed point with
   $\Lambda\ge1$ (`claim2.md` section 2).

**Dynamic solver** (any map, including stochastic and black-box ones). Mode (a) or (b) with a frozen population.
Forward threshold at horizon $T$: the smallest $\rho$ for which $m_B(T)<0$ starting from the benign initial
condition. Hysteresis: run to capture, then continue the same pool at a lower $\rho$ (convert zealots back to
benign agents that start at $-1$) and record the largest $\rho$ at which $m_B$ returns above zero.

**Tests for mode (d)** ($\beta=1.5$, $\bar J=1$, $\mu=0.3$ throughout):

- **D1, closed form.** $g=0$, $\sigma=0$: the saddle-node solves $\beta(1-\rho)(1-m^2)=1$ and
  $\operatorname{atanh}m=\beta[(1-\rho)m-\rho]+\mu$, giving $\rho_c^\uparrow=0.22704$, $m_c=0.37082$; the solver also
  returns $\rho_c^\downarrow=0.13234$. Tolerance $3\times10^{-4}$.
- **D2, heterogeneous Curie–Weiss.** $g=0$, $\sigma=0.2$: $\rho_c^\uparrow=0.2202$, $\rho_c^\downarrow=0.1539$.
- **D3, with disorder and reciprocity.** $\sigma=0.2$, $g=0.3$:

  | $\gamma$ | $\rho_c^\uparrow$ | $m_c$ | $\rho_c^\downarrow$ | simulation $\rho_c^\uparrow$, $\rho_c^\downarrow$ ($N=3000$, step 0.005) |
  |---|---|---|---|---|
  | −0.5 | 0.2039 | 0.223 | 0.1938 | 0.2050, 0.1900 |
  | 0 | 0.2108 | 0.275 | 0.1792 | 0.2100, 0.1725 |
  | 0.5 | 0.2189 | 0.307 | 0.1567 | 0.2200, 0.1475 |
  | 1 | 0.2278 | 0.336 | 0.1244 | 0.2300, 0.1125 |

  Tolerance 1% on $\rho_c^\uparrow$ and $\rho_c^\downarrow$ against the cavity column. The simulated backward
  thresholds sit below the cavity ones because a sweep of 150 rounds per step overshoots a saddle-node; do not tune
  the solver toward them.
- **D3b, no window.** $\sigma=0.2$, $g=0.45$, $\gamma=-0.5$: the solver must return `None` (the benign mean
  crosses zero continuously near $\rho=0.20$). At $g=0.45$, $\gamma=0,\ 0.5,\ 1$:
  $\rho_c^\uparrow=0.2023,\ 0.2134,\ 0.2261$ and $\rho_c^\downarrow=0.1966,\ 0.1708,\ 0.1220$.
- **D4, transient.** $g=0.3$, $\gamma=0.5$, $\sigma=0.2$, benign start at $\tanh(h^{\rm priv})$:
  $m_B(10)=0.809,\ 0.505,\ -0.151,\ -0.556,\ -0.624$ at $\rho=0.10,\ 0.20,\ 0.24,\ 0.28,\ 0.32$
  (`check_claim4_transient.py`; simulation at $N=3000$: $0.810,\ 0.504,\ -0.138,\ -0.560,\ -0.626$). Tolerance 0.005,
  except 0.02 at $\rho=0.24$, which lies in the bottleneck just above the threshold.

## 5. Error budget: what "<1%" means

Three error sources, with their measured sizes in the prototype:

| source | scaling | measured |
|---|---|---|
| Monte Carlo error of the solver | $c/\sqrt M$ | seed-to-seed difference at $M=4\times10^5$: 0.002 in $m$, 0.001 in $C$ (test A5); 0.001 against exact integration (A1) |
| sampling error of a simulation reference | $\sqrt{\mathrm{Var}/(N\,n_{\rm dis})}$, larger near instabilities | $N=4000$, 40 samples: 0.0028 in $m$, 0.0016 in $C$; stochastic agents 0.005 |
| finite-$N$ bias of a simulation reference | $O(1/N)$ | none detected at the 0.002 level between $N=500$, $2000$ and $4000$ in test A5 |
| quadrature | – | below $10^{-5}$ with 60 nodes |

Quantities such as $m_1\approx-0.04$ make a purely relative criterion meaningless, so the gate is defined as

- against an exact or quadrature reference (A1, A2, A4, C1, C2, D1): $|\text{solver}-\text{ref}|\le0.01\cdot\max(|\text{ref}|,0.2)$,
  which requires quadrature mode or $M\ge4\times10^5$;
- against a simulation reference (A5, C3, C4, D3, D4): $|\text{solver}-\text{sim}|\le0.01\cdot\max(|\text{sim}|,0.2)+3\,\mathrm{s.e.}$

The CLAUDE.md gate (mock Ising agent to <1%) is met by A1 in quadrature mode and A5 in Monte Carlo mode; A2, C1,
C2 and D1 are exact and cheap enough to run in the unit tests on every commit. A5 and D3 take minutes and belong in
a slower test tier.

## 6. Known limits of the specification

- Above the instability ($\Lambda>1$) the forward pass still runs and its transient is correct at $N=\infty$, but
  finite swarms at $N\le128$ deviate quickly and the static solvers do not apply. Report $\Lambda$ with every
  prediction.
- Mode (c) is unverified for nonlinear agents with $\gamma\neq0$ (`claim3.md` section 3). Its first use on such a
  case should be accompanied by a mock-agent simulation at the same parameters.
- Sparse graphs: for $K<8$ the Gaussian field is a poor approximation (4% in $m$ and 25% in a lag correlation at
  $K=3$, `claim5.md`). Use the pool-based live route or a surrogate simulation on the actual graph.
