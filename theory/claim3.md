# Claim 3: two-site cavity, participation ratio, and the early-warning claim

**Verdict.** The two-site cavity calculation carries over to discrete time and several populations, and the
resulting PR prediction matches exact results for linear swarms (with any reciprocity) and simulations of nonlinear
stochastic agents (at $\gamma=0$) to better than 1%. The safety claim built on it does not hold:

- *Approaching the bulk instability* ($\Lambda\to1$ of `claim2.md`) PR/N does fall, smoothly and to zero:
  $\mathrm{PR}/N=1-\Lambda^2$ for a homogeneous $\gamma=0$ swarm. Reciprocity makes it a weaker indicator, not a
  stronger one: at 90% of the critical gain PR/N is 0.34 for $\gamma=0$, 0.69 for $\gamma=0.6$ and 0.84 for
  $\gamma=1$.
- *Approaching a zealot-driven saddle-node* PR does **not** fall before the benign mean flips. The transition
  softens one collective mode, whose eigenvalue of the covariance matrix has to reach $\sqrt N$ times a typical one
  before PR moves, and noise tips the swarm long before that. In simulation, PR computed over the replicas that have
  not yet flipped stays flat or rises slightly right up to the threshold; PR over all replicas falls only as a
  by-product of some replicas having flipped. The variance of the benign mean across replicas rises 5- to 20-fold over
  the same range and is the right statistic.
- PR at a fixed round is an ensemble quantity. It needs $R\approx250$ replicas per condition at $N=32$ for a 9%
  standard error, with a bias-corrected estimator. It cannot be a per-run detector.

Sections 2–4 are derived and checked. Section 5 combines a derivation (the mean-mode formula), a scaling argument,
and a simulation. Section 6 is checked by Monte Carlo.

## 1. Definitions

Fix the couplings and the private fields, and let $\langle\cdot\rangle$ denote the average over replicas, runs that
differ only in the agents' sampling noise (and in the sampled initial answers). For agents $i,j$:

$$\mu_i(t)=\langle s_i(t)\rangle,\qquad\Sigma_{ij}(t,t')=\langle\delta s_i(t)\,\delta s_j(t')\rangle,\qquad
\chi_{ij}(t,t')=\frac{\partial\mu_i(t)}{\partial\theta_j(t')}, \tag{1}$$

with $\delta s=s-\mu$ and $\theta_j$ a probe added to $h_j$. The diagonal elements are of order one: $v_i(t,t')\equiv
\Sigma_{ii}(t,t')$ and $R_i(t,t')\equiv\chi_{ii}(t,t')$, the thermal covariance and thermal-averaged response of a
single agent with its quenched environment held fixed. Off-diagonal elements are $O(N^{-1/2})$ with random sign. The
participation ratio of the equal-time covariance $\Sigma(t)\equiv\Sigma(t,t)$ is

$$\mathrm{PR}(t)=\frac{(\mathrm{Tr}\,\Sigma)^2}{\mathrm{Tr}\,\Sigma^2},\qquad
\frac1N\mathrm{Tr}\,\Sigma^2=\underbrace{\langle v_i(t)^2\rangle}_{\text{single-site}}+\underbrace{\psi(t)}_{\text{two-site}},
\qquad\psi(t)\equiv N\,\big\langle\Sigma_{ij}(t,t)^2\big\rangle_{i\neq j}, \tag{2}$$

so PR is extensive, $\mathrm{PR}/N=\bar v^2/(\langle v^2\rangle+\psi)$, and the two-site cavity is needed for $\psi$.
(The mean coupling adds one collective mode to (2); section 5.)

A deterministic swarm started from fixed initial conditions has $\Sigma=0$. Everything here is driven by the agents'
sampling noise, so the single-site inputs are the *thermal* quantities $v_i$ and $R_i$, which differ from the
order parameters $C$ and $R$ of `claim1.md` by the frozen part: $\langle v_i(t,t')\rangle=C(t,t')-q(t,t')$ with $q$ the
cross-replica correlation of `claim1.md` eq. (9).

## 2. No reciprocity, no memory: an equal-time recursion

For $\gamma=0$ and a memoryless map $s_i(t+1)=F(h_i(t),\xi_i(t))$ the theory closes at equal times. The fields of two
agents $i\ne j$ at round $t$ have thermal covariance $c_{ij}(t)=\sum_{kl}J_{ik}J_{jl}\Sigma_{kl}(t)$, which is
$O(N^{-1/2})$, and to first order in it (Price's theorem for jointly Gaussian fields)

$$\Sigma_{ij}(t+1)=r_i(t)\,r_j(t)\,c_{ij}(t),\qquad r_i(t)=\frac{\partial\mu_i(t+1)}{\partial h_i(t)}=\Big\langle\frac{\partial}{\partial h}\mathbb E_\xi F(h,\xi)\Big\rangle_{\text{thermal }h_i(t)}. \tag{3}$$

Squaring and averaging over the couplings, with
$N\langle c_{ij}^2\rangle=g^4\big(\langle v_k^2\rangle+\psi\big)$:

$$\boxed{\;\psi(t+1)=g^4\,\langle r_i(t)^2\rangle^2\,\big[\langle v_i(t)^2\rangle+\psi(t)\big]\;},\qquad\psi(0)=0. \tag{4}$$

For populations, with $\psi_{ab}$ the same quantity for $i\in a$, $j\in b$:
$\psi_{ab}(t+1)=\langle r_a^2\rangle\langle r_b^2\rangle\big[\sum_c\rho_cg_{ac}^2g_{bc}^2\langle v_c^2\rangle+\sum_{c,e}\rho_c\rho_e\,g_{ac}^2g_{be}^2\,\psi_{ce}(t)\big]$
(derived by the same step; not simulated).

The single-site inputs need the split of the cavity field into the part frozen by the couplings (shared by all
replicas, covariance $g^2q(t,t')$) and the thermal part (redrawn, variance $g^2(C(t,t)-q(t,t))$). For $\pm1$ agents
with $P(+1)=(1+\tanh(\beta h+h^{\rm priv}))/2$:
$\mu_i(t+1)=\mathbb E_y\tanh\big(\beta(\eta^f_i(t)+g\sqrt{1-q(t,t)}\,y)+h^{\rm priv}_i\big)$,
$v_i=1-\mu_i^2$, $r_i=\beta\,\mathbb E_y[1-\tanh^2(\cdot)]$, $q(t,t')=\langle\mu_i(t)\mu_i(t')\rangle$.

**Check.** `theory/checks/check_claim3_binary.py`: $\beta=1.2$, $g=1$, $\sigma_{\rm priv}=0.5$; simulation with
$N=1000$, 600 replicas, 4 disorder samples, with unbiased estimates of the traces (`out_claim3_binary.txt`):

| $t$ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| $\psi$, eq. (4) | 0.1053 | 0.1286 | 0.1370 | 0.1382 |
| $\psi$, simulation (s.e. 0.001–0.003) | 0.1063 | 0.1295 | 0.1352 | 0.1366 |
| PR/N, theory | 0.8544 | 0.8419 | 0.8390 | 0.8387 |
| PR/N, simulation (s.e. 0.001–0.003) | 0.8528 | 0.8415 | 0.8409 | 0.8399 |

The largest relative error of PR/N over $t=0,\dots,8$ is 0.4%.

## 3. Reciprocity and memory: the general two-site equations

With $\gamma\neq0$ or memory, equal times do not close, because the reaction term makes each agent respond to its own
past. The objects are then two-time, and the algebra is cleanest with $n\times n$ matrices over the rounds
($n=T+1$): products below are matrix products in the time indices.

Add two agents $0$ and $1$ to a swarm of $N$. Each evolves by its own single-site process (with its own reaction
term), and they are linked in two ways. First, by an effective retarded coupling: the direct one plus every path
through the rest of the swarm,

$$\mathcal K_{01}(u,u')=J_{01}\,\delta_{uu'}+\sum_{j,k}J_{0j}\,\chi_{jk}(u,u')\,J_{k1},\qquad\chi_{01}=R_0\,\mathcal K_{01}\,R_1. \tag{5}$$

Second, by common input: their cavity fields have thermal covariance $c_{01}(u,u')=\sum_{j,k}J_{0j}J_{1k}\Sigma_{jk}(u,u')$.
To first order in these $O(N^{-1/2})$ quantities,

$$\Sigma_{01}=\underbrace{R_0\,c_{01}\,R_1^{\!\top}}_{\text{common input}}+\underbrace{R_0\,\mathcal K_{01}\,v_1}_{1\text{ drives }0}+\underbrace{v_0\,\mathcal K_{10}^{\!\top}R_1^{\!\top}}_{0\text{ drives }1}. \tag{6}$$

The second moments of $\mathcal K$ and $c$ over the couplings follow from (2) of `claim1.md`. Writing
$G_2=g^2$, $\tau=\gamma g^2$, and $\otimes$ for the product of two two-time objects with all four time indices free:

$$N\langle\mathcal K_{01}\otimes\mathcal K_{01}\rangle=G_2\,\mathbb 1\otimes\mathbb 1+G_2^2\big(\langle R\otimes R\rangle+\Psi^{\chi\chi}\big), \tag{7}$$
$$N\langle\mathcal K_{01}\otimes\mathcal K_{10}\rangle=\tau\,\mathbb 1\otimes\mathbb 1+\tau^2\big(\langle R\otimes R\rangle+\Psi^{\chi\chi^{\!\top}}\big), \tag{8}$$
$$N\langle c_{01}\otimes c_{01}\rangle=G_2^2\big(\langle v\otimes v\rangle+\Psi^{\Sigma\Sigma}\big), \tag{9}$$
$$N\langle c_{01}\otimes\mathcal K_{01}\rangle=G_2\,\tau\big(\langle v\otimes R\rangle+\Psi^{\Sigma\chi}\big), \tag{10}$$
$$N\langle c_{01}\otimes\mathcal K_{10}\rangle=G_2\,\tau\big(\langle v\otimes R\rangle+\Psi^{\Sigma\chi^{\!\top}}\big), \tag{11}$$

where the five four-point functions are the unknowns:

$$\Psi^{\Sigma\Sigma}=N\langle\Sigma_{01}\otimes\Sigma_{01}\rangle,\quad\Psi^{\chi\chi}=N\langle\chi_{01}\otimes\chi_{01}\rangle,\quad
\Psi^{\chi\chi^{\!\top}}=N\langle\chi_{01}\otimes\chi_{10}\rangle,\quad\Psi^{\Sigma\chi}=N\langle\Sigma_{01}\otimes\chi_{01}\rangle,\quad
\Psi^{\Sigma\chi^{\!\top}}=N\langle\Sigma_{01}\otimes\chi_{10}\rangle. \tag{12}$$

Inserting (5)–(6) into the definitions (12) and using (7)–(11) gives a closed linear system: each $\Psi$ is a sum of
terms of the form (site moments)·(moment of $\mathcal K$ or $c$)·(site moments), where the site moments are the
single-agent averages $\langle R\otimes R\rangle$, $\langle R\otimes v\rangle$, $\langle v\otimes v\rangle$ taken over the
distribution of agents (each agent's $R_i$ and $v_i$ are correlated, so these are not products of means). The nine
terms of $\Psi^{\Sigma\Sigma}$ and the index placement of every term are written out in
`theory/checks/two_site.py::four_point`, which is the reference for `solver_spec.md`. Because $R$ is strictly causal
the system is solved by iterating from $\Psi=0$; it terminates after at most $n$ sweeps. Then

$$\frac{\mathrm{PR}(t)}{N}=\frac{\bar v(t)^2}{\langle v_i(t)^2\rangle+\psi(t)},\qquad\psi(t)=\Psi^{\Sigma\Sigma}(t,t;t,t). \tag{13}$$

For several populations every $\Psi$ gains a pair of population indices and (7)–(11) gain sums over the
intermediate populations with weights $\rho_cg_{ac}^2g_{cb}^2$ and $\rho_c\rho_e\,g_{ac}^2g_{eb}^2$ (for (7)), and the
corresponding products of $\tau_{ab}=\gamma_{ab}g_{ab}g_{ba}$ and $g^2$ for (8)–(11). A frozen population (zealots) has
$R=0$ and $v=0$ and drops out of all sums, so for benign agents plus zealots one uses the single-population equations
with $G_2=\rho_Bg^2$, $\tau=\rho_B\gamma g^2$. Their solution is then normalized by the number of benign agents: it
is $\psi_B=N_B\langle\Sigma_{ij}^2\rangle=\rho_B\psi$, and $\mathrm{PR}/N_B=\bar v^2/(\langle v^2\rangle+\psi_B)$ (the `rho_b`
argument of the code; for a linear swarm with 40% frozen agents the script below gives PR/$N_B=0.7429$ against
$0.7409$ exact). The
general multi-population system is derived but not implemented or simulated.

**Stationary, single-frequency closed forms.** In a stationary state the same equations at one frequency give, for
homogeneous agents with dressed response $R(\omega)$ and spectrum $C(\omega)$,

$$N\langle|\chi_{ij}|^2\rangle=\frac{g^2|R|^4}{1-g^2|R|^2},\qquad N\langle\chi_{ij}\chi_{ji}\rangle=\frac{\gamma g^2R^4}{1-\gamma g^2R^2},
\qquad N\langle|\Sigma_{ij}(\omega)|^2\rangle\big|_{\gamma=0}=C^2\Big[\frac1{(1-g^2|R|^2)^2}-1\Big].$$

The first is the formula used for the spectral edge in `claim2.md`; it agrees with the Ward identity
$\frac1N\mathrm{Tr}\,GG^\dagger=-\mathrm{Im}\,\bar G/\mathrm{Im}\,\lambda$ at $\gamma=1$ and with the Ginibre resolvent at
$\gamma=0$. The third is the discrete-time counterpart of Clark, Abbott & Litwin-Kumar's (2023) result. Note that the
derivation needs the reservoir-mediated part of $\mathcal K$ in (5): dropping it loses terms at second order in
$g^2|R|^2$.

**Check.** `theory/checks/check_claim3_linear.py`. For a linear swarm $x(t+1)=a_i[Jx](t)+\sigma_i\xi_i(t)$, $x(0)=0$,
the covariance obeys $\Sigma(t+1)=A\Sigma(t)A^{\!\top}+\Sigma_\xi$ exactly for each sampled $J$, so
$\mathrm{Tr}\,\Sigma^2$ is available without sampling noise. $N=1000$, 4 samples, $n=12$ rounds
(`out_claim3_linear.txt`):

| case | max rel. error of $\psi(t)$ | max rel. error of PR/N | sample-to-sample s.d. of exact $\psi$ |
|---|---|---|---|
| homogeneous, $\gamma=0$, $a=0.8$ | 0.4% | 0.2% | 1.8% |
| homogeneous, $\gamma=0.6$, $a=0.55$ | 1.0% | 0.1% | 0.8% |
| homogeneous, $\gamma=-0.6$, $a=0.55$ | 1.3% | 0.2% | 3.3% |
| homogeneous, $\gamma=1$, $a=0.45$ | 0.4% | 0.04% | 1.6% |
| two agent types ($a=0.3,0.8$; $\sigma=1,0.5$), $\gamma=0.5$ | 0.7% | 0.25% | 3.4% |

The errors of $\psi$ are comparable to the sample-to-sample spread of the exact value. At $N=\infty$ the equal-time results are even in
$\gamma$, which the cavity equations reproduce exactly.

**Not checked:** equations (5)–(13) for nonlinear agents with $\gamma\neq0$. The nonlinear check of section 2 is at
$\gamma=0$ and the $\gamma\ne0$ checks are linear. The nonlinear, reciprocal case additionally needs the per-agent
thermal responses $R_i(t,t')$ from a nested single-site simulation (`solver_spec.md`, mode (c)).

## 4. PR approaching the instability

**Exact closed form at $\gamma=0$.** For the linear homogeneous swarm $\Sigma=\sigma^2\sum_kA^kA^{k\top}$. With
$A=aJ$ and $J$ Gaussian with independent entries, free probability gives
$\frac1N\mathrm{Tr}\,A^kA^{k\top}A^lA^{l\top}\to(ag)^{2(k+l)}(\min(k,l)+1)$, and summing,

$$\bar v=\frac{\sigma^2}{1-x},\qquad\frac1N\mathrm{Tr}\,\Sigma^2=\frac{\sigma^4}{(1-x)^2(1-x^2)},\qquad
\boxed{\;\frac{\mathrm{PR}}N=1-x^2\;},\qquad x=(ag)^2. \tag{14}$$

This is also the stationary solution of (4): $\psi=\bar v^2x^2/(1-x^2)$. For nonlinear, heterogeneous agents at
$\gamma=0$ the stationary form of (4) is

$$\frac{\mathrm{PR}}N=\frac{\bar v^2}{\langle v^2\rangle}\,(1-x^2),\qquad x=g^2\langle r_i^2\rangle, \tag{15}$$

with $x$ equal to the stability parameter $\Lambda$ of `claim2.md` at $\gamma=0$. So PR/N falls to zero linearly in
the distance from the boundary, $\mathrm{PR}/N\approx2(1-\Lambda)\bar v^2/\langle v^2\rangle$, while the replica
variance $\bar v$ diverges as $1/(1-\Lambda)$. From the chaotic side Clark et al. find $\mathrm{PR}/N\propto\epsilon^3$
in continuous time; we have not derived the discrete-time analogue.

**With reciprocity the fall is slower.** `theory/checks/check_claim3_critical.py` iterates the exact linear
recursion to stationarity at $N=600$ for gains at a fraction of the critical one, $a_c=1/(g(1+|\gamma|))$
(`out_claim3_critical.txt`):

| $a/a_c$ | PR/N, $\gamma=0$ (eq. 14) | PR/N, $\gamma=0.6$ | PR/N, $\gamma=1$ | $\bar v/\sigma^2$: $\gamma=0$, $0.6$, $1$ |
|---|---|---|---|---|
| 0.5 | 0.938 (0.938) | 0.989 | 0.995 | 1.33, 1.11, 1.07 |
| 0.7 | 0.759 (0.760) | 0.942 | 0.972 | 1.96, 1.27, 1.17 |
| 0.8 | 0.588 (0.590) | 0.873 | 0.937 | 2.79, 1.42, 1.25 |
| 0.9 | 0.336 (0.344) | 0.686 | 0.841 | 5.33, 1.70, 1.40 |
| 0.95 | sample unstable | 0.408 | 0.708 | –, 2.04, 1.53 |

At $\gamma=0$ every direction of the eigenvalue circle goes critical together. At $\gamma\neq0$ only the modes near
the tips of the ellipse do, so the variance grows slowly and the covariance stays high-dimensional until very close
to the boundary. (Numerical observation for the linear model, which the two-site equations reproduce; we have not
extracted the asymptotic exponents.) The practical reading: a swarm on a reciprocal graph can be within 10% of
the instability with PR/N still above 0.8. The amplification $\mathcal A$ and $\hat\Lambda$ of `claim2.md` section 5
measure the same proximity more directly and need two replicas per graph, not hundreds.

## 5. PR approaching a zealot-driven saddle-node

A positive mean coupling adds a collective mode that the random-coupling analysis above does not contain. Take
benign agents (fraction $\rho_B=1-\rho$) and zealots as in `claim4.md`. Fluctuations of the benign mean obey, to
linear order around the benign-majority fixed point,

$$\delta m(t+1)=\lambda\,\delta m(t)+\zeta(t),\qquad\lambda=\bar J\rho_B\langle r_i\rangle,\qquad\mathrm{Var}\,\zeta=\frac{\bar v}{N_B},$$

and the shared fluctuation $\delta m$ enters every benign agent's field, giving a rank-one term in the covariance,
$\Sigma_{ij}\ni r_ir_j(\bar J\rho_B)^2\mathrm{Var}(\delta m)$. Its eigenvalue and the resulting PR are

$$\Lambda_1=\frac{\kappa^2\,\bar v}{1-\lambda^2},\quad\kappa^2=(\bar J\rho_B)^2\langle r_i^2\rangle,\qquad
\boxed{\;\frac{\mathrm{PR}}{N_B}=\frac{\bar v^2}{\langle v^2\rangle+\rho_B\psi+\Lambda_1^2/N_B}\;},\qquad
N_B\mathrm{Var}(m_B)=\frac{\bar v}{1-\lambda^2}. \tag{16}$$

At the saddle-node $\lambda\to1$, with $1-\lambda\propto(\rho_c-\rho)^{1/2}$, so $\Lambda_1$ diverges. But it is one
eigenvalue among $N_B$, and it enters (16) divided by $N_B$.

**Scaling argument (not a theorem).** Write $\epsilon=1-\lambda$. PR halves when $\Lambda_1^2\approx N_B\langle v^2\rangle$,
that is at $\epsilon_{\rm PR}\approx1/(2\sqrt{N_B})$: 0.09 for $N_B=32$, 0.04 for $N_B=128$. Noise tips the swarm
earlier. Near a saddle-node the stable and unstable fixed points are $\epsilon/a$ apart ($a$ the curvature of the
mean-field map) and the barrier between them, in units of the noise $\bar v/N_B$, is of order
$N_B\epsilon^3/(3a^2\bar v)$, so escapes within the run become likely at
$\epsilon_{\rm esc}\sim(a^2\bar v/N_B)^{1/3}$: about 0.3 for $N_B=32$ and 0.2 for $N_B=128$ with $a,\bar v$ of order
0.5. At that point the mean-mode term in (16) is $\Lambda_1^2/(N_B\langle v^2\rangle)\approx1/(4N_B\epsilon_{\rm esc}^2)\propto
N_B^{-1/3}$, a few percent at these sizes and vanishing as $N$ grows, while $N_B\mathrm{Var}(m_B)\approx\bar v/2\epsilon_{\rm esc}$
has grown as $N_B^{1/3}$. The larger the swarm, the less PR moves before the flip.

**Check.** `theory/checks/check_claim3_flip.py`: stochastic $\pm1$ agents, $\beta=1.5$, $\bar J=1$,
$\mu_{\rm priv}=0.3$, $\sigma_{\rm priv}=0.2$ (mean-field threshold $\rho_c=0.220$), 2000 replicas, 8 disorder draws,
round $T=10$ (`out_claim3_flip.txt`). "Unflipped" restricts to replicas with $m_B(T)>0$:

| $N$ | $\rho$ | flipped by $T=10$ | PR/$N_B$, all replicas | PR/$N_B$, unflipped | $N_B\mathrm{Var}(m_B)$, unflipped | eq. (16): PR/$N_B$; $N_B$Var |
|---|---|---|---|---|---|---|
| 128 | 0 | 0 | 0.873 | 0.873 | 0.14 | 0.875; 0.14 |
| 128 | 0.102 | 0 | 0.895 | 0.895 | 0.33 | 0.897; 0.31 |
| 128 | 0.148 | 0.2% | 0.892 | 0.910 | 0.78 | 0.914; 0.51 |
| 128 | 0.172 | 1.9% | 0.821 | 0.907 | 1.48 | 0.925; 0.72 |
| 128 | 0.203 | 17% | 0.605 | 0.900 | 2.76 | 0.934; 1.50 |
| 32 | 0 | 0 | 0.860 | 0.860 | 0.16 | 0.875; 0.14 |
| 32 | 0.125 | 2.9% | 0.770 | 0.896 | 0.68 | 0.903; 0.39 |
| 32 | 0.156 | 8.6% | 0.673 | 0.902 | 1.05 | 0.914; 0.57 |
| 32 | 0.188 | 20% | 0.579 | 0.929 | 1.22 | 0.916; 0.96 |

Three things to read off. Among unflipped replicas PR/$N_B$ rises from 0.87 to about 0.91 as $\rho\to\rho_c$ (agents
desaturate, which raises $\bar v^2/\langle v^2\rangle$, and the mean mode is too weak to offset it), as (16) predicts.
The all-replica PR falls in step with the flipped fraction: at $N=128$ it is down 6% when 2% of replicas have
flipped and down 31% when 17% have. That fall is the flip itself, seen as a two-cluster mixture. And
$N_B\mathrm{Var}(m_B)$ among unflipped replicas rises from 0.14 to 2.8, a factor of 20, before a fifth of the replicas
have flipped. It exceeds the linear prediction of (16) near the threshold, as expected once fluctuations reach the
nonlinear part of the map. With coupling disorder ($g=0.3$, $\gamma=0.5$, $N=32$; third block of the output) the
picture is the same, with PR/$N_B$ of unflipped replicas rising from 0.65 to 0.92.

**Within a run** that does flip ($\rho>\rho_c$), the ensemble spreads because replicas pass the bottleneck at
different rounds. The all-replica PR is lowest while the replicas are split between the two sides, which is during
the flip, not before it.

**Conclusion for the experiments.** Report PR as a descriptive ensemble statistic with the prediction (13)/(16), and
use $N_B\mathrm{Var}(m_B)$ and the lag-1 autocorrelation of $m_B$ across replicas
(`analysis.metrics.ensemble_ews`) as the early-warning statistics. The claim that PR collapses "sharply prior to" the
flip should be dropped. The comparison with a transcript monitor should be made with $\mathrm{Var}(m_B)$, at the
level of conditions (a given $\rho$, graph and task), not runs.

## 6. Estimating PR from $R$ replicas

The plug-in estimator uses the sample covariance $\hat\Sigma$ with $n=R-1$ degrees of freedom. For Gaussian data

$$\mathbb E\,\mathrm{Tr}\,\hat\Sigma^2=\Big(1+\frac1n\Big)\mathrm{Tr}\,\Sigma^2+\frac1n(\mathrm{Tr}\,\Sigma)^2,\qquad
\mathbb E(\mathrm{Tr}\,\hat\Sigma)^2=(\mathrm{Tr}\,\Sigma)^2+\frac2n\mathrm{Tr}\,\Sigma^2, \tag{17}$$

so to leading order $1/\widehat{\mathrm{PR}}\approx1/\mathrm{PR}+1/n$: the plug-in value is
$\mathrm{PR}\cdot n/(n+\mathrm{PR})$, capped at $n$ and biased by an amount that grows with PR itself. Comparing
conditions with different PR at $R\lesssim N$ therefore compresses real differences. Solving (17) for the two
traces removes the bias for Gaussian data. Beliefs are bounded and often near-binary, so the distribution-free
version is preferable: for four distinct replicas, with $d_{12}=x_1-x_2$ and $d_{34}=x_3-x_4$,

$$\mathbb E\,(d_{12}\cdot d_{34})^2=4\,\mathrm{Tr}\,\Sigma^2,\qquad\mathbb E\,|d_{12}|^2|d_{34}|^2=4\,(\mathrm{Tr}\,\Sigma)^2, \tag{18}$$

averaged over all quadruples (a U-statistic, computable from the $R\times R$ Gram matrix). The ratio of the two
unbiased estimates has a residual bias of order $1/R$. This estimator already exists as
`analysis.metrics.participation_ratio_corrected` (Himeno–Yamada form); the independent implementation in
`theory/checks/pr_estimators.py` agrees with it to machine precision on a test array. Use
`analysis.metrics.subsample_ci` for intervals, since resampling replicas with replacement creates duplicate pairs
that break (18).

**Check and replica count.** `theory/checks/check_claim3_estimator.py`, $N=32$, 400 trials per cell, mean (s.d.)
(`out_claim3_estimator.txt`):

| data, true PR | $R$ | plug-in | $n\mathrm{PR}/(n+\mathrm{PR})$ | U-statistic |
|---|---|---|---|---|
| Gaussian, 12.0 | 32 | 8.99 (1.56) | 8.65 | 13.2 (3.3) |
| Gaussian, 12.0 | 128 | 11.08 (1.43) | 10.96 | 12.2 (1.7) |
| Gaussian, 12.0 | 256 | 11.61 (1.05) | 11.46 | 12.2 (1.2) |
| Gaussian, 24.0 | 32 | 13.51 (0.97) | 13.53 | 24.8 (3.1) |
| Gaussian, 24.0 | 256 | 21.94 (0.94) | 21.94 | 24.1 (1.1) |
| binary, 9.65 | 64 | 8.62 (1.28) | 8.37 | 9.90 (1.72) |
| binary, 9.65 | 256 | 9.46 (0.77) | 9.30 | 9.80 (0.83) |
| binary, 20.1 | 64 | 15.57 (1.42) | 15.23 | 20.6 (2.5) |
| binary, 20.1 | 256 | 18.80 (1.05) | 18.61 | 20.3 (1.2) |

The plug-in bias follows $n\mathrm{PR}/(n+\mathrm{PR})$ closely and reaches −44% at $R=N=32$ for PR $=24$. The
corrected estimator is within 6% of the truth at $R=64$ and within 2% from $R=128$ on (the remainder is the ratio
bias), and on binary data the U-statistic is less biased than the Gaussian correction. Its relative standard
deviation at $N=32$ is 17–19% at $R=64$, 12–14% at $R=128$, 8.5–9.5% at $R=256$ and 6–6.5% at $R=512$ for PR between
4 and 12 (about half that for PR above 20).

**Recommended $R$ at $N=32$:** 256 replicas per condition for a standard error of about 9%, which resolves a 30%
difference between two conditions at better than two standard errors; 500 if a 20% difference must be resolved.
With fewer than about 64 replicas the corrected estimate is too noisy to be useful, and the plug-in estimate should
not be reported at all without the factor $n/(n+\mathrm{PR})$ in view. One condition at $R=256$, $N=32$, $T=8$ is
about 65,000 agent calls.

## 7. Summary of what is established

| statement | status |
|---|---|
| equal-time recursion (4) and PR formula (15) for $\gamma=0$, nonlinear stochastic agents | derived; simulated, 0.4% |
| general two-site system (5)–(13) | derived; verified for linear dynamics with $\gamma\in\{-0.6,0,0.6,1\}$ and two agent types |
| general system for nonlinear agents with $\gamma\ne0$; multi-population form | derived; not verified |
| $\mathrm{PR}/N=1-(ag)^4$ (14) | exact; verified |
| PR falls more slowly toward the instability when $\gamma\neq0$ | numerical observation, linear model |
| mean-mode formula (16) | derived; verified away from the threshold |
| PR barely moves before a saddle-node flip; relative change $\propto N^{-1/3}$ | scaling argument; supported by simulation at $N=32,128$ |
| "PR collapses sharply before $m_B$ flips" | false in this model class |
| estimator bias $n\mathrm{PR}/(n+\mathrm{PR})$, U-statistic correction, $R\approx256$ at $N=32$ | verified by Monte Carlo |
