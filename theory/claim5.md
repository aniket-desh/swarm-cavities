# Claim 5: sparse graphs (brief)

**Verdict.** The machinery is known and the spec's two statements are true in the form that linear response makes
almost automatic: on a tree the susceptibility is a sum over paths, and the leading loop correction comes from
triangles, weighted by the product of their signs. What matters in practice is narrower. Directed graphs without
reciprocated edges are exactly solvable, and the solution is the same pool-of-trajectories procedure as the hybrid
LLM solver. Graphs with reciprocated edges need a closure. And the Physics of Agents graphs ($N=32$, mean degree
3.5) are too small to be tree-like, so for that data the useful predictions are per-graph and qualitative, not
ensemble cavity curves.

## 1. What is exactly solvable: directed graphs, $\gamma=0$

If agent $i$ reads $K$ others and none of them reads $i$ back (directly or through a short loop), the $K$ inputs
are independent of each other and of $i$'s own past. For memoryless $\pm1$ agents with
$P(s_i(t+1)=+1)=\frac12[1+\tanh(b\sum_k\epsilon_{ik}s_k(t)+h_i^{\rm priv})]$ and signs $\epsilon=+1$ with probability
$p_+$, each input contributes $+1$ with probability $\pi(t)=p_+\frac{1+m(t)}2+(1-p_+)\frac{1-m(t)}2$, and

$$m(t+1)=\sum_{n=0}^K\binom Kn\pi(t)^n(1-\pi(t))^{K-n}\,\big\langle\tanh\big(b(2n-K)+h^{\rm priv}\big)\big\rangle_{h^{\rm priv}}. \tag{1}$$

Two-time correlations, and agents with own-history memory, need whole trajectories, but the inputs stay
independent, so population dynamics over trajectories is exact:

$$\text{generation }d:\quad s^{(d)}(\tau+1)=F\big(h^{\rm priv},\{s^{(d)}(\le\tau)\},\ \textstyle\sum_{k=1}^K\epsilon_k\,s^{(d-1)}_{p_k}(\tau),\ \xi\big),\quad\tau<d, \tag{2}$$

where the parents $p_k$ are drawn at random from the pool of generation $d-1$, whose trajectories are valid up to
round $d-1$. After $T$ generations the pool samples the trajectory law up to round $T$, with cost $M\,T(T+1)/2$
single-agent updates for a pool of $M$. No Gaussian assumption enters. This is the sparse counterpart of the
single-site process of `claim1.md`, and with an LLM as $F$ it is the hybrid solver with real inboxes drawn from the
pool (`solver_spec.md`, mode (b)).

**Check.** `theory/checks/check_claim5_sparse.py`, part 1: $K=3$, $p_+=0.8$, $b=0.9$, simulation at $N=2\times10^5$
(`out_claim5_sparse.txt`). Equation (1) reproduces $m(t)$ to 0.0017 (simulation s.e. 0.0026) and the population
dynamics reproduces $C(t,t-1)$ to 0.003. The dense Gaussian DMFT with matched $\bar J=Kb(2p_+-1)$ and $g^2=Kb^2$ is
off by 0.006 in $m$ (4%) and by 0.011 in $C(t,t-1)$ (about a quarter of its value). That is the size of the
finite-$K$ error at $K=3$.

## 2. Reciprocated edges: what closure is needed

When $j$ reads $i$ back, $j$'s trajectory depends on $i$'s, and the message from $j$ to $i$ must be a conditional
law of $j$'s whole trajectory given $i$'s, $\mu_{j\to i}(s_j^{0..t}\mid s_i^{0..t-1})$. For binary agents and $T=8$ that
is a table of $2^9\times2^8$ entries per directed edge with an update cost exponential in $T$ and in the degree; for
continuous beliefs it is not tractable. The options, in order of usefulness here:

1. **Simulate a calibrated surrogate on the actual graph.** At $N=32$ this costs nothing and captures every loop. It
   is what Physics of Agents did with a fitted logistic rule, and it is the baseline any cavity prediction must beat.
2. **One-time (Markov) closure,** $\mu^t_{j\to i}(s_j(t)\mid s_i(t-1))$ (Neri & Bollé 2009; Aurell & Mahmoudi 2012).
   It keeps the two-round echo along each reciprocated edge, which is the sparse form of the reaction term, and
   drops longer memory. For symmetric couplings and parallel updates it is reported to be exact in the stationary
   state (from memory; not re-derived). Use it for stationary marginals on large sparse graphs.
3. **Dense DMFT with finite-$K$ corrections,** when $K\gtrsim8$. Three corrections are explicit:
   the field is non-Gaussian at $O(1/K)$ (size given by the check above); with a fixed in-degree the inputs are $K$
   draws, not a Poisson number, so the noise covariance is $g^2C(t,t')-(\bar J^2/K)\,m(t)m(t')$ in place of
   `claim1.md` eq. (5); and the reaction coefficient is a per-agent quantity,
   $\Gamma_i=\sum_{j\in\partial i}J_{ij}J_{ji}$, whose spread over agents is $\sqrt{r(1-r)/K}$ relative to $g^2$
   unless the number of reciprocated ties is fixed per agent. The last one can be included by sampling $\Gamma_i$ in
   the single-site process. (Derived by the same counting as `claim1.md` section 2; the second and third are not
   separately simulated.)

## 3. Leading loop correction

In linear response around a fixed point with gains $d_i$, the susceptibility matrix is $\chi=(D^{-1}-J)^{-1}$, a sum
over walks. On a tree the only closed walks are back-and-forth ones, and they resum into the cavity recursion

$$G_{i\to j}=\Big[\frac1{d_i}-\sum_{k\in\partial i\setminus j}J_{ik}J_{ki}\,G_{k\to i}\Big]^{-1},\qquad
\chi^{\rm tree}_{ii}=\Big[\frac1{d_i}-\sum_{k\in\partial i}J_{ik}J_{ki}\,G_{k\to i}\Big]^{-1}. \tag{3}$$

On a tree the result does not depend on the signs of symmetric ties, since only $J_{ik}J_{ki}=1$ enters. The first
walks that (3) misses are triangles:

$$\delta\chi_{ii}=\big(\chi^{\rm tree}_{ii}\big)^2\sum_{j\ne k\in\partial i}J_{ij}\,\chi^{\rm tree}_{jj}\,J_{jk}\,\chi^{\rm tree}_{kk}\,J_{ki}, \tag{4}$$

a sum over the triangles through $i$ (both orientations), each weighted by the product of its three signs. Balanced
triangles raise the susceptibility and frustrated ones lower it, so the correction is proportional to the *signed*
clustering around $i$, of relative order $2\,T_i\,(d\,\chi)^3/\chi$ for $T_i$ triangles. Squares enter at the next
order. This is the precise form of the spec's "corrections proportional to the local clustering coefficient".

**Check.** Part 2 of the same script, on the Physics of Agents ensemble ($N=32$, 56 undirected signed edges, 400
graphs, uniform gain $d=0.2$): a graph has 6.8 triangles on average; the network raises $\chi_{ii}$ by 22% over an
isolated agent; (3) is off by 1.8% on average (6.1% at the 95th percentile); adding (4) halves that to 0.9% (3.2%).

## 4. Consequences for the Physics of Agents data

The graphs have about 7 triangles, 19 squares and 50 pentagons on 56 edges ($k^L/2L$ with $k=3.5$), and
$\ln N/\ln k\approx2.8$, so a typical neighbourhood closes on itself within three steps. Ensemble cavity predictions
are therefore not the right comparison; the surrogate simulation is. The theory still makes parameter-free
qualitative predictions for this data, because the setup is a synchronous, symmetric ($\gamma=1$) swarm whose agents
do not see their own previous opinion (the opinion prompt in `lib/datagen/samplers.py` contains the persona and the
current inbox only):

1. **Period-2 flicker.** By `claim2.md` section 4, oscillating agents should have lag-2 agreement well above lag-1
   agreement (period index $\mathcal P<1$), and the excess should be absent in the asynchronous runs.
2. **Bipartite graphs flicker more.** On a bipartite graph the two sublattices at even and odd rounds form two
   histories that never interact, so they can settle on different answers and every agent then alternates. The
   square lattices are bipartite and the triangular ones are not, so at equal coupling the square lattices should
   show more period-2 oscillation.
3. **Coupling-induced overlap.** With four episodes on each of ten graphs per question, the same-graph minus
   other-graph replica overlap $\Delta q(t)$ of `claim2.md` section 5 can be computed directly.

**First look** (`theory/checks/check_poa_zero_cost.py`, all 9,600 communities, rounds 3–8, output in
`out_poa_zero_cost.txt`). With $E_k$ the mean squared $k$-round change of an agent's belief minus the part due to
sampling five votes, the ratio $E_2/E_1$ is 0.18–0.66 on the bipartite all-positive lattices, 0.22–0.70 on the random
signed graphs and 0.88–1.31 on the non-bipartite all-positive lattices, across the four models and both task types.
Predictions 1 and 2 hold in every one of the eight model-by-task cells. For GPT-4o-mini on objective questions the
one-round change on bipartite lattices is $E_1=1.32$ (the maximum possible is 4) against $E_2=0.24$. Prediction 3:
$\Delta q(0)=0.000\pm0.001$ and $\Delta q(t\ge1)$ is positive everywhere, from 0.015 (Qwen, objective) to 0.25–0.55
(GPT-4o-mini, subjective), where it alternates between odd and even rounds. Two caveats: square and triangular
lattices also differ in degree (4 against 6), so the bipartite contrast is not a controlled comparison, and the
asynchronous release, which is the proper control for prediction 1, has not been analysed here.
