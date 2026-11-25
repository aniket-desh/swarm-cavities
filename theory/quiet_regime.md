# Quiet regime: revised agent law, what still discriminates the theory, and the kick-and-echo measurement

**Verdict.**

1. *The agent anchors on its own last answer unless its inbox changed.* One law fits both the white-probe
   trajectories and the pilot swarm: the weight on the agent's own last answer is about 0.7 when the inbox is
   the one it already answered and falls to about 0.2 when the inbox is new (section 1). Fitted to the white probes
   alone, the law moves the predicted lag-1 correlation of replica differences from 0.29–0.33 (the old surrogate)
   to 0.54–0.60, against 0.68–0.80 in the pilot. With three numbers and the gains taken from the pilot itself it gives
   0.67–0.74, and reproduces ten pilot observables with a root-mean-square deviation of 1.2 to 2.5 simulated standard
   deviations (the old surrogate: 2.4 to 99). The second part of that statement is in-sample.
2. *The oscillation and amplification predictions are out of reach, and anchoring is the reason more than weak
   coupling.* Anchoring is a low-pass filter. It leaves the static boundary $d\,g(1+\gamma)=1$ where it was and moves
   the period-4 boundary at $\gamma=-1$ from $d\,g=0.5$ to $d\,g=\tfrac12\sqrt{(1+\rho)/(1-\rho)}=1.19$ (section 2).
   The period index $P_\delta$ of the pilot, 1.7–2.4, is $(1-c_2)/(1-c_1)$ of an anchored process and carries no
   information about period 4. What still isolates the cavity-specific term is a direct measurement of the reaction
   term, which is section 3.
3. *Kick and echo measures the reaction term $\Gamma R$ of `claim1.md` eq. (4) and nothing else to leading order.*
   The echo per unit kick is $E=\sum_{\rm types}(n_{\rm type}/K^2)\,a_{ij}a_{ji}\,\langle d_id_j\rangle$: exactly zero
   at $\gamma=0$, positive for concordant ties in both wordings, negative for discordant ties in the inverted wording
   and positive in the label wording. With the real-message gains, $|E|$ is 0.011 to 0.058 at $|\gamma|=1$; the
   anchoring law changes it to 0.003–0.067. About 2,000 twins per cell resolve every $|\gamma|=1$ echo predicted by
   linear response at 5 standard errors or more. The size of the echo alone separates linear response from the
   anchoring law in five of the eight $|\gamma|=1$ cells; the factorization slope and the dependence of the
   first-order response on kick size separate them at both $K$. The whole matrix costs about \$270, a first stage at
   $K=2$ about \$50.

Sections 2.2 and 3.2–3.4 are derived and checked on mock swarms. Section 1 is a fitted law with its fit quality
stated. The numbers in section 3.5 come from simulating that law and inherit its uncertainty. Section 3.9 lists what
is not verified. No Gaussian single-site DMFT of the new law was built: all swarm predictions here are simulations
of the law on actual graphs, which is also the right reference at $K=2$–$4$ where the field is far from Gaussian.

Code: `experiments/04_swarm_design/surrogate_v2.py` (law, fits, pilot comparison; output `surrogate_v2_out.txt`,
parameters `surrogate_v2_fit.json`) and `theory/checks/check_kick_echo.py` (output `out_kick_echo.txt`, frozen
numbers `kick_predictions.json`, surrogate table `kick_surrogate_table.json`).

## 1. Revised single-agent law

### 1.1 What the two data sets say

Notation as in `design_lib.py`: $s=2p-1$, $f_{A,i}=\frac1K\sum_{j\in A_i}s_j$ and $f_{D,i}=\frac1K\sum_{j\in D_i}s_j$
for the agree and disagree groups of agent $i$. Define the change of the inbox since the agent's last answer,

$$u_i(t)=\frac1K\sum_{j\in A_i\cup D_i}\big|s_j(t)-s_j(t-1)\big|. \tag{Q1}$$

In the white probes all $K=8$ template messages state the same probability, so $u=|\bar h(t)-\bar h(t-1)|$. The
weight on the own last answer can be read off both data sets without a model of the agent. In the white probes it
is the coefficient $\rho$ of $s(t+1)=\rho\,s(t)+b\,\bar h(t)+c$. In the pilot it is the coefficient of
$\delta_i(t+1)=\rho_{\rm eff}\,\delta_i(t)+b_A\,\delta f_{A,i}(t)+b_D\,\delta f_{D,i}(t)$ for replica differences
$\delta=s^\alpha-s^\beta$, which removes every agent-specific offset (the same agent, evidence and graph appear in
both replicas).

| inbox change $u$ | white: own-answer weight | white: exact repeats | white: residual s.d. | | $u$ | pilot $K=2$: $\rho_{\rm eff}$ | pilot $K=4$: $\rho_{\rm eff}$ |
|---|---|---|---|---|---|---|---|
| $<0.1$ | 0.59 | 13% | 0.057 | | $<0.03$ | 0.76 (0.03) | 0.71 (0.03) |
| 0.1–0.2 | 0.45 | 7% | 0.064 | | 0.03–0.08 | 0.67 (0.03) | 0.64 (0.02) |
| 0.2–0.4 | 0.20 | 2.5% | 0.082 | | 0.08–0.2 | 0.60 (0.04) | 0.42 (0.04) |
| $>0.4$ | 0.06 | 1.2% | 0.105 | | $>0.2$ | 0.45 (0.09) | 0.42 (0.07) |

The two columns lie on one curve. The white probes almost never offered a small $u$ (the field was redrawn with
s.d. 0.34 every round), which is why their pooled own-answer weight is 0.19, and the settled swarm almost never
offers a large one. The noise grows with the size of the move and exact repeats disappear when the inbox changes.
The old surrogate's i.i.d. noise of s.d. 0.10 around a static target corresponds to the last row only.

A second fact comes from the pilot's late rounds (6–10). The one-round moves are heavy-tailed: 26% ($K=2$) and 36%
($K=4$) are exact repeats, the median nonzero move is 0.04 and 0.02 (two and one percentage points of probability),
and 4.0% and 1.6% of moves are 0.15 or larger, with s.d. 0.22–0.25. The large moves are not undone in the next
round (reversal slope 0.02 and −0.11) and point toward the agent's level in the other replica (slope 0.38 and 0.79),
so they are occasional re-derivations of the answer, not excursions. The residual kurtosis is 9–13.

### 1.2 The law

$$\tau_i(t)=\tanh\!\big(a_+f_{A,i}(t)-a_-f_{D,i}(t)+a_p h^{\rm priv}_i+c\big) \tag{Q2}$$

$$\rho(u)=\rho_\infty+(\rho_0-\rho_\infty)\,e^{-u/u_0} \tag{Q3}$$

$$\mu_i(t)=\big(1-\rho(u_i(t))\big)\big(\tau_i(t)-s_i(t)\big) \tag{Q4}$$

$$\text{with probability }\pi_0\,e^{-|\mu_i|/\mu_\pi}:\quad s_i(t+1)=s_i(t) \tag{Q5}$$

$$\text{otherwise}:\quad s_i(t+1)=\mathrm{clip}\Big(s_i(t)+\mu_i+(\sigma_0+\sigma_1|\mu_i|)\,\xi+b\,\sigma_J\,\xi'\Big),\qquad
\xi,\xi'\sim\mathcal N(0,1),\ \ b\sim\mathrm{Bernoulli}(q_J). \tag{Q6}$$

$\tau$ is the static target of the two-channel fit, $\mu$ the intended move, (Q5) the exact repeat and the last term
of (Q6) the rare large move. The first inbox (round 1) has no predecessor, so $u=\infty$ and $\rho=\rho_\infty$.
The old surrogate is the special case of constant $\rho$, $\pi_0=\sigma_1=q_J=0$ and noise added around $\tau$.

The agent never sees its previous inbox, only its own last answer and message. The law is therefore a description:
the last message records what the agent had read, and the agent keeps its answer when the new inbox matches that
record. $u$ is the observable proxy that can be computed in both data sets and in a simulation.

**One-shot gains and target gains.** An agent that reads a new inbox with last answer $u_0$ returns
$u_0+(1-\rho_\infty)(\tau-u_0)$. A tanh fit to such answers, which is what the transfer check and the calibration
did, reports gains that are $(1-\rho_\infty)\approx0.81$ times the gains of $\tau$. Both are needed: the one-shot
gains are the response to a new message, the target gains set the level at which a swarm settles. Refitting the
transfer records in the one-shot form with $\rho_\infty=0.19$ gives the target gains (`surrogate_v2_out.txt`, part
B; s.e. in parentheses):

| | one-shot $a_+$ | one-shot $a_-$ | target $a_+$ | target $a_-$ | pilot fixed point $a_+$ | $a_-$ | pilot first inbox $a_+$ | $a_-$ |
|---|---|---|---|---|---|---|---|---|
| inverted, $K=2$ | 0.80 (0.03) | 0.15 (0.02) | 1.00 (0.03) | 0.19 (0.03) | 1.21 (0.05) | 0.24 (0.05) | 1.65 (0.05) | 0.23 (0.04) |
| inverted, $K=4$ | 0.66 (0.05) | 0.22 (0.04) | 0.82 (0.04) | 0.28 (0.04) | 1.13 (0.06) | 0.20 (0.06) | 2.39 (0.09) | 0.12 (0.07) |
| label, $K=2$ | 0.54 (0.03) | −0.24 (0.02) | 0.66 (0.03) | −0.30 (0.02) | no swarm run | | | |
| label, $K=4$ | 0.40 (0.03) | −0.21 (0.02) | 0.49 (0.03) | −0.26 (0.03) | no swarm run | | | |

Two findings in this table are new. The level at which the pilot settled (fixed point $s=\tau$ of the replica- and
round-averaged states of rounds 6–10) needs $a_+$ 20% ($K=2$) and 38% ($K=4$) above the transfer check's target
gain, and the fit does not change with $\gamma$ (1.42, 1.13, 1.16 at $K=2$; 1.15, 1.04, 1.15 at $K=4$), so it is not
an artefact of reciprocated ties. And the first inbox moves agents with $a_+=1.65$ and 2.39: round-0 messages quote
raw sensor reports, and a reader treats them as independent evidence, closer to adding the $K$ messages than to
averaging them ($a_+/K=0.83$ and 0.60). The settled level keeps part of that first response. The transfer check
never showed a round-0 message.

### 1.3 Parameters

| parameter | white probes (maximum likelihood, bootstrap s.e. over 200 trajectories) | swarm (in-sample, pilot) |
|---|---|---|
| $\rho_0$ | 0.54 (0.08) | 0.71, set by $\rho_{\rm eff}$ at $u<0.08$ |
| $\rho_\infty$ | 0.19 (0.05) | same |
| $u_0$ | 0.25 (0.21) | same |
| $\sigma_0$, $\sigma_1$ | 0.045 (0.003), 0.185 (0.015) | same |
| $\pi_0$, $\mu_\pi$ | 0.195 (0.033), 0.064 (0.010) | $\pi_0=0.36$ ($K=2$), 0.47 ($K=4$), set by the late repeat rate |
| $q_J$, $\sigma_J$ | 0 | $q_J=0.050$ ($K=2$), 0.031 ($K=4$), set by the late moves $\ge0.15$; $\sigma_J=0.25$ fixed |

The white-probe likelihood improves by 219 units over the same model without gating (constant $\rho$, $\sigma$,
$\pi$; four fewer parameters). The gating scale $u_0$ is the weak point: its s.e. is as large as its value, and
with a plain tanh target for the white agent the fit moves to $\rho_0=0.66$, $u_0=0.11$. The function is better
determined than its parameters: $\rho(u)$ at $u=0.05,0.15,0.3,0.6$ is 0.48, 0.38, 0.29, 0.22 in the adopted fit
and 0.51, 0.35, 0.27, 0.24 in the other. (The white agent's target is $A\tanh(\beta\bar h/A+c)$ with $A=0.42$:
uniform template inboxes saturate near $s=\pm0.45$. That form is 85 likelihood units better than $A=1$ and is used
for the white fit only; swarm fields stay below 0.4 where the two agree.)

The swarm column is fitted by simulated moments on six one-step statistics of the pilot (repeat rate, fraction of
large moves, $\rho_{\rm eff}$ at small $u$, for each $K$), not on the thermal observables of section 1.4.

### 1.4 Check against both data sets

*White probes*, data against the law driven by the same field sequences (`surrogate_v2_out.txt`, check 1): repeat
fraction 0.045 against 0.045 (0.005); own-answer weight by $u$ bin 0.59, 0.45, 0.20, 0.06 against 0.55, 0.42, 0.27,
0.14; residual s.d. by bin 0.057, 0.064, 0.082, 0.105 against 0.057, 0.064, 0.080, 0.115. The law keeps too much of
the own answer at large $u$, so the pooled own-answer weight is 0.233 (0.015) against 0.187 and the residual lag-1
autocorrelation 0.290 (0.025) against 0.211, both three simulated standard deviations high.

*Pilot*, on the pilot's own graphs and evidence (`cells.graph_for`, `cells.game_for`), 200 simulated replica pairs
per cell (check 3). $c_1,c_2,P_\delta$ are the replica-difference statistics of `design_lib.thermal_observables`,
"repeat" the fraction of exactly repeated answers and $D$ the replica distance, both over rounds 6–10, $m$ the
truth-aligned mean.

| cell | | $c_1$ | $c_2$ | $P_\delta$ | repeat | $D$ | $m$ |
|---|---|---|---|---|---|---|---|
| $K=2$, $\gamma=-1$ | pilot | 0.72 | 0.44 | 2.03 | 0.22 | 0.013 | 0.23 |
| | v1 | 0.32 | 0.06 | 1.36 | 0.00 | 0.025 | 0.21 |
| | v2 | 0.67 | 0.41 | 1.83 | 0.26 | 0.018 | 0.21 |
| $K=2$, $\gamma=0$ | pilot | 0.71 | 0.49 | 1.74 | 0.28 | 0.020 | 0.29 |
| | v1 | 0.29 | 0.09 | 1.29 | 0.00 | 0.024 | 0.32 |
| | v2 | 0.70 | 0.47 | 1.79 | 0.26 | 0.018 | 0.32 |
| $K=2$, $\gamma=+1$ | pilot | 0.80 | 0.60 | 1.90 | 0.29 | 0.025 | 0.24 |
| | v1 | 0.33 | 0.19 | 1.20 | 0.00 | 0.025 | 0.24 |
| | v2 | 0.73 | 0.54 | 1.72 | 0.27 | 0.019 | 0.25 |
| $K=4$, $\gamma=-1$ | pilot | 0.68 | 0.33 | 2.42 | 0.32 | 0.011 | 0.27 |
| | v1 | 0.31 | 0.07 | 1.33 | 0.00 | 0.023 | 0.20 |
| | v2 | 0.71 | 0.47 | 1.95 | 0.35 | 0.013 | 0.25 |
| $K=4$, $\gamma=0$ | pilot | 0.68 | 0.40 | 2.06 | 0.39 | 0.006 | 0.27 |
| | v1 | 0.30 | 0.09 | 1.28 | 0.00 | 0.023 | 0.22 |
| | v2 | 0.73 | 0.51 | 1.86 | 0.36 | 0.013 | 0.28 |
| $K=4$, $\gamma=+1$ | pilot | 0.80 | 0.61 | 2.09 | 0.38 | 0.010 | 0.25 |
| | v1 | 0.31 | 0.14 | 1.25 | 0.00 | 0.023 | 0.19 |
| | v2 | 0.74 | 0.53 | 1.85 | 0.36 | 0.013 | 0.25 |

v1 is the old law with the real one-shot gains; v2 is the swarm column of section 1.3 with the pilot's gains.
Root-mean-square $z=(\text{pilot}-\text{mean})/\text{s.d.}$ over the six cells:

| law, gains | $c_1$ | $c_2$ | $P_\delta$ | repeat | $\lvert\Delta s\rvert$ round 1 | $\lvert\Delta s\rvert$ late | $D(1)$ | $D$ late | $m(1)$ | $m$ late |
|---|---|---|---|---|---|---|---|---|---|---|
| v1, transfer (one-shot) | 8.7 | 7.0 | 10.1 | 99 | 9.3 | 20.7 | 2.4 | 5.3 | 4.4 | 6.5 |
| white-probe law, transfer (target): nothing from the pilot | 4.2 | 3.2 | 5.3 | 13.4 | 16.7 | 5.6 | 8.3 | 11.7 | 5.2 | 8.7 |
| v2, transfer (target) | 1.3 | 1.7 | 2.8 | 1.7 | 13.9 | 1.5 | 3.5 | 1.5 | 4.5 | 5.4 |
| v2, pilot | 1.2 | 1.5 | 1.8 | 1.5 | 2.5 | 1.8 | 1.4 | 1.6 | 2.1 | 2.1 |

The second row is the out-of-sample statement: the white-probe law with the transfer gains gives $c_1=0.54$–0.60 and
$c_2=0.27$–0.39, which closes about two thirds of the gap between v1 and the pilot in $c_1$. The third row shows
that the anchoring parameters, not the gains, carry the thermal observables. The transfer gains fail on round 1
(first-inbox gains) and on the settled mean.

**What v2 does not reproduce.**
- The dependence of $c_2$ on $\gamma$ at $K=4$: the pilot has $c_2(+1)-c_2(-1)=+0.29$, the law $+0.07$ (s.d. 0.07 per
  pair of graphs), a 3 s.d. shortfall on one graph per cell. At $K=2$ it is $+0.17$ against $+0.13$ (0.10).
- The response to replica differences in the inbox, $b_A=0.37$ and 0.34 in the pilot against 0.28 in the law.
- Round 1: the mean one-round change is off by 2.5 simulated s.d. (r.m.s. over cells), and $D(1)$ of the $K=4$,
  $\gamma=-1$ cell is 0.038 against 0.025.
- The shape of the noise: Gaussian core plus one jump component, against kurtosis 9–13; and the persistence of
  repeats (an agent that repeated last round repeats again with probability 0.36 and 0.54, otherwise 0.23 and 0.28).
- Agent-specific offsets of the target (the fixed-point fit leaves a residual s.d. of 0.09–0.11 that is constant
  across replicas). They cancel in every replica-difference observable and are not modelled.

### 1.5 The measurement that calibrates the law directly

The law's two weak points are the gating scale $u_0$ and the assumption (Q1) that a single flipped message among
$K$ acts like a uniform change of size $1/K$. Both were extrapolated from uniform template inboxes at $K=8$.
The direct measurement is a **step response on settled contexts**, with single calls and no swarm rounds:

- *Context.* A settled agent of a finished run at round $t\ge6$: its evidence, its own round-$t$ answer and
  message, and the real round-$t$ messages of its senders, which it would read next.
- *Inbox conditions.* (i) natural: the round-$t$ messages as they are; (ii) verbatim: the round-$(t-1)$ messages
  again, so that nothing at all is new; (iii) one agree message mirrored; (iv) one disagree message mirrored; (v)
  all $K$ messages mirrored. The size of a mirror step is $2|s_j|$, so the senders' own spread supplies step
  sizes from 0 to about 1.6 without any synthetic text.
- *Replicates.* Four samples of each (context, condition) with distinct session tags.
- *Size.* 120 contexts × 5 conditions × 4 replicates = 2,400 calls per (wording, $K$), 9,600 calls for the four
  combinations, about \$3.6 at the pilot's \$0.00037 per call. The label wording needs label base runs first, which
  the kick experiment needs anyway.

It returns the repeat probability and the noise variance at zero input change (conditions i and ii, which a kick
twin can never contain, because an unchanged prompt is replayed from the cache), the response per unit step as a
function of step size, which is $(1-\rho(u))(1-\pi)\,d\,a/K$, and from (iii) against (v) at matched $u$ whether the
mean change or the largest single change gates the anchoring. The first-order responses of the kick experiment
measure the same curve for condition (iii) in the running swarm (section 3.5).

## 2. What still discriminates the cavity theory

### 2.1 Where the swarm sits

Effective couplings from the one-shot gains (`design_lib.effective`), with the slope factor $d=1-s^2\approx0.78$ of
the settled pilot:

| | $\bar J$ | $g$ | $\gamma_{\rm eff}$ at $\gamma=-1$ | $d\,g\,(1+\lvert\gamma_{\rm eff}\rvert)$ at $\gamma=-1$ | at $\gamma=+1$ |
|---|---|---|---|---|---|
| inverted, $K=2$ | 0.33 | 0.41 | −0.36 | 0.43 | 0.64 |
| inverted, $K=4$ | 0.22 | 0.25 | −0.60 | 0.31 | 0.39 |
| label, $K=2$ | 0.39 | 0.29 | +0.75 | 0.40 | 0.46 |
| label, $K=4$ | 0.30 | 0.16 | +0.84 | 0.23 | 0.25 |

The inverted wording no longer gives $\bar J\approx0$ (the disagree channel is nearly off), and the label wording
has positive effective reciprocity at $\gamma=-1$ because its disagree ties attract.

### 2.2 Anchoring moves the oscillatory boundary and not the static one

Linearize (Q4) at fixed $\rho$ around a settled state: $\delta s(t+1)=\rho\,\delta s(t)+(1-\rho)\,d\,W\delta s(t)$,
with $W$ the matrix of target gains. The single-site response is

$$R(\omega)=\frac{(1-\rho)\,d\,e^{-i\omega}}{1-\rho\,e^{-i\omega}},\qquad
|R(0)|=d,\quad |R(\tfrac\pi2)|=\frac{(1-\rho)\,d}{\sqrt{1+\rho^2}},\quad |R(\pi)|=\frac{(1-\rho)\,d}{1+\rho}.$$

For Gaussian couplings of variance $g^2/N$ and reciprocity $\gamma$ the eigenvalues of $dW$ fill the ellipse
$\lambda=d\,g\,[(1+\gamma)\cos\theta+i(1-\gamma)\sin\theta]$, and the swarm is stable while

$$\max_\theta\ \big|\rho+(1-\rho)\,\lambda(\theta)\big|<1. \tag{B1}$$

At the rightmost point this is $d\,g(1+\gamma)<1$ for every $\rho$: the static boundary of `claim2.md` is unchanged,
and what anchoring changes there is the time to reach it. At the top of the ellipse, which for $\gamma=-1$ is the
period-4 mode, it is $\rho^2+(1-\rho)^2d^2g^2(1-\gamma)^2<1$, that is

$$d\,g\,(1+|\gamma|)<\sqrt{\frac{1+\rho}{1-\rho}}\qquad(\gamma=-1). \tag{B2}$$

`check_kick_echo.py` part D compares (B1) with the spectral radius of $N=1500$ matrices: 0.750 against 0.750 and
1.136 against 1.140 at $\rho=0.7$, $\gamma=-1$, $d\,g=0.45$ and 1.5. All eight cases at $\rho=0.7$ agree within
0.010 and the eight at $\rho=0$ within 0.05 (the finite-size edge of the ellipse). The critical $d\,g$ at
$\gamma=-1$ is 0.50 for $\rho=0$ and 1.19 for $\rho=0.7$. For the strongest cell (inverted, $K=2$,
$\gamma_{\rm eff}=-0.36$) the critical value moves from 0.73 to 1.51 at $\rho=0.71$, against $d\,g=0.40$ with the
target gains (`surrogate_v2_out.txt`, check 6). The period-4 signature is a factor 3.8 away in gain, not 1.8.

The static direction is closer than the table suggests. The level gains are the target gains, and on the actual
$K=2$, $\gamma=+1$ graphs (whose reciprocated ties form cycles) the largest eigenvalue of $dW$ is 0.73 with the
transfer gains and 0.84 with the pilot's (check 6). That cell has slow modes, and it is the one pilot cell in which
$D(t)$ grew. It helps little: in the simulated law $D(30)$ is 17–23% larger at $\gamma=+1$ than at $\gamma=-1$ for
$K=2$ and the same for $K=4$, because few modes are slow, and the per-graph spread of $D$ is about 30%.

### 2.3 The thermal observables measure the anchoring

- $c_1\approx\rho_{\rm eff}$. The pilot's 0.68–0.80 is the own-answer weight of section 1.1.
- For any stationary sequence $\sum_t(\delta(t)-\delta(t-k))^2=2\sum_t\delta(t)^2\,(1-c_k)$, so
  $P_\delta=(1-c_2)/(1-c_1)$, and $=1+\rho$ for an AR(1) process. The pilot has $P_\delta=2.03,1.74,1.90$ at $K=2$
  against $(1-c_2)/(1-c_1)=2.00,1.76,1.99$, and $2.42,2.05,2.09$ at $K=4$ against $2.11,1.86,1.89$ (there $D$ was
  still decaying). Values near 2 were to signal period 4; here they signal $\rho\approx0.7$.
- Amplification $\mathcal A=D/D_{\rm floor}$: anchoring alone gives $1/(1-\rho^2)\approx2$, the coupling changes
  the denominator by $(1-\rho)^2d^2g^2\le0.013$, and $D_{\rm floor}$ is itself state-dependent (section 1.1).
  $\hat\Lambda$ of `claim2.md` eq. (9) is not measurable.

### 2.4 Status of the original predictions

| prediction | status in the quiet, anchored regime |
|---|---|
| period 4 at $\gamma<0$, $\mathcal P>1.5$ (`claim2.md`, README test 2) | unmeasurable: boundary (B2) is a factor 3.8 away, and $P_\delta$ is a function of $c_1,c_2$ |
| period-2 flicker at $\gamma>0$ above the boundary | unmeasurable: $\lvert R(\pi)\rvert=0.17\,d$ |
| $\mathcal A=1/(1-\Lambda)$, $\hat\Lambda$ from the growth of $D$ | unmeasurable: dominated by $1/(1-\rho^2)$, floor not constant |
| sign change of $c_2$ at $\gamma=-1$ (old design) | gone: $c_2\ge0.33$ from anchoring; the shift between $\gamma=\pm1$ is predicted at $+0.13$ ($K=2$) and $+0.07$ ($K=4$) with per-pair s.d. 0.10 and 0.07, so 5 and 11 graph pairs give 3 s.e., but the size depends on the fitted law and the pilot's $K=4$ value is already 3 s.d. off |
| zero-shot DMFT curves from calibration (README test 3) | not zero-shot any more: the law's swarm parameters and level gains were fitted on the pilot. A new wording or $K$ is a fair test of the law, simulated on the graph |
| overlap excess $\Delta q$ | measurable: $+0.041$ (s.d. 0.016 per graph) at $K=2$, $+0.018$ (0.011) at $K=4$, $\gamma=0$; 2 and 4 graphs for 3 s.e. It tests the quenched-noise term (5) of `claim1.md`, not the reaction term |
| reaction term $\Gamma R$ | measurable directly by kick and echo (section 3) |

The echo is the one quantity here whose prediction is zero for a reason of structure ($\gamma=0$), changes sign
with a graph property and with the wording, and does not depend on the noise model.

## 3. Kick and echo

### 3.1 Protocol

Restore one replica of a finished run after round $t_0$. Replace agent $i$'s message, as its readers will see it,
by the mirrored message ($p\to1-p$, HEADS and TAILS exchanged), leaving $i$'s own history true; the kick is
$D_i=-2s_i(t_0)$. Run rounds $t_0+1$ and $t_0+2$. At $t_0+1$ agent $i$ sends its true answer again, so the kick is
an impulse. An agent whose prompt is unchanged is replayed from the cache and returns its base answer exactly; an
agent whose prompt changed is resampled. Three properties are exact:

1. $i$'s answer at $t_0+1$ is its base answer (it does not read itself).
2. At $t_0+1$ only the readers of $i$ change.
3. $i$'s answer at $t_0+2$ can change only if $i$ reads one of its readers. On the fixed in-degree graphs with
   $\gamma=0$ no tie is reciprocated and the echo is identically zero.

`check_kick_echo.py` part A verifies them through `envs.swarm` with the deterministic `MockBackend`: a twin built
from `Swarm._context` agrees with a direct numpy twin to $2\times10^{-16}$ for all 64 kicks, the kicked agent never
changes at $t_0+1$, only readers change, and the largest echo at $\gamma=0$ is 0.0. `envs.kick.kick_twin` (the
engine implementation) agrees with the numpy twin to $10^{-3}$, which is the two-decimal rounding of the mirrored
probability.

### 3.2 Linear response

Let $W_{jk}=a_+/K$ for an agree tie ($J_{jk}=+1$), $-a_-/K$ for a disagree tie, with the one-shot gains, and
$d_j=1-s_j^2$. To first order in the kick,

$$\Delta s_j(t_0+1)=d_j\,W_{ji}\,D_i, \tag{K1}$$

$$\Delta s_i(t_0+2)=d_i\sum_{j:\,J_{ij}\ne0,\,J_{ji}\ne0}W_{ij}\,d_j\,W_{ji}\;D_i. \tag{K2}$$

The estimators are slopes through the origin, which weight each twin by its kick and make near-zero kicks harmless:

$$r_{\rm type}=\frac{\sum\Delta s_j(t_0+1)\,D_i}{\sum D_i^2}\ \ (\text{readers }j\text{ with }J_{ji}=\text{type}),\qquad
E=\frac{\sum_i\Delta s_i(t_0+2)\,D_i}{\sum_iD_i^2}.$$

**Relation to the cavity equations.** In `claim1.md` eq. (4) the field on $i$ is mean $+$ noise $+$ reaction,
$\sum_b\Gamma_{ab}\sum_{t'<t}R_b(t,t')s_a(t')$. A kick that only $i$'s readers see changes $m$ by $D_i/N$, does not
touch the noise term (which is built from trajectories without $i$) and leaves $i$'s own history alone. It enters
through the reaction term only: $\delta h_i(t_0+1)=\Gamma R(t_0+1,t_0)D_i$, so

$$E=\langle d_i\rangle_w\ \Gamma\ R,\qquad \Gamma=\frac{r\,\langle a_{ij}a_{ji}\rangle_{\rm recip}}{K},\quad R=\langle d_j\rangle, \tag{K3}$$

with $r=n_r/K$ the fraction of reciprocated ties and $\langle\cdot\rangle_w$ the average with weights $D_i^2$. On a
graph with few ties the slope factors of partners are correlated with the tie type, and the form to use is

$$E=\sum_{\rm types}\frac{n_{\rm type}}{K^2}\,a_{ij}a_{ji}\,\mathcal D_{\rm type},\qquad
\mathcal D_{\rm type}=\frac{\sum_iD_i^2\,d_i\,\langle d_j\rangle_{j\ {\rm recip.\ of\ type}}}{\sum_iD_i^2}. \tag{K4}$$

At $p_-=0.5$ there are $n_r=\mathrm{round}(|\gamma|K)$ reciprocated ties per agent: for $\gamma>0$ half agree–agree
($a_{ij}a_{ji}=a_+^2$) and half disagree–disagree ($a_-^2$); for $\gamma<0$ all mixed ($-a_+a_-$). Hence the signs:

| | $\gamma>0$ | $\gamma=0$ | $\gamma<0$ |
|---|---|---|---|
| inverted ($a_->0$) | $+$ | 0 | $-$ |
| label ($a_-<0$) | $+$ | 0 | $+$ |

and two parameter-free relations: $E(\pm0.5)=\tfrac12E(\pm1)$ (echoes add over reciprocated ties) and
$E\propto1/K$ at fixed gains.

### 3.3 Check on the mock

Deterministic agents $s(t+1)=\tanh(Ws+a_ph^{\rm priv})$ on fixed in-degree graphs, $N=64$, gains set to the real
one-shot gains, 20 graphs × 64 kicks per cell, $t_0=8$ (`check_kick_echo.py` part B).

- *Per agent, small kick* ($D_i\to0.01D_i$): largest relative error of (K1) $6.2\times10^{-3}$, of (K2)
  $7.1\times10^{-3}$, both of the order of the kick; the echo is identically zero at $\gamma=0$.
- *Population.* $E$ from the exact twin, (K2) summed on the graph, and (K3):

| gains | $K$ | $\gamma$ | $\Gamma$ | $E$ exact | (K2) on graph | (K3) | $E$ at full mirror kick | ratio |
|---|---|---|---|---|---|---|---|---|
| inverted | 2 | −1 | −0.0600 | −0.0323 | −0.0323 | −0.0327 | −0.0319 | 0.99 |
| inverted | 2 | +1 | +0.1656 | +0.0779 | +0.0776 | +0.0788 | +0.1013 | 1.30 |
| inverted | 4 | −1 | −0.0363 | −0.0211 | −0.0211 | −0.0211 | −0.0208 | 0.99 |
| inverted | 4 | +1 | +0.0605 | +0.0324 | +0.0323 | +0.0325 | +0.0345 | 1.07 |
| label | 2 | −1 | +0.0648 | +0.0322 | +0.0322 | +0.0330 | +0.0348 | 1.08 |
| label | 2 | +1 | +0.0873 | +0.0445 | +0.0444 | +0.0452 | +0.0498 | 1.12 |
| label | 4 | −1 | +0.0210 | +0.0117 | +0.0117 | +0.0117 | +0.0119 | 1.02 |
| label | 4 | +1 | +0.0255 | +0.0137 | +0.0137 | +0.0138 | +0.0141 | 1.03 |

(K3) agrees with the exact twin to 3% in every cell, including the $|\gamma|=0.5$ cells in `out_kick_echo.txt`,
and the sign table holds. A full mirror kick is not small: through agree–agree ties at $K=2$ in the inverted
wording the tanh raises the echo by 30% at $\gamma=+1$ and 21% at $\gamma=+0.5$, because the kick moves the partner
toward zero where the slope is larger. In the other cells the nonlinearity is within 12%.

- *Additivity* (part C): $E(\pm0.5)/E(\pm1)$ is 0.48–0.52 in all eight cases.

### 3.4 The factorization test

The echo can be predicted from first-order responses measured in the same experiment, with no gain from any
calibration. The realized changes $\Delta s_j(t_0+1)$ of the reciprocated partners are what $i$ reads at $t_0+2$,
noise included, so they are the regressor, without an errors-in-variables problem:

$$\mathrm{pred}_i=d_i\sum_{j\ {\rm recip.}}c(J_{ij})\,\Delta s_j(t_0+1),\qquad
c({\rm type})=\frac{\sum\Delta s_j\,d_j\,D_i}{\sum d_j^2\,D_i^2},\qquad
F=\frac{\sum_i\Delta s_i(t_0+2)\,\mathrm{pred}_i}{\sum_i\mathrm{pred}_i^2}. \tag{K5}$$

$c$(type) is the first-order response per unit slope factor, taken from the other twins of the cell. $F=1$ says that
an agent responds to the returning change of its partner as a reader responds to a kick of the same edge type:
linear responses compose, which is what the reaction term assumes.

**The slope factor is needed.** With the type-mean response in place of $c\,d_i$, that is
$\mathrm{pred}_i=\sum_j r_{{\rm type}(ij)}\Delta s_j$, the slope is $\langle d_i\rangle_w/\langle d_j\rangle$ and not
1, because the estimators weight a kicked agent by $D_i^2=4s_i^2$ and confident agents have small $d_i$. On the
mock, where composition is exact, the raw slope is 0.79–0.87 (predicted 0.78–0.81), and (K5) gives 0.999–1.000 at
small kicks and 0.96–1.02 at full mirror kicks (part C). `experiments/07_kick/analyze_kick.py` uses the raw form
at the time of writing: its predicted echo would exceed the measured one by a factor of about 1.25 for tanh agents
that compose perfectly.

**Open two-step paths are the control.** At $t_0+2$ every agent $k\ne i$ that reads a reader of $i$ and is not
itself a reader gives the same regression with $\mathrm{pred}_k=d_k\sum_jc(J_{kj})\Delta s_j$, slope $F_{\rm open}$.
There are about 2 such agents per twin at $K=2$ and 11 at $K=4$ (4 and 14 at $\gamma=0$). A single-agent
nonlinearity lowers $F$ and $F_{\rm open}$ together; only a different treatment of the returning signal separates
them. In the deterministic control (part E, last block) $F$ and $F_{\rm open}$ are 1.03 and 0.92 at $K=2$,
$\gamma=+1$ (the tanh effect above, which acts on the closed path only) and within 0.03 of each other in the other
three cells.

**What anchoring does.** Under (Q3)–(Q5) the response per unit input is $G(u,\mu)=(1-\rho(u))(1-\pi(\mu))\,d\,a/K$
and grows with the size of the change. The first step is a large change of one message ($u\approx|D_i|/K$), the
second a small one ($u\approx r|D_i|/K$), so $F\approx G(u_2)/G(u_1)<1$ on closed and open paths alike, and the
first-order response itself is smaller for small kicks.

### 3.5 Predictions of the anchoring law

The protocol on the simulated law (`check_kick_echo.py` part E): swarm parameters of section 1.3, target gains from
the transfer check, 32 graphs × 2 replicas × $t_0\in\{6,8\}$ × agents with $|s_i|\ge0.1$, about 7,000 twins per
cell, every agent with a changed prompt resampled. With anchoring, repeats and noise switched off the same code
returns the mock numbers of section 3.3 ($E$ ratio to (K2) 1.31, 0.98, 1.07, 0.99).

| | $K$ | $r_A$ | $r_D$ | $E$ at $\gamma=-1$ | $-0.5$ | $+0.5$ | $+1$ | $E/E_{\rm lin}$ at $-1$, $+1$ | $F$ | $F_{\rm open}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| inverted | 2 | +0.30 | −0.05 | −0.0128 | −0.0056 | +0.0313 | +0.0672 | 0.40, 1.00 | 0.69, 0.83 | 0.75, 0.62 |
| inverted | 4 | +0.10 | −0.03 | −0.0063 | −0.0011 | +0.0044 | +0.0132 | 0.30, 0.42 | 0.69, 0.52 | 0.56, 0.51 |
| label | 2 | +0.20 | +0.09 | +0.0208 | +0.0102 | +0.0124 | +0.0307 | 0.65, 0.74 | 0.59, 0.66 | 0.54, 0.61 |
| label | 4 | +0.06 | +0.03 | +0.0030 | +0.0014 | +0.0012 | +0.0047 | 0.27, 0.37 | 0.31, 0.37 | 0.43, 0.38 |

Monte Carlo s.e. of $E$ is 0.0010–0.0015; $F$ and $F_{\rm open}$ are given at $\gamma=-1,+1$ with s.e. 0.02–0.06
(0.10 for label, $K=4$). Pooled over the $|\gamma|=1$ cells, $F=0.75\pm0.01$ and $F_{\rm open}=0.65\pm0.01$ at
$K=2$, $F=0.53\pm0.03$ and $F_{\rm open}=0.51\pm0.02$ at $K=4$. With the pilot's gains (inverted only) $E$ is
$-0.025$, $+0.113$ at $K=2$ and $-0.004$, $+0.029$ at $K=4$ for $\gamma=-1,+1$.

Three signatures separate the anchoring law from linear response:

- echoes through the weak channel shrink most: $E(-1)/E(+1)$ is $-0.19$ for inverted $K=2$ against $-0.49$ in linear
  response;
- $F\approx0.75$ ($K=2$) and 0.5 ($K=4$) against 1, on open paths as well;
- the agree-tie response to the smallest third of kicks ($|D|\le0.55$) is 0.64–0.77 of the response to the largest
  third ($|D|\ge1$), against 0.96–1.07 in the control. Every twin contributes $K$ readers to this ratio, so it is
  the best-determined of the three and calibrates $\rho(u)$ in the running swarm.

### 3.6 Noise of a single echo and number of kicks

One echo is the difference of two independent answers of the same settled agent to slightly different prompts,
plus the propagated resampling of its partners. Without a model, $\sigma_y^2\approx2\langle(s(t+1)-s(t))^2\rangle$
over the pilot's rounds 6–10: $\sigma_y=0.089$ ($K=2$), 0.066 ($K=4$). In the simulated law the residual s.d. of
an echo around $E\,D_i$ is 0.09–0.12 ($K=2$) and 0.07–0.08 ($K=4$), with kurtosis 8–19, and 5–7% ($K=2$) or 13–14%
($K=4$) of echoes are exactly zero (base and twin both repeat). With slopes through the origin,

$$\mathrm{s.e.}(E)=\frac{\sigma_y}{\sqrt{n\,\langle D^2\rangle}},\qquad n_{3}=\frac{9\,\sigma_y^2}{E^2\,\langle D^2\rangle},$$

with $\langle D^2\rangle=0.7$–1.1 at the settled state. The heavy tails do not invalidate this: the sandwich
s.e. and the spread over 400 random subsets of 200 kicks agree within 10% in every cell (last column of part E).
Kicks needed for $|E|=3$ s.e.:

| | $K$ | linear response: $\gamma=-1$ | $-0.5$ | $+0.5$ | $+1$ | anchoring law: $\gamma=-1$ | $-0.5$ | $+0.5$ | $+1$ |
|---|---|---|---|---|---|---|---|---|---|
| inverted | 2 | 100 | 399 | 96 | 24 | 609 | 2,731 | 100 | 30 |
| inverted | 4 | 133 | 531 | 236 | 59 | 1,797 | >10,000 | 3,050 | 397 |
| label | 2 | 93 | 371 | 266 | 67 | 226 | 837 | 610 | 115 |
| label | 4 | 430 | 1,721 | 1,380 | 345 | 6,852 | >10,000 | >10,000 | 2,612 |

For the factorization slope, s.e.$(F)=0.15$ needs 90–280 kicks (inverted, $K=2$), 230–390 (label, $K=2$), 610–1,200
(inverted, $K=4$) and about 3,500 (label, $K=4$) at $|\gamma|=1$ under the anchoring law. $F_{\rm open}$ has 2 to 11
times as many terms per twin and is always better determined than $F$.

**Recommendation: 2,048 twins per cell** (8 graphs × 2 replicas × 64 agents × 2 values of $t_0$). That gives
s.e.$(E)\approx0.0025$ at $K=2$ and 0.0019 at $K=4$. Every $|\gamma|=1$ echo of linear response is then at least
5.8 s.e. from zero (the smallest is label, $K=4$: 0.011). Under the anchoring law the $|\gamma|=1$ echoes are 5 to
25 s.e. at $K=2$, 3.2 and 6.8 s.e. for inverted $K=4$, and 1.7 and 2.7 s.e. for label $K=4$, which is the one
combination where the sign may stay open. The two predictions of $E$ differ by more than two combined standard
deviations (measurement and the uncertainty of the linear prediction) in five $|\gamma|=1$ cells: inverted $K=2$,
$\gamma=-1$ (2.8), inverted $K=4$ (3.2, 3.4) and label $K=4$ (2.7, 2.3). They do not differ usefully at $\gamma=+1$
for $K=2$ or in the label $K=2$ cells (1.0–1.8), where $F$ and $r_A(|D|)$ decide. s.e.$(F)$ is 0.03–0.07 at $K=2$,
0.08–0.12 for inverted $K=4$ and 0.2 for label $K=4$.

### 3.7 Design choices

**$t_0$: settled, with an early arm at $K=4$.** Settled kicks ($t_0\ge6$; use 6 and 8, and 5 and 7 if more twins
are wanted) are the primary measurement: the state is stationary, so the first-order responses measured at
$t_0+1$ are the right ones for the second step, and the frozen linear prediction applies. Early kicks ($t_0=0$,
mirroring a round-0 sensor report) meet the first-inbox gains and little anchoring. In the simulated law with the
pilot's gains they give $E=-0.042$, $+0.138$ at $K=2$ and $-0.019$, $+0.093$ at $K=4$, and need 71, 19, 283 and 21
kicks for 3 s.e. Settled kicks with the same gains need 175, 18, 5,161 and 90, so the gain is a factor 4 and 18 at
$K=4$ and at most 2.5 at $K=2$. They use the same base runs. Two cautions: the
factorization for $t_0=0$ needs the first-order responses of round 2, so it needs kicks at $t_0=1$ as well; and the
early prediction rests on first-inbox gains known only for the inverted wording, in-sample. Run $t_0\in\{0,1\}$ as
a secondary arm, mainly at $K=4$ and for the sign test.

**Kick size: the mirror, all agents with $|s_i|\ge0.1$.** The mirror is the only change of a message that keeps its
text consistent with its number. Its size $2|s_i|$ is set by the agent, and the slope estimators already weight by
it. Large kicks are where the transfer gains apply and small kicks are where anchoring shows, so the analysis
should report $r_A$ by thirds of $|D|$ and not select on size beyond dropping $|s_i|<0.1$, whose kicks are at most
0.2 and add almost nothing to $\sum D^2$. Selecting on anything that happens after $t_0$ would bias the echo.

**One kick per twin at $N=64$.** A twin costs only the calls of resampled agents, about $2K$ plus the open-path
agents: 7–8 calls at $K=2$ and 20–22 at $K=4$, \$0.003 and \$0.008. Two kicks in one twin cost the same calls as two
twins, so packing saves nothing in spend, only wall-clock time. Kicks $i\ne k$ can share a twin only if no agent
reads both, neither reads the other and neither reads a reader of the other (`envs.kick.compatible_kicks` has
these conditions), and the open-path control needs in addition that no agent reads readers of both. At $N=64$,
$K=4$ the two-step out-neighbourhood of one kick is about 20 agents, so at most 2 or 3 kicks fit. Pack only at
large $N$.

**Cost.** Per cell: 8 base runs (\$0.50 each) plus 2,048 twins, \$10 at $K=2$ and \$20 at $K=4$. The $\gamma=0$
cells need only 512 twins (their echo is zero without any call, and they serve the first-order and open-path
statistics), so the 20 cells cost about \$270. Stage 1, both wordings at $K=2$ and $\gamma\in\{-1,0,+1\}$, costs
about \$50 and already decides the sign pattern, the size of the weak-channel echo and $F$. These are estimates
from the pilot's \$0.00037 per call; gate (c) of `CLAUDE.md` applies. The step-response probes of
section 1.5 (\$3.6) should run first, since they fix $\sigma_y$ and the repeat rate independently.

### 3.8 Frozen predictions for pre-registration

Settled mirror kicks, $t_0\in\{6,8\}$, $N=64$, $p_-=0.5$, slopes through the origin over agents with $|s_i|\ge0.1$.
"Linear" is (K1) and (K4) with the one-shot real-message gains; its uncertainty is one s.d. from the gain s.e. of
the transfer check and from the slope factors (6% each for inverted, measured on the pilot's settled states: 
$d_A=0.68$, $d_D=0.76$, $\mathcal D=0.34$–0.47 at $K=2$ and 0.78, 0.82, 0.49–0.54 at $K=4$; 12% for label, where
they come from the simulated law). "Anchoring" is the simulated law of section 3.5. Source: `kick_predictions.json`
and `kick_surrogate_table.json`.

| wording | $K$ | $\gamma$ | $r_A$ linear | $r_D$ linear | $E$ linear | $E$ anchoring | sign |
|---|---|---|---|---|---|---|---|
| inverted | 2 | −1 | +0.275 ± 0.020 | −0.057 ± 0.009 | −0.0285 ± 0.0051 | −0.0128 | − |
| inverted | 2 | −0.5 | | | −0.0142 ± 0.0025 | −0.0056 | − |
| inverted | 2 | 0 | | | 0 exactly | 0 exactly | 0 |
| inverted | 2 | +0.5 | | | +0.0290 ± 0.0034 | +0.0313 | + |
| inverted | 2 | +1 | | | +0.0581 ± 0.0067 | +0.0672 | + |
| inverted | 4 | −1 | +0.128 ± 0.012 | −0.046 ± 0.008 | −0.0200 ± 0.0039 | −0.0063 | − |
| inverted | 4 | −0.5 | | | −0.0100 ± 0.0019 | −0.0011 | − |
| inverted | 4 | 0 | | | 0 exactly | 0 exactly | 0 |
| inverted | 4 | +0.5 | | | +0.0150 ± 0.0023 | +0.0044 | + |
| inverted | 4 | +1 | | | +0.0300 ± 0.0046 | +0.0132 | + |
| label | 2 | −1 | +0.203 ± 0.026 | +0.098 ± 0.015 | +0.0329 ± 0.0065 | +0.0208 | + |
| label | 2 | −0.5 | | | +0.0164 ± 0.0033 | +0.0102 | + |
| label | 2 | 0 | | | 0 exactly | 0 exactly | 0 |
| label | 2 | +0.5 | | | +0.0194 ± 0.0037 | +0.0124 | + |
| label | 2 | +1 | | | +0.0388 ± 0.0075 | +0.0307 | + |
| label | 4 | −1 | +0.079 ± 0.011 | +0.043 ± 0.007 | +0.0110 ± 0.0024 | +0.0030 | + |
| label | 4 | −0.5 | | | +0.0055 ± 0.0012 | +0.0014 | + |
| label | 4 | 0 | | | 0 exactly | 0 exactly | 0 |
| label | 4 | +0.5 | | | +0.0062 ± 0.0013 | +0.0012 | + |
| label | 4 | +1 | | | +0.0123 ± 0.0027 | +0.0047 | + |

$r_A$, $r_D$ do not depend on $\gamma$; under the anchoring law they are $+0.30$, $-0.05$ (inverted, $K=2$), $+0.10$,
$-0.03$ (inverted, $K=4$), $+0.20$, $+0.09$ (label, $K=2$), $+0.06$, $+0.03$ (label, $K=4$).

| relation | linear response | anchoring law |
|---|---|---|
| factorization slope $F$ of (K5), closed paths | 1 (mock: 0.96–1.02) | 0.75 ($K=2$), 0.53 ($K=4$) |
| $F_{\rm open}$, open two-step paths | 1 (mock: 0.92–0.99) | 0.65 ($K=2$), 0.51 ($K=4$) |
| $F/F_{\rm open}$ | 1, up to $+0.12$ at $K=2$, $\gamma=+1$ | 1.15 ($K=2$), 1.04 ($K=4$) |
| $r_A$ at $\lvert D\rvert\le0.55$ over $r_A$ at $\lvert D\rvert\ge1$ | 1 (control: 0.96–1.07) | 0.64–0.77 |
| $E(\pm0.5)/E(\pm1)$ | 0.5 (mock: 0.48–0.52) | 0.40–0.49 at $K=2$; not resolved at $K=4$ |
| $E(\gamma=-1)/E(\gamma=+1)$, inverted, $K=2$ | −0.49 | −0.19 |

**Reading the outcome.**
- *Structure of the theory.* The effective-coupling picture fails if the echo at $|\gamma|=1$, $K=2$ has the wrong
  sign at 3 s.e. in either wording, or if the label wording gives a negative echo at $\gamma=-1$ while its measured
  $r_D$ is positive. The sign prediction uses only the signs of the measured first-order responses.
- *Composition.* $F$ and $F_{\rm open}$ equal within their errors and the stated tanh offset: the returning signal
  is processed like any other, and the reaction term is $\Gamma R$ with the population's response, whatever the
  value of $F$. $F$ below $F_{\rm open}$ by more than 0.2: agents discount the return of their own influence, and
  the reaction term of `claim1.md` overstates the echo.
- *Single-agent law.* $F\approx1$ and a flat $r_A(|D|)$: linear response with the transfer gains is enough, and the
  anchoring seen in section 1 does not act on single-message changes. $F\approx0.5$–0.75 with $r_A$ rising with
  $|D|$: the anchoring law, with the measured $r_A(|D|)$ as its calibration.
- *Magnitude.* $E$ between the two columns is consistent with either at intermediate anchoring. $|E|$ above the
  linear column by more than 3 s.d. at $\gamma<0$ would mean the in-swarm gains exceed the transfer gains for
  single messages, as the pilot's level gains (20–38% higher) already suggest.

### 3.9 Not verified

- (Q1) for a single flipped message. The gating was fitted on uniform changes of eight template messages, and its
  scale has a relative s.e. near 1. All "anchoring" numbers in sections 3.5–3.8 inherit this, the $K=4$ ones most.
- Label wording in a swarm. Its level gains, first-inbox gains, settled states and anchoring parameters are
  assumed equal to the inverted wording's or taken from the transfer check; no label swarm has been run.
- Whether a mirrored settled message is read with the one-shot gains of the transfer check. The transfer probes
  showed such messages to an agent in its round-0 state, not to a settled one.
- Self-recognition. The kick returns to agent $i$ the consequence of a message it did not write. In the undisturbed
  swarm the returning signal carries $i$'s own sensor reports, which $i$ may recognize and discount. The kick
  measures the composed response; the reaction term of the natural dynamics equals it only if agents do not
  recognize their own content. The open-path control does not test this.
- The noise of an echo, $\sigma_y$, is inferred from late moves and from the law. The step-response probes measure
  it directly.
- Early kicks: predictions exist only for the inverted wording and use in-sample first-inbox gains.
- The persistence of repeats and the agent-specific offsets of section 1.4 are not in the law.
- No single-site DMFT of (Q1)–(Q6) was derived. Its field enters through $\tau$ and through $u$, a second linear
  statistic of the neighbours' increments, so it is a two-channel case of `claim1.md` eqs. (11)–(12) with
  $\psi_2=|s(t)-s(t-1)|$; this was not worked out or simulated.

## 4. Summary of what is established

| statement | status |
|---|---|
| own-answer weight falls with the inbox change, in white probes and pilot alike (section 1.1) | measured, model-free |
| law (Q1)–(Q6) with white-probe parameters closes two thirds of the $c_1$ gap | fitted on white probes, tested on the pilot |
| v2 reproduces ten pilot observables to r.m.s. $z=1.2$–2.5 | in-sample for three law parameters and the gains |
| one-shot gains are $(1-\rho_\infty)$ times target gains; first-inbox gains are 1.4–2.1 times the settled ones | follows from the law; gains fitted on the pilot |
| boundary (B1), (B2) with anchoring | derived, checked numerically |
| $P_\delta=(1-c_2)/(1-c_1)$ | identity for stationary sequences, checked on the pilot |
| (K1)–(K4), sign table, additivity, exact zero at $\gamma=0$ | derived, checked on the mock and through `envs.swarm` |
| (K5) needs the slope factor; raw slope is $\langle d_i\rangle_w/\langle d_j\rangle\approx0.8$ | derived, checked on the mock |
| echo magnitudes, $F$, kicks needed under anchoring | simulation of a fitted law |
| $\sigma_y$, s.e. formula | from pilot moves and simulation; sandwich formula checked against subsampling |
