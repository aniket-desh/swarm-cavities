# Claim 6: the mechanistic bridge (brief)

**Verdict.** As written the claim is close to unfalsifiable in one direction and self-contradictory in another. A
monotone relation between "a projection onto $\mathbf v_{\rm soc}$" and the behavioural kernel $R$ is guaranteed by a
confound (both shrink when the agent is confident), and the spec's recipe for $\mathbf v_{\rm soc}$ extracts a
direction that encodes *what the peers said*, whose steering shifts the agent's belief like an external field and
does not change the coupling $g_{\rm eff}$. A meaningful version exists and its core can be tested on a single agent
in a few GPU-hours, before any swarm is run. Nothing in this note has been checked numerically; it is analysis and
experimental design.

## 1. How a residual-stream direction relates to $R$

For an open-weight agent answering a binary question, let $\ell$ be the logit difference at the answer position, so
$s=\tanh(\ell/2)$. The one-round behavioural kernel is

$$R=\frac{\partial\langle s\rangle}{\partial h}=\tfrac12(1-s^2)\,\frac{\partial\ell}{\partial h},\qquad
\frac{\partial\ell}{\partial h}=\Big\langle\nabla_{a_L}\ell,\ \frac{\partial a_L}{\partial h}\Big\rangle, \tag{1}$$

where $a_L$ is the residual stream at layer $L$ (all positions at and after the peer messages) and $h$ is the social
field, varied by changing the peers' reported beliefs. Split the Jacobian along a unit direction $\mathbf v$:
$\partial a_L/\partial h=e\,\mathbf v+a_\perp$. Then

$$\frac{\partial\ell}{\partial h}=\underbrace{e}_{\text{encoding}}\cdot\underbrace{\langle\nabla_{a_L}\ell,\mathbf v\rangle}_{\text{readout }\varrho}
+\langle\nabla_{a_L}\ell,a_\perp\rangle,\qquad f\equiv\frac{e\,\varrho}{\partial\ell/\partial h}. \tag{2}$$

$\hat R_{\rm mech}=\tfrac12(1-s^2)\,e\,\varrho$ equals $R$ exactly when the mediation fraction $f=1$, that is when the
part of the social signal orthogonal to $\mathbf v$ is not read out. $f$ is measurable without gradients: project
$\mathbf v$ out of $a_L$ and remeasure the slope of $\ell$ against $h$; the slope that remains is $(1-f)$ of the
original. So the bridge claim, stated properly, is "$f$ is close to 1 and stable across prompts and tasks".

## 2. What the claim needs in order to mean something

1. **Remove the confidence confound.** $R$ contains the factor $\tfrac12(1-s^2)$: confident agents respond less,
   whatever the mechanism. Any probe correlated with confidence will "monotonically predict" $R$. The test has to be
   on the logit slope $\partial\ell/\partial h$, or on $R$ with $1-s^2$ partialled out, and must beat the baseline
   that predicts $R$ from $s$ alone.
2. **Distinguish a content direction from a gain direction.** The spec extracts $\mathbf v_{\rm soc}$ from "contrastive
   pairs with the same private evidence and opposing peer consensus". That contrast isolates the representation of
   the peers' opinion, i.e. of $h$ itself. Adding $-\alpha\mathbf v$ along it is equivalent to shifting $h$ by a
   constant: the conformity curve moves sideways, its slope is unchanged, and so are $g_{\rm eff}$, $\chi$ and the
   stability parameter $\Lambda$ of `claim2.md`. Such steering can move a swarm across the zealot saddle-node of
   `claim4.md`, but as a bias toward one answer, the same as changing the private evidence. A dial on the coupling
   needs a *gain* direction: the contrast between contexts in which the agent defers and contexts in which it
   anchors, at matched and sign-balanced peer content (for example, high against low measured slope, or prompts that
   assign high against low trust to peers).
3. **Check that additive steering changes a gain at all.** Adding a vector changes a slope only through a
   downstream multiplicative interaction (for instance attention to peer tokens gated by a trust feature). This is
   plausible and not guaranteed; it is the empirical content of the claim.
4. **Measure the side effects in the same units.** The private-evidence slope $\partial\ell/\partial h^{\rm priv}$ and
   accuracy on peer-free questions under the same steering. "Cleanly across the phase boundary" means the social
   slope changes by the required factor while these stay within a stated tolerance.

If a gain direction exists with dose-response $b(\alpha)=\partial\ell/\partial h$, the macroscopic consequences need no
further mechanism: $g_{\rm eff}(\alpha)=g\,b(\alpha)/b(0)$, hence $\Lambda(\alpha)=\Lambda(0)\,[b(\alpha)/b(0)]^2$
(`claim2.md` eq. (8)), and the zealot thresholds follow from `claim4.md` eqs. (2)–(5) with $\beta\to\beta\,b(\alpha)/b(0)$.
A swarm steering run then tests the DMFT with a measured input. It does not add evidence for the bridge beyond the
single-agent measurement.

## 3. The most falsifiable cheap version

One 8B open-weight model, single agent, no swarm:

1. Measure the conformity curve $\ell(h)$ at seven field levels, two private-evidence levels and both own-history
   states, 20 prompt variants each; fit the slope $b$ and intercept $\ell_0$.
2. Extract a gain direction at a mid-to-late layer from the difference of mean activations between high-trust and
   low-trust framings, averaged over peer content of both signs. Extract the content direction of the spec as a
   comparison, and a random direction of equal norm as a control.
3. Steer at five strengths $\alpha$ and repeat step 1 (about 4000 forward passes in total, minutes on one GPU).

Predictions, each with its falsifier:

| prediction | falsified if |
|---|---|
| P1: the gain direction changes the social slope monotonically, by at least 30% within the usable range | the slope moves by less than 30% before peer-free accuracy drops by 5 points |
| P2: it acts as a gain, not a bias: the intercept moves by less than 20% of $b\times$(field range) | the curve shifts sideways with unchanged slope |
| P3: the content direction does the opposite: intercept shift, slope unchanged | its slope change is comparable to that of the gain direction |
| P4: mediation: projecting the gain direction out removes at least half of $b$ | $f<0.2$, the social response does not pass through it |
| P5: across at least 30 prompt-by-task conditions, $e\,\varrho$ ranks $\partial\ell/\partial h$ better than confidence does | Spearman correlation no higher than the confidence-only baseline |

Only if P1, P2 and P4 hold is the swarm-level steering experiment worth its cost, and its predictions are then the
quantitative ones above ($\Lambda(\alpha)$, $\rho_c(\alpha)$). If they fail, the result is still a clean negative:
social deference in that model is not a single steerable gain.
