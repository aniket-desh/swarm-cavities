# Test 1: single-agent calibration of claude-haiku-5-5 (2026-11-25)

The single agent the swarm runs (`envs/prompts.py`, own last answer shown, H=1) was fed synthetic inboxes of K=8
peers: 3,920 probe calls plus 1,800 white-probe calls. All parsed, with 0 refusals. Total cost was **$2.10**
(ledger experiment_ids `calib-20261125-*`). Code is in `run.py` and `analyze.py`; numbers are in
`results/calibration/summary.json` and the figure is `paper/figures/calibration.png`.

## Answers

**1. Peers don't act through the nominal scalar field.** The nominal field is h = Σ_agree s_j − Σ_disagree s_j. In
the sufficiency design (5 field levels × 6 compositions × 2 evidence levels × 2 history states × 25), the share of
response variance due to composition is η²_comp = 0.48 (0.47 after the noise correction). The pass criterion was
below 0.05, and the largest single contrast is 97% of the response range across field levels. Almost all of that is
**route B**: the same nominal field delivered by peers "you tend to disagree with" who hold the opposite view. Its
contrast with uniform inboxes is +0.31, +0.21, −0.01, −0.18 and −0.23 across the five levels (s.e. ≤ 0.013), so it
moves the agent the opposite way to the nominal field. The shuffled-order null is within 0.015 of zero. The other
compositions differ from uniform by up to 0.11 (polarized) and −0.09 (few strong).

**2. A two-channel field nearly suffices.** Fitting separate gains to the agree and disagree groups,
E[s] = tanh(a_A f_A − a_D f_D + a_p h_priv + a_u u + c), gives a held-out cell-mean RMSE of 0.058. The single-channel
fit gives 0.159, and the sampling floor is 0.022. A two-channel field is still linear in the peers' states, so it's a
scalar field with *effective* couplings a₊/K for agree ties and −a₋/K for disagree ties. The DMFT of
theory/claim1.md therefore applies with effective (J̄, g, γ) from eq. (13), and the genuinely multi-channel
equations aren't needed. The remaining misfit (2.6× the floor) is non-additivity, from the polarized and few-strong
compositions, and it bounds how quantitative the DMFT can be at K=8.

**3. Channel gains, and a wording that inverts the disagree channel.** These are agree-only vs. disagree-only
inboxes, 200 calls each:

| disagree wording | a₊ | a₋ | a₋/a₊ |
|---|---|---|---|
| label only ("sources you tend to disagree with", from Physics of Agents) | 0.68 ± 0.02 | −0.30 ± 0.02 | −0.43 |
| inverted ("...have miscalibrated sensors... tend to point the wrong way") | 0.99 ± 0.03 | +0.57 ± 0.03 | +0.58 |

With the label wording, disagree sources still *attract*, at about 40% of the strength of agree sources. That matches
the small open models in Physics of Agents (b₋/b₊ from −0.59 to +0.15). The inverted wording gives a negative
coupling and also raises the agree gain. Only the inverted wording makes J̄ ≈ 0 and γ < 0 reachable with signed
inboxes (claim1.md §5).

**4. Conformity slope.** Uniform agree inboxes over h̄ ∈ [−0.8, 0.8] at two evidence levels and both history states
give β_resp = 0.77 ± 0.01 per unit h̄. The curve saturates strongly above h̄ ≈ 0.4 on the side the agent's own
evidence already favors.

**5. Memory kernel.** The white-probe estimate (M=200 trajectories, T=8, ε²=0.023) of R(lag) per unit h̄ is
0.50 ± 0.04 at lag 1, 0.15 ± 0.04 at lag 2 and 0.08 ± 0.05 at lag 3, then consistent with zero up to lag 6 (lags 7–8
are noisy). So there is real one-to-two-round memory. In the static probes, though, the shown own answer has no
effect (a_u = −0.015 ± 0.006, and the history-state offsets differ by less than 0.1). The lag-2 response must
therefore come through the *content* of the agent's own previous message, which restates what it read. That's the
"hidden state" case of claim1.md §4.2.

**6. Regime.** On a fixed in-degree graph with K=8 and the label wording, the effective couplings are J̄ ≈ 0.58,
g ≈ 0.21 at 25% disagree ties. With the inverted wording at 50% disagree ties, J̄ ≈ 0.05 and g ≈ 0.22. With a mean
slope factor d ≈ 0.8, the stability parameter is Λ ≈ d·g·(1+|γ|) ≈ 0.35 even at |γ| = 1. **Haiku swarms at K=8 sit
deep in the quiet regime**, where reciprocity effects are O(g²) ≈ 0.05. Getting near the boundary needs g about 0.6,
i.e. K ≈ 1–2 with the inverted wording, or a steeper response.

## Caveats

The synthetic peer messages are short templates; real swarm messages carry arguments and sensor counts, which may
change the gains. All probes use one evidence rendering at two strengths, and Haiku only.
