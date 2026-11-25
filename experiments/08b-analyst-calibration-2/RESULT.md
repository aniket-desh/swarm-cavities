# 08b-analyst-calibration-2: neither fix works at K = 16; reversing faded calls makes analysts follow faded colleagues more, and no prompt variant removes the SELL bias

The three-world contrast stays a no-go in every variant. Reversing the faded colleagues' calls before showing them
doesn't make the faded channel repel. With the notes kept, it makes it attract three times more strongly:
a₋ = −0.52 (s.e. 0.04) against −0.15 (0.04) for the ordinary wording, so r = a₋/a₊ = −0.29 and −0.30 against −0.08.
With the faded notes removed, a₋ = −0.17 (0.04), the same as the ordinary wording. All four r are more than 2 s.e.
below the 0.3 threshold.

No prompt variant brings the bias within |c| < 0.2. Listing SELL before BUY is the best variant (c = −0.38, s.e.
0.01, against −0.64). Asking whether the asset will close down makes it worse (c = −0.99). Removing all pilot note
text helps a little (c = −0.49). The YES/NO token control failed: Qwen kept writing BUY or SELL, and 89% of those
probes were missing.

With the measured gains, the frozen go rule gives:
- contrast: no in all 8 valid cells;
- predicted echo resolvable (|Γ|(1−q)² > 0.03): no in any cell at its measured bias, and in the mutual world only once
  the bias is removed;
- Λ_mutual: 0.003 to 0.032 at the measured bias, ≤ 0.11 at c = 0.

All numbers come from 1,000 base probes per cell, all at K = 16 with the "max" wording, with bootstrap s.e. over base
probes.

**On what the reversal tests.** Inverting the faded calls, by construction, tests the sign structure of the reaction
term given negative couplings. It doesn't test whether agents act on distrust. Here the construction itself failed:
the readers did not take the reversed calls at face value, so the negative couplings it was meant to realize never
appeared.

## Fading: experimenter-reversed calls (K = 16, max wording)

| cell | what the reader sees for a faded colleague | a₊ | a₋ | r = a₋/a₊ | r > 0.3 |
|---|---|---|---|---|---|
| P0 | ordinary: "usually wrong; treat each call as strong evidence for the opposite" | 1.74 (0.04) | −0.15 (0.04) | −0.08 (0.02) | fails |
| F1 | the call reversed by the desk, labelled "(reversed)", with their note | 1.80 (0.04) | −0.52 (0.04) | −0.29 (0.02) | fails |
| F2 | as F1, plus "a reversed call is as informative as an ally's call" | 1.74 (0.04) | −0.52 (0.03) | −0.30 (0.02) | fails |
| F3 | as F2, notes not shown | 1.94 (0.04) | −0.17 (0.04) | −0.09 (0.02) | fails |
| T0 | ordinary wording, all notes templates (no pilot text) | 1.65 (0.04) | +0.10 (0.03) | +0.06 (0.02) | fails |

a₋ is defined against the colleague's actual call. So in F1–F3 a reader that took the displayed (reversed) call at
face value would show a₋ ≈ a₊ ≈ +1.7. What happened instead:
- **No notes (F3).** The response to the displayed call is negative in its own terms: the reader leans toward the
  colleague's original call by 0.17, as much as with no reversal at all. The reader seems to undo the desk's
  reversal and then treat the faded colleague as it did in 08, slightly following them.
- **With notes (F1, F2).** The faded colleague's note argues for their original call (89% of the SELL notes in the
  pool name a downward direction). The pull toward the original call triples, from −0.17 to −0.52. Next to a
  reversed label, the note seems to carry more weight, not less.
- **The informativeness statement.** "As informative as an ally's" changes nothing (F2 against F1).

Across 08 and 08b, the only positive a₋ appears when the faded notes are templates that carry nothing but the call:
+0.10 (0.03) here in T0, and +0.05 in 08's template arm. So it is the real notes' content that pulls readers toward
faded colleagues. With bare calls, the wording fades them weakly.

## Bias: where it comes from, and the gains after removing it

| cell | variant | c | a₊ | a₋ | ν (ν_nom 0.5) | a0 | κ | ā |
|---|---|---|---|---|---|---|---|---|
| P0 | close up, BUY listed first (08's prompt) | −0.64 (0.01) | 1.74 | −0.15 | 0.30 | 0.09 | −0.10 | 1.23 |
| P1 | close up, SELL first | **−0.38 (0.01)** | 1.58 | −0.11 | 0.27 | 0.09 | −0.17 | 1.12 |
| P2 | close down, BUY first | −0.99 (0.01) | 1.72 | −0.11 | 0.26 | 0.09 | −0.11 | 1.22 |
| P3 | close down, SELL first | −0.57 (0.01) | 1.29 | −0.07 | 0.17 | 0.06 | −0.13 | 0.92 |
| T0 | P0 with template notes everywhere | −0.49 (0.01) | 1.65 | +0.10 | 0.39 | 0.14 | +0.04 | 1.17 |
| P4 | YES/NO to "close up" | 896 of 1,000 missing | | | | | | |
| P5 | YES/NO to "close down" | 886 of 1,000 missing | | | | | | |

The s.e. are 0.03 to 0.04 for the gains and 0.01 to 0.02 for the rest. The decomposition (pre-registered):
- **Option order.** Listing SELL first moves c by +0.34, so the bias favours the option listed last.
- **Question framing.** Asking about closing down moves c by −0.27.
- **Pilot note text.** The pilot notes add −0.15 (T0 − P0), because the pilot swarm was bearish.
- **What is left.** About −0.35 to −0.5 that the order, framing and notes don't explain. Whether it sits on the SELL
  token or on the content "down" stays open. In the YES/NO cells the leftover 18% of the mass went to BUY and SELL
  themselves, so valid mass was 0.82 and those probes counted as missing. On the ~11% that passed, the estimates are
  a selected sample and aren't used.

"Gains after removing the bias": no variant removes it. Combining the effects (SELL first and template notes) would
predict c ≈ −0.2 if they add, but that wasn't tested, and a real swarm writes its own notes. Removing the bias
mostly changes the saturation, not the gains: a₊ = 1.6 to 1.7 and a₋ ≈ −0.1 in every valid polarity cell. The clean
way to remove it is on the experimenter's side, by sampling calls from tanh(x − ĉ). The c = 0 columns below show what
that would give.

## Frozen go rule with the new gains (`theory/analyst_scenario.md` section 7; analyst_lib)

Γ = g² = (a₊² + a₋²)/32 in the mutual world and −a₊a₋/16 in the antagonistic world. ρ₂ is from the solver (50,000
sites); Λ_mutual is from the solver (3,000 sites × 64 copies). s.e. from 100 bootstrap draws.

| cell | ā | r | contrast go | |Γ|(1−q)² mutual / antag. | ρ₂ mutual / antag. | Λ_mutual | at c = 0: |Γ|(1−q)² mut. / antag. | at c = 0: ρ₂ mut. / antag. | at c = 0: Λ_mutual |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 1.23 (0.03) | −0.08 | no | 0.007 / 0.001 | +0.010 / +0.003 | 0.010 | 0.068 / 0.011 | +0.075 / +0.018 | 0.096 |
| P1 | 1.12 (0.03) | −0.07 | no | 0.024 / 0.004 | +0.034 / +0.012 | 0.032 | 0.057 / 0.008 | +0.079 / +0.029 | 0.090 |
| P2 | 1.22 (0.03) | −0.06 | no | 0.002 / 0.000 | +0.005 / +0.003 | 0.003 | 0.068 / 0.009 | +0.077 / +0.017 | 0.100 |
| P3 | 0.92 (0.02) | −0.05 | no | 0.010 / 0.001 | +0.016 / +0.006 | 0.013 | 0.043 / 0.005 | +0.058 / +0.019 | 0.063 |
| T0 | 1.17 (0.03) | +0.06 | no | 0.014 / 0.002 | +0.018 / −0.000 | 0.021 | 0.057 / 0.007 | +0.059 / −0.005 | 0.071 |
| F1 | 1.33 (0.03) | −0.29 | no | 0.003 / 0.002 | +0.005 / +0.003 | 0.005 | 0.080 / 0.043 | +0.085 / +0.047 | 0.105 |
| F2 | 1.28 (0.03) | −0.30 | no | 0.003 / 0.002 | +0.004 / +0.003 | 0.005 | 0.076 / 0.043 | +0.084 / +0.050 | 0.110 |
| F3 | 1.38 (0.03) | −0.09 | no | 0.003 / 0.001 | +0.006 / +0.001 | 0.006 | 0.086 / 0.015 | +0.088 / +0.017 | 0.109 |

The s.e. are 0.000 to 0.001 for |Γ|(1−q)² at the measured bias, 0.002 to 0.005 for ρ₂, and 0.001 to 0.013 for Λ.

- **Contrast (ā ≥ 1 and r > 0.3):** no in every cell. ā passes everywhere except P3, but r ≤ 0.06.
- **Echo resolvable (|Γ|(1−q)² > 0.03):** no in any cell at its measured bias; the best is P1, mutual, at 0.024.
  With the bias removed, the mutual echo passes in every cell (0.043 to 0.086, ρ₂ ≈ +0.06 to +0.09). So a
  mutual-against-one-way echo test becomes feasible if the experimenter de-biases the call sampling. The
  antagonistic echo then passes only in F1 and F2 (0.043), and there its sign is **positive**: because a₋ < 0,
  Γ = −a₊a₋/K > 0, so the "antagonistic" graph acts as a weak second mutual world. That is the theory's own sign rule
  (falsifier 2: the antagonistic echo has the sign of −a₊a₋), but it isn't the designed three-world contrast.
- **Boundary (Λ_mutual ≥ 0.7):** no, by an order of magnitude.
- **Herding first (J̄ > 1.3):** no anywhere. The highest is F1 at 1.16.

Figures: `paper/figures/analyst_calib2_bias.{pdf,png}` (c per polarity cell), `analyst_calib2_fade.{pdf,png}` (a₊, a₋
per fading variant), `analyst_calib2_go.{pdf,png}` (r against ā with the contrast thresholds).

## Criteria (pre-registered, three-way)

1. **Bias removed (|c| < 0.2):** fails in every valid cell; inconclusive in P4 and P5 (≥ 5% missing).
2. **Reversed fading works (r > 0.3):** fails in F1, F2 and F3. There was no separate combined cell: the selection
   picked P1 with ordinary fading, because no reversed cell beat the ordinary r.
3. **Go rule:** see the table above.

Overall: **not supported**. Criterion 2 fails in every reversed cell.

## Checks and caveats

- **Reproducibility.** P0 repeats 08's max/real probes. The median |Δx| is 4 × 10⁻⁸ and the maximum 0.125 (two bf16
  logit steps); a₊, a₋ and c are identical to 08 to the second decimal.
- **Selection and missing data.** The pre-registered selection excluded P4 and P5 (missing ≥ 5%; P5 also has
  ā = 0.09) and P3 (ā < 0.8 ā(P0)). Valid mass was ≥ 0.998 in every other cell, with nothing censored.
- **Unit tests.** 56 analyst tests pass, including a frozen hash of the default prompt and no forbidden words in any
  of the 32 variants. `analyze.py selftest` reproduces the note's Λ = 0.60 and ρ₂ = ±0.26 at (4, 4).
- **Not verified:**
  - whether de-biasing the call sampling (tanh(x − ĉ)) behaves in a swarm as the c = 0 numbers assume, since the
    notes would still be written from the biased belief;
  - why readers undo the desk's reversal (an inference from F3, not observed directly);
  - whether the order and template-note effects on c add;
  - small K, which was dropped by the user's scope change.
- **Scope change.** The amendment is recorded in EXPERIMENT.md. During review, it emerged that 1 − 1/A overstates Λ at
  small K, because A = 1 + KΛ/((K−1)(1−Λ)) on a K-regular tree. This matters for the deferred small-K experiment.

## Cost

Pod: one A40 at $0.59/h for about 0.6 h (setup 6 min, 10 cells at about 165 s each): ≈ $0.35, against a $0.30
projection and a $5 budget. No API spend.
