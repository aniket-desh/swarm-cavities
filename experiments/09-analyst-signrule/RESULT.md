# 09-analyst-signrule: an analyst's call comes back to it only through reciprocated ties, with the sign and ordering the reaction term predicts; the antagonistic echo is 3× the zero-shot value

The sign rule holds in Qwen3.5-9B analyst swarms, given the couplings the swarm itself has. The memory-robust echo
X₂ = ρ₂ − ρ₁² (rounds 5–20) is:
- mutual: +0.076 (s.e. 0.004);
- one-way: +0.002 (0.003), zero within error;
- antagonistic: +0.033 (0.003), positive.

Kick twins agree: E = +0.069 (0.014) mutual, +0.031 (0.007) antagonistic, and 0 in the one-way world, where the
kicked analyst's prompt is unchanged. Every pre-registered falsifier F1–F6 is consistent, so the verdict-table label
is **supported**: reciprocity, ordering and the antagonistic sign confirmed, and one-step memory holds.

**The main caveat is magnitude.** The antagonistic echo is 3.3 times its zero-shot prediction (+0.0096). The reason is
that faded colleagues attract far more strongly inside the swarm than in the single-agent probes. The first-order kick
responses measure the faded pull at u_D = 0.42 (s.e. 0.02), against 0.146 in the calibration. With the swarm's own
couplings, the reaction coefficient predicts the antagonistic echo (Ĝd = +0.043 against a measured contrast of +0.031)
and the mutual one (+0.057 against +0.074) within the pre-registered tolerance.

**The asynchronous control moves the echo to lag 1 as predicted, but the lag-2 test is not resolved.** The lag-1
contrast is D₁(async) = +0.044 (0.011), against a predicted +0.041. The lag-2 test statistic is T₇ = +0.010 (0.007),
1.4 s.e. against a 3 s.e. criterion, because the asynchronous lag-2 contrast stayed at 0.36 of the synchronous one
(predicted 0.21).

**What this tests, and what it doesn't.**
- It tests the reaction term's sign rule given the couplings measured in the swarm. It does not test whether agents
  act on distrust: they don't. Faded colleagues attract in every wording tried, so the "antagonistic" graph is
  antagonistic in its labels only, and its positive echo is what the rule predicts for a₋ < 0.
- A sign-blind reciprocity effect would predict the same signs. Telling the two apart needs a cell where faded
  colleagues repel, which no prompt produced.
- The calls were de-biased by the experimenter, but every note was still written by the model from its biased belief.

**Option (a) was a no-go, whatever this result says.** The three-world contrast as originally designed, with faded
colleagues who repel, could not be set up. In 08 and 08b, Qwen followed its faded colleagues under every trust wording:
- a₋ = −0.12 to −0.27;
- with the experimenter reversing their calls, a₋ = −0.52, so readers undid the reversal;
- coupling stayed far below the stability boundary (Λ_mutual ≤ 0.03).

## Design (as run)

- **Cell.** 08b P0: wording "max", default prompt, fictional asset, news ν_nom = 0.5, research note a0_nom = 0.3.
- **De-biasing.** Calls were drawn from tanh(x − ĉ) with ĉ = −0.637 and a stored uniform per (replica, round,
  analyst).
- **Runs.** N = 256, K = 16, p₋ = ½, T = 20, 2 replicas in lockstep sharing graph, notes and news.
  - Synchronous: 2 mutual, 5 one-way, 5 antagonistic.
  - Asynchronous (Amendment 1): block-random schedule with 16 blocks of 16; 4 mutual and 4 one-way.
  - Each run had its own graph.
- **Kick twins:** 100 mutual + 25 null, 20 one-way, 400 antagonistic + 25 null, at t0 ∈ {8, 12}, each with both
  replays (all notes rewritten, notes kept).
- **Data volume:** 215,040 base decisions and notes (12 synchronous and 8 asynchronous runs × 21 rounds × 2 × 256),
  plus about 20,000 twin decisions. 0 missing, 0 censored, 0 refusals. Three notes matched the refusal
  regex, all ordinary uses of "unable to".

Predictions came from `theory/analyst_signrule.md` (frozen 20:52 UTC) and its Amendment 1 (21:52 UTC, before any
asynchronous run). The analysis is `analyze.py`, fixed before data; the numbers are in `summary.json`.

## Echo per world against the frozen predictions (synchronous runs)

World means over runs, with s.e. from the pooled run-to-run s.d. (9 d.f.). The prediction is frozen law M1 (calls
only) with its calibration s.d. z = (obs − pred) / √(s.e.² + calib s.d.²).

| statistic | world | observed (s.e.) | frozen M1 (calib. s.d.) | residual | z | frozen M2 |
|---|---|---|---|---|---|---|
| X₂ | mutual | **+0.0763 (0.0044)** | +0.0571 (0.0024) | +0.019 | +3.8 | +0.054 |
| | one-way | **+0.0020 (0.0028)** | −0.0005 (0.0003) | +0.003 | +0.9 | −0.000 |
| | antagonistic | **+0.0333 (0.0028)** | +0.0096 (0.0024) | +0.024 | +6.5 | +0.017 |
| ρ₂ | mutual / one-way / antagonistic | +0.077 / +0.004 / +0.035 | +0.063 / +0.005 / +0.015 | | +2.8 / −0.4 / +5.3 | |
| ρ₁ | mutual / one-way / antagonistic | −0.018 / −0.043 / −0.042 (0.006 / 0.004 / 0.004) | −0.074 / −0.075 / −0.074 | | +5.1 / +3.1 / +3.2 | |
| q | mutual / one-way / antagonistic | 0.252 / 0.239 / 0.243 | 0.226 / 0.223 / 0.223 | | +2.2 / +1.7 / +2.2 | |
| m | mutual / one-way / antagonistic | −0.229 / −0.124 / −0.074 (0.070 / 0.045 / 0.045) | 0.00 (0.03) | | −3.0 / −2.4 / −1.5 | +0.37 / +0.35 / +0.32 |
| c₂ (replica differences) | mutual / one-way / antagonistic | +0.15 / +0.15 / +0.19 (0.10 / 0.06 / 0.06) | +0.097 / +0.014 / +0.029 | | +0.5 / +2.2 / +2.5 | |

- **The contrasts the rule is about** (differences from the one-way world):
  - mutual +0.074 (0.005), 14 s.e.;
  - antagonistic +0.031 (0.004), 8 s.e.;
  - mutual minus antagonistic +0.043 (0.005), 8 s.e.

  All three have the predicted sign and order. One-way X₂ is +0.002 (0.003): the one-step memory premise holds.
- **Magnitudes.** The mutual echo is 30% above the frozen M1 value, and just above its zero-shot interval
  [0.03, 0.07]. The antagonistic echo is 3.3 times the frozen value, and 0.010 above its interval [0.005, 0.021].
  Both lie within the pre-registered 3 s.e. of their intervals (3 s.e. = 0.016 and 0.012), so F5 records
  "calibration transfers". The antagonistic case is borderline, and the central zero-shot value missed it by 6.5 s.e.
- **Own memory is weaker in the swarm.** ρ₁ = −0.04 in the one-way world against −0.075 predicted. The in-swarm
  own-call weight is κ = −0.03 (0.002), against −0.095 in the probes.
- **An unpredicted synchronous lag-1 contrast.** ρ₁(mutual) − ρ₁(one-way) = +0.025 (0.007), where the law gives
  +0.001. Synchronous rounds carry no reply at lag 1, so this is outside the law. It comes from 2 mutual runs.
- **The mean belief matches neither law.** It is negative in every synchronous world (−0.23, −0.12, −0.07), so about
  δc ≈ −0.05 to −0.10 of SELL bias survived de-biasing. Under the note-channel law M2 it should have been +0.35. The
  run-to-run s.d. of m is 0.10, four times the law's 0.025. Per the fixed rule, ĉ was not changed.
- **c₂ is uninformative.** Replica-difference correlations are large in every world, the one-way world included
  (c₁ ≈ 0.1 to 0.13, c₂ ≈ 0.15 to 0.19, pooled run s.d. 0.15). The replicas of a run share their session tag, so
  their prompts and their residuals coincide whenever their states do. The note anticipated this kind of shared
  residual. c₂ is therefore not used for any decision, and was not planned as one.

Figure: `paper/figures/analyst_signrule_echo.{pdf,png}` (X₂ per world: runs, means, frozen M1, zero-shot band, Ĝd
from the swarm). `analyst_signrule_residuals.{pdf,png}` shows every residual against M1 and M2.

## Kick twins against the frozen predictions

Per unit kick, on the sampling belief. The s.e. is the larger of the cluster bootstraps over slices and over (run,
replica).

| | mutual (100 twins) | one-way (20) | antagonistic (400) |
|---|---|---|---|
| E, notes rewritten | **+0.069 (0.014)**; M1 +0.055, M2 +0.055 | +0.004 (0.004) | **+0.031 (0.007)**; M1 +0.009, M2 +0.017 |
| E, notes kept | +0.061 (0.013) | **0 exactly** | **+0.029 (0.005)**; M2 (kept) +0.004 |
| first-order reader logit w_A (ally) | +0.111 (0.007); pred +0.108 | +0.139 (0.018) | +0.124 (0.004) |
| w_D (faded) | **+0.024 (0.002)**; pred +0.009 | +0.024 (0.006) | **+0.026 (0.001)**; pred +0.009 |
| composition ratio | E/((K/2)(r_A² + r_D²)) = 1.16 | – | E/(K r_A r_D) = 1.01 (calls-only law 0.96; M2 5.1) |
| null twins (note noise, logit at t0+2) | mean −0.12 (0.05), s.d. 0.27 | – | mean +0.04 (0.07), s.d. 0.37 |

- **Mutual.** The kick echo agrees with the frozen value (z = +1.0) and with the echo function (F6 consistent).
  Mutual twins are powered by the pre-registered rule.
- **Antagonistic.** E is positive at 4.3 s.e. and agrees with the echo function's +0.031. The twins are "unpowered"
  by the rule: they could not resolve the frozen +0.009 at 3 s.e., because rewritten notes add 0.27 to 0.37 in logit
  of noise. The note assumed at most 0.2. The measured echo is 3.4 times larger, so the sign is resolved anyway, and
  the notes-kept replay gives +0.029 (0.005) with the sign of r_A·r_D.
- **The kick measures the coupling directly.** A faded-tie reader moves toward the flipped call by w_D = +0.026 per
  unit kick (13 s.e. above the calibration value). That is the faded attraction behind the larger antagonistic echo.
- **The note channel is small for the echo.** The composition ratios are near 1, the calls-only value (M2 predicts
  about 5), and the kept and rewritten replays agree. Even so, the in-swarm leak regression gives a belief pull
  through faded notes of b_LD ≈ +0.47 (0.03).
- **One-way protocol check.** The kept replay is exactly 0 in all 20 twins. The rewritten replay is 0 in 14 of 20 and
  ±0.03125 in logit in 6. That is one bf16 step, from batch-position non-determinism on identical prompts in the same
  batch; the paired recompute does not remove it. So E = +0.004 (0.004) there is engine noise, not an echo.

Figure: `paper/figures/analyst_signrule_kick.{pdf,png}`.

## Falsifiers (pre-registered decision rules)

| | rule | numbers | outcome |
|---|---|---|---|
| F1 mutual echo | X₂(mut) − X₂(one-way) > 3 se and E_mut > 3 se | +0.074 (0.005); E +0.069 (0.014); Ĝ_M d = +0.057 | **consistent** |
| F2 one-way echo (one-step memory) | \|X₂(one-way)\| ≤ 3 se | +0.002 (0.003) | **consistent** |
| F3 antagonistic sign, conditional on the swarm's couplings | sign of Ĝ_A d = +0.043 (0.002); opposite sign at 3 se falsifies | X₂ contrast +0.031 (0.004); E +0.031 (0.007); notes kept +0.029 (0.005), r_A r_D > 0 | **consistent, sign resolved** (both replays) |
| F4 ordering | X₂(mut) − X₂(ant) > 3 se (kick part only if both worlds powered) | +0.043 (0.005); E difference +0.038 (0.016), not counted | **consistent** |
| F5 magnitudes | dX₂ in zero-shot interval within 3 se; else dX₂ against Ĝd | mutual +0.074 in [0.03, 0.07]; antagonistic +0.031 in [0.005, 0.021] (both within 3 se); Ĝd 0.057 and 0.043 | calibration "transfers" by the rule (borderline for the antagonistic world); **single-site reduction consistent** with in-swarm couplings |
| F6 kick against echo function | \|E − dX₂\| ≤ 3 se + 15% where powered | mutual 0.069 against 0.074 | **consistent** (antagonistic unpowered, values agree) |
| F7 asynchronous control | T₇ = ½D₂(sync) − D₂(async) > 3 se; positive check D₁(async) > 3 se | see next section | **T₇ not resolved; positive check passes** |

Verdict table: **supported**. Reciprocity, ordering and the antagonistic sign are confirmed, and one-step memory
holds. F6 is consistent where powered. F7: T₇ not resolved, positive check passes.

## Asynchronous control (Amendment 1)

Mutual minus one-way contrasts. The synchronous values come from this experiment's runs; the asynchronous ones from
4 + 4 runs (block-random, B = 16), with s.e. from their run-to-run spread.

| | observed | frozen (M1) |
|---|---|---|
| D₂(sync) | +0.073 (0.005) | +0.058 |
| D₁(async) | **+0.044 (0.011)**, 4.0 s.e. | +0.041 |
| D₂(async) | +0.026 (0.007) | +0.012 |
| D₂(async)/D₂(sync) | 0.36 | 0.21 (95% of draws 0.19–0.25) |
| D₁(async)/D₂(sync) | 0.60 | 0.71 |
| T₇ = ½D₂(sync) − D₂(async) | +0.010 (0.007): 1.4 s.e. (Welch d.f. 5.5, one-sided p = 0.11) | +0.017 |
| D₁ − D₂, asynchronous | +0.018 (0.009), 2.1 s.e. | +0.029 |

- **The positive check passes.** Under asynchronous updates a lag-1 echo of the predicted size appears, while the
  synchronous lag-1 contrast is near zero, as the theory says (measured +0.025, which is itself unexpected; see
  above).
- **T₇ is not resolved.** The lag-2 echo fell from 0.073 to 0.026, not to the predicted 0.012, so T₇ sits at 1.4 s.e.,
  short of the 3 s.e. needed to confirm. It is also far from the −3 s.e. that would falsify.
- **Why the power fell short.** The asynchronous mutual runs were noisier than the law predicts: the run s.d. of ρ₁ is
  0.021 against 0.006. The leftover lag-2 echo, at 0.36 of the synchronous one, may reflect analysts reading a
  colleague's latest note together with older ones, which the law does not model; that is an inference.
- **Mean belief.** The asynchronous runs drifted to BUY: m = +0.07 to +0.44, mean +0.20, against −0.07 to −0.23 in
  the synchronous runs. That is unexplained.

Figure: `paper/figures/analyst_signrule_async.{pdf,png}`.

## In-swarm couplings (falsifier 3's inputs; calls-only regression of x − ĉ, pooled per world)

| world | a₊ | a₋ | κ | with leak: b_LA, b_LD |
|---|---|---|---|---|
| mutual | 1.83 (0.03) | −0.43 (0.02) | −0.005 (0.004) | −0.13 (0.05), +0.44 (0.06) |
| one-way | 1.95 (0.02) | −0.49 (0.01) | −0.030 (0.002) | +0.01 (0.03), +0.52 (0.03) |
| antagonistic | 1.98 (0.02) | −0.54 (0.01) | −0.035 (0.003) | +0.03 (0.03), +0.47 (0.03) |

The calibration probes had a₊ = 1.74 and a₋ = −0.146. In the swarm the ally gain is similar, but faded colleagues
attract 3 to 4 times more. The pilot's in-swarm fit in 08 already pointed this way (1.9 times).

## Gaussian-cavity flags (guardrail 5)

- **Bimodality:** none. The bimodality coefficient is ≤ 0.38 in every world, under the 0.555 threshold.
- **Platykurtosis:** no flag for one-way conditional kurtosis below −0.3.
- **Heavy tails, which the pre-registered flags didn't cover:** given a previous BUY call, the field x − ĉ is
  right-skewed and has positive excess kurtosis (skew 0.44 to 0.59, excess kurtosis 0.33 to 0.58). Given a previous
  SELL it is close to Gaussian (|kurtosis| < 0.1). The single-site solver assumes a Gaussian field. At K = 16 this
  asymmetry is moderate, and it did not stop Ĝd from matching the echoes.

## How it was checked

- **Unit tests:** 70 analyst tests and 13 vLLM backend tests, all passing on the coordinator and on every worker.
  They include:
  - the de-biased loop equal to the note's own simulation;
  - `kick_slice` equal to the note's `kick_twins`;
  - the block schedule equal to Amendment 1's simulator, and to `analyst_lib`'s asynchronous mode at B = N;
  - crash resume, including mid-round in the block schedule.
- **Dry run:** a full-size `dryrun.py` under the frozen law, in two residual variants, recovers the frozen numbers and
  gives "supported", with F7 confirmed.
- **Review:** a blind reviewer passed the definition (three rounds) and Amendment 1 (one round plus fixes) before any
  data.
- **Pre-registered:** the power rule, the completeness check and the verdict table were fixed before data.
- **Not verified:**
  - Why faded colleagues attract more in the swarm than in the probes. Candidates are real notes from a balanced swarm
    and the inbox composition; nothing here tests them.
  - The origin of the synchronous lag-1 contrast, and of the asynchronous BUY drift.
  - Whether the leftover asynchronous lag-2 echo is a mixed-recency effect.
  - c₂, which is dominated by shared prompts between replicas.
  - Small K, which was deferred by the user.

## Spend

Pods for 09, all on A40s at $0.59/h, $10.95 billed in total ($11.45 estimated), against the coordinator's $8–15:
- W1 (synchronous mutual and one-way): $3.79;
- W2 (synchronous antagonistic): $3.72 billed ($4.20 estimated);
- W3 and W4 (asynchronous): $1.73 and $1.71.

No API spend. All workers are deleted.
