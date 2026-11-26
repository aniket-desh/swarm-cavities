"""Slide-sized figures for slides/slides.tex, redrawn from the verified summaries (no new analysis).

Sources (all written by the frozen analyses):
  results/kick1k-20261125-summary.json                 kick stage 1 (experiments/07_kick/RESULT.md)
  results/kick2k-20261125-summary.json                 kick stage 2 (experiments/07_kick/RESULT_stage2.md)
  results/kick2-additivity.json                        R+- = E(+-0.5)/E(+-1)
  results/kick2e-20261125-t0-{0,1}-summary.json        early kicks (exploratory)
  experiments/07_kick/refrozen_predictions_stage2.json frozen slopes and asymmetry ratios per law
  experiments/09-analyst-signrule/summary.json         Qwen analyst sign-rule test
  theory/checks/analyst_signrule{,_async}_predictions.json   its frozen predictions
Numbers typed in below are copied from tables in the cited notes; each block names its table.

Encodings, the same in every figure: black filled = measured (bars: 1 s.e. unless stated), gray = single runs or
cells, blue = prediction frozen before the data (linear response / calls-only law), orange = the anchoring law.
Figures are drawn at their printed size on a 160 mm x 90 mm slide, so font sizes are final sizes.
Run: uv run python slides/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
R = ROOT / "results"
load = lambda p: json.loads(Path(p).read_text())  # noqa: E731

BLUE, ORANGE, GRAY = "#1f5fa8", "#c8641e", "0.55"
for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyreheros-*.otf"):
    matplotlib.font_manager.fontManager.addfont(str(f))       # the deck's sans serif (tgheros in slides.tex)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm", "font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.markersize": 5.5,
    "errorbar.capsize": 2.5, "legend.frameon": False, "legend.fontsize": 9, "pdf.fonttype": 3})   # Type 3: the OpenType (CFF) Heros files do not embed cleanly as Type 42
W, H = 5.04, 2.13     # inches = 128 mm x 54 mm: a full-width figure under a one-line title and a context line
SIDE = (3.15, 2.2)    # 80 mm x 56 mm: a figure beside a column of notes


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def meas(ax, x, y, e, **kw):
    ax.errorbar(x, y, e, fmt="o", color="black", zorder=4, **kw)


def pred(ax, x, y, e=None, **kw):
    ax.errorbar(x, y, e, fmt="o", mfc="white", color=BLUE, zorder=3, **kw)


def ivw(vals):
    """Inverse-variance mean and s.e. of [(value, s.e.), ...] (cells of one condition share the agents' prompt)."""
    v, s = np.array(vals).T
    w = 1 / s**2
    return float(np.sum(w * v) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))


S1 = load(R / "kick1k-20261125-summary.json")
S2 = load(R / "kick2k-20261125-summary.json")
ADD = load(R / "kick2-additivity.json")
c1 = {(c["wording"], c["gamma"]): c for c in S1["cells"]}
c2 = {(c["K"], c["gamma"]): c for c in S2["cells"]}

# ---- 1. stage 1: the echo by wording and reciprocity (DESIGN.md revision 3 frozen linear values, +-1 s.d.) ----
LIN1 = {("inverted", -1.0): (-0.0285, 0.0051), ("inverted", 1.0): (0.0581, 0.0067),
        ("label", -1.0): (0.0329, 0.0065), ("label", 1.0): (0.0388, 0.0075)}
fig, ax = plt.subplots(1, 2, figsize=(W, H), sharey=True)
for a, (w, ttl) in zip(ax, [("inverted", "disagree ties repel\n(“inverted” wording)"),
                            ("label", "disagree ties attract\n(“label” wording)")]):
    for k, g in enumerate((-1.0, 0.0, 1.0)):
        c = c1[(w, g)]
        if g != 0:
            pred(a, k - 0.14, *LIN1[(w, g)], label="frozen linear prediction" if k == 0 else None)
        else:
            a.plot(k - 0.14, 0, "o", mfc="white", color=BLUE, zorder=3)
        meas(a, k + 0.14, c["echo"], c["echo_se"], label="measured" if k == 0 else None)
    a.axhline(0, color="0.75", lw=0.6, zorder=0)
    a.set_xticks(range(3), ["−1", "0", "+1"])
    a.set_xlim(-0.6, 2.6)
    a.set_xlabel("reciprocity $\\gamma$")
    a.set_title(ttl, fontsize=9.5)
    a.set_yticks([-0.02, 0, 0.02, 0.04, 0.06])
    a.annotate("exactly 0", (1.14, 0), (1.3, 0.016), ha="center", fontsize=8.5, color="0.3",
               arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))
ax[0].set_ylabel("echo $E$ per unit kick")
ax[1].legend(loc="lower right", borderaxespad=0.1, handletextpad=0.3, labelspacing=0.3)
fig.tight_layout(pad=0.4, w_pad=1.2)
save(fig, "echo_sign")

# ---- 2. stage 2: additivity, and K = 4 (DESIGN_stage2.md frozen values) ----
LIN2 = {(4, -1.0): (-0.0200, 0.0039), (4, 1.0): (0.0300, 0.0046)}
fig, ax = plt.subplots(1, 2, figsize=(W, H))
a = ax[0]
for k, key in enumerate(("R+", "R-")):
    a.errorbar(k, ADD[key]["R"], 2 * ADD[key]["se"], fmt="o", color="black", zorder=4)
a.axhline(0.5, color=BLUE, lw=1.2)
a.text(1.4, 0.515, "echoes add: ½", color=BLUE, fontsize=9, ha="left", va="bottom")
for y, t in ((0.15, "needs both ties"), (0.85, "one tie saturates")):
    a.axhline(y, color=GRAY, lw=0.7, ls="--")
    a.text(1.4, y + (0.02 if y > 0.5 else -0.02), t, color="0.35", fontsize=8.5, ha="left",
           va="bottom" if y > 0.5 else "top")
a.set_xticks([0, 1], ["$\\frac{E(+1/2)}{E(+1)}$", "$\\frac{E(-1/2)}{E(-1)}$"], fontsize=12)
a.set_xlim(-0.5, 2.95)
a.set_ylim(0, 1.05)
a.set_ylabel("echo ratio (bars: 2 s.e.)")
a.set_title("$K=2$: one reciprocated tie against two", fontsize=9.5)
a = ax[1]
for k, g in enumerate((-1.0, 0.0, 1.0)):
    c = c2[(4, g)]
    if g != 0:
        pred(a, k - 0.14, *LIN2[(4, g)])
    else:
        a.plot(k - 0.14, 0, "o", mfc="white", color=BLUE, zorder=3)
    meas(a, k + 0.14, c["echo"], c["echo_se"])
a.axhline(0, color="0.75", lw=0.6, zorder=0)
a.set_xticks(range(3), ["−1", "0", "+1"])
a.set_xlim(-0.6, 2.6)
a.set_xlabel("reciprocity $\\gamma$")
a.set_ylabel("echo $E$ per unit kick")
a.set_title("$K=4$, disagree ties repel", fontsize=9.5)
a.text(1.68, LIN2[(4, 1.0)][0], "frozen linear\nprediction", color=BLUE, fontsize=9, ha="right", va="center")
a.text(0.32, c2[(4, -1.0)]["echo"] - 0.004, "measured", fontsize=9, ha="left", va="top")
fig.tight_layout(pad=0.4, w_pad=1.6)
save(fig, "additivity_k4")

# ---- 3. single-message responses against the calibrated (transfer-check) values, quiet_regime.md section 3.8 ----
# (condition, marker, [(predicted, s.d.) agree, disagree], [cells measured])
rows = [
    ("disagree repel, $K=2$", "o", ((0.275, 0.020), (-0.057, 0.009)), [c1[("inverted", g)] for g in (-1.0, 0.0, 1.0)]),
    ("disagree repel, $K=2$ (new graphs)", "o", ((0.275, 0.020), (-0.057, 0.009)), [c2[(2, g)] for g in (-0.5, 0.5)]),
    ("disagree attract, $K=2$", "s", ((0.203, 0.026), (0.098, 0.015)), [c1[("label", g)] for g in (-1.0, 0.0, 1.0)]),
    ("disagree repel, $K=4$", "^", ((0.128, 0.012), (-0.046, 0.008)), [c2[(4, g)] for g in (-1.0, 0.0, 1.0)]),
]
fig, a = plt.subplots(figsize=SIDE)
xx = np.array([-0.09, 0.31])
a.plot(xx, xx, color=BLUE, lw=1.1, zorder=1)
a.plot(xx, xx / 2, color=GRAY, lw=0.8, ls="--", zorder=1)
a.text(0.21, 0.222, "calibration holds", color=BLUE, fontsize=8.5, rotation=37, ha="right", va="bottom")
a.text(0.30, 0.137, "half", color="0.35", fontsize=8.5, rotation=21, ha="right", va="top")
seen = set()
for name, mk, preds, cells in rows:
    for (p, ps), key, filled in zip(preds, ("r_agree", "r_disagree"), (True, False)):
        m, s = ivw([(c[key], c[key + "_se"]) for c in cells])
        lab = name.replace(" (new graphs)", "") if (name.split(" (")[0] not in seen and filled) else None
        a.errorbar(p, m, s, xerr=ps, fmt=mk, color="black", mfc="black" if filled else "white", zorder=4, label=lab,
                   capsize=0, elinewidth=0.7)
    seen.add(name.split(" (")[0])
a.axhline(0, color="0.8", lw=0.5, zorder=0)
a.axvline(0, color="0.8", lw=0.5, zorder=0)
a.set_xlabel("response predicted from calibration")
a.set_ylabel("measured response")
a.set_xlim(-0.09, 0.31)
a.set_ylim(-0.11, 0.31)
a.legend(loc="lower right", fontsize=8.5, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.3)
a.text(-0.083, 0.30, "filled: agree ties\nopen: disagree ties", fontsize=8.5, color="0.3", ha="left", va="top")
fig.tight_layout(pad=0.4)
save(fig, "first_order")

# ---- 4. confirm / disconfirm asymmetry (open-path slope ratio); nulls from refrozen_predictions_stage2.json ----
PR = load(ROOT / "experiments" / "07_kick" / "refrozen_predictions_stage2.json")
E0 = load(R / "kick2e-20261125-t0-0-summary.json")["pooled"]["pooled_open_confirm_over_disconfirm"]
E1 = load(R / "kick2e-20261125-t0-1-summary.json")["pooled"]["pooled_open_confirm_over_disconfirm"]
pts = [("$K=2$\nfirst data set", (0.60, 0.05), "2", True),      # stage 1, lagged slope factors (DESIGN_stage2.md)
       ("$K=2$\nreplication", S2["pooled_by_K"]["2"]["pooled_open_confirm_over_disconfirm"], "2", True),
       ("$K=4$", S2["pooled_by_K"]["4"]["pooled_open_confirm_over_disconfirm"], "4", True),
       ("$K=4$\nround-0 kick", E0, None, False), ("$K=4$\nround-1 kick", E1, None, False)]
fig, a = plt.subplots(figsize=(W, H))
for k, (lab, (v, se), K, settled) in enumerate(pts):
    if K:
        lin = [PR[f"{law}|pooled_K{K}"]["open_confirm_over_disconfirm"] for law in ("linear+noise", "linear+noise2")]
        a.fill_between([k - 0.33, k + 0.33], min(lin), max(lin), color=BLUE, alpha=0.75, lw=0,
                       label="linear response (two noise levels)" if k == 0 else None)
        an = PR[f"anchoring|pooled_K{K}"]["open_confirm_over_disconfirm"]
        a.plot([k - 0.33, k + 0.33], [an, an], color=ORANGE, lw=1.6, label="anchoring law" if k == 0 else None)
    a.errorbar(k, v, se, fmt="o", color="black", mfc="black" if settled else "white", zorder=4,
               label={0: "settled agents (rounds 6, 8)", 3: "early kicks (exploratory)"}.get(k))
a.axhline(1, color="0.8", lw=0.5, zorder=0)
a.axvline(2.5, color="0.8", lw=0.5)
a.set_xticks(range(5), [p[0] for p in pts], fontsize=9)
a.set_xlim(-0.55, 4.55)
a.set_ylim(0.35, 1.55)
a.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
a.set_ylabel("confirming / disconfirming\nresponse slope")
h, l = a.get_legend_handles_labels()
order = [l.index(s) for s in ("settled agents (rounds 6, 8)", "early kicks (exploratory)",
                              "linear response (two noise levels)", "anchoring law")]
a.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", ncol=2, columnspacing=1.5, fontsize=8.5,
         borderaxespad=0.1, labelspacing=0.3)
fig.tight_layout(pad=0.4)
save(fig, "asymmetry")

# ---- 5. distance to the stability boundary, every calibrated cell ----
# Haiku: quiet_regime.md section 2.1, d g (1 + |gamma_eff|), larger of gamma = -1, +1 (no s.e. in that table).
# Qwen, same game: experiments/06-qwen-calibration/RESULT.md, Lambda_max (s.e.).
# Qwen analysts, K = 16: 08 RESULT.md, Lambda_mutual from the solver per wording (s.e. 0.002); 08b: at most 0.032.
groups = [("Haiku 5.5\nsensor game", [("repel, 2", 0.64, None), ("repel, 4", 0.39, None), ("attract, 2", 0.46, None),
                                       ("attract, 4", 0.25, None)]),
          ("Qwen3.5-9B\nsensor game", [("", 0.29, 0.04), ("", 0.35, 0.05), ("", 0.32, 0.04), ("", 0.35, 0.04),
                                       ("", 0.45, 0.05), ("", 0.22, 0.05)]),
          ("Qwen3.5-9B\nanalysts, $K=16$", [("", 0.010, 0.002), ("", 0.010, 0.002), ("", 0.015, 0.002),
                                            ("", 0.011, 0.002), ("", 0.032, None)])]
fig, a = plt.subplots(figsize=(W, H))
x, ticks = 0.0, []
for name, vals in groups:
    xs = x + np.arange(len(vals)) * 0.42
    for xi, (_, v, se) in zip(xs, vals):
        a.errorbar(xi, v, se, fmt="o", color="black", zorder=4)
    ticks.append((xs.mean(), name))
    x = xs[-1] + 1.3
a.axhline(1, color=BLUE, lw=1.2)
a.text(x - 1.0, 1.02, "instability: $\\Lambda=1$", color=BLUE, fontsize=9, ha="right", va="bottom")
a.axhline(0.7, color=GRAY, lw=0.7, ls="--")
a.text(x - 1.0, 0.715, "threshold we set for attempting a boundary sweep", color="0.35", fontsize=8.5, ha="right",
       va="bottom")
a.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
a.set_xlim(-0.6, x - 0.9)
a.set_ylim(0, 1.15)
a.set_ylabel("stability parameter $\\Lambda$")
fig.tight_layout(pad=0.4)
save(fig, "stability")

# ---- 6. Qwen analysts: the pull of distrusted colleagues, r = a_-/a_+ (08 and 08b RESULT.md tables) ----
vars_ = [("mild", -0.20, 0.03), ("moderate", -0.17, 0.02), ("strong", -0.12, 0.03), ("strongest", -0.08, 0.02),
         ("calls\nreversed", -0.29, 0.02), ("reversed,\nexplained", -0.30, 0.02),
         ("reversed,\nno notes", -0.09, 0.02), ("template\nnotes", 0.06, 0.02)]
fig, a = plt.subplots(figsize=(W, H))
for k, (lab, v, se) in enumerate(vars_):
    meas(a, k, v, se)
a.axhline(0, color="0.6", lw=0.7)
a.axhline(0.3, color=BLUE, lw=1.2)
a.text(7.45, 0.315, "needed for an antagonistic world", color=BLUE, fontsize=9, ha="right", va="bottom")
a.text(-0.45, 0.025, "above 0: distrusted colleagues repel", fontsize=8.5, color="0.35", va="bottom")
a.text(-0.45, -0.025, "below 0: they attract", fontsize=8.5, color="0.35", va="top")
a.axvline(3.5, color="0.8", lw=0.5)
a.text(1.5, -0.40, "strength of the distrust instruction", fontsize=8.5, color="0.35", ha="center")
a.text(5.5, -0.40, "experimenter interventions", fontsize=8.5, color="0.35", ha="center")
a.set_xticks(range(len(vars_)), [v[0] for v in vars_], fontsize=8.5)
a.set_xlim(-0.5, 7.5)
a.set_ylim(-0.44, 0.42)
a.set_ylabel("$a_-\\,/\\,a_+$")
fig.tight_layout(pad=0.4)
save(fig, "analyst_fade")

# ---- 7. Qwen analysts, de-biased calls: echo per world (as experiments/09-analyst-signrule/plot.py: echo) ----
S9 = load(ROOT / "experiments" / "09-analyst-signrule" / "summary.json")
P9 = load(ROOT / "theory" / "checks" / "analyst_signrule_predictions.json")
A9 = load(ROOT / "theory" / "checks" / "analyst_signrule_async_predictions.json")
W3 = [("mutual", "mutual\n$\\gamma=+1$"), ("oneway", "one-way\n$\\gamma=0$"), ("antagonistic", "“antagonistic”\n$\\gamma=-1$")]
fig, a = plt.subplots(figsize=SIDE)
for k, (w, lab) in enumerate(W3):
    X = S9["worlds"][w]["X2"]
    runs = X["runs"]
    a.scatter(k - 0.2 + np.linspace(-0.06, 0.06, len(runs)), runs, s=9, c="0.6", zorder=2,
              label="single runs" if k == 0 else None)
    meas(a, k, X["mean"], X["se"], label="mean over runs" if k == 0 else None)
    p = P9["predictions"][w]["M1"]["X2"]
    pred(a, k + 0.2, p["value"], p["calib_sd"], label="frozen, from probes" if k == 0 else None)
    if w != "oneway":
        rx = S9["reaction"][w]
        key = "Gd_M" if w == "mutual" else "Gd_A"
        a.errorbar(k + 0.4, rx[key] + S9["worlds"]["oneway"]["X2"]["mean"], rx[key + "_se"], fmt="D", ms=4.5,
                   mfc="white", color="black", zorder=3, label="from in-swarm couplings" if k == 0 else None)
a.axhline(0, color="0.75", lw=0.6, zorder=0)
a.set_xticks(range(3), [x[1] for x in W3], fontsize=9)
a.set_xlim(-0.5, 2.7)
a.set_ylim(-0.008, 0.106)
a.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
a.set_ylabel("two-round echo $\\rho_2-\\rho_1^2$")
a.legend(loc="upper right", fontsize=8.5, handletextpad=0.1, borderaxespad=0.0, labelspacing=0.3)
fig.tight_layout(pad=0.4)
save(fig, "signrule_echo")

# ---- 8. asynchronous control (as plot.py: asynchronous); D_k = rho_k(mutual) - rho_k(one-way) ----
F7, Ws = S9["falsifiers"]["F7"], S9["worlds"]
M1 = P9["predictions"]
D1s = (Ws["mutual"]["rho1"]["mean"] - Ws["oneway"]["rho1"]["mean"],
       float(np.hypot(Ws["mutual"]["rho1"]["se"], Ws["oneway"]["rho1"]["se"])))
rows = [(0, D1s, M1["mutual"]["M1"]["rho1"]["value"] - M1["oneway"]["M1"]["rho1"]["value"]),
        (1, (F7["D2_sync"]["diff"], F7["D2_sync"]["se"]),
         M1["mutual"]["M1"]["rho2"]["value"] - M1["oneway"]["M1"]["rho2"]["value"]),
        (2.4, (F7["D1_async"]["diff"], F7["D1_async"]["se"]), A9["predictions"]["M1"]["mutual"]["D1"]["value"]),
        (3.4, (F7["D2_async"]["diff"], F7["D2_async"]["se"]), A9["predictions"]["M1"]["mutual"]["D2"]["value"])]
fig, a = plt.subplots(figsize=SIDE)
for k, (xk, (v, se), pr) in enumerate(rows):
    a.plot([xk - 0.3, xk + 0.3], [pr, pr], color=BLUE, lw=1.6, label="frozen prediction" if k == 0 else None)
    meas(a, xk, v, se, label="measured" if k == 0 else None)
a.axhline(0, color="0.75", lw=0.6, zorder=0)
a.set_xticks([r[0] for r in rows], ["lag 1", "lag 2", "lag 1", "lag 2"])
a.text(0.5, -0.031, "synchronous", ha="center", fontsize=9.5, color="0.1")
a.text(2.9, -0.031, "asynchronous", ha="center", fontsize=9.5, color="0.1")
a.set_xlim(-0.6, 4.0)
a.set_ylim(-0.005, 0.098)
a.set_ylabel("echo, mutual minus one-way")
a.legend(loc="upper right", fontsize=8.5, borderaxespad=0.0, ncol=2, columnspacing=0.8, handletextpad=0.3)
fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.25)
save(fig, "signrule_async")
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
