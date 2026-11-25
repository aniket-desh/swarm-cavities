"""Figures for kick stage 1 (experiments/07_kick/RESULT.md), from results/kick1k-20261125-summary.json and the frozen
predictions in DESIGN.md (quiet_regime.md section 3.8; refrozen F values from refrozen_predictions.json).

Figure 1 (kick1_echo): per |gamma| = 1 cell, the measured echo E (black, graph-bootstrap s.e.) against the frozen
linear-response prediction (open circle with its s.d.) and the anchoring-law prediction (cross); lower panel: the
residual measured - linear in units of the combined s.d. Also the first-order responses r_A, r_D against their linear
predictions.
Figure 2 (kick1_composition): pooled F and F_open (black) against the noisy-linear and anchoring values, and the
confirming / disconfirming split.
Run: uv run python experiments/07_kick/plot_kick1.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
S = json.loads((ROOT / "results" / "kick1k-20261125-summary.json").read_text())

# frozen predictions (DESIGN.md revision 3)
LIN = {("inverted", -1): (-0.0285, 0.0051), ("inverted", 1): (0.0581, 0.0067),
       ("label", -1): (0.0329, 0.0065), ("label", 1): (0.0388, 0.0075)}
ANCH = {("inverted", -1): -0.0128, ("inverted", 1): 0.0672, ("label", -1): 0.0208, ("label", 1): 0.0307}
R_LIN = {"inverted": ((0.275, 0.020), (-0.057, 0.009)), "label": ((0.203, 0.026), (0.098, 0.015))}
F_LIN, FO_LIN, F_AN, FO_AN = 0.98, 0.945, 0.67, 0.63

cells = {(c["wording"], int(c["gamma"])): c for c in S["cells"]}
order = [("inverted", -1), ("inverted", 1), ("label", -1), ("label", 1)]
labels = ["inverted\nγ = −1", "inverted\nγ = +1", "label\nγ = −1", "label\nγ = +1"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [3, 1.3]})
x = np.arange(4)
for k, key in enumerate(order):
    c = cells[key]
    m, s = LIN[key]
    ax[0, 0].errorbar(k - 0.15, m, s, fmt="o", mfc="white", color="tab:blue", capsize=2, label="linear (frozen)" if k == 0 else None)
    ax[0, 0].plot(k + 0.15, ANCH[key], "x", color="tab:orange", ms=7, label="anchoring law (frozen)" if k == 0 else None)
    ax[0, 0].errorbar(k, c["echo"], c["echo_se"], fmt="o", color="black", capsize=2, label="measured" if k == 0 else None)
    z = (c["echo"] - m) / np.hypot(c["echo_se"], s)
    ax[1, 0].bar(k, z, color="0.5", width=0.5)
ax[0, 0].axhline(0, color="0.7", lw=0.6)
ax[0, 0].set_xticks(x, [""] * 4)
ax[0, 0].set_ylabel("echo E per unit kick")
ax[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
ax[0, 0].set_title("two-round echo (reaction term)", fontsize=9)
ax[1, 0].axhline(0, color="0.7", lw=0.6)
for y in (-3, 3):
    ax[1, 0].axhline(y, color="0.7", lw=0.6, ls="--")
ax[1, 0].set_xticks(x, labels, fontsize=7)
ax[1, 0].set_ylabel("(meas − lin) / s.d.")

# first-order responses, pooled over gamma within wording (cells at gamma = -1, 0, +1 share the agents' prompt)
for k, (w, ty, nm) in enumerate([("inverted", 0, "r_agree"), ("inverted", 1, "r_disagree"),
                                 ("label", 0, "r_agree"), ("label", 1, "r_disagree")]):
    m, s = R_LIN[w][ty]
    ax[0, 1].errorbar(k - 0.15, m, s, fmt="o", mfc="white", color="tab:blue", capsize=2)
    vals = [(cells[(w, g)][nm], cells[(w, g)][nm + "_se"]) for g in (-1, 0, 1)]
    v, se = np.array(vals).T
    wt = 1 / se**2
    mean, sem = np.sum(wt * v) / np.sum(wt), 1 / np.sqrt(np.sum(wt))
    for j, (vv, ss) in enumerate(vals):
        ax[0, 1].errorbar(k + 0.05 + 0.07 * j, vv, ss, fmt=".", color="0.6", capsize=1)
    ax[0, 1].errorbar(k + 0.3, mean, sem, fmt="o", color="black", capsize=2)
    ax[1, 1].bar(k, (mean - m) / np.hypot(sem, s), color="0.5", width=0.5)
ax[0, 1].axhline(0, color="0.7", lw=0.6)
ax[0, 1].set_xticks(x, [""] * 4)
ax[0, 1].set_ylabel("first-order response per unit kick")
ax[0, 1].set_title("readers' response at t0+1\n(gray: each γ, black: pooled)", fontsize=9)
ax[1, 1].axhline(0, color="0.7", lw=0.6)
for y in (-3, 3):
    ax[1, 1].axhline(y, color="0.7", lw=0.6, ls="--")
ax[1, 1].set_xticks(x, ["inverted\nagree", "inverted\ndisagree", "label\nagree", "label\ndisagree"], fontsize=7)
ax[1, 1].set_ylabel("(meas − lin) / s.d.")
fig.tight_layout(pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"kick1_echo.{ext}", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
P = S["pooled"]
for k, (key, lin, an) in enumerate([("F", F_LIN, F_AN), ("F_open", FO_LIN, FO_AN)]):
    ax[0].plot([k - 0.25, k + 0.25], [lin, lin], color="tab:blue", label="noisy linear (frozen)" if k == 0 else None)
    ax[0].plot([k - 0.25, k + 0.25], [an, an], color="tab:orange", label="anchoring law (frozen)" if k == 0 else None)
    ax[0].errorbar(k, P[key][0], P[key][1], fmt="o", color="black", capsize=2, label="measured, pooled" if k == 0 else None)
ax[0].set_xticks([0, 1], ["F (closed paths)", "F_open (open paths)"])
ax[0].set_ylim(0.4, 1.2)
ax[0].set_ylabel("factorization slope")
ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("composition of responses", fontsize=9)
for k, (a, b) in enumerate([("F_confirm", "F_open_confirm"), ("F_disconfirm", "F_open_disconfirm")]):
    ax[1].errorbar(k - 0.1, P[a][0], P[a][1], fmt="o", color="black", capsize=2, label="closed" if k == 0 else None)
    ax[1].errorbar(k + 0.1, P[b][0], P[b][1], fmt="s", mfc="white", color="black", capsize=2, label="open" if k == 0 else None)
ax[1].axhline(1, color="0.7", lw=0.6)
ax[1].set_xticks([0, 1], ["confirming input", "disconfirming input"])
ax[1].set_ylabel("slope")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("alignment split (pooled)", fontsize=9)
fig.tight_layout(pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"kick1_composition.{ext}", dpi=200)
print("wrote", OUT / "kick1_echo.png", OUT / "kick1_composition.png")
