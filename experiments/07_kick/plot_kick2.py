"""Figures for kick stage 2 (experiments/07_kick/RESULT_stage2.md), from results/kick2k-20261125-summary.json,
results/kick2-additivity.json and the frozen predictions in DESIGN_stage2.md (echoes: quiet_regime.md section 3.8;
factorization slopes and asymmetry ratios: refrozen_predictions_stage2.json). Same conventions as plot_kick1.py.

Figure 1 (kick2_echo): left, per settled cell the measured echo E (black, graph-bootstrap s.e.) against the frozen
linear prediction (open circle, its s.d.) and the anchoring law (cross), with the residual measured - linear in units
of the combined s.d. below; right, the additivity ratios R+- = E(+-0.5)/E(+-1) (raw and slope-factor normalized)
against the additive value 0.5 and the decision band edges 0.15 and 0.85.
Figure 2 (kick2_composition): left, pooled F and F_open (all four cells, and K = 4 only) against the linear laws
(sigma0 0.045 and 0.063) and the anchoring law; right, the open-path confirm/disconfirm ratio at K = 2 and K = 4
against the linear nulls and the anchoring law, with stage 1's K = 2 value (lagged slope factors) for reference.
Run: uv run python experiments/07_kick/plot_kick2.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
S = json.loads((ROOT / "results" / "kick2k-20261125-summary.json").read_text())
A = json.loads((ROOT / "results" / "kick2-additivity.json").read_text())
PR = json.loads((Path(__file__).with_name("refrozen_predictions_stage2.json")).read_text())

# frozen echo predictions (DESIGN_stage2.md, Predictions)
LIN = {(2, -0.5): (-0.0142, 0.0025), (2, 0.5): (0.0290, 0.0034), (4, -1.0): (-0.0200, 0.0039), (4, 1.0): (0.0300, 0.0046)}
ANCH = {(2, -0.5): -0.0056, (2, 0.5): 0.0313, (4, -1.0): -0.0063, (4, 1.0): 0.0132}
STAGE1_RATIO = (0.60, 0.05)     # stage 1, K = 2, lagged slope factors (DESIGN_stage2.md)

cells = {(c["K"], c["gamma"]): c for c in S["cells"]}
order = [(2, -0.5), (2, 0.5), (4, -1.0), (4, 1.0)]
labels = ["K = 2\nγ = −0.5", "K = 2\nγ = +0.5", "K = 4\nγ = −1", "K = 4\nγ = +1"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [3, 1.3]})
gs = ax[0, 1].get_gridspec()
ax[0, 1].remove(); ax[1, 1].remove()
ax_r = fig.add_subplot(gs[:, 1])        # additivity panel spans both rows
x = np.arange(4)
for k, key in enumerate(order):
    c = cells[key]
    m, s = LIN[key]
    ax[0, 0].errorbar(k - 0.15, m, s, fmt="o", mfc="white", color="tab:blue", capsize=2, label="linear (frozen)" if k == 0 else None)
    ax[0, 0].plot(k + 0.15, ANCH[key], "x", color="tab:orange", ms=7, label="anchoring law (frozen)" if k == 0 else None)
    ax[0, 0].errorbar(k, c["echo"], c["echo_se"], fmt="o", color="black", capsize=2, label="measured" if k == 0 else None)
    ax[1, 0].bar(k, (c["echo"] - m) / np.hypot(c["echo_se"], s), color="0.5", width=0.5)
ax[0, 0].axhline(0, color="0.7", lw=0.6)
ax[0, 0].set_xticks(x, [""] * 4)
ax[0, 0].set_ylabel("echo E per unit kick")
ax[0, 0].legend(frameon=False, fontsize=7, loc="center", bbox_to_anchor=(0.45, 0.45))
ax[0, 0].set_title("two-round echo, inverted wording", fontsize=9)
ax[1, 0].axhline(0, color="0.7", lw=0.6)
for y in (-3, 3):
    ax[1, 0].axhline(y, color="0.7", lw=0.6, ls="--")
ax[1, 0].set_xticks(x, labels, fontsize=7)
ax[1, 0].set_ylabel("(meas − lin) / s.d.")

for k, key in enumerate(("R+", "R-")):
    r = A[key]
    ax_r.errorbar(k - 0.1, r["R"], 2 * r["se"], fmt="o", color="black", capsize=2, label="R, 2 s.e." if k == 0 else None)
    ax_r.errorbar(k + 0.1, r["R_norm"], 2 * r["se_norm"], fmt="s", mfc="white", color="black", capsize=2,
                      label="slope-factor normalized" if k == 0 else None)
ax_r.axhline(0.5, color="tab:blue", lw=1, label="additive (0.5)")
for y in (0.15, 0.85):
    ax_r.axhline(y, color="0.7", lw=0.6, ls="--")
ax_r.set_xticks([0, 1], ["R+ = E(+0.5)/E(+1)", "R− = E(−0.5)/E(−1)"])
ax_r.set_xlim(-0.6, 1.6)
ax_r.set_ylim(0, 1.1)
ax_r.set_ylabel("echo ratio, K = 2")
ax_r.legend(frameon=False, fontsize=7, loc="upper left")
ax_r.set_title("additivity over reciprocated ties\n(dashed: decision-band edges)", fontsize=9)
fig.tight_layout(pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"kick2_echo.{ext}", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
P, PK = S["pooled"], S["pooled_by_K"]
laws = [("linear+noise", "tab:blue", "-", "linear σ₀ = 0.045"), ("linear+noise2", "tab:blue", ":", "linear σ₀ = 0.063"),
        ("anchoring", "tab:orange", "-", "anchoring law")]
groups = [("F", P["F"], "pooled"), ("F_open", P["F_open"], "pooled"), ("F", PK["4"]["F"], "pooled_K4"),
          ("F_open", PK["4"]["F_open"], "pooled_K4")]
for k, (stat, (v, se), grp) in enumerate(groups):
    for law, col, ls, nm in laws:
        ax[0].plot([k - 0.3, k + 0.3], [PR[f"{law}|{grp}"][stat]] * 2, color=col, ls=ls, label=nm if k == 0 else None)
    ax[0].errorbar(k, v, se, fmt="o", color="black", capsize=2, label="measured" if k == 0 else None)
ax[0].set_xticks(range(4), ["F\nall cells", "F_open\nall cells", "F\nK = 4", "F_open\nK = 4"], fontsize=7)
ax[0].set_ylim(0.3, 1.35)
ax[0].set_ylabel("factorization slope")
ax[0].legend(frameon=False, fontsize=6, loc="lower left", ncol=2)
ax[0].set_title("composition of responses", fontsize=9)

for k, K in enumerate(("2", "4")):
    grp = f"pooled_K{K}"
    for law, col, ls, nm in laws:
        ax[1].plot([k - 0.3, k + 0.3], [PR[f"{law}|{grp}"]["open_confirm_over_disconfirm"]] * 2, color=col, ls=ls)
    v, se = PK[K]["pooled_open_confirm_over_disconfirm"]
    ax[1].errorbar(k, v, se, fmt="o", color="black", capsize=2)
ax[1].errorbar(-0.15, *STAGE1_RATIO, fmt="o", mfc="white", color="0.5", capsize=2, label="stage 1, K = 2")
ax[1].set_xticks([0, 1], ["K = 2", "K = 4"])
ax[1].set_ylim(0.3, 1.2)
ax[1].set_ylabel("confirming / disconfirming slope")
ax[1].legend(frameon=False, fontsize=7, loc="upper left")
ax[1].set_title("confirm/disconfirm asymmetry", fontsize=9)
fig.tight_layout(pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"kick2_composition.{ext}", dpi=200)
print("wrote", OUT / "kick2_echo.png", OUT / "kick2_composition.png")
