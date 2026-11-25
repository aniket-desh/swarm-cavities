"""Figures of 08b-analyst-calibration-2 from results/analyst-calib2/summary.json (analyze.py computes everything).
One question per figure:

  analyst_calib2_bias   Where does the SELL bias come from, and does any variant remove it?  (c per phase-A polarity
                        cell with s.e.; band |c| < 0.2)
  analyst_calib2_fade   Does experimenter-reversed fading make the faded channel repel like the allies?  (a_+ and a_-
                        for ordinary, reversed, reversed_equal; dashed line a_- = a_+ of the ordinary cell)
  analyst_calib2_go     Which variants pass the frozen go rule for the three-world contrast?  (r = a_-/a_+ against
                        abar per cell with s.e.; gray lines at abar = 1 and r = 0.3)

Cells with >= 5% missing probes (the YES/NO cells P4, P5) are left out, as in the analysis.
Conventions (paper-writing figure rules): black = estimates with s.e.; gray = reference lines. PDF and PNG.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
SUMMARY = ROOT / "results" / "analyst-calib2" / "summary.json"
LABELS = {"P0": "up, BUY first", "P1": "up, SELL first", "P2": "down, BUY first", "P3": "down, SELL first",
          "P4": "YES/NO, up", "P5": "YES/NO, down", "F1": "reversed", "F2": "reversed +\nequal",
          "F3": "reversed + equal,\nno notes", "T0": "template notes", "best": "best (combined)"}


def valid(S, c):
    x = S["cells"].get(c)
    return x is not None and x["n_missing"] < 0.05 * (x["n"] + x["n_missing"])


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bias(S):
    cells = [c for c in ("P0", "P1", "P2", "P3", "P4", "P5", "T0") if valid(S, c)]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    e = [S["cells"][c]["c"]["est"] for c in cells]
    se = [S["cells"][c]["c"]["se"] for c in cells]
    ax.axhspan(-0.2, 0.2, color="0.92", zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.errorbar(range(len(cells)), e, yerr=se, fmt="o", color="k", ms=5, capsize=2)
    ax.set_xticks(range(len(cells)), [LABELS[c] for c in cells], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"bias $c$ toward BUY (half log-odds)")
    save(fig, "analyst_calib2_bias")


def fade(S):
    cells = [c for c in ("P0", "F1", "F2", "F3", "best") if c in S["cells"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(cells))
    for par, dx, mfc, lab in (("a_plus", -0.08, "k", r"$a_+$ (allies)"), ("a_minus", 0.08, "w", r"$a_-$ (faded)")):
        ax.errorbar(x + dx, [S["cells"][c][par]["est"] for c in cells],
                    yerr=[S["cells"][c][par]["se"] for c in cells], fmt="o", color="k", mfc=mfc, ms=5,
                    capsize=2, label=lab)
    ax.axhline(S["cells"]["P0"]["a_plus"]["est"], ls="--", color="0.6", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x, ["ordinary" if c == "P0" else LABELS[c] for c in cells], fontsize=8)
    ax.set_ylabel("gain (half log-odds per unit field)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "analyst_calib2_fade")


def go(S):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for nm, c in S["cells"].items():
        if not valid(S, nm):
            continue
        ax.errorbar(c["abar"]["est"], c["r"]["est"], xerr=c["abar"]["se"], yerr=c["r"]["se"], fmt="o", color="k",
                    ms=4, capsize=2)
        ax.annotate(nm, (c["abar"]["est"], c["r"]["est"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axvline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axhline(0.3, color="0.6", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.axhline(1.0, color="0.8", lw=0.6, ls=":")
    ax.set_xlim(0.5, 1.6)
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlabel(r"$\bar a=\sqrt{(a_+^2+a_-^2)/2}$")
    ax.set_ylabel(r"$r=a_-/a_+$")
    ax.text(0.98, 0.32, "contrast go: right of 1, above 0.3", transform=ax.get_yaxis_transform(), ha="right",
            fontsize=7, color="0.4")
    save(fig, "analyst_calib2_go")


if __name__ == "__main__":
    S = json.loads(SUMMARY.read_text())
    bias(S)
    fade(S)
    go(S)
    print("figures in", FIG)
