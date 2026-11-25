"""Figures of 08b-analyst-calibration-2 from results/analyst-calib2/summary.json (analyze.py computes everything).
One question per figure:

  analyst_calib2_bias   Where does the SELL bias come from, and does any variant remove it?  (c per phase-A polarity
                        cell with s.e.; band |c| < 0.2)
  analyst_calib2_fade   Does experimenter-reversed fading make the faded channel repel like the allies?  (a_+ and a_-
                        for ordinary, reversed, reversed_equal; dashed line a_- = a_+ of the ordinary cell)
  analyst_calib2_K      Can a small-K swarm reach the boundary?  (left: a_+, a_- against K for the best variant;
                        right: Lambda_mutual against K, graph simulation (filled) and solver (open), at the measured
                        bias and with c = 0, and the 0.7 threshold)

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
          "P4": "YES/NO, up", "P5": "YES/NO, down", "F1": "reversed", "F2": "reversed +\nequal"}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def bias(S):
    cells = [c for c in ("P0", "P1", "P2", "P3", "P4", "P5") if f"{c}|K16" in S["cells"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    e = [S["cells"][f"{c}|K16"]["c"]["est"] for c in cells]
    se = [S["cells"][f"{c}|K16"]["c"]["se"] for c in cells]
    ax.axhspan(-0.2, 0.2, color="0.92", zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.errorbar(range(len(cells)), e, yerr=se, fmt="o", color="k", ms=5, capsize=2)
    ax.set_xticks(range(len(cells)), [LABELS[c] for c in cells], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"bias $c$ toward BUY (half log-odds)")
    save(fig, "analyst_calib2_bias")


def fade(S):
    cells = [c for c in ("P0", "F1", "F2") if f"{c}|K16" in S["cells"]]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(cells))
    for par, dx, mfc, lab in (("a_plus", -0.08, "k", r"$a_+$ (allies)"), ("a_minus", 0.08, "w", r"$a_-$ (faded)")):
        ax.errorbar(x + dx, [S["cells"][f"{c}|K16"][par]["est"] for c in cells],
                    yerr=[S["cells"][f"{c}|K16"][par]["se"] for c in cells], fmt="o", color="k", mfc=mfc, ms=5,
                    capsize=2, label=lab)
    ax.axhline(S["cells"]["P0|K16"]["a_plus"]["est"], ls="--", color="0.6", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x, ["ordinary" if c == "P0" else LABELS[c] for c in cells], fontsize=8)
    ax.set_ylabel("gain (half log-odds per unit field)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "analyst_calib2_fade")


def smallK(S):
    sel = S["selection"]
    rows = []
    for K in (2, 4, 8, 16):
        key = f"best|K{K}" if f"best|K{K}" in S["cells"] else (f"{sel['best_is_cell']}|K16" if K == 16 and sel["best_is_cell"] else None)
        if key and key in S["cells"]:
            rows.append((K, S["cells"][key]))
    Ks = np.array([k for k, _ in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 2.9))
    for par, mfc, lab in (("a_plus", "k", r"$a_+$"), ("a_minus", "w", r"$a_-$")):
        a1.errorbar(Ks, [c[par]["est"] for _, c in rows], yerr=[c[par]["se"] for _, c in rows], fmt="o-", color="k",
                    mfc=mfc, ms=5, capsize=2, lw=0.6, label=lab)
    a1.set_xscale("log", base=2)
    a1.set_xticks(Ks, [str(k) for k in Ks])
    a1.axhline(0, color="0.6", lw=0.6)
    a1.set_xlabel("K (colleagues read)")
    a1.set_ylabel("gain per unit normalized field")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(Ks, [c["Lam_graph"] for _, c in rows], "o-", color="k", ms=5, lw=0.8, label="graph sim., measured c")
    a2.plot(Ks, [c["Lam_graph_c0"] for _, c in rows], "s--", color="k", ms=4, lw=0.8, label="graph sim., c = 0")
    a2.errorbar(Ks, [c["Lam_solver"]["est"] for _, c in rows], yerr=[c["Lam_solver"]["se"] for _, c in rows],
                fmt="o", color="k", mfc="w", ms=5, capsize=2, label="solver, measured c")
    a2.plot(Ks, [c["Lam_solver_c0"] for _, c in rows], "s", color="k", mfc="w", ms=4, label="solver, c = 0")
    a2.axhline(0.7, ls=":", color="0.5", lw=0.8)
    a2.set_xscale("log", base=2)
    a2.set_xticks(Ks, [str(k) for k in Ks])
    a2.set_ylim(0, 1)
    a2.set_xlabel("K (colleagues read)")
    a2.set_ylabel(r"$\Lambda_{\rm mutual}$")
    a2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "analyst_calib2_K")


if __name__ == "__main__":
    S = json.loads(SUMMARY.read_text())
    bias(S)
    fade(S)
    smallK(S)
    print("figures in", FIG)
