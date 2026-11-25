"""Figures of 08-analyst-calibration from results/analyst-calib/summary.json and the raw probe records (analyze.py
computes every fitted number; this file only draws). One question per figure:

  analyst_calib_gains      How strongly does each trust wording couple a Qwen analyst to its allies and faded
                           colleagues, and do template notes overstate it?  (a_+, a_- per wording, real vs template)
  analyst_calib_linearity  Are the log-odds linear in the ally / faded call counts?  (partial residuals of x against
                           the net count per group: gray probes, black bin means with s.e., line = the M1 slope)
  analyst_calib_gonogo     Does any wording reach the stability boundary?  (Lambda_mutual against abar: the theory's
                           curve at the measured nu, a0, equal gains; black points = wordings with s.e.; 0.7 line)

Conventions (paper-writing figure rules): gray = individual realizations, black = means or estimates with s.e., lines
= theory or fitted model. PDF and PNG in paper/figures/.

Usage:  uv run python experiments/08-analyst-calibration/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import analyze as an  # noqa: E402

FIG = ROOT / "paper" / "figures"
WORDINGS = an.WORDINGS


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def gains(S):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    x = np.arange(len(WORDINGS))
    for ax, par, title in ((axs[0], "a_plus", r"ally gain $a_+$"), (axs[1], "a_minus", r"fade gain $a_-$")):
        for arm, dx, mfc in (("real", -0.08, "k"), ("template", 0.08, "w")):
            e = [S["probes"][f"{w}|{arm}"][par]["est"] for w in WORDINGS]
            se = [S["probes"][f"{w}|{arm}"][par]["se"] for w in WORDINGS]
            ax.errorbar(x + dx, e, yerr=se, fmt="o", color="k", mfc=mfc, ms=5, capsize=2, label=f"{arm} notes")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(x, WORDINGS)
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("half log-odds per unit normalized field")
    axs[0].legend(frameon=False, fontsize=8)
    save(fig, "analyst_calib_gains")


def linearity(S):
    design, recs = an.load_probes()
    fig, axs = plt.subplots(2, len(WORDINGS), figsize=(10, 4.6), sharey="row")
    for j, w in enumerate(WORDINGS):
        rows = an.rows_for(design, recs, w, "real", sorted(design))
        f = S["probes"][f"{w}|real"]
        coef = {k: f[k]["est"] for k in an.M1_NAMES}
        X = np.array([an._cols(r, "M1") for r in rows])
        y = np.array([r["x"] for r in rows])
        w_vec = np.array([coef[k] for k in an.M1_NAMES])
        for i, (col, par, lab) in enumerate(((0, "a_plus", "allies"), (1, "a_minus", "faded"))):
            ax = axs[i, j]
            net = np.round(X[:, col] * an.K * (1 if col == 0 else -1)).astype(int)   # net BUY count in the group
            part = y - X @ w_vec + X[:, col] * coef[par]                              # partial residual
            ax.scatter(net + np.random.default_rng(0).uniform(-0.25, 0.25, len(net)), part, s=2, c="0.75", lw=0)
            ks = np.arange(net.min(), net.max() + 1)
            mk = [(k, part[net == k].mean(), part[net == k].std(ddof=1) / np.sqrt((net == k).sum()))
                  for k in ks if (net == k).sum() >= 5]
            kk, mm, ss = map(np.array, zip(*mk))
            ax.errorbar(kk, mm, yerr=ss, fmt="o", ms=3, color="k", capsize=1.5)
            sgn = 1 if col == 0 else -1
            ax.plot(ks, sgn * coef[par] * ks / an.K, "-", color="k", lw=0.9, label="law (A1), M1 fit")
            ax.set_xlabel(f"net BUY calls among {lab}")
            if j == 0:
                ax.set_ylabel(r"partial residual of $x$")
            if i == 0:
                ax.set_title(w, fontsize=10)
    axs[0, 0].legend(frameon=False, fontsize=7)
    save(fig, "analyst_calib_linearity")


def gonogo(S):
    G = S["gonogo"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    mild = S["probes"]["mild|real"]
    nu, a0 = abs(mild["nu_05"]["est"]), abs(mild["a0"]["est"])
    ab = np.linspace(0.25, max(7.0, max(g["abar"] for g in G.values()) * 1.15), 30)
    lam = [an.lam_mutual(a, a, nu, a0, 0.0, 0.0, M=2000, copies=48)["Lam"] for a in ab]
    ax.plot(ab, lam, "-", color="k", lw=0.9, label=fr"theory, $a_+=a_-$, $\nu$={nu:.2f}, $a_0$={a0:.2f}")
    for w, g in G.items():
        ax.errorbar(g["abar"], g["Lam_mutual"], xerr=g["abar_se"], yerr=g["Lam_mutual_se"], fmt="o", color="k",
                    ms=4, capsize=2)
        ax.annotate(w, (g["abar"], g["Lam_mutual"]), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.axhline(0.7, ls="--", color="0.5", lw=0.8)
    ax.text(ab[0], 0.72, "boundary-sweep threshold 0.7", fontsize=7, color="0.4")
    ax.set_xlabel(r"$\bar a=\sqrt{(a_+^2+a_-^2)/2}$")
    ax.set_ylabel(r"$\Lambda_{\rm mutual}$ (single-site solver)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "analyst_calib_gonogo")


if __name__ == "__main__":
    S = json.loads((an.OUT / "summary.json").read_text())
    gains(S)
    linearity(S)
    gonogo(S)
    print("figures in", FIG)
