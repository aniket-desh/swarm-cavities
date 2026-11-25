"""Figures of the Qwen3.5-9B calibration from results/qwen-calib/summary.json and the raw conformity records
(analyze.py computes everything; this file only draws). Writes paper/figures/qwen_calibration.{pdf,png} and
paper/figures/qwen_pilot.{pdf,png}.

Conventions (paper-writing figure rules): gray = individual realizations (single probe answers, single cells),
black = means with s.e. bars, lines = model curves (the tanh fit) or reference values (slope 1, Lambda = 0.7, 1).

Usage:  uv run python experiments/06-qwen-calibration/plot.py [--root results]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
WORDINGS, KS = ("inverted", "label"), (2, 4, 8)
HAIKU_REAL = {("inverted", 2): (0.80, 0.03, 0.15, 0.02), ("inverted", 4): (0.66, 0.05, 0.22, 0.04),   # HANDOFF.md
              ("label", 2): (0.54, 0.03, -0.24, 0.02), ("label", 4): (0.40, 0.03, -0.21, 0.02)}


def conformity_panels(axs, s, root):
    for ax, (wording, K) in zip(axs, [(w, k) for w in WORDINGS for k in KS]):
        recs = json.loads((root / "qwen-calib" / f"conformity_{wording}.json").read_text())
        for arm, mk in (("real", "o"), ("template", "s")):
            key = f"{wording}|K{K}|{arm}"
            if key not in s["conformity"]:
                continue
            o = s["conformity"][key]
            v = o["by_ev"]["0.4"]
            if arm == "real":
                rs = [r for r in recs if r["K"] == K and r["arm"] == arm and r["ev_level"] == 0.4 and r["hist"] == 1
                      and r["s"] is not None]
                jit = np.random.default_rng(0).uniform(-0.015, 0.015, len(rs))
                ax.scatter(np.array([r["hbar"] for r in rs]) + jit, [r["s"] for r in rs], s=4, c="0.7", lw=0, zorder=1)
            ax.errorbar(v["hbar"], v["mean_s"], yerr=v["se_s"], fmt=mk, ms=4, color="k", capsize=2, zorder=3,
                        mfc="k" if arm == "real" else "w", label=f"{arm} texts")
            hb = np.linspace(-0.85, 0.85, 200)
            c = [val for kk, val in o["offsets"].items() if kk.startswith("0.4")][0]
            ax.plot(hb, np.tanh(o["beta_resp"] * hb + c), "-" if arm == "real" else "--", color="k", lw=0.8, zorder=2)
        vr = s["conformity"][f"{wording}|K{K}|real"]["by_ev"]["0.4"]
        k = int(np.argmax(vr["slopes"]))
        x0, y0 = np.mean(vr["steepest_between"]), np.mean(vr["mean_s"][k:k + 2])
        ax.plot([x0 - 0.3, x0 + 0.3], [y0 - 0.3, y0 + 0.3], ":", color="k", lw=0.8, label="slope 1")
        ax.set_title(f"{wording}, K={K}: steepest {vr['steepest_slope']:.2f}$\\pm${vr['steepest_slope_se']:.2f}",
                     fontsize=9)
        ax.set_xlabel(r"inbox field $\bar h$ (all agree)")
        ax.set_ylim(-1, 1)
    axs[0].set_ylabel(r"answer $s = 2p-1$")
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")


def gains_panel(ax, s):
    x = np.arange(len(WORDINGS) * len(KS))
    labels = []
    for j, (wording, K) in enumerate([(w, k) for w in WORDINGS for k in KS]):
        labels.append(f"{wording[:3]}\nK={K}")
        t = s["transfer"].get(f"{wording}|K{K}")
        if t is None:
            continue
        t = t["s"]
        for nm, dx, mk in (("a_plus", -0.12, "o"), ("a_minus", 0.12, "v")):
            ax.errorbar(j + dx, t[f"real_{nm}"], yerr=t[f"real_{nm}_se"], fmt=mk, color="k", ms=5, capsize=2)
            ax.errorbar(j + dx, t[f"template_{nm}"], yerr=t[f"template_{nm}_se"], fmt=mk, color="k", mfc="w", ms=5,
                        capsize=2)
        if (wording, K) in HAIKU_REAL:
            hp, hps, hm, hms = HAIKU_REAL[(wording, K)]
            ax.errorbar([j - 0.12, j + 0.12], [hp, hm], yerr=[hps, hms], fmt="x", color="0.55", ms=5, capsize=2)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(r"gain per unit field")
    ax.set_title(r"$a_+$ (o), $a_-$ (v): Qwen real (filled), template (open), Haiku real (gray x)", fontsize=8)


def lambda_panel(ax, s):
    for j, (wording, K) in enumerate([(w, k) for w in WORDINGS for k in KS]):
        t = s["transfer"].get(f"{wording}|K{K}")
        if t is None:
            continue
        st = t["stability"]
        ax.errorbar(j - 0.1, st["probe_d"]["Lambda_max"], yerr=st["probe_d"]["Lambda_max_se"], fmt="o", color="k",
                    ms=5, capsize=2)
        ax.errorbar(j + 0.1, st["pilot_d"]["Lambda_max"], yerr=st["pilot_d"]["Lambda_max_se"], fmt="o", color="k",
                    mfc="w", ms=5, capsize=2)
    for y, ls in ((0.7, "--"), (1.0, "-")):
        ax.axhline(y, color="k", lw=0.8, ls=ls)
    ax.set_xticks(range(6), [f"{w[:3]}\nK={k}" for w in WORDINGS for k in KS], fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(r"$\Lambda = d\,g\,(1+|\gamma_{\rm eff}|)$, max over $\gamma$")
    ax.set_title(r"d from probes (filled) or pilot (open); lines 0.7, 1", fontsize=8)


def pilot_figure(s):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    t = np.arange(1, 11)
    for name, key, ax in (("Qwen", "qwen_unchanged", axs[0]), ("Haiku", "haiku_unchanged", axs[0])):
        rows = [r[key]["same_displayed"] for r in s["pilot"].values() if key in r]
        if not rows:
            continue
        rows = np.array(rows)
        for r in rows:
            ax.plot(t, r, color="0.75", lw=0.7)
        ax.errorbar(t, rows.mean(0), yerr=rows.std(0, ddof=1) / np.sqrt(len(rows)), color="k",
                    fmt="-o" if name == "Qwen" else "--s", ms=3, capsize=2, label=name)
    axs[0].set_xlabel("round t")
    axs[0].set_ylabel("fraction with p(t) = p(t-1) (2 decimals)")
    axs[0].legend(frameon=False, fontsize=8)
    for name, key, fmt in (("Qwen", "qwen_unchanged", "-o"), ("Haiku", "haiku_unchanged", "--s")):
        rows = np.array([r[key]["mean_abs_dp"] for r in s["pilot"].values() if key in r])
        if not len(rows):
            continue
        for r in rows:
            axs[1].plot(t, r, color="0.75", lw=0.7)
        axs[1].errorbar(t, rows.mean(0), yerr=rows.std(0, ddof=1) / np.sqrt(len(rows)), color="k", fmt=fmt, ms=3,
                        capsize=2, label=name)
    axs[1].set_xlabel("round t")
    axs[1].set_ylabel(r"mean $|p(t)-p(t-1)|$")
    axs[1].set_yscale("log")
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Pilot swarms (inverted, p_neg=0.5, N=64, 6 cells; gray = single cells, black = mean over cells)",
                 fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"qwen_pilot.{ext}", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    s = json.loads((a.root / "qwen-calib" / "summary.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(2, 6)
    axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
    conformity_panels(axs, s, a.root)
    gains_panel(fig.add_subplot(gs[1, :3]), s)
    lambda_panel(fig.add_subplot(gs[1, 3:]), s)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"qwen_calibration.{ext}", dpi=150)
    pilot_figure(s)


if __name__ == "__main__":
    main()
