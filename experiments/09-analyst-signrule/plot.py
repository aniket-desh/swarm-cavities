"""Figures of 09-analyst-signrule from results/analyst-signrule/summary.json (analyze.py computes everything).
One question per figure:

  analyst_signrule_echo       Does the memory-robust echo X2 follow the sign rule per world?  (gray: runs; black: world
                              mean with s.e.; theory: frozen M1 value with its calibration s.d. (bar), the zero-shot
                              interval (gray band), and G d from the swarm's own couplings plus the one-way X2, since
                              G d predicts the difference from the one-way world (open diamond))
  analyst_signrule_kick       Do kick twins return the same echo?  (E per world, both replays, with s.e.; frozen M1
                              and M2 values as ticks)
  analyst_signrule_mean       Which law holds for the mean belief?  (m(t): gray runs, black world means; frozen M1
                              and M2 values as lines)
  analyst_signrule_residuals  Where do the observations depart from the frozen predictions?  ((obs - pred) / s.e. per
                              statistic and world, against M1 (filled) and M2 (open); +-3 lines)

Conventions (paper-writing figure rules): gray = realizations, black = means with s.e., lines/bands = theory.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
SUMMARY = ROOT / "results" / "analyst-signrule" / "summary.json"
PRED = json.loads((ROOT / "theory" / "checks" / "analyst_signrule_predictions.json").read_text())
W3 = ("mutual", "oneway", "antagonistic")
LAB = {"mutual": "mutual", "oneway": "one-way", "antagonistic": "antagonistic"}
INTERVAL = {"mutual": (0.03, 0.07), "oneway": (0.0, 0.0), "antagonistic": (0.005, 0.021)}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def echo(S):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, w in enumerate(W3):
        lo, hi = INTERVAL[w]
        ax.add_patch(plt.Rectangle((k - 0.3, lo), 0.6, max(hi - lo, 0.0006), color="0.9", zorder=0))
        p = PRED["predictions"][w]["M1"]["X2"]
        ax.errorbar(k + 0.18, p["value"], yerr=p["calib_sd"], fmt="_", color="0.4", ms=14, capsize=0, lw=2,
                    label="frozen M1 (calib. s.d.)" if k == 0 else None)
        runs = S["worlds"][w]["X2"]["runs"]
        ax.scatter(np.full(len(runs), k - 0.15) + np.linspace(-0.05, 0.05, len(runs)), runs, s=10, c="0.6", zorder=2,
                   label="runs" if k == 0 else None)
        m = S["worlds"][w]["X2"]
        ax.errorbar(k, m["mean"], yerr=m["se"], fmt="o", color="k", ms=5, capsize=2, zorder=3,
                    label="world mean (s.e.)" if k == 0 else None)
        if "reaction" in S and w != "oneway":
            g = S["reaction"][w]["Gd_M" if w == "mutual" else "Gd_A"] + S["worlds"]["oneway"]["X2"]["mean"]
            ax.plot(k + 0.3, g, "D", mfc="w", color="k", ms=5,
                    label=r"$\hat G d$ + $X_2$(one-way), swarm couplings" if k == 0 else None)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xticks(range(3), [LAB[w] for w in W3])
    ax.set_ylabel(r"$X_2=\rho_2-\rho_1^2$ (rounds 5-20)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "analyst_signrule_echo")


def kick(S):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, w in enumerate(W3):
        kt = S["kicks"].get(w)
        if not kt:
            continue
        for key, dx, mfc, lab in (("E", -0.08, "k", "rewritten notes"), ("E_kept", 0.08, "w", "notes kept")):
            ax.errorbar(k + dx, kt[key]["est"], yerr=kt[key]["se"], fmt="o", color="k", mfc=mfc, ms=5, capsize=2,
                        label=lab if k == 0 else None)
        P1, P2 = PRED["predictions"][w]["M1"]["kick"], PRED["predictions"][w]["M2"]["kick"]
        ax.plot([k - 0.3, k + 0.3], [P1["E"]] * 2, "-", color="0.4", lw=1.2, label="frozen M1" if k == 0 else None)
        ax.plot([k - 0.3, k + 0.3], [P2["E"]] * 2, ":", color="0.4", lw=1.2, label="frozen M2" if k == 0 else None)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xticks(range(3), [LAB[w] for w in W3])
    ax.set_ylabel("kick echo $E$ per unit kick")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "analyst_signrule_kick")


def mean(S):
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, w in zip(axs, W3):
        mts = [np.array(v["m_t"]) for k, v in S["runs"].items() if k.startswith(w + ":")]
        for m in mts:
            ax.plot(m, color="0.7", lw=0.8)
        if mts:
            M = np.array(mts)
            ax.errorbar(np.arange(M.shape[1]), M.mean(0), yerr=M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else None,
                        color="k", lw=1.2, capsize=0)
        ax.axhline(PRED["predictions"][w]["M1"]["m"]["value"], color="0.4", lw=0.8, ls="-", label="frozen M1")
        ax.axhline(PRED["predictions"][w]["M2"]["m"]["value"], color="0.4", lw=0.8, ls=":", label="frozen M2")
        ax.set_title(LAB[w], fontsize=9)
        ax.set_xlabel("round")
    axs[0].set_ylabel(r"mean sampling belief $m(t)$")
    axs[0].legend(frameon=False, fontsize=7)
    save(fig, "analyst_signrule_mean")


def residuals(S):
    keys = ("rho1", "rho2", "X2", "c2", "Xc2", "q", "m", "kick_E", "kick_E_kept", "kick_wA", "kick_wD")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for k, w in enumerate(W3):
        for law, mfc, dx in (("M1", "k", -0.12), ("M2", "w", 0.12)):
            zs = [S["residuals"].get(f"{w}|{law}|{s}", {}).get("z") for s in keys]
            xs = np.arange(len(keys)) + (k - 1) * 0.28 + dx / 3
            ax.scatter(xs, [np.nan if z is None else np.clip(z, -12, 12) for z in zs], marker="osD"[k], s=16,
                       facecolors=mfc, edgecolors="k", label=f"{LAB[w]}, {law}")
    for y in (-3, 3):
        ax.axhline(y, color="0.6", ls="--", lw=0.7)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("(obs - pred) / s.e. (clipped at 12)")
    ax.legend(frameon=False, fontsize=6, ncol=3)
    save(fig, "analyst_signrule_residuals")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    if len(sys.argv) > 2:
        FIG = Path(sys.argv[2])
    S = json.loads(path.read_text())
    echo(S)
    kick(S)
    mean(S)
    residuals(S)
    print("figures in", FIG)
