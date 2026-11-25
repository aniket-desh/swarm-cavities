"""Figures for the zero-cost PoA analyses, from the CSVs in results/poa/ (no computation here).

  fig_field_sufficiency  (a) held-out cross-entropy of each inbox model relative to the scalar field M1, per
                         model x task; (b) two-channel gains b+, b- with 95% bootstrap CIs.
  fig_flicker            (a) GPT-4o-mini spin period index (a2/a1, divided by the schedule's memoryless null),
                         observed and simulated with the fitted agent, sync against async; (b) belief E2/E1 by
                         graph group for all model x task cells.
  fig_stability          (a) linear stability of the synchronous map on the actual graphs, max(lam+, lam-) of D W,
                         at round 0 and in rounds 3-7; (b) replica amplification A(t) by graph class.

Also writes field_sufficiency_summary.csv (shares of peer information and contrasts, see summary_table).

Run: uv run python analysis/poa/plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.poa.common import MODEL_ORDER, RESULTS  # noqa: E402

# categorical slots 1-4 of the dataviz reference palette, in fixed order
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "legend.frameon": False})
CELLS = [(m, t) for m in MODEL_ORDER for t in ("objective", "subjective")]
SHORT_NAME = {"gpt-4o-mini": "GPT", "Qwen3.5-9B": "Qwen", "Llama-3.1-8B": "Llama", "gemma-3n-E4B": "Gemma"}
CELL_LABEL = [f"{SHORT_NAME[m]}\n{t[:3]}." for m, t in CELLS]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_field_sufficiency():
    cv = pd.read_csv(RESULTS / "field_sufficiency_cv.csv")
    cv = cv[cv.ref == "M1"]
    gains = pd.read_csv(RESULTS / "field_sufficiency_gains.csv").set_index(["model", "mode"])
    specs = [("M0", "no peers (M0)", MUTED), ("M2", "two channels (M2)", BLUE), ("M2avg", "two channels, averaged",
             AQUA), ("M2cat", "two channels, nonlinear", YELLOW), ("M4", "4 counts + quadratic (M4)", ORANGE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(len(CELLS))
    w = 0.16
    for k, (spec, lab, col) in enumerate(specs):
        y, e = [], []
        for m, t in CELLS:
            r = cv[(cv.model == m) & (cv["mode"] == t) & (cv.spec == spec)].iloc[0]
            y.append(r.dce_vs_ref_mnats)
            e.append(r.se_mnats)
        a.bar(x + (k - 2) * w, y, w * 0.9, yerr=e, color=col, label=lab, error_kw=dict(lw=0.6, ecolor=INK))
    a.axhline(0, color=INK, lw=0.6)
    a.set_xticks(x, CELL_LABEL, fontsize=6.5)
    a.set_ylabel("held-out CE minus CE of scalar field M1\n(millinats per vote)")
    a.set_title("(a) the literal field $Js$ (M1) misses what two gains capture", fontsize=8, loc="left")
    a.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for k, (col, lab, sign) in enumerate([(BLUE, "$b_+$ (agree tie)", 1), (ORANGE, "$b_-$ (disagree tie)", 1)]):
        key = "b_plus" if k == 0 else "b_minus"
        y = np.array([gains.loc[c, key] for c in CELLS])
        lo = np.array([gains.loc[c, key + "_lo"] for c in CELLS])
        hi = np.array([gains.loc[c, key + "_hi"] for c in CELLS])
        b.errorbar(x + (k - 0.5) * 0.25, y, yerr=[y - lo, hi - y], fmt="o", ms=4, color=col, lw=1, label=lab)
    b.axhline(0, color=INK, lw=0.6)
    b.set_xticks(x, CELL_LABEL, fontsize=6.5)
    b.set_ylabel("logit pull per neighbour")
    b.set_title("(b) gains (95% bootstrap CI over statements)", fontsize=8, loc="left")
    b.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_field_sufficiency")


def fig_flicker():
    sim = pd.read_csv(RESULTS / "async_sim.csv")
    obs = pd.read_csv(RESULTS / "flicker_async.csv")
    obs = obs[obs.group == "all random"]
    bip = pd.read_csv(RESULTS / "flicker_bip_contrast.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1, 1.8]))
    xs = {("objective", "sync"): 0, ("objective", "async"): 1, ("subjective", "sync"): 2.6, ("subjective", "async"): 3.6}
    for (mode, sched), x0 in xs.items():
        o = obs[(obs["mode"] == mode) & (obs.schedule == sched)].iloc[0]
        s = sim[(sim["mode"] == mode) & (sim.schedule == sched) & (sim.form == "M2")].iloc[0]
        a.bar(x0 - 0.2, o.P_corrected, 0.36, yerr=2 * o.P_corrected_se, color=BLUE if sched == "sync" else AQUA,
              error_kw=dict(lw=0.6, ecolor=INK))
        a.bar(x0 + 0.2, s.P_corrected, 0.36, color="none", edgecolor=BLUE if sched == "sync" else AQUA, hatch="////",
              lw=1)
    a.axhline(1, color=INK, lw=0.6, ls="--")
    a.set_xticks(list(xs.values()), ["sync", "async", "sync", "async"])
    a.text(0.5, -0.2, "objective", ha="center", transform=a.get_xaxis_transform(), fontsize=7)
    a.text(3.1, -0.2, "subjective", ha="center", transform=a.get_xaxis_transform(), fontsize=7)
    a.set_ylabel("spin period index / null\n(1 = no temporal structure)")
    a.set_title("(a) GPT-4o-mini: flicker vanishes when async", fontsize=8, loc="left")
    a.legend(handles=[Patch(color=MUTED, label="observed (±2 s.e.)"),
                      Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="fitted agent, simulated")],
             fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    a.set_ylim(0, 1.3)
    groups = [("square lattice", "square lattice (bipartite, deg 3.25)", BLUE),
              ("random, agree-bipartite", "random, agree subgraph bipartite (3.5)", ORANGE),
              ("random, agree-non-bipartite", "random, agree subgraph not bipartite (3.5)", AQUA),
              ("triangular lattice", "triangular lattice (deg 4.56)", YELLOW)]
    x = np.arange(len(CELLS))
    for k, (g, lab, col) in enumerate(groups):
        y, e = [], []
        for m, t in CELLS:
            r = bip[(bip.cell == f"{m} | {t}") & (bip.group == g)].iloc[0]
            y.append(r.E2_over_E1)
            e.append(2 * r.se)
        b.errorbar(x + (k - 1.5) * 0.17, y, yerr=e, fmt="o", ms=3.5, color=col, lw=0.8, label=lab)
    b.axhline(1, color=INK, lw=0.6, ls="--")
    b.set_xticks(x, CELL_LABEL, fontsize=6.5)
    b.set_ylabel("belief $E_2/E_1$ (sync, rounds 3-8)")
    b.set_title("(b) flicker on every graph but the triangular lattice", fontsize=8, loc="left")
    b.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    b.set_ylim(0, 1.6)
    fig.tight_layout()
    save(fig, "fig_flicker")


def fig_stability():
    ec = pd.read_csv(RESULTS / "effective_coupling.csv")
    ec = ec[(ec.form == "M2") & (ec.graph_class != "all")]
    amp = pd.read_csv(RESULTS / "effective_coupling_amplification.csv")
    cls_col = [("square lattice", BLUE), ("random signed", ORANGE), ("triangular lattice", YELLOW)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    x = np.arange(len(CELLS))
    for k, (cls, col) in enumerate(cls_col):
        r0, late = [], []
        for m, t in CELLS:
            r = ec[(ec.model == m) & (ec["mode"] == t) & (ec.graph_class == cls)].iloc[0]
            r0.append(max(r.lam_plus_r0, r.lam_minus_r0))
            late.append(max(r.lam_plus_late, r.lam_minus_late))
        xx = x + (k - 1) * 0.22
        a.scatter(xx, r0, s=18, color=col, zorder=3)
        a.scatter(xx, late, s=18, facecolor="white", edgecolor=col, zorder=3)
        a.vlines(xx, late, r0, color=col, lw=0.6)
    a.axhline(1, color=INK, lw=0.6, ls="--")
    a.set_yscale("log")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a.set_yticks([0.2, 0.5, 1, 2, 5])
    a.set_xticks(x, CELL_LABEL, fontsize=6.5)
    a.set_ylabel("largest |eigenvalue| of $DW$ on the graph")
    a.set_title("(a) linear stability: round 0 (filled), rounds 3-7 (open)", fontsize=8, loc="left")
    a.legend(handles=[Patch(color=col, label=cls) for cls, col in cls_col], fontsize=6, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.2))
    for cls, col in cls_col:
        for m, t in CELLS:
            g = amp[(amp.model == m) & (amp["mode"] == t) & (amp.graph_class == cls)]
            b.plot(g.t, g.A, color=col, lw=1.2 if m == "gpt-4o-mini" else 0.6,
                   alpha=1.0 if m == "gpt-4o-mini" else 0.5)
    for cls, col in cls_col:
        b.plot([], [], color=col, label=cls)
    b.axhline(1, color=INK, lw=0.6, ls="--")
    b.set_yscale("log")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xlabel("round t")
    b.set_ylabel("replica amplification $A(t)=D/D_{floor}$")
    b.set_title("(b) replica distance over sampling floor (thick: GPT)", fontsize=8, loc="left")
    b.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "fig_stability")


def summary_table():
    """field_sufficiency_summary.csv: per cell, the held-out gain of each model over M0 as a fraction of the gain of
    the most flexible inbox model M4 (share of the peer information each form captures), and the contrasts that
    answer the sub-questions, with s.e. over statements: M2 vs M1 (two gains), M2avg vs M2 (average vs sum),
    M2cat vs M2 (saturation within a channel), M4 vs M3sq (interaction between counts), M6 vs M4 (own state)."""
    cv = pd.read_csv(RESULTS / "field_sufficiency_cv.csv")
    rows = []
    for m, t in CELLS:
        c = cv[(cv.model == m) & (cv["mode"] == t)]
        ce = c[c.ref == "M1"].set_index("spec").ce_nats

        def contrast(spec, ref):
            r = c[(c.spec == spec) & (c.ref == ref)].iloc[0]
            return r.dce_vs_ref_mnats, r.se_mnats
        full = ce["M0"] - ce["M4"]
        row = dict(model=m, mode=t, gain_M4_over_M0_mnats=1e3 * full)
        for spec in ("M1", "M2", "M2avg", "M2cat", "M3"):
            row[f"share_{spec}"] = (ce["M0"] - ce[spec]) / full
        for spec, ref in (("M2", "M1"), ("M2avg", "M2"), ("M2avg_sep", "M2"), ("M2cat", "M2"), ("M3", "M2"),
                          ("M4", "M3sq"), ("M4", "M2"), ("M6", "M2")):
            d, se = contrast(spec, ref)
            row[f"d_{spec}_vs_{ref}"], row[f"se_{spec}_vs_{ref}"] = d, se
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "field_sufficiency_summary.csv", index=False)


if __name__ == "__main__":
    summary_table()
    fig_field_sufficiency()
    fig_flicker()
    fig_stability()
