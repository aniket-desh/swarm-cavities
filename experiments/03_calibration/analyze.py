"""Analysis of test 1 (experiments/03_calibration/run.py): what single-agent calibration says about the
claude-haiku-5-5 agent of envs/prompts.py, in the units the theory needs.

Outputs results/calibration/summary.json and paper/figures/calibration.{pdf,png}:
  (a) conformity curve E[s](hbar) per (evidence, history) cell, with the tanh fit (beta_resp);
  (b) channel response, agree vs disagree arm, label vs inverted wording (a_+, a_-);
  (c) sufficiency: sbar(h, c) for the six compositions (eta^2_comp, max contrast / range);
  (d) white-probe kernel R(lag) (solver_spec.md eq. (S1)), lag-averaged.
Derived quantities (theory/claim1.md eqs. (10), (13)): with K peers read and the agent averaging its inbox,
the per-tie coupling in the field h = sum_j J_ij s_j is a_+-/K, so for a fixed in-degree graph with sign fractions
p_+-:  Jbar = p_+ a_+ - p_- a_-,  g^2 = (p_+ a_+^2 + p_- a_-^2)/K, and the stability parameter of claim2.md
for homogeneous memoryless agents is Lambda ~ d g (1 + |gamma|), d = <1 - s^2> the mean local slope factor.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cavity.probes as P  # noqa: E402

D = ROOT / "results" / "calibration"
FIG = ROOT / "paper" / "figures"


def load(stage):
    f = D / f"{stage}.json"
    return json.loads(f.read_text()) if f.exists() else None


def counts(recs):
    return {s: sum(r["status"] == s for r in recs) for s in sorted({r["status"] for r in recs})}


def tojson(x):
    if isinstance(x, dict):
        return {str(k): tojson(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tojson(v) for v in x]
    if isinstance(x, np.ndarray):
        return tojson(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    out = {}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))

    conf = load("conf")
    if conf:
        r = conf["records"]
        fc = P.fit_conformity(r)
        out["conformity"] = {**fc, "status": counts(r)}
        for cell in sorted({(x["ev_level"], x["hist"]) for x in r}):
            rs = [x for x in r if (x["ev_level"], x["hist"]) == cell and np.isfinite(x["s"])]
            hb = sorted({x["hbar"] for x in rs})
            m = [np.mean([x["s"] for x in rs if x["hbar"] == h]) for h in hb]
            se = [np.std([x["s"] for x in rs if x["hbar"] == h], ddof=1) / np.sqrt(sum(x["hbar"] == h for x in rs))
                  for h in hb]
            ax[0].errorbar(hb, m, se, marker="o", ms=3, lw=1, label=f"evidence {cell[0]}, last answer {cell[1]:+d}")
            c = fc["offsets"][f"{cell[0]}|{cell[1]}"]
            hh = np.linspace(min(hb), max(hb), 50)
            ax[0].plot(hh, np.tanh(fc["beta_resp"] * hh + c), "k-", lw=0.5)
        ax[0].set(xlabel=r"normalized field $\bar h$", ylabel=r"mean response $s=2p-1$",
                  title=rf"(a) conformity, $\beta_{{resp}}={fc['beta_resp']:.2f}\pm{fc['beta_resp_se']:.2f}$")
        ax[0].legend(fontsize=6)

    for stage, ls in (("gains", "-"), ("gains_inv", "--")):
        g = load(stage)
        if not g:
            continue
        r = g["records"]
        fg = P.fit_channel_gains(r)
        out[stage] = {**fg, "status": counts(r), "disagree_style": g["disagree_style"]}
        for arm, col in (("agree", "C0"), ("disagree", "C3")):
            xs = sorted({x["x"] for x in r})
            m = [np.nanmean([x["s"] for x in r if x["arm"] == arm and x["x"] == v]) for v in xs]
            ax[1].plot(xs, m, ls, color=col, marker="o", ms=3, label=f"{arm}, {g['disagree_style']} wording")
    ax[1].set(xlabel="state of all K peers", ylabel="mean response", title="(b) channel gains")
    ax[1].legend(fontsize=6)

    suff = load("suff")
    if suff:
        r = suff["records"]
        e = P.eta2_comp(r)
        out["sufficiency"] = {**{k: e[k] for k in ("eta2", "eta2_corrected", "max_contrast_frac", "range_over_levels",
                                                  "eta2_by_cell", "n_used", "n_missing")},
                              "route_A_minus_B": e["route_A_minus_B"], "noise_floor_shuffled": e["noise_floor_shuffled"],
                              "contrast_vs_uniform": e["contrast_vs_uniform"], "contrast_se": e["contrast_se"],
                              "levels": e["levels"], "compositions": e["compositions"], "status": counts(r),
                              "pass": bool(e["eta2"] < 0.05 and e["max_contrast_frac"] < 0.2)}
        hb = np.array(e["levels"]) * 0.1
        for b, c in enumerate(e["compositions"]):
            ax[2].errorbar(hb + 0.004 * (b - 2.5), e["sbar"][:, b], e["se"][:, b], marker="o", ms=3, lw=1, label=c)
        ax[2].set(xlabel=r"field level $\bar h$", ylabel=r"$\bar s(h, c)$",
                  title=rf"(c) sufficiency, $\eta^2_{{comp}}={e['eta2']:.3f}$")
        ax[2].legend(fontsize=6)
        # surrogate fitted on everything with a field (suff + conf + gains, label wording)
        allr = r + (conf["records"] if conf else []) + (load("gains")["records"] if load("gains") else [])
        out["surrogate_two_channel"] = P.fit_surrogate(allr, two_channel=True)
        out["surrogate_one_channel"] = P.fit_surrogate(allr, two_channel=False)

    white = load("white")
    if white:
        w = {k: np.array(v, float) if isinstance(v, list) else v for k, v in white.items()}
        k = P.white_probe_kernel(w)
        out["white_kernel"] = {"R_lag": k["R_lag"], "R_lag_se": k["R_lag_se"], "lags": k["lags"], "eps2": k["eps2"],
                               "R": k["R"], "missing": int(np.isnan(w["s"]).sum())}
        ax[3].errorbar(k["lags"], k["R_lag"], k["R_lag_se"], marker="o", ms=3)
        ax[3].axhline(0, color="k", lw=0.5)
        ax[3].set(xlabel=r"lag $t-t'$", ylabel=r"$R(t,t')$ per unit $\bar h$", title="(d) white-probe kernel")

    # derived couplings for an option-B graph, K = 8, label wording, as a function of p_neg
    if "gains" in out and "conformity" in out:
        K = 8
        ap, am = out["gains"]["a_plus"], out["gains"]["a_minus"]
        derived = {}
        for p_neg in (0.0, 0.25, 0.5):
            pp = 1 - p_neg
            derived[str(p_neg)] = {"Jbar": pp * ap - p_neg * am, "g": float(np.sqrt((pp * ap**2 + p_neg * am**2) / K))}
        out["derived_option_B_K8_label"] = derived
        if "gains_inv" in out:
            ai = out["gains_inv"]["a_minus"]
            out["derived_option_B_K8_inverted"] = {
                str(p): {"Jbar": (1 - p) * ap - p * ai, "g": float(np.sqrt(((1 - p) * ap**2 + p * ai**2) / K))}
                for p in (0.25, 0.5)}

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"calibration.{ext}", dpi=150)
    (D / "summary.json").write_text(json.dumps(tojson(out), indent=1))
    print(json.dumps(tojson({k: v for k, v in out.items() if k != "white_kernel"}), indent=1)[:6000])
    if "white_kernel" in out:
        print("R_lag", np.round(out["white_kernel"]["R_lag"], 3), "+-", np.round(out["white_kernel"]["R_lag_se"], 3))


if __name__ == "__main__":
    main()
