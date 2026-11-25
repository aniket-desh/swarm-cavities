"""Surrogates of the calibrated claude-haiku-5-5 agent, fitted to the test-1 probe records (results/calibration/).

Two response surfaces for the mean answer, both of the form E[s] = tanh(x), with the inbox of K peers split into
the agree group A and the disagree group D (cavity/probes.py units: f_A = (1/K) sum_A s_j, f_D = (1/K) sum_D s_j):

    lin    x = a_A f_A - a_D f_D + a_p hpriv + c                                   (experiments/03_calibration, item 2)
    step   x = (1/K) sum_A [al_A s_j + de_A sgn(s_j)] + (1/K) sum_D [al_D s_j + de_D sgn(s_j)] + a_p hpriv + c

"step" adds a per-peer term in the side a peer leans to (sgn = 0 when the peer's stated probability is within 0.03
of 0.5, the band in which the probe text was neutral). It is still additive over peers, so it is an effective
scalar field with influence function phi(s) = al s + de sgn(s) (theory/claim1.md section 4.2).

The inverted disagree wording was probed only in the gains design (agree-only and disagree-only inboxes, 200
calls), so its surrogate is the "lin" form with (a_A, a_D) = (a_+, a_-) of that fit and (a_p, c) of the label fit.

Run: uv run python experiments/04_swarm_design/surrogate.py      (prints the fits and writes surrogate_fits.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cavity.probes import synth_states  # noqa: E402

CAL = ROOT / "results" / "calibration"
HERE = Path(__file__).resolve().parent
NEUTRAL = 0.06          # |s| below this counts as no lean (stated probability within 0.03 of 0.5)


def sgn(s):
    return np.where(np.abs(s) < NEUTRAL, 0.0, np.sign(s))


def _features(r):
    """Per-record sums over the peers of each group: (sum s, sum sgn s) / K for A and D."""
    K = r["K"]
    if r["set"] == "gains":
        a, d = ([r["x"]] * K, []) if r["arm"] == "agree" else ([], [r["x"]] * K)
    else:
        a, d = synth_states(r["hbar"], r.get("comp", "uniform"), K)
    a, d = np.round(np.array(a, float) / 2 + 0.5, 2) * 2 - 1, np.round(np.array(d, float) / 2 + 0.5, 2) * 2 - 1
    return [a.sum() / K, sgn(a).sum() / K, d.sum() / K, sgn(d).sum() / K, r["h_priv"]]


def load(names=("suff", "conf", "gains")):
    recs = []
    for n in names:
        recs += [r for r in json.loads((CAL / f"{n}.json").read_text())["records"] if r["status"] == "ok"]
    X = np.array([_features(r) for r in recs])
    y = np.array([r["s"] for r in recs])
    cell = np.array([r["id"].rsplit("|", 1)[0] for r in recs])
    return X, y, cell


def fit(X, y, cell, kind, seed=0, holdout=0.2):
    """Least-squares fit of tanh(X w + c); returns parameters and held-out cell-mean RMSE (cells held out whole)."""
    cols = {"lin": [0, 2, 4], "step": [0, 1, 2, 3, 4]}[kind]
    rng = np.random.default_rng(seed)
    uc = np.unique(cell)
    test = np.isin(cell, rng.choice(uc, int(round(holdout * len(uc))), replace=False))
    Z = X[:, cols]
    res = least_squares(lambda w: np.tanh(Z[~test] @ w[:-1] + w[-1]) - y[~test], np.r_[np.ones(len(cols)) * 0.5, 0.0])
    w = res.x
    pred = np.tanh(Z @ w[:-1] + w[-1])
    tc = uc[np.isin(uc, cell[test])]
    obs_c = np.array([y[cell == c].mean() for c in tc]) if len(tc) else np.array([np.nan])
    pre_c = np.array([pred[cell == c].mean() for c in tc]) if len(tc) else np.array([np.nan])
    names = {"lin": ["a_A", "a_D_signed", "a_p"], "step": ["al_A", "de_A", "al_D", "de_D", "a_p"]}[kind]
    return dict(kind=kind, **dict(zip(names, map(float, w[:-1]))), c=float(w[-1]),
                rmse_cellmeans_heldout=float(np.sqrt(np.mean((obs_c - pre_c) ** 2))),
                resid_sd=float((y[~test] - pred[~test]).std()), n=int(len(y)))


if __name__ == "__main__":
    X, y, cell = load()
    out = {}
    for kind in ("lin", "step"):
        fits = [fit(X, y, cell, kind, seed=s) for s in range(10)]
        f = fit(X, y, cell, kind, holdout=0.0)          # parameters from all records
        f["rmse_cellmeans_heldout"] = float(np.mean([g["rmse_cellmeans_heldout"] for g in fits]))
        out[kind] = f
        print(kind, json.dumps(f))
    # sampling floor: s.d. within cells (replicate calls on identical inputs)
    uc = np.unique(cell)
    within = np.sqrt(np.mean([y[cell == c].var(ddof=1) for c in uc if (cell == c).sum() > 1]))
    out["within_cell_sd"] = float(within)
    print("within-cell s.d. of s (pure sampling noise of the agent):", round(within, 4))
    summ = json.loads((CAL / "summary.json").read_text())
    out["label_gains"] = {k: summ["gains"][k] for k in ("a_plus", "a_minus")}
    out["inverted_gains"] = {k: summ["gains_inv"][k] for k in ("a_plus", "a_minus")}
    out["white_kernel_R_lag"] = summ["white_kernel"]["R_lag"][:4]
    (HERE / "surrogate_fits.json").write_text(json.dumps(out, indent=1))
