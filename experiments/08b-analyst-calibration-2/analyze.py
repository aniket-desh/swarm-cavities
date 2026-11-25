"""Analysis of 08b-analyst-calibration-2 (coordinator, CPU; fit_cell and load_cell are also used on the worker for
the selection). Fixed before the probe data exist. All cells are at K = 16.

Per cell (prompt variant), from its probes: M1 of experiments/08-analyst-calibration/analyze.py by OLS on x (the half
log-odds of the up-call BUY),
    x = a_+ f_A - a_- f_D + nu_.5 xi 1[nu_nom=.5] + nu_1 xi 1[nu_nom=1] + a0 h0 + kappa sigma_own + c,
f = sum_group sigma / 16 with the TRUE calls (in the reversed-fade cells the reader is shown -sigma for faded
colleagues; a_- > 0 still means leaning against their actual call). Bootstrap over base probes, B = 1,000.

The frozen go rule of theory/analyst_scenario.md (section 7 and (A4), (A9)), per cell, at the measured parameters and
again with c = 0 (the bias removed, e.g. by sampling calls from tanh(x - c_hat)):
    abar = sqrt((a_+^2 + a_-^2)/2), r = a_-/a_+, Jbar = (a_+ - a_-)/2;
    contrast go iff abar >= 1 and r > 0.3;
    Gamma = g^2 = (a_+^2 + a_-^2)/(2K) (mutual), -a_+ a_-/K (antagonistic); q = <s^2> from the solver in that world;
    echo resolvable iff |Gamma| (1 - q)^2 > 0.03 (the leading-order echo (A9)), per world;
    rho_2 per world from analyst_lib.dmft (copies = 1, 50,000 sites, T = 30; the theory's solver value);
    Lambda_mutual = max_omega of analyst_lib.dmft(..., "mutual") (3,000 sites x 64 copies, T = 40);
    boundary go iff Lambda_mutual >= 0.7; herding first iff Jbar > 1.3.
s.e. of Lambda, rho_2 and the echo criterion from 100 bootstrap draws (smaller solver runs).
Criteria (EXPERIMENT.md), three-way on the bootstrap s.e. (holds iff on the right side of the threshold by > 2 s.e.,
fails iff on the wrong side by > 2 s.e., else inconclusive): bias removed iff |c| < 0.2; fading works iff r > 0.3.

Polarity decomposition: order effect = mean c(SELL first) - mean c(BUY first) over P0-P3; framing effect =
mean c("down") - mean c("up") over P0-P3; token part (c(P5) - c(P4))/2, content part (c(P4) + c(P5))/2 (a bias toward
the token NO flips sign between P4 and P5, a bias toward "down" does not); T0 - P0: what pilot note text contributes.
Reproducibility: P0 against experiments/08-analyst-calibration max/real (same prompts and probes).

Usage: uv run python experiments/08b-analyst-calibration-2/analyze.py
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.ledger import read_jsonl  # noqa: E402
from theory.checks import analyst_lib as al  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN08 = _load("analyze08", ROOT / "experiments" / "08-analyst-calibration" / "analyze.py")
TAG = "analyst-calib2-20261125"
DIR = ROOT / "results" / TAG
OUT = ROOT / "results" / "analyst-calib2"
NAMES = AN08.M1_NAMES
B = 1000


def load_cell(name: str, K: int):
    design = {d["b"]: d for d in json.loads((DIR / f"design_K{K}.json").read_text())}
    recs = {r["b"]: r for r in read_jsonl(DIR / f"{name}_K{K}.jsonl")}
    rows = [{**design[b], "x": r["x"], "p_buy": r["p_buy"]} for b, r in sorted(recs.items()) if r["status"] == "ok"]
    return rows, recs


def fit_cell(name: str, K: int) -> dict:
    return AN08.fit(load_cell(name, K)[0], "M1")


def agent(p: dict, c=None) -> al.Agent:
    return al.Agent(p["a_plus"], p["a_minus"], nu=abs(p["nu_05"]), a0=abs(p["a0"]), c=p["c"] if c is None else c,
                    kappa=p["kappa"], K=16)


def theory_numbers(p: dict, c=None, small=False, seed=0) -> dict:
    """The go-rule quantities of the module docstring for one parameter set."""
    e = {"abar": float(np.sqrt((p["a_plus"] ** 2 + p["a_minus"] ** 2) / 2)), "r": float(p["a_minus"] / p["a_plus"]),
         "Jbar": float((p["a_plus"] - p["a_minus"]) / 2)}
    P = agent(p, c)
    d = al.dmft(P, "mutual", T=40, M=1000 if small else 3000, copies=32 if small else 64, seed=seed)
    e["Lam_mutual"] = float(d["Lam"].max())
    for w in ("mutual", "antagonistic"):
        dw = al.dmft(P, w, T=30, M=10000 if small else 50000, seed=seed + 1)
        q = float(dw["q"][10:].mean())
        G = al.effective(P, w)["Gamma"]
        e[f"q_{w}"], e[f"rho2_{w}"], e[f"echo_lo_{w}"] = q, float(dw["rho"][1]), float(abs(G) * (1 - q) ** 2)
    return e


def three_way(est, se, tau, below=True) -> str:
    """holds iff est is on the right side of tau by more than 2 se (below: est < tau), fails iff on the wrong side by
    more than 2 se, else inconclusive."""
    d = (tau - est) if below else (est - tau)
    return "holds" if d > 2 * se else "fails" if d < -2 * se else "inconclusive"


def analyze_cell(name: str, K: int = 16, n_boot=B, n_th=100) -> dict:
    rows, recs = load_cell(name, K)
    X = np.array([AN08._cols(r, "M1") for r in rows])
    y = np.array([r["x"] for r in rows])
    bids = np.array([r["b"] for r in rows])
    est = AN08.fit_arrays(X, y, "M1")
    rng = np.random.default_rng(0)
    allb = np.array(sorted(recs))
    draws = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.choice(allb, len(allb)), minlength=allb.max() + 1)[bids].astype(float)
        draws.append(AN08.fit_arrays(X, y, "M1", cnt))
    out = {k: {"est": float(est[k]), "se": float(np.std([d[k] for d in draws], ddof=1))} for k in NAMES + ["resid_sd"]}
    for tag, c in (("", None), ("_c0", 0.0)):
        th = theory_numbers(est, c)
        thb = [theory_numbers(d, c, small=True, seed=i) for i, d in enumerate(draws[:n_th])]
        for k, v in th.items():
            out[k + tag] = {"est": v, "se": float(np.std([t[k] for t in thb], ddof=1))}
        out["contrast_go" + tag] = bool(th["abar"] >= 1 and th["r"] > 0.3)
        out["boundary_go" + tag] = bool(th["Lam_mutual"] >= 0.7)
        out["herding_first" + tag] = bool(th["Jbar"] > 1.3)
        for w in ("mutual", "antagonistic"):
            out[f"echo_resolvable_{w}{tag}"] = bool(th[f"echo_lo_{w}"] > 0.03)
    out["label_bias_removed"] = three_way(abs(out["c"]["est"]), out["c"]["se"], 0.2, below=True)
    out["label_fading_works"] = three_way(out["r"]["est"], out["r"]["se"], 0.3, below=False)
    out.update(n=len(rows), n_missing=sum(r["status"] != "ok" for r in recs.values()),
               n_censored=sum(bool(r["censored"]) for r in recs.values() if r["status"] == "ok"),
               valid_mass_min=float(min(r["valid_mass"] for r in recs.values())))
    if out["n_missing"] >= 0.05 * len(recs):
        out["label_bias_removed"] = out["label_fading_works"] = "inconclusive (missing >= 5%)"
    return out


def selftest():
    """theory_numbers reproduces the note's reference-design values at gains (4, 4): Lambda_mutual 0.60 (table 3.3),
    rho_2 +0.256 mutual and -0.258 antagonistic (solver, table of section 2.4)."""
    t = theory_numbers(dict(a_plus=4.0, a_minus=4.0, nu_05=0.5, a0=0.3, c=0.0, kappa=0.0))
    print({k: round(v, 3) for k, v in t.items()})
    assert abs(t["Lam_mutual"] - 0.60) < 0.04 and abs(t["rho2_mutual"] - 0.256) < 0.01
    assert abs(t["rho2_antagonistic"] + 0.258) < 0.01
    assert three_way(0.1, 0.02, 0.2) == "holds" and three_way(0.5, 0.05, 0.3, below=False) == "holds"
    assert three_way(0.25, 0.05, 0.2) == "inconclusive" and three_way(0.0, 0.05, 0.3, below=False) == "fails"
    print("selftest ok")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = sorted({(p.stem.rsplit("_K", 1)[0], int(p.stem.rsplit("_K", 1)[1])) for p in DIR.glob("*_K*.jsonl")})
    S = {"selection": json.loads((DIR / "selection.json").read_text()) if (DIR / "selection.json").exists() else None,
         "cells": {}}
    for name, K in cells:
        S["cells"][name] = c = analyze_cell(name, K)
        print(f"{name}: n {c['n']} miss {c['n_missing']} cens {c['n_censored']} | " + " ".join(
            f"{k} {c[k]['est']:+.2f}({c[k]['se']:.2f})" for k in NAMES + ["abar", "r", "Jbar"]) +
            f" | Lam {c['Lam_mutual']['est']:.3f} rho2 mut {c['rho2_mutual']['est']:+.3f} ant "
            f"{c['rho2_antagonistic']['est']:+.3f} echo_lo mut {c['echo_lo_mutual']['est']:.3f} ant "
            f"{c['echo_lo_antagonistic']['est']:.3f} | c=0: Lam {c['Lam_mutual_c0']['est']:.3f} rho2 mut "
            f"{c['rho2_mutual_c0']['est']:+.3f} ant {c['rho2_antagonistic_c0']['est']:+.3f} | contrast "
            f"{c['contrast_go']} bias {c['label_bias_removed']} fade {c['label_fading_works']}", flush=True)
    # polarity decomposition
    cc = {nm: S["cells"][nm]["c"]["est"] for nm in ("P0", "P1", "P2", "P3", "P4", "P5", "T0") if nm in S["cells"]}
    if len(cc) >= 6:
        S["polarity"] = {"order_effect": (cc["P1"] + cc["P3"] - cc["P0"] - cc["P2"]) / 2,
                         "framing_effect": (cc["P2"] + cc["P3"] - cc["P0"] - cc["P1"]) / 2,
                         "yesno_up": cc["P4"], "yesno_down": cc["P5"],
                         "token_part": (cc["P5"] - cc["P4"]) / 2, "content_part": (cc["P4"] + cc["P5"]) / 2,
                         "T0_minus_P0": cc["T0"] - cc["P0"] if "T0" in cc else None}
        print("polarity", {k: round(v, 3) for k, v in S["polarity"].items()})
    # reproducibility against 08
    p08 = ROOT / "results" / "analyst-probes-20261125" / "max_real.jsonl"
    if p08.exists() and (DIR / "P0_K16.jsonl").exists():
        a = {r["b"]: r for r in read_jsonl(p08)}
        b = {r["b"]: r for r in read_jsonl(DIR / "P0_K16.jsonl")}
        d = np.array([abs(a[k]["x"] - b[k]["x"]) for k in b if k in a and a[k]["status"] == b[k]["status"] == "ok"])
        S["repro_vs_08"] = {"n": int(len(d)), "median_abs_dx": float(np.median(d)), "max_abs_dx": float(d.max()),
                            "frac_identical": float((d == 0).mean())}
        print("reproducibility P0 vs 08 max/real:", S["repro_vs_08"])
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
