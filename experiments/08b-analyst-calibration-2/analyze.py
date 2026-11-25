"""Analysis of 08b-analyst-calibration-2 (coordinator, CPU; fit_cell is also used on the worker for the selection).
Fixed before the probe data exist.

Per cell (prompt variant x K), from the real-note probes: M1 of experiments/08-analyst-calibration/analyze.py by OLS on
x (the half log-odds of the up-call BUY),
    x = a_+ f_A - a_- f_D + nu_.5 xi 1[nu_nom=.5] + nu_1 xi 1[nu_nom=1] + a0 h0 + kappa sigma_own + c,
f = sum_group sigma / K with the TRUE calls (in the reversed-fade variants the reader sees -sigma for faded
colleagues; a_- > 0 still means leaning against their actual call). Bootstrap over base probes, B = 1,000.

Derived per cell: abar = sqrt((a_+^2 + a_-^2)/2), r = a_-/a_+, Jbar = (a_+ - a_-)/2, and Lambda_mutual three ways:
  - solver: max_omega of theory/checks/analyst_lib.dmft(Agent(..., K), "mutual") (Gaussian closure, K = infinity
    in form; T = 40, 3,000 sites x 64 copies; s.e. from 100 bootstrap draws at 1,000 x 32), also with c set to 0;
  - tree:   (K - 1)/2 [tanh^2(a_+/K) + tanh^2(a_-/K)]  (analyst_scenario.md (A13); exact on a tree, no news, notes,
            bias or memory);
  - graph:  1 - 1/A, A the amplification of analyst_lib.chi_sg on simulated runs of the law with the measured
            parameters on fixed_indegree_signed_couplings(1024, K, +1, 1/2) graphs (3 graphs x 4 replicas, T = 60,
            rounds 10-60), the reference at small K where the Gaussian closure is not expected to hold.
Go/no-go (analyst_scenario.md section 7): contrast iff abar >= 1 and r > 0.3; boundary iff Lambda >= 0.7 (solver,
reported with the other two); herding first iff Jbar > 1.3.

Polarity decomposition (phase A, K = 16): order effect = mean c(SELL first) - mean c(BUY first) over P0-P3; framing
effect = mean c(question "down") - mean c(question "up") over P0-P3; token vs content from the YES/NO pair: a bias
toward the token NO gives c(P4) < 0 and c(P5) > 0, a bias toward the content "down" gives c < 0 in both.
Reproducibility: P0 against experiments/08-analyst-calibration max/real (same prompts and probes).

Usage: uv run python experiments/08b-analyst-calibration-2/analyze.py [selftest]
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


def lam_all(p: dict, K: int, M=3000, copies=64, seed=0) -> dict:
    P = al.Agent(p["a_plus"], p["a_minus"], nu=abs(p["nu_05"]), a0=abs(p["a0"]), c=p["c"], kappa=p["kappa"], K=K)
    d = al.dmft(P, "mutual", T=40, M=M, copies=copies, seed=seed)
    return {"Lam": float(d["Lam"].max()), "q": float(d["q"][10:].mean())}


def lam_graph(p: dict, K: int, N=1024, T=60, R=4, n_graphs=3, seed=0, c=None) -> float:
    """1 - 1/A from chi_sg on simulated runs of the law with these parameters (module docstring)."""
    P = al.Agent(p["a_plus"], p["a_minus"], nu=abs(p["nu_05"]), a0=abs(p["a0"]), c=p["c"] if c is None else c,
                 kappa=p["kappa"], K=K)
    rng = np.random.default_rng(seed)
    As = []
    for _ in range(n_graphs):
        J = al.trust_graph(N, K, "mutual", rng)
        o = al.simulate(al.couplings(J, P), P, T, al.draw_inputs(N, T, R, rng))
        As.append(al.chi_sg(o["sig"], burn=10)[0])
    return float(1 - 1 / np.mean(As))


def lam_tree(p: dict, K: int) -> float:
    return float((K - 1) / 2 * (np.tanh(p["a_plus"] / K) ** 2 + np.tanh(p["a_minus"] / K) ** 2))


def derived(p: dict) -> dict:
    return {"abar": float(np.sqrt((p["a_plus"] ** 2 + p["a_minus"] ** 2) / 2)), "r": float(p["a_minus"] / p["a_plus"]),
            "Jbar": float((p["a_plus"] - p["a_minus"]) / 2)}


def analyze_cell(name: str, K: int, n_boot=B, n_lam=100) -> dict:
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
        f = AN08.fit_arrays(X, y, "M1", cnt)
        draws.append({**f, **derived(f)})
    out = {k: {"est": float(est[k]), "se": float(np.std([d[k] for d in draws], ddof=1))} for k in NAMES + ["resid_sd"]}
    dv = derived(est)
    out.update({k: {"est": v, "se": float(np.std([d[k] for d in draws], ddof=1))} for k, v in dv.items()})
    L = lam_all(est, K)
    L0 = lam_all({**est, "c": 0.0}, K)
    lb = [lam_all(d, K, M=1000, copies=32, seed=i)["Lam"] for i, d in enumerate(draws[:n_lam])]
    out.update(n=len(rows), n_missing=sum(r["status"] != "ok" for r in recs.values()),
               n_censored=sum(bool(r["censored"]) for r in recs.values() if r["status"] == "ok"),
               valid_mass_min=float(min(r["valid_mass"] for r in recs.values())),
               Lam_solver={"est": L["Lam"], "se": float(np.std(lb, ddof=1))}, Lam_solver_c0=L0["Lam"], q=L["q"],
               Lam_tree=lam_tree(est, K), Lam_graph=lam_graph(est, K), Lam_graph_c0=lam_graph(est, K, c=0.0))
    out["contrast_go"] = bool(dv["abar"] >= 1 and dv["r"] > 0.3)
    out["boundary_go"] = bool(L["Lam"] >= 0.7)
    out["herding_first"] = bool(dv["Jbar"] > 1.3)
    return out


def selftest():
    """lam_graph reproduces the theory note's graph value 1 - 1/A = 0.58 at gains (4, 4), reference design, K = 16
    (analyst_scenario.md table 3.3, N = 1,024, T = 40), within the run-to-run scatter (about 0.05)."""
    p = dict(a_plus=4.0, a_minus=4.0, nu_05=0.5, a0=0.3, c=0.0, kappa=0.0)
    v = lam_graph(p, 16, T=40, n_graphs=3, seed=1)
    print("lam_graph (4,4), K=16:", round(v, 3), "(note: 0.58); tree", round(lam_tree(p, 16), 3))
    assert abs(v - 0.58) < 0.08
    p2 = dict(a_plus=2.0, a_minus=2.0, nu_05=0.0, a0=0.0, c=0.0, kappa=0.0)
    v2, t2 = lam_graph(p2, 16, T=60, n_graphs=2, seed=2), lam_tree(p2, 16)
    print("lam_graph (2,2), no news, K=16:", round(v2, 3), "tree (exact on a tree):", round(t2, 3))
    assert abs(v2 - t2) < 0.06


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = sorted({(p.stem.rsplit("_K", 1)[0], int(p.stem.rsplit("_K", 1)[1])) for p in DIR.glob("*_K*.jsonl")})
    S = {"selection": json.loads((DIR / "selection.json").read_text()) if (DIR / "selection.json").exists() else None,
         "cells": {}}
    for name, K in cells:
        S["cells"][f"{name}|K{K}"] = c = analyze_cell(name, K)
        print(f"{name} K={K}: n {c['n']} miss {c['n_missing']} cens {c['n_censored']} | " + " ".join(
            f"{k} {c[k]['est']:+.2f}({c[k]['se']:.2f})" for k in NAMES + ["abar", "r", "Jbar"]) +
            f" | Lam solver {c['Lam_solver']['est']:.3f}({c['Lam_solver']['se']:.3f}) c0 {c['Lam_solver_c0']:.3f} "
            f"tree {c['Lam_tree']:.3f} graph {c['Lam_graph']:.3f} graph_c0 {c['Lam_graph_c0']:.3f}", flush=True)
    # polarity decomposition
    cc = {nm: S["cells"][f"{nm}|K16"]["c"]["est"] for nm in ("P0", "P1", "P2", "P3", "P4", "P5") if f"{nm}|K16" in S["cells"]}
    if len(cc) == 6:
        S["polarity"] = {"order_effect": (cc["P1"] + cc["P3"] - cc["P0"] - cc["P2"]) / 2,
                         "framing_effect": (cc["P2"] + cc["P3"] - cc["P0"] - cc["P1"]) / 2,
                         "yesno_up": cc["P4"], "yesno_down": cc["P5"],
                         "token_part": (cc["P5"] - cc["P4"]) / 2, "content_part": (cc["P4"] + cc["P5"]) / 2}
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
