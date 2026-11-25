"""Predicted observables for every candidate swarm design, from the calibrated surrogate (design_lib.py):
direct simulation on fixed in-degree graphs (N = 32, 64; two replicas per graph) and the Gaussian DMFT.

Writes predictions.json (one record per cell) and prints the tables used in DESIGN.md.

Run: uv run python experiments/04_swarm_design/predict.py [--quick]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from design_lib import GAINS, SIG_XI, dmft, effective, simulate

HERE = Path(__file__).resolve().parent
T = 10
QUICK = "--quick" in sys.argv
NG = 80 if QUICK else 400


def feasible(K, gamma, p_neg):
    r = round(abs(gamma) * K) / K
    if abs(r - abs(gamma)) > 1e-9:
        return False
    return gamma >= 0 or r <= 2 * min(p_neg, 1 - p_neg) + 1e-9


def cell(wording, scale, K, p_neg, gamma, N, mode="sync", seed=0, with_dmft=True):
    ap, am = GAINS[(wording, scale)]
    sim = simulate(N, K, gamma, p_neg, ap, am, T=T, mode=mode, n_graphs=NG, seed=seed)
    rec = dict(wording=wording, scale=scale, K=K, p_neg=p_neg, gamma=gamma, N=N, mode=mode,
               eff={k: float(v) for k, v in effective(ap, am, K, p_neg, gamma).items()})
    stat = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1)))
    rec["sim"] = dict(
        radius=stat(sim["radius"]), angle=float(np.median(sim["angle"])), mean_mode=stat(sim["mean_mode"]),
        amp=stat(sim["D"][:, T] / (2 * SIG_XI**2)), c1=stat(sim["c1"]), c2=stat(sim["c2"]),
        P_delta=stat(sim["P_delta"]), P_raw=stat(sim["P_raw"]), dq=stat(sim["dq"][:, T]),
        mT=stat(sim["m"][:, T]), CT2=stat(sim["C2"][:, T]), qT=stat(sim["q"][:, T]),
        m=sim["m"].mean(0).tolist(), Cd=sim["Cd"].mean(0).tolist(), C0=sim["C0"].mean(0).tolist(),
        q=sim["q"].mean(0).tolist(), D=sim["D"].mean(0).tolist(), dq_t=sim["dq"].mean(0).tolist(),
        m_sd=sim["m"].std(0, ddof=1).tolist(), Cd_sd=sim["Cd"].std(0, ddof=1).tolist(), q_sd=sim["q"].std(0, ddof=1).tolist())
    if with_dmft and mode == "sync":
        d = dmft(K, gamma, p_neg, ap, am, T=T, seed=seed)
        d0 = dmft(K, gamma, p_neg, ap, am, T=T, seed=seed, shared_couplings=False)
        rec["dmft"] = dict(Lam=d["Lam"], amp=float(d["D"][T] / (2 * SIG_XI**2)), c1=float(d["c1"]), c2=float(d["c2"]),
                           P_delta=float(d["P_delta"]), P_raw=float(d["P_raw"]),
                           dq=float(d["q"][T, T] - d0["q"][T, T]), mT=float(d["m"][T]), CT2=float(d["C"][T, T - 2]),
                           qT=float(d["q"][T, T]), m=d["m"].tolist(), Cd=np.diag(d["C"]).tolist(), C0=d["C"][:, 0].tolist(),
                           q=np.diag(d["q"]).tolist(), D=d["D"].tolist(),
                           dq_t=(np.diag(d["q"]) - np.diag(d0["q"])).tolist(), R_lag=d["R"][T, T - 1::-1][:4].tolist())
    return rec


if __name__ == "__main__":
    cells = []
    grid = {"label": [0.25, 0.5], "inverted": [0.5, 0.625]}
    for wording, scale in [("label", "small"), ("label", "large"), ("inverted", "large"), ("inverted", "small")]:
        for K, p_neg, gamma in itertools.product([2, 4, 8], grid[wording], [-1.0, -0.5, 0.0, 0.5, 1.0]):
            if not feasible(K, gamma, p_neg):
                continue
            for N in (32, 64):
                cells.append(cell(wording, scale, K, p_neg, gamma, N))
    # asynchronous control for the cells that carry the gamma predictions
    for scale, K, gamma, N in itertools.product(["large", "small"], [2, 4], [-1.0, 0.0, 1.0], [64]):
        cells.append(cell("inverted", scale, K, 0.5, gamma, N, mode="async"))
    (HERE / "predictions.json").write_text(json.dumps(cells))

    f = lambda t: f"{t[0]:6.3f}({t[1]:.3f})"
    print(f"sim: mean(per-graph s.d.) over {NG} graphs, 2 replicas each, T={T}; dmft: Gaussian single-site theory")
    for c in cells:
        if c["N"] != 64:
            continue
        s, e = c["sim"], c["eff"]
        line = (f"{c['wording'][:3]}/{c['scale'][:2]} K={c['K']} p-={c['p_neg']:.3f} g={c['gamma']:+.1f} {c['mode'][:2]} | "
                f"Jb {e['Jbar']:.2f} g {e['g']:.2f} ge {e['gamma_eff']:+.2f} | rad {s['radius'][0]:.2f}@{s['angle']:3.0f} mm {s['mean_mode'][0]:.2f} | "
                f"amp {f(s['amp'])} c1 {f(s['c1'])} c2 {f(s['c2'])} Pd {f(s['P_delta'])} Pr {f(s['P_raw'])} dq {f(s['dq'])} mT {f(s['mT'])}")
        if "dmft" in c:
            d = c["dmft"]
            line += (f" || dmft Lam {max(d['Lam'].values()):.2f} amp {d['amp']:.2f} c1 {d['c1']:.3f} c2 {d['c2']:.3f} "
                     f"Pd {d['P_delta']:.2f} dq {d['dq']:.3f} mT {d['mT']:.3f}")
        print(line)
