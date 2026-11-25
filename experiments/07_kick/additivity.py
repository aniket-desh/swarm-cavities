"""Additivity test of kick stage 2 (experiments/07_kick/DESIGN_stage2.md): echoes add over reciprocated ties, so at
fixed K and gains E(gamma = +-0.5) = 0.5 E(gamma = +-1) (theory/quiet_regime.md section 3.2; mock 0.48-0.52).

R_+- = E(+-0.5, stage 2) / E(+-1, stage 1), inverted wording, K = 2, s.e. by the delta method from the two cells'
graph-bootstrap s.e. (the stage-1 and stage-2 graphs differ; cells with the same graph index share the evidence draw,
a mild positive correlation the delta method ignores, which makes this s.e. slightly conservative).

Slope-factor normalization (design review P2): the echo carries the cell's slope factors, 𝒟 = sum D_i^2 d_i <d_j> /
sum D_i^2 over eligible kicks (d_i = 1 - s_i(t0+1)^2, d_j = 1 - s_j(t0)^2 over i's reciprocated partners, D_i = -2 s_i(t0),
|s_i(t0)| >= 0.1, t0 in {6, 8}, both replicas), computed from the base runs only. R_norm = R * 𝒟(+-1) / 𝒟(+-0.5) removes
the difference in settled confidence between the cells.
Run: uv run python experiments/07_kick/additivity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
sys.path.insert(0, str(ROOT / "experiments" / "07_kick"))
from analyze_kick import S_MIN, base_state  # noqa: E402
from cells import cell_id, graph_for  # noqa: E402


def slope_factor(base_tag: str, gamma: float) -> float:
    man = json.loads((ROOT / "results" / base_tag / "manifest.json").read_text())
    num = den = 0.0
    for c in man["cells"]:
        if not (c["wording"] == "inverted" and c["K"] == 2 and c["gamma"] == gamma):
            continue
        J = graph_for(c)
        exp = f"{base_tag}-{cell_id(c)}"
        for rep in (0, 1):
            for t0 in (6, 8):
                s0, s1 = base_state(exp, rep, t0), base_state(exp, rep, t0 + 1)
                for i in np.flatnonzero(np.abs(s0) >= S_MIN):
                    partners = [j for j in np.flatnonzero(J[:, i]) if J[i, j] != 0]
                    if not partners:
                        continue
                    w = 4 * s0[i] ** 2
                    num += w * (1 - s1[i] ** 2) * np.mean([1 - s0[j] ** 2 for j in partners])
                    den += w
    return num / den


if __name__ == "__main__":
    s1 = {(c["wording"], c["K"], c["gamma"]): c for c in
          json.loads((ROOT / "results" / "kick1k-20261125-summary.json").read_text())["cells"]}
    s2 = {(c["wording"], c["K"], c["gamma"]): c for c in
          json.loads((ROOT / "results" / "kick2k-20261125-summary.json").read_text())["cells"]}
    out = {}
    for sign in (+1, -1):
        a, b = s2[("inverted", 2, 0.5 * sign)], s1[("inverted", 2, 1.0 * sign)]
        R = a["echo"] / b["echo"]
        se = abs(R) * np.hypot(a["echo_se"] / a["echo"], b["echo_se"] / b["echo"])
        Dh, Df = slope_factor("kick2-20261125", 0.5 * sign), slope_factor("kick1-20261125", 1.0 * sign)
        key = f"R{'+' if sign > 0 else '-'}"
        out[key] = {"R": R, "se": se, "R_norm": R * Df / Dh, "se_norm": se * Df / Dh, "slope_factor_half": Dh,
                    "slope_factor_full": Df, "E_half": a["echo"], "E_half_se": a["echo_se"], "E_full": b["echo"],
                    "E_full_se": b["echo_se"], "interval_2se": [R - 2 * se, R + 2 * se]}
        print(f"{key} = {R:.3f} ({se:.3f}), 2-s.e. interval [{R - 2 * se:.3f}, {R + 2 * se:.3f}]; slope-factor "
              f"normalized {R * Df / Dh:.3f} (𝒟 half {Dh:.3f}, full {Df:.3f})  [E(+-0.5) {a['echo']:+.4f} "
              f"({a['echo_se']:.4f}) / E(+-1) {b['echo']:+.4f} ({b['echo_se']:.4f})]")
    (ROOT / "results" / "kick2-additivity.json").write_text(json.dumps(out, indent=1))
