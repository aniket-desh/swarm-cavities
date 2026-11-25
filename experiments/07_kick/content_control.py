"""Content control for the kick-size ratio (experiments/07_kick/DESIGN.md, reviewer P1): do fresh agents respond less,
per unit of stated lean, to real messages that lean weakly than to messages that lean strongly?

Small kicks come from senders with |s| <= 0.275 (|D| = 2|s| <= 0.55); a smaller per-unit response to such messages
would lower the kick-size ratio without any anchoring in the reader. The transfer check (experiments/05_swarm/
transfer.py, arm "real") showed fresh agents (round-1 state) inboxes of real pilot messages, so each probe's agree
field can be split by the messages' |s|:

    E[s] = tanh(a_weak f_A,weak + a_mid f_A,mid + a_strong f_A,strong - a_- f_D + a_p h_priv + c),
    f_A,bin = sum over agree messages in the bin of s_j / K,  bins |s_j| <= 0.275, (0.275, 0.5), >= 0.5.

Reported: a_weak / a_strong with a bootstrap s.e. over probes, per (wording, K). Zero cost (existing data).
Run: uv run python experiments/07_kick/content_control.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cavity.probes import _fit_tanh  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
import transfer as TR  # noqa: E402

WEAK, STRONG = 0.275, 0.5

if __name__ == "__main__":
    out = {}
    for wording in TR.WORDINGS:
        recs = [r for r in json.loads((TR.OUT / f"{wording}.json").read_text()) if r["arm"] == "real" and r["s"] is not None]
        for K in TR.KS:
            bps = {bp["b"]: bp for bp in TR.base_probes(K, TR.N_BASE)}
            X, y = [], []
            for r in (r for r in recs if r["K"] == K):
                ctx = TR.context(bps[r["b"]], "real", 0)
                fa = np.zeros(3)
                for m in ctx.agree:
                    s = 2 * m.prob - 1
                    fa[0 if abs(s) <= WEAK else 2 if abs(s) >= STRONG else 1] += s / K
                fD = sum(2 * m.prob - 1 for m in ctx.disagree) / K
                X.append([*fa, -fD, r["h_priv"]]); y.append(r["s"])
            X, y = np.array(X), np.array(y)
            w, _, _ = _fit_tanh(X, y, np.zeros(len(y)), [0.8, 0.8, 0.8, 0.2, 0.4])
            rng = np.random.default_rng(0)
            ratios = []
            for _ in range(500):
                b = rng.integers(len(y), size=len(y))
                wb, _, _ = _fit_tanh(X[b], y[b], np.zeros(len(y)), w)
                ratios.append(wb[0] / wb[2])
            n_weak = int((np.abs(X[:, 0]) > 0).sum())
            out[f"{wording}|K{K}"] = {"a_weak": w[0], "a_mid": w[1], "a_strong": w[2], "ratio_weak_strong": w[0] / w[2],
                                      "ratio_se": float(np.std(ratios, ddof=1)), "n": len(y), "n_probes_with_weak": n_weak}
            o = out[f"{wording}|K{K}"]
            print(f"{wording} K={K}: a_weak {w[0]:.2f} a_mid {w[1]:.2f} a_strong {w[2]:.2f} | weak/strong "
                  f"{o['ratio_weak_strong']:.2f} ({o['ratio_se']:.2f}) | n {len(y)}, probes with a weak agree message {n_weak}")
    (ROOT / "results" / "transfer" / "content_control.json").write_text(json.dumps(out, indent=1, default=float))
