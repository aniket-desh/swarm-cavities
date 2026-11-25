"""Analysis of 08-analyst-calibration (coordinator, CPU). Reads the raw records written by run.py; writes
results/analyst-calib/summary.json and prints the tables of RESULT.md. Fixed before the probe data exist.

Half log-odds of a probe, from the decision token (envs/analyst.py (D1)):  x = (log P(BUY) - log P(SELL)) / 2.

M1, the law (A1) of theory/analyst_scenario.md, by OLS on x, per wording and text arm:
    x = a_+ f_A - a_- f_D + nu_.5 xi 1[nu_nom = .5] + nu_1 xi 1[nu_nom = 1] + a0 h0 + kappa sigma_own + c
    f_A = sum_allies sigma_j / K, f_D = sum_faded sigma_j / K (K = 16); xi, h0 the realized fields ((N1), (N2)).
    nu_.5 is the theory's nu at the pilot's news amplitude; nu_1 the same at twice the nominal amplitude.
M2, leak (does the note carry the sender's belief s_j, not only its call?): y_j = sigma_j + lam (s_j - sigma_j),
    x = a_+ (f_A + lam_A L_A) - a_- (f_D + lam_D L_D) + ...,   L = sum_group (s_j - sigma_j) / K,
    fitted as M1 + b_A L_A + b_D L_D; lam_A = b_A / a_+, lam_D = -b_D / a_-. In the template arm the notes carry
    no s_j, so b_A, b_D there are a placebo (expected 0).
M3, linearity in the counts: M1 + f_A^3 + f_D^3 (curvature/saturation) + sign f_A + sign f_D (majority rule).
    The law has all four at 0. Reported: the coefficients, and the response to a full 8-colleague bloc
    (f = 1/2) under M3 against a_+/2 under M1.
M4, adding vs averaging: M1 with f_A, f_D replaced by the pairs (f_A, g_A), (f_D, g_D), g = sum_group sigma / n_group
    (the group mean; 0 for an empty group). The law (W = a/K per colleague) has the g coefficients at 0; averaging
    within a group puts the weight on g instead.
Censoring: a probe whose BUY or SELL token fell outside the top-20 logprobs has x at a bound (lower in magnitude
    than the truth); M1 uses the bound, and a robustness fit drops censored probes and fits tanh on s = 2P - 1.
Uncertainty: bootstrap over base probes (B = 1000), resampled jointly for every wording and arm (paired).

Premise labels (EXPERIMENT.md "Success and failure criteria"), per wording, real arm, each test three-way with a
threshold tau on |est|:  holds iff |est| + 2 se < tau;  fails iff |est| > tau and |est| > 2 se;  else inconclusive.
    linear:  |sign coefficient| < 0.25 a/2 for each group; |bloc - a/2| < 0.2 a/2 for each group, on all probes AND on
             uncensored probes (if the two disagree the linearity label is inconclusive)
    adding:  |g| < 0.25 a 8/16 for each group
    call:    |lam| < 0.25 for each group; untestable if the pool's hesitant-confident gap in mean sigma s < 0.4
A premise holds for a wording iff all its tests hold, fails iff any fails, else inconclusive. A (wording, arm) with
>= 5% missing probes is labelled inconclusive throughout. The overall verdict rests on wordings with abar >= 1.

Go/no-go (analyst_scenario.md section 7), per wording, from M1 on the real arm:
    abar = sqrt((a_+^2 + a_-^2) / 2), r = a_- / a_+, Jbar = (a_+ - a_-) / 2,
    Lambda_mutual = max_omega of analyst_lib.dmft(Agent(a_+, a_-, nu_.5, a0, c, kappa), "mutual") (solver,
    T = 40, 3000 sites x 64 copies; bootstrap from 100 draws at 1000 x 32), also at nu_1,
    rho_1 = kappa <1 - s^2> with <s^2> from the same solver run (mutual world),
    rules: three-world contrast go iff abar >= 1 and r > 0.3; boundary sweep go iff Lambda_mutual >= 0.7;
    herding first iff Jbar > 1.3; memory flag iff rho_1 > 0.3.

Pilot (exploratory, excluded from confirmatory use): missing and censoring rates, note status, refusal or
disclaimer phrases in notes, seconds per round, m(t) and q(t), echo rho_1..4 (analyst_lib.echo_function), and the
in-swarm M1 regression per world (theory section 1, "the running swarm measures the same parameters").

Usage: uv run python experiments/08-analyst-calibration/analyze.py [selftest]
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from engine.ledger import read_jsonl  # noqa: E402
from theory.checks import analyst_lib as al  # noqa: E402

PILOT_TAG = "analyst-pilot-20261125"
PROBE_DIR = ROOT / "results" / "analyst-probes-20261125"
OUT = ROOT / "results" / "analyst-calib"
WORLDS = ("mutual", "oneway", "antagonistic")
WORDINGS = ("mild", "moderate", "strong", "max")
ARMS = ("real", "template")
K = 16
B = 1000
M1_NAMES = ["a_plus", "a_minus", "nu_05", "nu_1", "a0", "kappa", "c"]
REFUSAL = re.compile(r"financial advice|cannot (provide|help|assist)|can't (provide|help|assist)|unable to|as an ai|"
                     r"i'm sorry|i am sorry|not able to", re.I)


# ---------------------------------------------------------------- design matrices

def _cols(d: dict, model: str) -> list[float]:
    nu5 = d["xi_real"] * (d["nu_nom"] == 0.5)
    nu1 = d["xi_real"] * (d["nu_nom"] == 1.0)
    base = [d["fA"], -d["fD"], nu5, nu1, d["h0_real"], d["own"], 1.0]
    if model == "M1":
        return base
    if model == "M2":
        return base + [d["sA"] - d["fA"], d["sD"] - d["fD"]]
    if model == "M3":
        return base + [d["fA"] ** 3, d["fD"] ** 3, np.sign(d["fA"]), np.sign(d["fD"])]
    if model == "M4":
        nA, nD = d["n_allies"], K - d["n_allies"]
        gA = d["fA"] * K / nA if nA else 0.0
        gD = d["fD"] * K / nD if nD else 0.0
        return base + [gA, -gD]
    raise ValueError(model)


EXTRA = {"M1": [], "M2": ["b_LA", "b_LD"], "M3": ["fA3", "fD3", "signA", "signD"], "M4": ["g_A", "g_D"]}


def ols(X, y, w=None):
    """Least squares; integer weights w (bootstrap multiplicities) are the same as repeating rows."""
    if w is None:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.sqrt(w)
    return np.linalg.lstsq(X * r[:, None], y * r, rcond=None)[0]


def fit_arrays(X, y, model: str, w=None) -> dict:
    coef = ols(X, y, w)
    out = dict(zip(M1_NAMES + EXTRA[model], coef))      # column 2 is -f_D, so its coefficient is a_minus
    res = y - X @ coef
    ww = np.ones_like(y) if w is None else w
    out["resid_sd"] = float(np.sqrt((ww * res**2).sum() / max(ww.sum() - X.shape[1], 1)))
    return out


def fit(rows: list[dict], model: str = "M1") -> dict:
    X = np.array([_cols(r, model) for r in rows])
    return fit_arrays(X, np.array([r["x"] for r in rows]), model)


def fit_tanh(rows: list[dict]) -> dict:
    """M1 on the belief scale: least squares of s = 2P - 1 on tanh(X w) (insensitive to extreme x)."""
    X = np.array([_cols(r, "M1") for r in rows])
    s = np.array([2 * r["p_buy"] - 1 for r in rows])
    w = least_squares(lambda th: np.tanh(X @ th) - s, ols(X, np.array([r["x"] for r in rows]))).x
    return dict(zip(M1_NAMES, w))


# ---------------------------------------------------------------- probes

def load_probes() -> tuple[dict, dict]:
    design = {d["b"]: d for d in json.loads((PROBE_DIR / "design.json").read_text())}
    recs = {}
    for w in WORDINGS:
        for arm in ARMS + ("dup",):
            p = PROBE_DIR / f"{w}_{arm}.jsonl"
            if p.exists():
                recs[(w, arm)] = {r["b"]: r for r in read_jsonl(p)}
    return design, recs


def rows_for(design, recs, w, arm, bs, drop_censored=False):
    out = []
    for b in bs:
        r = recs[(w, arm)].get(b)
        if r is None or r["status"] != "ok" or (drop_censored and r["censored"]):
            continue
        out.append({**design[b], "x": r["x"], "p_buy": r["p_buy"]})
    return out


def lam_from(f: dict) -> dict:
    return {"lam_A": f["b_LA"] / f["a_plus"] if f["a_plus"] else np.nan,
            "lam_D": -f["b_LD"] / f["a_minus"] if f["a_minus"] else np.nan}


def bloc(f3: dict) -> dict:
    """Response of x to a full ally bloc (8 colleagues, all BUY: f_A = 1/2) under M3, and the same for faded."""
    return {"bloc_A": f3["a_plus"] * 0.5 + f3["fA3"] * 0.125 + f3["signA"],
            "bloc_D": f3["a_minus"] * 0.5 - f3["fD3"] * 0.125 - f3["signD"]}


def build(design, recs) -> dict:
    """Design matrices of every model for every (wording, arm), ok probes only."""
    data = {}
    for (w, arm), rr in recs.items():
        if arm not in ARMS:
            continue
        rows = rows_for(design, recs, w, arm, sorted(design))
        data[(w, arm)] = {"b": np.array([r["b"] for r in rows]), "y": np.array([r["x"] for r in rows]),
                          "X": {m: np.array([_cols(r, m) for r in rows]) for m in EXTRA}}
    return data


def probe_stats(data: dict, counts=None) -> dict:
    """Every number of the calibration table; counts[b] = multiplicity of base probe b in a bootstrap draw (None:
    the estimate)."""
    out = {}
    for key, D in data.items():
        wt = None if counts is None else counts[D["b"]].astype(float)
        f1, f2, f3, f4 = (fit_arrays(D["X"][m], D["y"], m, wt) for m in ("M1", "M2", "M3", "M4"))
        out[key] = {**f1, **{f"M2_{k}": v for k, v in {**f2, **lam_from(f2)}.items()},
                    **{f"M3_{k}": v for k, v in {**f3, **bloc(f3)}.items()},
                    **{f"M4_{k}": v for k, v in f4.items()}, "n": len(D["y"])}
    return out


def boot(data: dict, bs, nb=B, seed=0) -> list[dict]:
    """Bootstrap over base probes, the same draw for every wording and arm (paired)."""
    rng = np.random.default_rng(seed)
    bs = np.array(bs)
    return [probe_stats(data, np.bincount(rng.choice(bs, len(bs)), minlength=bs.max() + 1)) for _ in range(nb)]


def se_of(draws, key_path):
    vals = []
    for d in draws:
        v = d
        for k in key_path:
            v = v[k]
        vals.append(v)
    return float(np.nanstd(vals, ddof=1))


# ---------------------------------------------------------------- go/no-go

def lam_mutual(a_plus, a_minus, nu, a0, c, kappa, M=3000, copies=64, seed=0) -> dict:
    d = al.dmft(al.Agent(a_plus, a_minus, nu=abs(nu), a0=abs(a0), c=c, kappa=kappa), "mutual", T=40, M=M,
                copies=copies, seed=seed)
    q = float(d["q"][10:].mean())
    return {"Lam": float(d["Lam"].max()), "Lam_omega": d["Lam"].tolist(), "q": q, "d": 1 - q,
            "rho1": kappa * (1 - q)}


def gonogo(est: dict, draws: list[dict], wordings) -> dict:
    out = {}
    for w in wordings:
        if (w, "real") not in est:
            continue
        e = est[(w, "real")]
        g = {}
        for k, f in [("abar", lambda e: np.sqrt((e["a_plus"] ** 2 + e["a_minus"] ** 2) / 2)),
                     ("r", lambda e: e["a_minus"] / e["a_plus"]), ("Jbar", lambda e: (e["a_plus"] - e["a_minus"]) / 2)]:
            g[k] = float(f(e))
            g[k + "_se"] = float(np.std([f(d[(w, "real")]) for d in draws], ddof=1))
        L = lam_mutual(e["a_plus"], e["a_minus"], e["nu_05"], e["a0"], e["c"], e["kappa"])
        L1 = lam_mutual(e["a_plus"], e["a_minus"], e["nu_1"], e["a0"], e["c"], e["kappa"])
        bl = [lam_mutual(d[(w, "real")]["a_plus"], d[(w, "real")]["a_minus"], d[(w, "real")]["nu_05"],
                         d[(w, "real")]["a0"], d[(w, "real")]["c"], d[(w, "real")]["kappa"], M=1000, copies=32, seed=i)
              for i, d in enumerate(draws[:100])]
        g.update(Lam_mutual=L["Lam"], Lam_mutual_se=float(np.std([b["Lam"] for b in bl], ddof=1)),
                 Lam_omega=L["Lam_omega"], Lam_mutual_nu1=L1["Lam"], q_mutual=L["q"], rho1=L["rho1"],
                 rho1_se=float(np.std([b["rho1"] for b in bl], ddof=1)),
                 Lam_tree_nonews=float((K - 1) / 2 * (np.tanh(e["a_plus"] / K) ** 2 + np.tanh(e["a_minus"] / K) ** 2)))
        g["contrast_go"] = bool(g["abar"] >= 1 and g["r"] > 0.3)
        g["mutual_vs_oneway_go"] = bool(g["abar"] >= 1 and 0 < g["r"])
        g["boundary_go"] = bool(g["Lam_mutual"] >= 0.7)
        g["herding_first"] = bool(g["Jbar"] > 1.3)
        g["memory_flag"] = bool(g["rho1"] > 0.3)
        g["far_above"] = bool(g["abar"] > 7)
        out[w] = g
    return out


# ---------------------------------------------------------------- premise labels

def three_way(est: float, se: float, tau: float) -> str:
    if abs(est) + 2 * se < tau:
        return "holds"
    if abs(est) > tau and abs(est) > 2 * se:
        return "fails"
    return "inconclusive"


def combine(labels) -> str:
    labels = list(labels)
    if "fails" in labels:
        return "fails"
    return "holds" if all(lb == "holds" for lb in labels) else "inconclusive"


def premises(table: dict, gng: dict, pool_stats: dict | None) -> dict:
    out = {}
    gap = None if pool_stats is None else min(pool_stats["BUY|gap"], pool_stats["SELL|gap"])
    for w in WORDINGS:
        t = table.get((w, "real"))
        if t is None:
            continue
        ap, am = abs(t["a_plus"]["est"]), abs(t["a_minus"]["est"])
        tests = {}

        def lin(src):
            return [three_way(src["signA"]["est"], src["signA"]["se"], 0.25 * ap / 2),
                    three_way(src["signD"]["est"], src["signD"]["se"], 0.25 * am / 2),
                    three_way(src["bloc_A"]["est"] - src["a_plus"]["est"] / 2, src["bloc_A"]["se"], 0.2 * ap / 2),
                    three_way(src["bloc_D"]["est"] - src["a_minus"]["est"] / 2, src["bloc_D"]["se"], 0.2 * am / 2)]

        m3 = {k: t[f"M3_{k}"] for k in ("a_plus", "a_minus", "signA", "signD", "bloc_A", "bloc_D")}
        l_all = combine(lin(m3))
        l_unc = combine(lin(t["M3_uncensored"])) if t.get("M3_uncensored") else l_all
        tests["linear"] = {"label": l_all if l_all == l_unc else "inconclusive", "all_probes": l_all, "uncensored": l_unc}
        tests["adding"] = {"label": combine([three_way(t["M4_g_A"]["est"], t["M4_g_A"]["se"], 0.25 * ap / 2),
                                             three_way(t["M4_g_D"]["est"], t["M4_g_D"]["se"], 0.25 * am / 2)])}
        if gap is not None and gap < 0.4:
            tests["call"] = {"label": "untestable", "gap": gap}
        else:
            tests["call"] = {"label": combine([three_way(t["M2_lam_A"]["est"], t["M2_lam_A"]["se"], 0.25),
                                               three_way(t["M2_lam_D"]["est"], t["M2_lam_D"]["se"], 0.25)]), "gap": gap}
        if t["n_missing"] >= 0.05 * (t["n"] + t["n_missing"]):
            tests = {k: {**v, "label": "inconclusive", "why": "missing >= 5%"} for k, v in tests.items()}
        tests["in_verdict"] = bool(gng.get(w, {}).get("abar", 0) >= 1)
        out[w] = tests
    return out


# ---------------------------------------------------------------- pilot (exploratory)

def pilot_stats() -> dict:
    out = {}
    for w in WORLDS:
        d = ROOT / "results" / f"{PILOT_TAG}-{w}"
        if not (d / "analyst_calls.jsonl").exists():
            continue
        recs = list(read_jsonl(d / "analyst_calls.jsonl"))
        tim = list(read_jsonl(d / "analyst_timings.jsonl"))
        bel = np.load(d / "beliefs.npz")
        s, sig = bel["s"], bel["sig"].astype(float)
        ok = [r for r in recs if r["status"] == "ok"]
        notes = [r["note"] for r in recs if r["note_status"] == "ok"]
        h0r = np.load(d / "inputs.npz")["h0_real"]
        reg_rows = [{"fA": r["fA"], "fD": r["fD"], "xi_real": r["xi_real"], "nu_nom": 0.5, "own": r["own_call"],
                     "h0_real": float(h0r[r["agent_id"]]), "x": r["x"]} for r in ok if r["round_t"] >= 1]
        f1 = fit(reg_rows, "M1") if reg_rows else {}
        sf = np.nan_to_num(s, nan=0.0)
        out[w] = {
            "n_calls": len(recs), "n_missing": len(recs) - len(ok),
            "n_censored": sum(bool(r["censored"]) for r in ok),
            "valid_mass_min": min(r["valid_mass"] for r in recs), "valid_mass_median": float(np.median([r["valid_mass"] for r in recs])),
            "note_status": {k: sum(r["note_status"] == k for r in recs) for k in ("ok", "truncated", "carried")},
            "n_refusal_phrase": sum(bool(REFUSAL.search(n)) for n in notes),
            "refusal_examples": [n for n in notes if REFUSAL.search(n)][:5],
            "sec_per_round_inbox": float(np.median([t["total_s"] for t in tim if t["round_t"] >= 1])),
            "sec_per_round_0": float(np.median([t["total_s"] for t in tim if t["round_t"] == 0])),
            "belief_s_median": float(np.median([t["belief_s"] for t in tim if t["round_t"] >= 1])),
            "note_s_median": float(np.median([t["note_s"] for t in tim if t["round_t"] >= 1])),
            "m_t": np.nanmean(s, (1, 2)).tolist(), "q_t": np.nanmean(s ** 2, (1, 2)).tolist(),
            "call_mean_t": sig.mean((1, 2)).tolist(),
            "frac_extreme_P": float(np.mean([min(r["p_buy"], 1 - r["p_buy"]) < 0.01 for r in ok])),
            "rho_1_4": al.echo_function(sf, sig, burn=2, L=4).tolist(),
            "in_swarm_M1": {k: float(v) for k, v in f1.items()},
            "note_words_median": float(np.median([len(n.split()) for n in notes])) if notes else None,
        }
    return out


# ---------------------------------------------------------------- self-test on synthetic probes

def selftest():
    """Recover known parameters from probes generated by the law with a leak, on the run.py design distribution."""
    rng = np.random.default_rng(1)
    truth = dict(a_plus=3.0, a_minus=2.0, nu_05=0.6, nu_1=1.1, a0=0.4, kappa=0.5, c=0.1)
    lam = 0.3
    design, recs = {}, {(w, a): {} for w in WORDINGS for a in ARMS}
    for b in range(3000):
        faded = rng.random(K) < 0.5
        m = rng.uniform(-1, 1, 2)
        calls = np.where(rng.random(K) < (1 + np.where(faded, m[1], m[0])) / 2, 1, -1)
        s_send = np.clip(calls * rng.uniform(-0.2, 1.0, K), -1, 1)
        d = dict(b=b, n_allies=int((~faded).sum()), fA=calls[~faded].sum() / K, fD=calls[faded].sum() / K,
                 sA=s_send[~faded].sum() / K, sD=s_send[faded].sum() / K, own=int(rng.choice([-1, 1])),
                 h0_real=rng.standard_normal(), xi_real=rng.standard_normal(), nu_nom=float(rng.choice([0.5, 1.0])))
        design[b] = d
        for w in WORDINGS:
            for arm in ARMS:
                lk = lam if arm == "real" else 0.0
                yA = d["fA"] + lk * (d["sA"] - d["fA"])
                yD = d["fD"] + lk * (d["sD"] - d["fD"])
                nu = truth["nu_05"] if d["nu_nom"] == 0.5 else truth["nu_1"]
                x = (truth["a_plus"] * yA - truth["a_minus"] * yD + nu * d["xi_real"] + truth["a0"] * d["h0_real"]
                     + truth["kappa"] * d["own"] + truth["c"] + 0.3 * rng.standard_normal())
                recs[(w, arm)][b] = {"b": b, "status": "ok", "x": x, "p_buy": (1 + np.tanh(x)) / 2, "censored": None}
    bs = sorted(design)
    est = probe_stats(build(design, recs))
    e = est[("mild", "template")]
    for k, v in truth.items():
        assert abs(e[k] - v) < 0.05, (k, e[k], v)
    er = est[("mild", "real")]
    assert abs(er["M2_lam_A"] - lam) < 0.08 and abs(er["M2_lam_D"] - lam) < 0.08, er          # s.e. about 0.04
    assert abs(e["M2_lam_A"]) < 0.08                                    # placebo in the template arm
    print({k: round(float(e[f"M3_{k}"]), 3) for k in ("fA3", "fD3", "signA", "signD", "bloc_A", "bloc_D")})
    assert abs(e["M3_signA"]) < 0.05 and abs(e["M3_signD"]) < 0.05       # cubic terms are imprecise (s.e. ~0.2)
    assert abs(e["M3_bloc_A"] - 1.5) < 0.05 and abs(e["M3_bloc_D"] - 1.0) < 0.05
    assert abs(e["M4_g_A"]) < 0.05 and abs(e["M4_a_plus"] - 3.0) < 0.1
    # an averaging agent: x = a * (group mean) puts the weight on g
    for b in bs:
        d = design[b]
        nA, nD = d["n_allies"], K - d["n_allies"]
        x = 1.5 * (d["fA"] * K / nA if nA else 0) - 1.0 * (d["fD"] * K / nD if nD else 0) + 0.3 * rng.standard_normal()
        recs[("mild", "template")][b]["x"] = x
    e = probe_stats(build(design, recs))[("mild", "template")]
    assert abs(e["M4_g_A"] - 1.5) < 0.1 and abs(e["M4_a_plus"]) < 0.3, e
    assert three_way(0.05, 0.02, 0.2) == "holds" and three_way(0.5, 0.1, 0.2) == "fails"
    assert three_way(0.0, 0.5, 0.2) == "inconclusive" and three_way(0.3, 0.2, 0.2) == "inconclusive"
    L = lam_mutual(4, 4, 0.5, 0.3, 0, 0)
    assert abs(L["Lam"] - 0.60) < 0.05, L                               # analyst_scenario.md table 3.3, (4,4)
    print("selftest ok: M1-M4 recover the generating parameters; Lambda(4,4) =", round(L["Lam"], 3))


# ---------------------------------------------------------------- main

def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"pilot": pilot_stats()}
    for w, p in summary["pilot"].items():
        print(f"pilot {w}: missing {p['n_missing']}/{p['n_calls']}, censored {p['n_censored']}, refusal phrases "
              f"{p['n_refusal_phrase']}, notes {p['note_status']}, {p['sec_per_round_inbox']:.0f} s/round "
              f"(belief {p['belief_s_median']:.0f}, notes {p['note_s_median']:.0f}), rho_1..4 "
              f"{np.round(p['rho_1_4'], 3).tolist()}, m(T) {p['m_t'][-1]:+.2f} q(T) {p['q_t'][-1]:.2f}")
        print("   in-swarm M1:", {k: round(v, 2) for k, v in p["in_swarm_M1"].items()})
    if not (PROBE_DIR / "design.json").exists():
        (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
        return
    design, recs = load_probes()
    bs = sorted(design)
    data = build(design, recs)
    est = probe_stats(data)
    draws = boot(data, bs)
    table = {}
    for key, e in est.items():
        table[key] = {k: {"est": float(v), "se": se_of(draws, [key, k])} for k, v in e.items() if k != "n"}
        table[key]["n"] = e["n"]
        all_r = list(recs[key].values())
        table[key]["n_missing"] = sum(r["status"] != "ok" for r in all_r)
        table[key]["n_censored"] = sum(bool(r["censored"]) for r in all_r if r["status"] == "ok")
        table[key]["valid_mass_min"] = min(r["valid_mass"] for r in all_r)
        nc = rows_for(design, recs, *key, bs, drop_censored=True)
        table[key]["M1_uncensored"] = {k: float(v) for k, v in fit(nc, "M1").items()} if len(nc) > 20 else None
        table[key]["M3_uncensored"] = None
        if len(nc) > 20 and table[key]["n_censored"]:
            Xn = np.array([_cols(r, "M3") for r in nc])
            yn = np.array([r["x"] for r in nc])
            dn = {b: i for i, b in enumerate(r["b"] for r in nc)}
            est3 = {**fit_arrays(Xn, yn, "M3")}
            est3.update(bloc(est3))
            rng = np.random.default_rng(1)
            bb = np.array([r["b"] for r in nc])
            dr = []
            for _ in range(B):
                cnt = np.bincount(rng.choice(np.array(bs), len(bs)), minlength=max(bs) + 1)[bb].astype(float)
                f3 = fit_arrays(Xn, yn, "M3", cnt)
                dr.append({**f3, **bloc(f3)})
            table[key]["M3_uncensored"] = {k: {"est": float(est3[k]), "se": float(np.std([d[k] for d in dr], ddof=1))}
                                           for k in ("a_plus", "a_minus", "signA", "signD", "bloc_A", "bloc_D")}
        table[key]["M1_tanh"] = {k: float(v) for k, v in fit_tanh(rows_for(design, recs, *key, bs)).items()}
    # paired differences real - template, and between wordings
    diffs = {}
    for w in WORDINGS:
        if (w, "real") in est and (w, "template") in est:
            for k in ("a_plus", "a_minus", "nu_05", "a0", "kappa", "M2_b_LA", "M2_b_LD"):
                diffs[(w, k)] = {"est": est[(w, "real")][k] - est[(w, "template")][k],
                                 "se": float(np.std([d[(w, "real")][k] - d[(w, "template")][k] for d in draws], ddof=1))}
    if ("mild", "dup") in recs:
        dd = [abs(recs[("mild", "dup")][b]["x"] - recs[("mild", "real")][b]["x"]) for b in recs[("mild", "dup")]
              if recs[("mild", "dup")][b]["status"] == "ok" and recs[("mild", "real")][b]["status"] == "ok"]
        dp = [abs(recs[("mild", "dup")][b]["p_buy"] - recs[("mild", "real")][b]["p_buy"]) for b in recs[("mild", "dup")]
              if recs[("mild", "dup")][b]["status"] == "ok" and recs[("mild", "real")][b]["status"] == "ok"]
        summary["repeat"] = {"n": len(dd), "x_absdiff_median": float(np.median(dd)), "x_absdiff_max": float(max(dd)),
                             "p_absdiff_max": float(max(dp))}
    summary["probes"] = table
    summary["diffs"] = diffs
    summary["gonogo"] = gonogo(est, draws, WORDINGS)
    ps_path = PROBE_DIR / "pool_stats.json"
    summary["pool_stats"] = json.loads(ps_path.read_text()) if ps_path.exists() else None
    summary["premises"] = premises(table, summary["gonogo"], summary["pool_stats"])
    for w, p in summary["premises"].items():
        print("premises", w, {k: v["label"] for k, v in p.items() if isinstance(v, dict)}, "| in verdict:", p["in_verdict"])
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, default=float))
    for key, t in table.items():
        print(key, "n", t["n"], "missing", t["n_missing"], "censored", t["n_censored"], " ".join(
            f"{k} {t[k]['est']:+.2f}({t[k]['se']:.2f})" for k in M1_NAMES + ["resid_sd", "M2_lam_A", "M2_lam_D",
                                                                               "M3_fA3", "M3_fD3", "M3_signA",
                                                                               "M3_signD", "M4_a_plus", "M4_g_A",
                                                                               "M4_a_minus", "M4_g_D"]))
    for w, g in summary["gonogo"].items():
        print(w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k != "Lam_omega"})
    if "repeat" in summary:
        print("repeat", summary["repeat"])


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
