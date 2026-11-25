"""Analysis of 09-analyst-signrule (coordinator, CPU). Fixed before any swarm data exist; the decision rules are the
falsifiers of theory/analyst_signrule.md section 8, implemented here as written in EXPERIMENT.md.

Per run (theory/checks/analyst_signrule_check.run_stats, rounds 5..20, all on the SAMPLING belief s = tanh(x - chat)):
  rho_1..4 (echo function), X2 = rho2 - rho1^2, c1, c2 of replica differences, Xc2 = c2 - c1^2, q = <s^2>, m = <s>.
  A missing decision (valid mass < 0.95) has no x; it is filled with the analyst's previous logit for the estimators
  (counted and reported; the swarm itself carried the previous call).
Per world: mean over runs; s.e. = pooled within-world run-to-run s.d. / sqrt(n_runs) (pooled over the three worlds,
  9 degrees of freedom); a comparison's s.e. = sqrt(se1^2 + se2^2).
Kick twins (envs/analyst.kick_slice; results/<TAG>/kicks_<world>_g<g>.jsonl, one line per slice): per unit kick
  (Delta * D / 4) on the sampling belief; r_A, r_D, w_A, w_D from the readers at t0+1 by tie type (J[reader, kicked]
  = +1 ally, -1 faded); E (rewritten and notes-kept replays) and E_x of the kicked analyst at t0+2. s.e. by cluster
  bootstrap (B = 2,000) over slices and over (run, replica); the larger of the two is used in every decision. Null
  twins: mean and s.d. of the kicked analyst's logit change (note noise).
Gaussian-cavity flags (guardrail 5): per world, excess kurtosis and Sarle's bimodality coefficient
  (skew^2 + 1) / (kurt + 3) of the de-biased logit x - chat (rounds 5..20) conditional on the analyst's previous call
  (analyst_scenario.md 2.4); flags: one-way conditional excess kurtosis < -0.3, or bimodality coefficient > 0.555.
In-swarm regression (analyst_signrule_check.inswarm_fit, pooled over a world's runs and replicas): the de-biased
  logit x(t+1) - chat on f_A, -f_D, own call, 1, with and without the leak terms L = sum_group (s^r_j - sigma_j)/K;
  s.e. by cluster bootstrap over analysts (B = 500).
Reaction coefficient from the swarm (falsifier 3; (S4)): u_A = K w_A, u_D = K w_D from the world's kick twins, v from
  that world's leak regression (b_LA, b_LD), d = <1 - s^2>, d^r = <1 - s_r^2>; tau_g = u_g d + v_g d^r;
  G_M = (u_A tau_A + u_D tau_D)/2K, G_A = (u_A tau_D + u_D tau_A)/2K; predicted X2 = G d.

Power of the kick parts: a world's twins are "powered" iff the frozen M1 kick echo E_pred (mutual +0.0553,
antagonistic +0.0092) is at least 3 times the observed s.e. of E. An unpowered kick part is reported but does not
decide a falsifier (an echo the twins could not have resolved is not evidence against it); the falsifier is then
decided on the echo function alone.

Falsifiers (EXPERIMENT.md "Fixed rules"):
  F1 mutual echo: X2(mut) - X2(one-way) > 3 se AND (E_mut > 3 se or twins unpowered) -> consistent; either negative by
     > 3 se -> falsified; otherwise (not resolved while G_M d > 0.03) -> falsified ("not positive").
  F2 one-way echo (the one-step memory premise): |X2(one-way)| > 3 se -> falsified. X2 = 0 in the one-way world holds
     only if the own memory is one step in the call; a path through the analyst's own notes would show here, and the
     kick echo (own history held at the base run's) excludes it. So F5, F6 and the G d check use X2(world) - X2(one-way).
  F3 antagonistic sign vs sign(G_A) from the swarm: |G_A d| < 0.004 -> no sign predicted (second null); else the echo
     (X2(ant) - X2(one-way), and E_ant rewritten) opposite to sign(G_A) by > 3 se -> falsified. Notes-kept: E_kept
     opposite in sign to r_A r_D by > 3 se -> falsified.
  F4 ordering: X2(mut) - X2(ant) < 3 se, or (both worlds' twins powered and E_mut - E_ant < 3 se) -> falsified.
  F5 magnitudes, on dX2 = X2(world) - X2(one-way): mutual dX2 outside [0.03, 0.07] or antagonistic dX2 outside
     [0.005, 0.021] by > 3 se -> calibration did not transfer; then dX2 against G d per world: |dX2 - G d| > 0.25 |G d|
     + 3 se -> single-site reduction fails.
  F6 kick vs echo function: |E - dX2| > 3 se + 0.15 |dX2| in a world whose twins are powered -> falsified; unpowered
     twins: "unpowered".
  F7 asynchronous control: not run.
Recorded: m (M1 0.00 +- 0.04, M2 +0.35 +- 0.15), composition ratios E / (K r_A r_D) (antagonistic) and
E / ((K/2)(r_A^2 + r_D^2)) (mutual), missing and refusal rates, residuals of every observable against the frozen
M1 and M2 predictions.

Verdict (EXPERIMENT.md "Fixed rules", the mapping table): verdict() below.

Usage: uv run python experiments/09-analyst-signrule/analyze.py
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.ledger import read_jsonl  # noqa: E402
from theory.checks import analyst_signrule_check as SR  # noqa: E402

PRED = json.loads((ROOT / "theory" / "checks" / "analyst_signrule_predictions.json").read_text())
CHAT = float(PRED["design"]["chat"])
TAG = "analyst-signrule-20261125"
RES = ROOT / "results"
OUT = RES / "analyst-signrule"
W3 = ("mutual", "oneway", "antagonistic")
K, N, T, R, BURN = 16, 256, 20, 2, 4
STATS = ("rho1", "rho2", "rho3", "rho4", "X2", "c1", "c2", "Xc2", "q", "m")
REFUSAL = re.compile(r"financial advice|cannot (provide|help|assist)|can't (provide|help|assist)|unable to|as an ai|"
                     r"i'm sorry|i am sorry|not able to", re.I)
INTERVAL = {"mutual": (0.03, 0.07), "antagonistic": (0.005, 0.021)}


class _Chat:
    chat = CHAT


def load_run(world: str, g: int):
    d = RES / f"{TAG}-{world}-g{g}"
    if not (d / "analyst_calls.jsonl").exists():
        return None
    J = np.load(d / "inputs.npz")["J"].astype(int)
    X = np.full((T + 1, R, N), np.nan)
    sig = np.zeros((T + 1, R, N))
    notes, n_miss, n_cens = [], 0, 0
    for r in read_jsonl(d / "analyst_calls.jsonl"):
        t, rep, i = r["round_t"], r["replica"], r["agent_id"]
        sig[t, rep, i] = r["call"]
        if r["status"] == "ok":
            X[t, rep, i] = r["x"]
            n_cens += bool(r.get("censored"))
        else:
            n_miss += 1
        if r["note_status"] == "ok":
            notes.append(r["note"])
    if np.isnan(X[0]).any():
        X[0] = np.where(np.isnan(X[0]), CHAT, X[0])
    for t in range(1, T + 1):                                     # fill missing logits with the previous one
        X[t] = np.where(np.isnan(X[t]), X[t - 1], X[t])
    o = dict(X=X, s=np.tanh(X - CHAT), sr=np.tanh(X), sig=sig)
    st = SR.run_stats(o)
    tim = list(read_jsonl(d / "analyst_timings.jsonl"))
    return dict(o=o, J=J, stats=st, n_missing=n_miss, n_censored=n_cens, n_calls=(T + 1) * R * N,
                n_refusal=sum(bool(REFUSAL.search(n)) for n in notes), n_notes=len(notes),
                sec_per_round=float(np.median([t_["total_s"] for t_ in tim if t_["round_t"] >= 1])) if tim else None,
                m_t=o["s"].mean((1, 2)).tolist(), q_t=(o["s"] ** 2).mean((1, 2)).tolist())


def world_table(runs: dict) -> dict:
    """Per-world means with pooled s.e. (module docstring)."""
    out = {}
    pooled = {}
    for k in STATS:
        ss, df = 0.0, 0
        for w in W3:
            v = np.array([r["stats"][k] for r in runs[w]])
            if len(v) > 1:
                ss += ((v - v.mean()) ** 2).sum()
                df += len(v) - 1
        pooled[k] = float(np.sqrt(ss / df)) if df else float("nan")
    for w in W3:
        out[w] = {"n_runs": len(runs[w])}
        for k in STATS:
            v = np.array([r["stats"][k] for r in runs[w]])
            out[w][k] = {"mean": float(v.mean()) if len(v) else float("nan"),
                         "se": pooled[k] / np.sqrt(len(v)) if len(v) else float("nan"),
                         "runs": v.tolist(), "sd_world": float(v.std(ddof=1)) if len(v) > 1 else None}
    out["pooled_run_sd"] = pooled
    return out


def inswarm(runs_w, leak: bool, nb=500, seed=0):
    """Pooled in-swarm regression of one world; cluster bootstrap over analysts (all runs)."""
    rows = []
    for r in runs_w:
        o, J = r["o"], r["J"]
        A_, D_ = (J > 0).astype(float) / K, (J < 0).astype(float) / K
        sig, sr, X = o["sig"][BURN:-1], o["sr"][BURN:-1], o["X"][BURN + 1:] - CHAT
        cols = [sig @ A_.T, -(sig @ D_.T), sig, np.ones_like(sig)]
        if leak:
            cols += [(sr - sig) @ A_.T, (sr - sig) @ D_.T]
        M = np.stack(cols, -1)                                       # (T', R, N, p)
        rows.append((M.transpose(2, 0, 1, 3).reshape(N, -1, M.shape[-1]), X.transpose(2, 0, 1).reshape(N, -1)))
    Xs = np.concatenate([m for m, _ in rows])                        # (n_analysts, obs, p)
    ys = np.concatenate([y for _, y in rows])
    names = ["a_plus", "a_minus", "kappa", "c"] + (["b_LA", "b_LD"] if leak else [])

    def fit(idx):
        return np.linalg.lstsq(Xs[idx].reshape(-1, Xs.shape[-1]), ys[idx].ravel(), rcond=None)[0]

    est = fit(np.arange(len(ys)))
    rng = np.random.default_rng(seed)
    bs = np.array([fit(rng.integers(0, len(ys), len(ys))) for _ in range(nb)])
    return {n: {"est": float(e), "se": float(s)} for n, e, s in zip(names, est, bs.std(0, ddof=1))}


def kick_table(world: str, nb=2000, seed=0):
    lines = [ln for f in sorted((RES / TAG).glob(f"kicks_{world}_g*.jsonl")) for ln in read_jsonl(f)]
    if not lines:
        return None
    recs = [dict(r, g=ln["g"], slice=ln["slice"]) for ln in lines for r in ln["twins"]]
    kicks = [r for r in recs if r["kind"] == "kick"]
    nulls = [r for r in recs if r["kind"] == "null"]

    def stats(ks):
        rd = [(rr, r["D"]) for r in ks for rr in r["readers"] if rr["ok"]]
        f = lambda tie, key: float(np.mean([rr[key] * D / 4 for rr, D in rd if rr["tie"] == tie])) if any(
            rr["tie"] == tie for rr, _ in rd) else float("nan")
        e = lambda key: float(np.nanmean([r[key] for r in ks])) if ks else float("nan")
        return dict(rA=f(1, "ds"), rD=f(-1, "ds"), wA=f(1, "dx"), wD=f(-1, "dx"), E=e("E_rewritten"),
                    E_kept=e("E_kept"), E_x=e("Ex_rewritten"), E_x_kept=e("Ex_kept"))

    est = stats(kicks)
    rng = np.random.default_rng(seed)
    se = {}
    for name, keyf in (("slice", lambda r: (r["g"], r["slice"])), ("run_rep", lambda r: (r["g"], r["rep"]))):
        cl = sorted({keyf(r) for r in kicks})
        by = {c: [r for r in kicks if keyf(r) == c] for c in cl}
        draws = [stats([r for k in rng.integers(0, len(cl), len(cl)) for r in by[cl[k]]]) for _ in range(nb)]
        se[name] = {k: float(np.nanstd([d[k] for d in draws], ddof=1)) for k in est}
        se[name]["n_clusters"] = len(cl)
    out = {k: {"est": v, "se": max(se["slice"][k], se["run_rep"][k]), "se_slice": se["slice"][k],
               "se_run_rep": se["run_rep"][k]} for k, v in est.items()}
    out.update(n_kicks=len(kicks), n_nulls=len(nulls), n_slices=len(lines),
               n_clusters={k: v["n_clusters"] for k, v in se.items()},
               n_missing=sum(not (r["ok_rewritten"] and r["ok_kept"]) for r in kicks),
               n_reader_missing=sum(not rr["ok"] for r in kicks for rr in r["readers"]),
               frac_identical_prompt=float(np.mean([r["identical_rewritten"] for r in kicks])) if kicks else None,
               max_abs_E=float(np.nanmax(np.abs([r["E_rewritten"] for r in kicks]))) if kicks else None,
               mean_call_flips=float(np.mean([r["n_call_flips"] for r in kicks])) if kicks else None,
               E_sd_per_twin=float(np.nanstd([r["E_rewritten"] for r in kicks], ddof=1)) if len(kicks) > 1 else None)
    if nulls:
        dx = np.array([r["dx"] for r in nulls if r["ok"]])
        out["null"] = {"n": int(len(dx)), "mean_dx": float(dx.mean()), "sd_dx": float(dx.std(ddof=1)),
                       "se_mean": float(dx.std(ddof=1) / np.sqrt(len(dx)))}
    return out


def gauss_flags(runs_w):
    """Excess kurtosis and bimodality coefficient of x - chat (rounds 5..20) conditional on the previous call."""
    out = {}
    for prev in (+1, -1):
        v = np.concatenate([(r["o"]["X"][BURN + 1:] - CHAT)[r["o"]["sig"][BURN:-1] == prev] for r in runs_w])
        z = (v - v.mean()) / v.std()
        sk, ku = float((z**3).mean()), float((z**4).mean() - 3)
        out[f"prev{prev:+d}"] = {"n": int(len(v)), "skew": sk, "excess_kurtosis": ku,
                                 "bimodality_coef": (sk**2 + 1) / (ku + 3)}
    return out


def reaction(world_runs, kick_w, leak_fit):
    """G of (S4) from the swarm's own couplings (module docstring)."""
    s = np.concatenate([r["o"]["s"][BURN:].ravel() for r in world_runs])
    sr = np.concatenate([r["o"]["sr"][BURN:].ravel() for r in world_runs])
    d, dr = float((1 - s**2).mean()), float((1 - sr**2).mean())
    uA, uD = K * kick_w["wA"]["est"], K * kick_w["wD"]["est"]
    vA, vD = leak_fit["b_LA"]["est"], leak_fit["b_LD"]["est"]
    tA, tD = uA * d + vA * dr, uD * d + vD * dr
    return dict(uA=uA, uD=uD, vA=vA, vD=vD, d=d, dr=dr, G_M=(uA * tA + uD * tD) / (2 * K),
                G_A=(uA * tD + uD * tA) / (2 * K), Gd_M=(uA * tA + uD * tD) / (2 * K) * d,
                Gd_A=(uA * tD + uD * tA) / (2 * K) * d)


def gd_se(world_runs, kick_w, leak_fit, nb=1000, seed=1):
    """s.e. of G d from the kick and regression s.e. (independent normal draws; d, d^r taken as known)."""
    rng = np.random.default_rng(seed)
    out = {"Gd_M": [], "Gd_A": []}
    for _ in range(nb):
        kw = {k: {"est": kick_w[k]["est"] + kick_w[k]["se"] * rng.standard_normal()} for k in ("wA", "wD")}
        lf = {k: {"est": leak_fit[k]["est"] + leak_fit[k]["se"] * rng.standard_normal()} for k in ("b_LA", "b_LD")}
        g = reaction(world_runs, kw, lf)
        out["Gd_M"].append(g["Gd_M"])
        out["Gd_A"].append(g["Gd_A"])
    return {k: float(np.std(v, ddof=1)) for k, v in out.items()}


def comp(a, b):
    return {"diff": a["mean"] - b["mean"], "se": float(np.hypot(a["se"], b["se"]))}


E_PRED = {w: PRED["predictions"][w]["M1"]["kick"]["E"] for w in W3}


def powered(KT, w) -> bool:
    return abs(E_PRED[w]) >= 3 * KT[w]["E"]["se"]


def falsifiers(W, KT, G) -> dict:
    F = {"powered": {w: powered(KT, w) for w in ("mutual", "antagonistic")}}
    m_o = comp(W["mutual"]["X2"], W["oneway"]["X2"])
    a_o = comp(W["antagonistic"]["X2"], W["oneway"]["X2"])
    m_a = comp(W["mutual"]["X2"], W["antagonistic"]["X2"])
    Em, Ea = KT["mutual"], KT["antagonistic"]
    # F1
    pw = F["powered"]["mutual"]
    pos = m_o["diff"] > 3 * m_o["se"] and (Em["E"]["est"] > 3 * Em["E"]["se"] or not pw)
    neg = m_o["diff"] < -3 * m_o["se"] or Em["E"]["est"] < -3 * Em["E"]["se"]
    gm = G["mutual"]["Gd_M"]
    F["F1"] = dict(X2_mut_minus_one=m_o, E_mut=Em["E"], Gd_M=gm,
                   outcome="consistent" if pos else "falsified (negative)" if neg else
                   ("falsified (not positive while G_M d > 0.03)" if gm > 0.03 else "unresolved (G_M d <= 0.03)"))
    # F2
    o = W["oneway"]["X2"]
    F["F2"] = dict(X2_oneway=o, outcome="falsified" if abs(o["mean"]) > 3 * o["se"] else "consistent")
    # F3
    gA = G["antagonistic"]["Gd_A"]
    rArD = Ea["rA"]["est"] * Ea["rD"]["est"]
    if abs(gA) < 0.004:
        out3 = "no sign predicted (|G_A d| < 0.004): antagonistic world is a second null"
        if abs(a_o["diff"]) > 3 * a_o["se"] or abs(Ea["E"]["est"]) > 3 * Ea["E"]["se"]:
            out3 += "; but an echo is resolved (see values)"
    else:
        sgn = np.sign(gA)
        bad = sgn * a_o["diff"] < -3 * a_o["se"] or sgn * Ea["E"]["est"] < -3 * Ea["E"]["se"]
        good = sgn * a_o["diff"] > 3 * a_o["se"] or sgn * Ea["E"]["est"] > 3 * Ea["E"]["se"]
        out3 = "falsified" if bad else "consistent (sign resolved)" if good else "consistent (sign not resolved at 3 se)"
    kept_bad = np.sign(rArD) * Ea["E_kept"]["est"] < -3 * Ea["E_kept"]["se"]
    kept_good = np.sign(rArD) * Ea["E_kept"]["est"] > 3 * Ea["E_kept"]["se"]
    F["F3"] = dict(Gd_A=gA, X2_ant_minus_one=a_o, E_ant=Ea["E"], E_ant_kept=Ea["E_kept"], rA_rD=rArD, outcome=out3,
                   notes_kept_outcome="falsified" if kept_bad else "consistent (sign resolved)" if kept_good
                   else "consistent (sign not resolved at 3 se)")
    # F4
    e_ma = {"diff": Em["E"]["est"] - Ea["E"]["est"], "se": float(np.hypot(Em["E"]["se"], Ea["E"]["se"]))}
    both = F["powered"]["mutual"] and F["powered"]["antagonistic"]
    F["F4"] = dict(X2_mut_minus_ant=m_a, E_mut_minus_ant=e_ma, kick_part_counts=both,
                   outcome="falsified" if (m_a["diff"] < 3 * m_a["se"] or (both and e_ma["diff"] < 3 * e_ma["se"]))
                   else "consistent")
    # F5 (on dX2 = X2(world) - X2(one-way))
    f5 = {}
    for w, (lo, hi) in INTERVAL.items():
        dd = comp(W[w]["X2"], W["oneway"]["X2"])
        x = {"mean": dd["diff"], "se": dd["se"]}
        out_by = max(lo - x["mean"], x["mean"] - hi, 0.0)
        transfer = out_by <= 3 * x["se"]
        gd = G[w]["Gd_M" if w == "mutual" else "Gd_A"]
        gse = G[w]["Gd_M_se" if w == "mutual" else "Gd_A_se"]
        se_tot = float(np.hypot(x["se"], gse))
        red_fail = abs(x["mean"] - gd) > 0.25 * abs(gd) + 3 * se_tot
        f5[w] = dict(dX2=x, interval=[lo, hi], calibration_transfers=bool(transfer), Gd=gd, Gd_se=gse,
                     single_site_fails=bool(red_fail))
    F["F5"] = dict(per_world=f5, outcome="; ".join(
        f"{w}: " + ("in interval" if v["calibration_transfers"] else "calibration did not transfer") +
        (", single-site reduction fails" if v["single_site_fails"] else ", X2 consistent with G d") for w, v in f5.items()))
    # F6
    f6 = {}
    for w in ("mutual", "antagonistic"):
        dd = comp(W[w]["X2"], W["oneway"]["X2"])
        e = KT[w]["E"]
        se = float(np.hypot(dd["se"], e["se"]))
        bad = abs(e["est"] - dd["diff"]) > 3 * se + 0.15 * abs(dd["diff"])
        f6[w] = dict(E=e, dX2=dd, powered=F["powered"][w],
                     outcome="falsified" if (bad and F["powered"][w]) else "consistent" if F["powered"][w] else "unpowered")
    outs = [v["outcome"] for v in f6.values()]
    F["F6"] = dict(per_world=f6, outcome="falsified" if "falsified" in outs else
                   "consistent" if all(o == "consistent" for o in outs) else
                   "consistent where powered; " + ", ".join(f"{w} unpowered" for w, v in f6.items() if v["outcome"] == "unpowered")
                   if "consistent" in outs else "unpowered")
    F["verdict"] = verdict(F)
    F["F7"] = dict(outcome="not run (no asynchronous control in this design)")
    return F


def verdict(F) -> dict:
    """The mapping table of EXPERIMENT.md "Fixed rules" from falsifier outcomes to the label."""
    f1, f3, f4, f6 = F["F1"]["outcome"], F["F3"]["outcome"], F["F4"]["outcome"], F["F6"]["outcome"]
    falsified = [k for k in ("F1", "F3", "F4", "F6") if F[k]["outcome"].startswith("falsified")]
    if F["F3"]["notes_kept_outcome"] == "falsified":
        falsified.append("F3 (notes kept)")
    if falsified:
        label, text = "not supported", "falsified: " + ", ".join(falsified)
    elif f1.startswith("unresolved"):
        label, text = "inconclusive", "mutual echo unresolved and not predicted above 0.03"
    elif f3 == "consistent (sign resolved)":
        label, text = "supported", "reciprocity, ordering and the antagonistic sign confirmed"
    elif f3.startswith("no sign predicted") and "but an echo is resolved" not in f3:
        label, text = "supported", "reciprocity and ordering confirmed; antagonistic world a second null as predicted"
    else:
        label, text = "inconclusive (antagonistic part)", ("reciprocity (mutual against one-way) and the ordering confirmed; "
                                                           "the antagonistic sign not resolved at 3 se")
    memory = "one-step memory holds" if F["F2"]["outcome"] == "consistent" else "one-step memory fails (F2)"
    return {"label": label, "text": text, "memory": memory, "F6": f6, "F5": F["F5"]["outcome"]}


def residuals(W, KT):
    out = {}
    for w in W3:
        for law in ("M1", "M2"):
            P = PRED["predictions"][w][law]
            for k in ("rho1", "rho2", "X2", "c2", "Xc2", "q", "m"):
                if k in P:
                    p, obs = P[k], W[w][k]
                    se = float(np.hypot(obs["se"], p["calib_sd"]))
                    out[f"{w}|{law}|{k}"] = dict(obs=obs["mean"], obs_se=obs["se"], pred=p["value"], calib_sd=p["calib_sd"],
                                                 resid=obs["mean"] - p["value"], z=(obs["mean"] - p["value"]) / se)
            if KT.get(w):
                P = PRED["predictions"][w][law]["kick"]
                for k, pk in (("E", "E"), ("E_kept", "E_notes_kept"), ("rA", "rA"), ("rD", "rD"), ("wA", "wA"), ("wD", "wD")):
                    o = KT[w][k]
                    out[f"{w}|{law}|kick_{k}"] = dict(obs=o["est"], obs_se=o["se"], pred=P[pk], resid=o["est"] - P[pk],
                                                      z=(o["est"] - P[pk]) / o["se"] if o["se"] else None)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {w: [] for w in W3}
    meta = {}
    for w in W3:
        for g in range(5):
            r = load_run(w, g)
            if r is not None:
                runs[w].append(r)
                meta[f"{w}:{g}"] = {k: r[k] for k in ("n_missing", "n_censored", "n_calls", "n_refusal", "n_notes",
                                                      "sec_per_round", "m_t", "q_t")} | {"stats": r["stats"]}
    S = {"runs": meta, "chat": CHAT}
    W = world_table(runs)
    S["worlds"] = W
    for w in W3:
        print(f"{w}: {W[w]['n_runs']} runs | " + " ".join(f"{k} {W[w][k]['mean']:+.4f} ({W[w][k]['se']:.4f})" for k in STATS))
    S["inswarm"] = {w: {"calls_only": inswarm(runs[w], False), "leak": inswarm(runs[w], True)} for w in W3 if runs[w]}
    for w, f in S["inswarm"].items():
        print(f"in-swarm {w}: calls-only " + " ".join(f"{k} {v['est']:+.3f}({v['se']:.3f})" for k, v in f["calls_only"].items())
              + " | leak " + " ".join(f"{k} {v['est']:+.3f}({v['se']:.3f})" for k, v in f["leak"].items()))
    S["gauss"] = {w: gauss_flags(runs[w]) for w in W3 if runs[w]}
    if "oneway" in S["gauss"]:
        g = S["gauss"]
        S["gauss_flags"] = {"oneway_conditional_kurtosis_below_-0.3": any(v["excess_kurtosis"] < -0.3 for v in g["oneway"].values()),
                            "bimodal_any_world": any(v["bimodality_coef"] > 0.555 for w in g for v in g[w].values())}
        print("Gaussian-cavity flags:", S["gauss_flags"], {w: {k: round(v["excess_kurtosis"], 2) for k, v in g[w].items()} for w in g})
    KT = {w: kick_table(w) for w in W3}
    S["kicks"] = KT
    for w, k in KT.items():
        if k:
            print(f"kicks {w}: n {k['n_kicks']} (+{k['n_nulls']} null) | " + " ".join(
                f"{kk} {k[kk]['est']:+.4f} ({k[kk]['se']:.4f})" for kk in ("rA", "rD", "wA", "wD", "E", "E_kept", "E_x"))
                + f" | null {k.get('null')} | identical prompts {k['frac_identical_prompt']}")
    if all(KT[w] for w in W3) and all(runs[w] for w in W3):
        G = {}
        for w in ("mutual", "antagonistic"):
            G[w] = reaction(runs[w], KT[w], S["inswarm"][w]["leak"])
            G[w].update({k + "_se": v for k, v in gd_se(runs[w], KT[w], S["inswarm"][w]["leak"]).items()})
        S["reaction"] = G
        print("reaction from the swarm:", {w: {k: round(v, 4) for k, v in g.items()} for w, g in G.items()})
        S["falsifiers"] = falsifiers(W, KT, G)
        print("twins powered (frozen E_pred >= 3 observed se):", S["falsifiers"]["powered"])
        for k, v in S["falsifiers"].items():
            if k.startswith("F") and k[1:].isdigit():
                print(k, v["outcome"], v.get("notes_kept_outcome", ""))
        S["composition"] = {
            "antagonistic_E_over_KrArD": KT["antagonistic"]["E"]["est"] / (K * KT["antagonistic"]["rA"]["est"] * KT["antagonistic"]["rD"]["est"]),
            "mutual_E_over_halfK_r2": KT["mutual"]["E"]["est"] / (K / 2 * (KT["mutual"]["rA"]["est"] ** 2 + KT["mutual"]["rD"]["est"] ** 2))}
        print("composition", S["composition"])
        print("VERDICT", S["falsifiers"]["verdict"])
    S["residuals"] = residuals(W, KT)
    S["mean_belief_law"] = {w: {"m": W[w]["m"], "M1": PRED["predictions"][w]["M1"]["m"], "M2": PRED["predictions"][w]["M2"]["m"]}
                            for w in W3 if runs[w]}
    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
