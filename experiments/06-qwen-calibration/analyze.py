"""Analysis of the Qwen3.5-9B calibration (run.py writes the raw records; this file only reads them, CPU only).

Writes results/qwen-calib/summary.json, which plot.py and RESULT.md read. Sections:

  transfer    E[s] = tanh(a_+ f_A - a_- f_D + a_p h_priv + c)  (experiments/05_swarm/transfer.py eq. (1), its fit()),
              per (wording, K) and arm, with a paired bootstrap over base probes (both arms resampled together).
              Done twice: on the logprob belief s (what the swarm propagates) and on s_sampled (one draw of the
              stated number from the same digit distribution: what a sampled answer would have said).
  stability   g = sqrt((p_+ a_+^2 + p_- a_-^2) / K), p_+ = p_- = 1/2, and gamma_eff from design_lib.effective at
              gamma = -1, 0, +1; Lambda = d g (1 + |gamma_eff|) (theory/claim2.md boundary d g (1 + |gamma|) = 1),
              with d = <1 - s^2> the mean local slope of tanh. d is taken from the real-arm probe answers of the same
              (wording, K) (their peers are mirrored at random, so |s| is smaller and d larger than in a settled
              swarm: an upper bound on Lambda), and from the pilot swarm as a cross-check. Bootstrap over (a_+, a_-).
  conformity  per (wording, K, text arm, evidence level): level means of s, local slopes between adjacent hbar
              levels (bootstrap within levels), the steepest one, and the tanh fit beta_resp (probes.fit_conformity).
  noise       within-level s.d. of s and s_sampled over the 20 conformity replicates (inputs identical up to the
              template wording and the session tag), against the same statistic on the Haiku conformity run
              (results/calib-20261125-conf, label wording, K=8); residual s.d. of the transfer fits, Qwen and Haiku.
  pilot       thermal_observables (experiments/04_swarm_design/design_lib.py, via experiments/05_swarm/analyze.load_run)
              per cell; per-round mean |delta p| and fractions of unchanged answers (displayed probability equal to
              two decimals, sampled number equal, public message identical); the same fractions on the Haiku pilot.

Usage:  uv run python experiments/06-qwen-calibration/analyze.py [--root results] [--n-boot 1000]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
sys.path.insert(0, str(ROOT / "experiments" / "04_swarm_design"))

import importlib.util  # noqa: E402

import cavity.probes as P  # noqa: E402

# experiments/05_swarm/analyze.py has this file's name, so load it by path
_spec = importlib.util.spec_from_file_location("swarm05_analyze", ROOT / "experiments" / "05_swarm" / "analyze.py")
_m05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m05)
load_run = _m05.load_run
from design_lib import effective, thermal_observables  # noqa: E402
from transfer import fit  # noqa: E402  (experiments/05_swarm/transfer.py)

WORDINGS, TRANSFER_KS, CONF_KS, ARMS = ("inverted", "label"), (2, 4, 8), (2, 4, 8), ("real", "template")
PILOT_TAG, HAIKU_PILOT_TAG = "qwen-pilot-20261125", "pilot-20261125"


def _se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


# ---------------------------------------------------------------- transfer and stability

def transfer_section(root: Path, n_boot: int, d_pilot: float) -> dict:
    out = {}
    for wording in WORDINGS:
        recs = json.loads((root / "qwen-calib" / f"transfer_{wording}.json").read_text())
        for K in TRANSFER_KS:
            rk = [r for r in recs if r["K"] == K]
            if not rk:
                continue
            key = f"{wording}|K{K}"
            o = {"n_calls": len(rk), "n_missing": sum(r["s"] is None for r in rk),
                 "missing_reasons": {m: sum(r["reason"] == m for r in rk) for m in {r["reason"] for r in rk} if m}}
            for belief in ("s", "s_sampled"):
                ok_b = sorted(b for b in {r["b"] for r in rk}
                              if all(r[belief] is not None for r in rk if r["b"] == b))
                by = {(r["b"], r["arm"]): {**r, "s": r[belief]} for r in rk if r["b"] in set(ok_b)}
                bs = np.array(ok_b)
                est = {arm: fit([by[(b, arm)] for b in bs]) for arm in ARMS}
                rng = np.random.default_rng(0)
                boot = {arm: [] for arm in ARMS}
                for _ in range(n_boot):
                    pick = rng.choice(bs, len(bs))
                    for arm in ARMS:
                        boot[arm].append(fit([by[(b, arm)] for b in pick]))
                boot = {a: np.array(v) for a, v in boot.items()}
                o[belief] = {"n_pairs": int(len(bs))}
                for arm in ARMS:
                    for i, nm in enumerate(("a_plus", "a_minus", "a_p")):
                        o[belief][f"{arm}_{nm}"] = float(est[arm][i])
                        o[belief][f"{arm}_{nm}_se"] = _se(boot[arm][:, i])
                    rs = [by[(b, arm)] for b in bs]
                    X = np.array([[r["f_A"], -r["f_D"], r["h_priv"]] for r in rs])
                    y = np.array([r["s"] for r in rs])
                    w, cs, _ = P._fit_tanh(X, y, np.zeros(len(y)), est[arm])
                    o[belief][f"{arm}_resid_sd"] = float(np.std(np.tanh(X @ w + next(iter(cs.values()))) - y))
                    o[belief][f"{arm}_mean_slope_d"] = float(np.mean(1 - y**2))
                for i, nm in enumerate(("a_plus", "a_minus", "a_p")):
                    o[belief][f"diff_{nm}"] = float(est["real"][i] - est["template"][i])
                    o[belief][f"diff_{nm}_se"] = _se(boot["real"][:, i] - boot["template"][:, i])
                if belief == "s":
                    yr = np.array([by[(b, "real")]["s"] for b in bs])
                    o["stability"] = stability(boot["real"], est["real"], K, o[belief]["real_mean_slope_d"], d_pilot,
                                               float(np.sqrt(np.mean((1 - yr**2) ** 2))))
            out[key] = o
    return out


def stability(boot: np.ndarray, est: np.ndarray, K: int, d_probe: float, d_pilot: float, d_rms: float) -> dict:
    """g, Jbar, gamma_eff and Lambda = d g (1 + |gamma_eff|) at gamma = -1, 0, +1 for the real-message gains."""
    def one(ap, am, d):
        r = {}
        for gamma in (-1.0, 0.0, 1.0):
            e = effective(ap, am, K, 0.5, gamma)
            r[f"{gamma:+.0f}"] = {"gamma_eff": float(e["gamma_eff"]), "Lambda": float(d * e["g"] * (1 + abs(e["gamma_eff"])))}
        e0 = effective(ap, am, K, 0.5, 0.0)
        r["g"], r["Jbar"], r["d_Jbar"] = float(e0["g"]), float(e0["Jbar"]), float(d * e0["Jbar"])
        r["Lambda_max"] = max(v["Lambda"] for k, v in r.items() if isinstance(v, dict))
        return r

    # probe_d is the primary (fixed rules); pilot_d and rms_d (sqrt<d^2>, theory/claim2.md section 2) are cross-checks.
    # d is held fixed in the bootstrap, so the s.e. covers the gains only.
    out = {"d_probe": d_probe, "d_pilot": d_pilot, "d_rms": d_rms}
    for name, d in (("probe_d", d_probe), ("pilot_d", d_pilot), ("rms_d", d_rms)):
        pt = one(est[0], est[1], d)
        bt = [one(b[0], b[1], d) for b in boot]
        out[name] = {**pt, "g_se": _se([b["g"] for b in bt]), "Lambda_max_se": _se([b["Lambda_max"] for b in bt]),
                     "d_Jbar_se": _se([b["d_Jbar"] for b in bt]),
                     **{f"Lambda{k}_se": _se([b[k]["Lambda"] for b in bt]) for k in ("-1", "+0", "+1")}}
    return out


# ---------------------------------------------------------------- conformity and noise

def conformity_section(root: Path, n_boot: int) -> dict:
    out = {}
    for wording in WORDINGS:
        recs = json.loads((root / "qwen-calib" / f"conformity_{wording}.json").read_text())
        for K in CONF_KS:
            for arm in ("template", "real"):
                rs = [r for r in recs if r["K"] == K and r["arm"] == arm]
                if not rs:
                    continue
                key = f"{wording}|K{K}|{arm}"
                both = [{**r, "s": r["s"] if r["s"] is not None else np.nan} for r in rs]
                rs = [r for r in rs if r["hist"] == 1]                    # curves and beta_resp: own last 0.70 HEADS
                ok = [r for r in both if r["hist"] == 1]
                fc = P.fit_conformity(ok)
                o = {"n_calls": len(rs), "n_missing": sum(r["s"] is None for r in rs),
                     "beta_resp": fc["beta_resp"], "beta_resp_se": fc["beta_resp_se"], "offsets": fc["offsets"],
                     "by_ev": {}}
                if {r["hist"] for r in both} == {1, -1}:
                    # weight of the own last answer at fixed inbox: E[s] = tanh(beta hbar + kappa_u u + c_ev),
                    # u = 2 p_own - 1 = +-0.4 (cavity.probes.HISTORY_STATES)
                    fin = [r for r in both if np.isfinite(r["s"])]
                    X = np.array([[r["hbar_realized"], r["u"]] for r in fin])
                    w, _, se = P._fit_tanh(X, np.array([r["s"] for r in fin]),
                                           np.array([str(r["ev_level"]) for r in fin]), [1.0, 0.3])
                    o["kappa_u"], o["kappa_u_se"], o["beta_resp_both_hist"] = float(w[1]), float(se[1]), float(w[0])
                for ev in sorted({r["ev_level"] for r in rs}):
                    hb = sorted({r["hbar"] for r in rs if r["ev_level"] == ev})
                    lv = {h: np.array([r["s"] for r in rs if r["ev_level"] == ev and r["hbar"] == h and r["s"] is not None])
                          for h in hb}
                    lvs = {h: np.array([r["s_sampled"] for r in rs if r["ev_level"] == ev and r["hbar"] == h
                                        and r["s_sampled"] is not None]) for h in hb}
                    means = np.array([lv[h].mean() for h in hb])
                    slopes = np.diff(means) / np.diff(hb)
                    rng = np.random.default_rng(1)
                    bs = []
                    for _ in range(n_boot):
                        m = np.array([rng.choice(lv[h], len(lv[h])).mean() for h in hb])
                        bs.append(np.diff(m) / np.diff(hb))
                    bs = np.array(bs)
                    k = int(np.argmax(slopes))
                    # fixed points of the mean-field map hbar -> E[s](hbar) on all-agree inboxes: sign changes of
                    # mean_s - hbar that are resolved (both neighbours outside 2 s.e.), and roots of the fitted tanh
                    se_m = np.array([_se(lv[h]) / np.sqrt(len(lv[h])) for h in hb])
                    dm = means - np.array(hb)
                    resolved = np.abs(dm) > 2 * se_m
                    n_cross = int(sum(np.sign(dm[j]) != np.sign(dm[j + 1]) and resolved[j] and resolved[j + 1]
                                      for j in range(len(hb) - 1)))
                    c_ev = [v for kk, v in o["offsets"].items() if kk.startswith(f"{ev}|")]
                    grid = np.linspace(-1, 1, 4001)
                    n_fix = (int(np.sum(np.diff(np.sign(np.tanh(o["beta_resp"] * grid + c_ev[0]) - grid)) != 0))
                             if c_ev else None)
                    o["by_ev"][str(ev)] = {
                        "hbar": hb, "mean_s": means.tolist(), "se_s": [_se(lv[h]) / np.sqrt(len(lv[h])) for h in hb],
                        "n": [len(lv[h]) for h in hb], "slopes": slopes.tolist(), "slopes_se": bs.std(0, ddof=1).tolist(),
                        "steepest_slope": float(slopes[k]), "steepest_slope_se": float(bs[:, k].std(ddof=1)),
                        "steepest_between": [hb[k], hb[k + 1]],
                        # max over the bootstrap of the max slope, to show the selection of the steepest pair
                        "steepest_slope_boot_max_mean": float(bs.max(1).mean()),
                        "n_resolved_crossings": n_cross, "n_fixed_points_fit": n_fix,
                        "within_sd_s": float(np.sqrt(np.mean([lv[h].var(ddof=1) for h in hb]))),
                        "within_sd_s_sampled": float(np.sqrt(np.mean([lvs[h].var(ddof=1) for h in hb]))),
                    }
                out[key] = o
    return out


def haiku_conformity_noise(root: Path) -> dict:
    """Within-level s.d. of Haiku's stated s over the conformity replicates (calib-20261125-conf, K=8, label),
    per (evidence, history) cell, pooled over hbar levels; same statistic as within_sd_s above."""
    f = root / "calib-20261125-conf" / "checkpoints.jsonl"
    if not f.exists():
        return {}
    cells = {}
    for line in open(f):
        r = json.loads(line)
        if r["status"] != "ok" or not str(r["agent_id"]).startswith("conf|"):
            continue
        _, ev, hist, hb, _ = str(r["agent_id"]).split("|")
        cells.setdefault((ev, hist), {}).setdefault(hb, []).append(2 * r["calibrated_belief_prob"] - 1)
    return {f"{ev}|{hist}": float(np.sqrt(np.mean([np.var(v, ddof=1) for v in lv.values() if len(v) > 1])))
            for (ev, hist), lv in cells.items()}


def haiku_transfer_resid(root: Path) -> dict:
    out = {}
    for wording in WORDINGS:
        f = root / "transfer" / f"{wording}.json"
        if not f.exists():
            continue
        recs = json.loads(f.read_text())
        for K in (2, 4):
            rs = [r for r in recs if r["K"] == K and r["arm"] == "real" and r["s"] is not None]
            X = np.array([[r["f_A"], -r["f_D"], r["h_priv"]] for r in rs])
            y = np.array([r["s"] for r in rs])
            w, cs, _ = P._fit_tanh(X, y, np.zeros(len(y)), [1.0, 0.5, 0.4])
            out[f"{wording}|K{K}"] = {"resid_sd": float(np.std(np.tanh(X @ w + next(iter(cs.values()))) - y)), "a": w.tolist(),
                                      "mean_slope_d": float(np.mean(1 - y**2))}
    return out


# ---------------------------------------------------------------- pilot

def unchanged_fractions(exp_dir: Path, qwen: bool) -> dict:
    """Per round t >= 1, over agents and replicas: fraction whose displayed probability (2 decimals) equals the
    previous round's, fraction whose sampled number equals the previous (Qwen only), and fraction whose public
    message is identical to the previous one; plus the mean |delta p| of the recorded beliefs."""
    if qwen:
        calls = {(r["seed"], r["round_t"], r["agent_id"]): r for r in map(json.loads, open(exp_dir / "vllm_calls.jsonl"))}
    else:
        calls = {(r["seed"], r["round_t"], r["agent_id"]): r for r in map(json.loads, open(exp_dir / "checkpoints.jsonl"))}
    meta = json.loads((exp_dir / "swarm_meta.json").read_text())
    R, T, N = meta["config"]["replicas"], meta["config"]["T"], len(meta["h_priv"])

    def get(rep, t, i):
        # round 0 is shared: replica 0's call (envs/swarm.py share_initial)
        return calls.get((0 if t == 0 else rep, t, i)) or calls.get((rep, t, i))

    def mode(r):
        return max(r["masses"].items(), key=lambda kv: kv[1])[0] if r.get("masses") else None

    rows = {"same_displayed": [], "same_sampled": [], "same_mode": [], "same_message": [], "mean_abs_dp": []}
    for t in range(1, T + 1):
        sd, ss, sm, dp, smo = [], [], [], [], []
        for rep in range(R):
            for i in range(N):
                a, b = get(rep, t - 1, i), get(rep, t, i)
                if a is None or b is None or a["status"] != "ok" or b["status"] != "ok":
                    continue
                pa, pb = a["calibrated_belief_prob"], b["calibrated_belief_prob"]
                sd.append(round(pa, 2) == round(pb, 2))
                dp.append(abs(pb - pa))
                sm.append(a["public_message"].strip() == b["public_message"].strip())
                if qwen and a.get("p_sampled") is not None and b.get("p_sampled") is not None:
                    ss.append(a["p_sampled"] == b["p_sampled"])
                if qwen and mode(a) is not None and mode(b) is not None:
                    smo.append(mode(a) == mode(b))       # most likely stated number: the analogue of restating it
        rows["same_displayed"].append(float(np.mean(sd)))
        rows["same_message"].append(float(np.mean(sm)))
        rows["mean_abs_dp"].append(float(np.mean(dp)))
        rows["same_sampled"].append(float(np.mean(ss)) if ss else None)
        rows["same_mode"].append(float(np.mean(smo)) if smo else None)
    return rows


def pilot_section(root: Path) -> tuple[dict, float]:
    import cells as haiku_cells
    man = json.loads((root / PILOT_TAG / "manifest.json").read_text())
    out, ds = {}, []
    for c in man["cells"]:
        cid = haiku_cells.cell_id(c)
        d = root / f"{PILOT_TAG}-{cid}"
        S, info = load_run(d)
        obs = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in thermal_observables(S).items()}
        dlate = float(np.mean(1 - S[:, 3:] ** 2))
        ds.append(dlate)
        calls = [json.loads(line) for line in open(d / "vllm_calls.jsonl")]
        vm = np.array([r["valid_mass"] for r in calls if r.get("valid_mass") is not None])
        rec = {**{k: c[k] for k in ("wording", "K", "p_neg", "gamma", "N", "T", "mode", "graph")}, **info, **obs,
               "d_late": dlate, "valid_mass_median": float(np.median(vm)), "valid_mass_min": float(vm.min()),
               "missing_reasons": {m: sum(r.get("missing_reason") == m for r in calls) for m in
                                   ("malformed_sample", "low_valid_mass")},
               "qwen_unchanged": unchanged_fractions(d, True)}
        hd = root / f"{HAIKU_PILOT_TAG}-{cid}"
        if (hd / "checkpoints.jsonl").exists():
            hS, _ = load_run(hd)
            rec["haiku_unchanged"] = unchanged_fractions(hd, False)
            rec["haiku_c1"], rec["haiku_c2"] = (float(thermal_observables(hS)[k]) for k in ("c1", "c2"))
            rec["haiku_d_late"] = float(np.mean(1 - hS[:, 3:] ** 2))
        out[cid] = rec
    return out, float(np.mean(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    pilot, d_pilot = pilot_section(a.root)
    summary = {"pilot": pilot, "d_pilot_mean": d_pilot,
               "transfer": transfer_section(a.root, a.n_boot, d_pilot),
               "conformity": conformity_section(a.root, a.n_boot),
               "haiku_conformity_within_sd": haiku_conformity_noise(ROOT / "results"),
               "haiku_transfer_real_fit": haiku_transfer_resid(ROOT / "results")}
    summary["verdicts"] = verdicts(summary)
    (a.root / "qwen-calib" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print_tables(summary)


def verdicts(s: dict) -> dict:
    """The fixed rules of EXPERIMENT.md. Near the boundary (clear): Lambda_max (probe d) - 2 s.e. >= 0.7; straddling:
    the 2 s.e. interval contains 0.7; below: upper bound < 0.7. Bistable: real-text conformity at evidence 0.4 with
    steepest slope - 2 s.e. > 1, beta_resp > 1 and three fixed points (resolved crossings of the level means, or the
    fitted tanh). Secondary (labelled, not a criterion): Lambda / (1 - d kappa_u) for gamma >= 0 (theory/claim2.md
    section 1, static threshold with own-answer memory)."""
    out = {"cells": {}}
    for key, o in s["transfer"].items():
        st = o["stability"]["probe_d"]
        lo, hi = st["Lambda_max"] - 2 * st["Lambda_max_se"], st["Lambda_max"] + 2 * st["Lambda_max_se"]
        lam = "clear" if lo >= 0.7 else "straddles" if hi >= 0.7 else "below"
        wording, K = key.split("|")
        c = s["conformity"].get(f"{wording}|{K}|real")
        bist, mem = None, None
        if c is not None:
            v = c["by_ev"]["0.4"]
            slope_ok = v["steepest_slope"] - 2 * v["steepest_slope_se"] > 1
            three = v["n_resolved_crossings"] >= 3 or v["n_fixed_points_fit"] == 3
            bist = ("yes" if slope_ok and c["beta_resp"] > 1 and three else
                    "inconclusive" if v["steepest_slope"] > 1 and not (slope_ok and c["beta_resp"] > 1 and three) else "no")
            if "kappa_u" in c:
                kl = o["stability"]["d_probe"] * c["kappa_u"]
                mem = {"kappa_lin": kl, **{g: st[g]["Lambda"] / (1 - kl) for g in ("+0", "+1")}}
        out["cells"][key] = {"Lambda_max": st["Lambda_max"], "Lambda_lo": lo, "Lambda_hi": hi, "Lambda": lam,
                             "bistable": bist, "with_memory_secondary": mem}
    cs = out["cells"].values()
    out["overall"] = ("supported" if any(c["Lambda"] == "clear" or c["bistable"] == "yes" for c in cs) else
                      "inconclusive" if any(c["Lambda"] == "straddles" or c["bistable"] == "inconclusive" for c in cs)
                      else "not supported")
    return out


def print_tables(s: dict):
    print("\n== pilot (inverted, p_neg=0.5, N=64, T=10, 2 replicas)")
    for cid, r in s["pilot"].items():
        u = r["qwen_unchanged"]
        hu = r.get("haiku_unchanged", {})
        print(f"K={r['K']} g={r['gamma']:+.0f} | missing {r['missing']}/{r['n']} | m(T) {r['m'][-1]:+.3f} c1 {r['c1']:.2f} "
              f"c2 {r['c2']:+.2f} P_delta {r['P_delta']:.2f} | d_late {r['d_late']:.2f} | |dp| t=1 {u['mean_abs_dp'][0]:.3f} "
              f"t>=6 {np.mean(u['mean_abs_dp'][5:]):.3f} | unchanged(disp) t>=6 {np.mean(u['same_displayed'][5:]):.2f} "
              f"(mode {np.mean([x for x in u['same_mode'][5:] if x is not None]):.2f}, sampled {np.mean([x for x in u['same_sampled'][5:] if x is not None]):.2f}, msg {np.mean(u['same_message'][5:]):.2f})"
              + (f" | Haiku unchanged t>=6 {np.mean(hu['same_displayed'][5:]):.2f} msg {np.mean(hu['same_message'][5:]):.2f}"
                 if hu else ""))
    print("\n== transfer, logprob belief (paired bootstrap s.e.)")
    for key, o in s["transfer"].items():
        t, st = o["s"], o["stability"]["probe_d"]
        print(f"{key}: n={t['n_pairs']} miss {o['n_missing']} | real a+ {t['real_a_plus']:.2f}({t['real_a_plus_se']:.2f}) "
              f"a- {t['real_a_minus']:.2f}({t['real_a_minus_se']:.2f}) ap {t['real_a_p']:.2f}({t['real_a_p_se']:.2f}) | "
              f"tpl a+ {t['template_a_plus']:.2f}({t['template_a_plus_se']:.2f}) a- {t['template_a_minus']:.2f}"
              f"({t['template_a_minus_se']:.2f}) | diff a+ {t['diff_a_plus']:+.2f}({t['diff_a_plus_se']:.2f}) "
              f"a- {t['diff_a_minus']:+.2f}({t['diff_a_minus_se']:.2f}) | g {st['g']:.2f}({st['g_se']:.2f}) "
              f"d {o['stability']['d_probe']:.2f} Lmax {st['Lambda_max']:.2f}({st['Lambda_max_se']:.2f}) "
              f"dJbar {st['d_Jbar']:+.2f} | resid sd {t['real_resid_sd']:.3f} (sampled {o['s_sampled']['real_resid_sd']:.3f})")
    print("\n== conformity")
    for key, o in s["conformity"].items():
        evs = " | ".join(f"ev {ev}: steepest {v['steepest_slope']:.2f}({v['steepest_slope_se']:.2f}) at "
                         f"{v['steepest_between']} within-sd {v['within_sd_s']:.3f} (sampled {v['within_sd_s_sampled']:.3f})"
                         for ev, v in o["by_ev"].items())
        print(f"{key}: beta_resp {o['beta_resp']:.2f}({o['beta_resp_se']:.2f}) miss {o['n_missing']}/{o['n_calls']} | {evs}")
    print("\n== verdicts (fixed rules)")
    for key, v in s["verdicts"]["cells"].items():
        print(key, v)
    print("overall:", s["verdicts"]["overall"])
    print("\nHaiku conformity within-level sd:", s["haiku_conformity_within_sd"])
    print("Haiku transfer real-arm fits:", s["haiku_transfer_real_fit"])


if __name__ == "__main__":
    main()
