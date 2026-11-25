"""Task 1: is a scalar field enough to describe how a PoA agent responds to its inbox? (claim1.md sec. 4.1)

For every transition (agent i, round t -> t+1, t = 1..7) we regress the K = 5 votes at t+1 on the inbox counts
n_a+, n_a-, n_d+, n_d- (agree / disagree inbox, sender holding +1 / -1 at round t; see common.transitions).
Every model has one intercept alpha per (statement, agent), the quenched private field h_priv of claim1.md eq. (1)
(persona x question; the same persona sits at the same agent index on all ten graphs of a statement). Logit of
P(vote = +1):

  M0  private field only        alpha
  M1  scalar field              alpha + beta h,              h = m_a - m_d = sum_j J_ij s_j
  M2  two channels              alpha + b+ m_a - b- m_d,     m_a = n_a+ - n_a-, m_d = n_d+ - n_d-
      (b+: logit pull per agreeing-tie message; a disagree-tie message has coupling -b-, claim1.md sec. 4.2)
  PoA PoA's 3-coupling drive     alpha + bpos (J+ s) + bneg (J- s) + b0 (|J| s), fit in PoA's two stages
      (res/utils.py two_stage_fit). |J|s = J+s - J-s, so this spans exactly M2; b+ = bpos + b0, b- = bneg - b0.
  M2avg two channels, averaged   alpha + b+ m_a/k - b- m_d/k, k = inbox size (non-nested alternative to M2:
      does the agent sum or average what it reads? This decides the 1/sqrt(K) in claim1.md eq. (10))
  M2avg_sep  each inbox averaged on its own: alpha + b+ m_a/k_a - b- m_d/k_d
  M2cat two channels, additive but nonlinear: alpha + f(m_a) + g(m_d), f, g free on the levels -4..4
      (|m| >= 4 pooled); M2cat vs M2 measures saturation within a channel
  M3  four free counts          alpha + c1 n_a+ + c2 n_a- + c3 n_d+ + c4 n_d-   (adds inbox-size effects)
  M3sq M3 + the four squares    (saturation in each count)
  M4  M3 + all 10 quadratic and interaction terms of the four counts; M4 vs M3sq is the pure interaction
      (non-additivity) part
  M5  M4 + s_i(t)                own current state (not shown in the prompt)
  M6  M5 + s_i(t-1)              own state two rounds back (memory; PoA is nominally memoryless)
  M2+s, M2+s+s' : M2 with s_i(t), then s_i(t-1), for readable own-state coefficients.

Held-out score: 5-fold cross-validation by graph (fold = graph index mod 5, so each fold holds out 2 of the 10
graphs of every statement; the alpha of each (statement, agent) is learned from the other 8 graphs). Reported as
cross-entropy per vote, CE = -loglik / #votes (nats), and differences in millinats per vote with standard errors
over statements (statements are the independent units: personas and questions are shared within one).
Gain CIs: percentile bootstrap over statements (B resamples), full-data fits.

Run: uv run python analysis/poa/field_sufficiency.py [--boot 200] [--selftest]
Outputs (results/poa/): field_sufficiency_cv.csv, field_sufficiency_cv_by_class.csv, field_sufficiency_gains.csv,
field_sufficiency_coefs.json, fits_M2.npz (alpha, b+, b- per cell for M2 and M2avg, statement names; used by
effective_coupling.py and async_sim.py), selftest_fit.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from analysis.poa.common import RESULTS, cells, fit_logit_fe, loglik_rows, transitions  # noqa: E402

N_AGENTS = 32


def safe_div(a, b):
    return np.where(b > 0, a / np.maximum(b, 1), 0.0)


def design(T: dict, name: str):
    """Design matrix (rows x p) and column names of each model; see module docstring."""
    nap, nam, ndp, ndm = T["nap"], T["nam"], T["ndp"], T["ndm"]
    m_a, m_d = T["m_a"], T["m_d"]
    four = [nap, nam, ndp, ndm]
    four_names = ["n_a+", "n_a-", "n_d+", "n_d-"]
    quad = [four[i] * four[j] for i, j in combinations_with_replacement(range(4), 2)]
    quad_names = [f"{four_names[i]}*{four_names[j]}" for i, j in combinations_with_replacement(range(4), 2)]
    k = T["k_a"] + T["k_d"]
    sq_names = [f"{n}^2" for n in four_names]
    # one-hot levels of m_a and m_d, reference level 0, |m| >= 4 pooled into +-4 (rare): additive in the two
    # channels but free to saturate
    lev = [-4, -3, -2, -1, 1, 2, 3, 4]
    ca, cd = np.clip(m_a, -4, 4), np.clip(m_d, -4, 4)
    cat_cols = [(ca == v).astype(float) for v in lev] + [(cd == v).astype(float) for v in lev]
    cat_names = [f"m_a={v}" for v in lev] + [f"m_d={v}" for v in lev]
    cols = {
        "M0": ([], []),
        "M1": ([m_a - m_d], ["h"]),
        "M2": ([m_a, m_d], ["m_a", "m_d"]),
        "M2avg": ([safe_div(m_a, k), safe_div(m_d, k)], ["m_a/k", "m_d/k"]),
        "M2avg_sep": ([safe_div(m_a, T["k_a"]), safe_div(m_d, T["k_d"])], ["m_a/k_a", "m_d/k_d"]),
        "M2cat": (cat_cols, cat_names),
        "M3": (four, four_names),
        "M3sq": (four + [x * x for x in four], four_names + sq_names),
        "M4": (four + quad, four_names + quad_names),
        "M5": (four + quad + [T["s_now"]], four_names + quad_names + ["s(t)"]),
        "M6": (four + quad + [T["s_now"], T["s_prev"]], four_names + quad_names + ["s(t)", "s(t-1)"]),
        "M2+s": ([m_a, m_d, T["s_now"]], ["m_a", "m_d", "s(t)"]),
        "M2+s+s'": ([m_a, m_d, T["s_now"], T["s_prev"]], ["m_a", "m_d", "s(t)", "s(t-1)"]),
    }[name]
    X = np.stack(cols[0], 1) if cols[0] else np.zeros((len(m_a), 0))
    return X, cols[1]


CV_MODELS = ["M0", "M1", "M2", "M2avg", "M2avg_sep", "M2cat", "M3", "M3sq", "M4", "M5", "M6", "M2+s", "M2+s+s'"]
BOOT_MODELS = ["M1", "M2", "M2avg", "M3", "M2+s", "M2+s+s'"]


def fit(T, idx, name, group=None, n_groups=None):
    X, names = design(T, name)
    g = T["group"][idx] if group is None else group
    G = T["n_groups"] if n_groups is None else n_groups
    beta, alpha, ll = fit_logit_fe(X[idx], T["npos"][idx], T["ntot"][idx], g, G)
    return beta, alpha, ll, names


def fit_poa_two_stage(T, idx):
    """PoA's two-stage fit (res/utils.py two_stage_fit) with our (statement, agent) intercepts in place of their
    persona-question field: stage 1 fits b0 on the unsigned drive |J|s = m_a + m_d; stage 2 fits bpos, bneg on
    J+s = m_a and J-s = -m_d with b0 |J|s as a fixed offset."""
    us = (T["m_a"] + T["m_d"])[idx]
    g, G = T["group"][idx], T["n_groups"]
    b1, _, _ = fit_logit_fe(us[:, None], T["npos"][idx], T["ntot"][idx], g, G)
    b0 = b1[0]
    X2 = np.stack([T["m_a"][idx], -T["m_d"][idx]], 1)
    b2, alpha, ll = fit_logit_fe(X2, T["npos"][idx], T["ntot"][idx], g, G, offset=b0 * us)
    return dict(beta_pos=b2[0], beta_neg=b2[1], beta_0=b0, b_plus=b2[0] + b0, b_minus=b2[1] - b0, ll=ll)


def cross_validate(T, models=CV_MODELS, n_folds=5):
    """Held-out loglik per row for each model (rows of fold f predicted by a fit on the other folds)."""
    fold = T["graph"] % n_folds
    out = {}
    for name in models:
        X, _ = design(T, name)
        ll_rows = np.empty(len(fold))
        for f in range(n_folds):
            tr, te = fold != f, fold == f
            beta, alpha, _ = fit_logit_fe(X[tr], T["npos"][tr], T["ntot"][tr], T["group"][tr], T["n_groups"])
            u = X[te] @ beta + alpha[T["group"][te]]
            ll_rows[te] = loglik_rows(u, T["npos"][te], T["ntot"][te])
        out[name] = ll_rows
    return out


def ce_table(T, ll, ref="M1", by=None):
    """CE per vote for each model, and the difference from ``ref`` (millinats/vote) with a cluster s.e. over
    statements: dCE = sum_s dLL_s / sum_s n_s, se from the ratio estimator's linearization."""
    stmt = T["stmt"]
    rows = []
    masks = {"all": np.ones(len(stmt), bool)} if by is None else {v: T[by] == v for v in np.unique(T[by])}
    for lab, m in masks.items():
        n_s = np.bincount(stmt[m], T["ntot"][m])
        S = (n_s > 0)
        for name, l in ll.items():
            L_s = np.bincount(stmt[m], l[m], len(n_s))
            D_s = L_s - np.bincount(stmt[m], ll[ref][m], len(n_s))
            ce = -L_s[S].sum() / n_s[S].sum()
            d = -D_s[S].sum() / n_s[S].sum()                    # CE(model) - CE(ref): negative = better
            resid = -D_s[S] - d * n_s[S]
            ns = S.sum()
            se = np.sqrt(ns / (ns - 1) * np.sum(resid**2)) / n_s[S].sum()
            rows.append(dict(subset=lab, spec=name, ce_nats=ce, dce_vs_ref_mnats=1e3 * d, se_mnats=1e3 * se,
                             ref=ref, n_votes=int(n_s[S].sum()), n_statements=int(ns)))
    return rows


def bootstrap(T, B, rng, models=BOOT_MODELS):
    """Percentile bootstrap over statements of the full-data coefficients."""
    stmt = T["stmt"]
    n_stmt = stmt.max() + 1
    rows_of = [np.nonzero(stmt == s)[0] for s in range(n_stmt)]
    draws = {m: [] for m in models}
    for _ in range(B):
        pick = rng.integers(0, n_stmt, n_stmt)
        idx = np.concatenate([rows_of[s] for s in pick])
        slot = np.concatenate([np.full(len(rows_of[s]), k) for k, s in enumerate(pick)])
        group = slot * N_AGENTS + T["agent"][idx]
        for m in models:
            beta, _, _, _ = fit(T, idx, m, group=group, n_groups=n_stmt * N_AGENTS)
            draws[m].append(beta)
    return {m: np.array(v) for m, v in draws.items()}


def selftest(T, rng, B=40):
    """Recovery check on synthetic votes with the real design (counts, groups, folds): truth is M2 with b+ = 1.5,
    b- = 0.5 and the intercepts of a real M2 fit. Expect: M2 recovered inside its bootstrap CI; held-out CE of
    M1 worse than M2; M3, M4, M6 not better than M2 beyond noise; the PoA two-stage fit equal to M2 in loglik."""
    lines = []
    _, alpha, _, _ = fit(T, np.arange(len(T["npos"])), "M2")
    u = 1.5 * T["m_a"] - 0.5 * T["m_d"] + alpha[T["group"]]
    Ts = dict(T)
    Ts["npos"] = rng.binomial(T["ntot"].astype(int), 1 / (1 + np.exp(-u))).astype(float)
    allrows = np.arange(len(u))
    b, _, ll, _ = fit(Ts, allrows, "M2")
    poa = fit_poa_two_stage(Ts, allrows)
    bs = bootstrap(Ts, B, rng, models=["M2"])["M2"]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    lines.append(f"truth b+ = 1.5, m_d coef = -0.5;  fit {b.round(4)}, 95% CI {lo.round(3)} .. {hi.round(3)}")
    lines.append(f"PoA two-stage: b+ = {poa['b_plus']:.4f}, b- = {poa['b_minus']:.4f}, beta_0 = {poa['beta_0']:.4f};"
                 f" loglik diff from M2 = {poa['ll'] - ll:.2e}")
    ll_cv = cross_validate(Ts, models=["M1", "M2", "M3", "M4", "M6"])
    for r in ce_table(Ts, ll_cv, ref="M2"):
        lines.append(f"  {r["spec"]:4s} CE {r['ce_nats']:.5f}  dCE vs M2 {r['dce_vs_ref_mnats']:+.3f} "
                     f"({r['se_mnats']:.3f}) mnat/vote")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    C = cells()

    if args.selftest:
        T = transitions(C[("gpt-4o-mini", "objective")])
        lines = ["Self-test of fit_logit_fe / cross-validation on synthetic M2 votes (gpt-4o-mini objective design)"]
        lines += selftest(T, rng)
        txt = "\n".join(lines)
        print(txt)
        (RESULTS / "selftest_fit.txt").write_text(txt + "\n")
        return

    cv_rows, cls_rows, gain_rows, coefs, fits = [], [], [], {}, {}
    for (model, mode), c in sorted(C.items()):
        T = transitions(c)
        cell = f"{model} | {mode}"
        print(f"\n== {cell}: {len(T['npos'])} transitions, {int(T['ntot'].sum())} votes", flush=True)
        ll = cross_validate(T)
        for r in ce_table(T, ll, ref="M1"):
            cv_rows.append(dict(model=model, mode=mode, **r))
            print(f"  {r["spec"]:8s} CE {r['ce_nats']:.4f}  dCE vs M1 {r['dce_vs_ref_mnats']:+7.2f} "
                  f"({r['se_mnats']:.2f}) mnat/vote")
        for ref in ("M2", "M3sq"):          # s.e. of the channel, saturation and interaction contrasts
            for r in ce_table(T, ll, ref=ref):
                if r["spec"] != ref:
                    cv_rows.append(dict(model=model, mode=mode, **r))
        for r in ce_table(T, ll, ref="M2", by="gclass"):
            cls_rows.append(dict(model=model, mode=mode, **r))

        allrows = np.arange(len(T["npos"]))
        full = {}
        for m in CV_MODELS:
            beta, alpha, llf, names = fit(T, allrows, m)
            full[m] = dict(beta=beta, names=names, ll=llf)
            if m == "M2":
                fits[f"{model}|{mode}|alpha"] = alpha
            if m == "M2avg":
                fits[f"{model}|{mode}|alpha_avg"] = alpha
                fits[f"{model}|{mode}|b_avg"] = np.array([beta[0], -beta[1]])
        fits[f"{model}|{mode}|stmt_names"] = np.array(c["stmt_names"])
        poa = fit_poa_two_stage(T, allrows)
        boot = bootstrap(T, args.boot, rng)
        coefs[cell] = {}
        for m in CV_MODELS:
            ent = {"names": full[m]["names"], "beta": full[m]["beta"].tolist(), "loglik": full[m]["ll"]}
            if m in boot:
                lo, hi = np.percentile(boot[m], [2.5, 97.5], axis=0)
                ent["ci95_lo"], ent["ci95_hi"] = lo.tolist(), hi.tolist()
                ent["boot_sd"] = boot[m].std(0, ddof=1).tolist()
            coefs[cell][m] = ent
        coefs[cell]["PoA_two_stage"] = {k: float(v) for k, v in poa.items()}
        coefs[cell]["PoA_two_stage"]["loglik_minus_M2"] = float(poa["ll"] - full["M2"]["ll"])
        coefs[cell]["n_transitions"], coefs[cell]["n_votes"] = len(T["npos"]), int(T["ntot"].sum())

        # gains: b+ = coef(m_a), b- = -coef(m_d); bootstrap percentiles transformed the same way
        b2 = boot["M2"]
        bp, bm = full["M2"]["beta"][0], -full["M2"]["beta"][1]
        bpa, bma = boot["M2avg"][:, 0], -boot["M2avg"][:, 1]
        bs_s = boot["M2+s+s'"]
        gain_rows.append(dict(
            model=model, mode=mode,
            b_plus=bp, b_plus_lo=np.percentile(b2[:, 0], 2.5), b_plus_hi=np.percentile(b2[:, 0], 97.5),
            b_minus=bm, b_minus_lo=np.percentile(-b2[:, 1], 2.5), b_minus_hi=np.percentile(-b2[:, 1], 97.5),
            ratio=bm / bp, ratio_lo=np.percentile(-b2[:, 1] / b2[:, 0], 2.5),
            ratio_hi=np.percentile(-b2[:, 1] / b2[:, 0], 97.5),
            beta_scalar=full["M1"]["beta"][0],
            beta_scalar_lo=np.percentile(boot["M1"][:, 0], 2.5), beta_scalar_hi=np.percentile(boot["M1"][:, 0], 97.5),
            b_plus_avg=full["M2avg"]["beta"][0], b_plus_avg_lo=np.percentile(bpa, 2.5),
            b_plus_avg_hi=np.percentile(bpa, 97.5),
            b_minus_avg=-full["M2avg"]["beta"][1], b_minus_avg_lo=np.percentile(bma, 2.5),
            b_minus_avg_hi=np.percentile(bma, 97.5),
            coef_s_now=full["M2+s+s'"]["beta"][2], coef_s_now_lo=np.percentile(bs_s[:, 2], 2.5),
            coef_s_now_hi=np.percentile(bs_s[:, 2], 97.5),
            coef_s_prev=full["M2+s+s'"]["beta"][3], coef_s_prev_lo=np.percentile(bs_s[:, 3], 2.5),
            coef_s_prev_hi=np.percentile(bs_s[:, 3], 97.5),
            poa_b_plus=poa["b_plus"], poa_b_minus=poa["b_minus"], poa_beta_0=poa["beta_0"],
        ))
        fits[f"{model}|{mode}|b"] = np.array([bp, bm])
        g = gain_rows[-1]
        print(f"  b+ = {bp:.3f} [{g['b_plus_lo']:.3f}, {g['b_plus_hi']:.3f}]   "
              f"b- = {bm:.3f} [{g['b_minus_lo']:.3f}, {g['b_minus_hi']:.3f}]   "
              f"s(t) {g['coef_s_now']:+.3f}  s(t-1) {g['coef_s_prev']:+.3f}", flush=True)

    pd.DataFrame(cv_rows).to_csv(RESULTS / "field_sufficiency_cv.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(RESULTS / "field_sufficiency_cv_by_class.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(RESULTS / "field_sufficiency_gains.csv", index=False)
    (RESULTS / "field_sufficiency_coefs.json").write_text(json.dumps(
        {"boot": args.boot, "seed": args.seed, "cells": coefs}, indent=1))
    np.savez(RESULTS / "fits_M2.npz", **fits)


if __name__ == "__main__":
    main()
