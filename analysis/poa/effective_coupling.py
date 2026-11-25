"""Task 3: effective couplings, stability parameter Lambda and regime of each (model, task) in the PoA swarms.

Effective couplings (claim1.md sec. 4.2). With the two-channel fit M2 the logit of agent i is
alpha_i + sum_j W_ij s_j with W_ij = b+ [J_ij = +1] - b- [J_ij = -1] (logit units per neighbour). Mapping the actual
graph onto claim1 eq. (2), J_ij = Jbar/N + g z_ij / sqrt(N):
    Jbar = < sum_j W_ij >_i,    g^2 = < sum_j W_ij^2 >_i - Jbar^2 / N,
    gamma = sum_ij (W_ij - w)(W_ji - w) / sum_ij (W_ij - w)^2  (w = Jbar/N) = 1 for the symmetric PoA graphs.
For M2avg (agents average their inbox) W_ij is divided by the receiver's degree k_i, which makes gamma < 1 on
graphs with heterogeneous degree.

Bare gain. The spin is the majority of K = 5 votes, so E[s_i(t+1)] = phi(u_i) and the bare one-round response is
d_i = phi'(u_i) = 60 [p(1-p)]^3, p = sigmoid(u_i) (common.dphi_maj; 0.9375 at u = 0). Three operating points:
    balanced : u_i = alpha_i (an inbox with no net pull; what claim2 calls the quiet state)
    round 0  : u_i = alpha_i + (W s(0))_i, the field produced by the round-0 answers (start of the dynamics)
    late     : u_i(t) = alpha_i + (W s(t))_i over rounds t = 3..7 (the states the agents actually respond to)

Lambda. claim2 eq. (8) for memoryless agents with gamma = 1, solved at the most unstable frequency by
theory/dmft_solver._lambda_static over the empirical distribution of d_i; inf means no solution on the branch
continuous with weak coupling (already beyond the boundary). Also the homogeneous approximation
g (1 + |gamma|) sqrt<d^2>, and the mean-field (consensus) gain Jbar <d>. Because N = 32 and the graphs have
degree 3-4.6, the dense-graph DMFT is only indicative, so we also give the direct linear stability of the
synchronous map on the actual graphs: the eigenvalues of D W (D = diag d_i), real because W is symmetric,
split into the static mode lambda_+ = max eig and the period-2 mode lambda_- = -min eig, at the late operating
point, averaged over communities and rounds. The swarm is linearly unstable to a static pattern if lambda_+ > 1
and to a period-2 alternation if lambda_- > 1.

Empirical check without the fitted kernel (claim2.md eq. (9)): the replica distance D(t) = <(s^a_i - s^b_i)^2>
over the 6 replica pairs of each (statement, graph), against the sampling floor D_floor(t) = <2 - phi(u^a)^2 -
phi(u^b)^2> from the fitted u at t-1 (round 0, which has no inbox: twice the across-community variance of s_i(0)); amplification A(t) = D(t)/D_floor(t), and Lambda_hat(t) =
(D(t+1) - D_floor(t+1)) / D(t). A(T) ~ 1 means quiet, A(T) >> 1 beyond the boundary.

Run: uv run python analysis/poa/effective_coupling.py
Outputs: results/poa/effective_coupling.csv (per model x task x graph class), effective_coupling_amplification.csv,
effective_coupling.json, effective_coupling_log.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.poa.common import (CLASS_SHORT, RESULTS, cells, coupling_matrix, dphi_maj, fit_m2,  # noqa: E402
                                 phi_maj)
from theory.dmft_solver import _lambda_static  # noqa: E402

T_LATE = (3, 7)          # states s(t), t = 3..7, which drive rounds 4..8


def couplings(W):
    """Jbar, g, gamma of claim1 eq. (2) for one effective coupling matrix W (N x N)."""
    N = len(W)
    Jbar = W.sum(1).mean()
    w = Jbar / N
    off = ~np.eye(N, dtype=bool)
    dW = np.where(off, W - w, 0.0)
    g2 = (W**2).sum(1).mean() - Jbar**2 / N
    gamma = (dW * dW.T).sum() / (dW**2).sum()
    return Jbar, np.sqrt(g2), gamma


def lam_dmft(d, g, gamma):
    """claim2 (8) over the empirical d_i (uniform weights)."""
    return _lambda_static(np.asarray(d, float), np.full(len(d), 1.0 / len(d)), g, gamma, 1.0)


def scale_matrix(J, avg):
    """Divide row i by its degree for the averaging agent (M2avg); identity otherwise."""
    if not avg:
        return 1.0
    k = (J != 0).sum(1, keepdims=True)
    return np.where(k > 0, 1.0 / np.maximum(k, 1), 0.0)


def main():
    C = cells()
    rows, amp_rows, js, lines = [], [], {}, []
    for (model, mode), c in sorted(C.items()):
        R, T1, N = c["S"].shape
        for form, avg in (("M2", False), ("M2avg", True)):
            bp, bm, alpha = fit_m2(c, averaged=avg)
            a_comm = alpha.reshape(-1, N)[c["stmt"]]                         # (R, N) alpha of each agent
            # u_i(t) = alpha_i + (W s(t))_i for every community and round
            Ws = np.stack([coupling_matrix(J, bp, bm) * scale_matrix(J, avg) for J in c["J"]])   # (R, N, N)
            u = a_comm[:, None, :] + np.einsum("rij,rtj->rti", Ws, c["S"].astype(float))           # (R, T+1, N)
            d_all = dphi_maj(u)
            for cls in ["all"] + sorted(set(c["gclass"])):
                m = np.ones(R, bool) if cls == "all" else c["gclass"] == cls
                cname = cls if cls == "all" else CLASS_SHORT[cls]
                Jb, g, ga = np.mean([couplings(W) for W in Ws[m]], axis=0)
                d_bal = dphi_maj(a_comm[m]).ravel()
                d_r0 = d_all[m, 0].ravel()
                d_late = d_all[m, T_LATE[0]:T_LATE[1] + 1].ravel()
                # finite-graph spectra of D W at the late operating point
                lp, lm = [], []
                for r in np.nonzero(m)[0][::2]:
                    for t in range(T_LATE[0], T_LATE[1] + 1):
                        sq = np.sqrt(d_all[r, t])
                        if avg:   # D W with row-scaled W is not symmetric; its eigenvalues are still those of D W
                            ev = np.linalg.eigvals(d_all[r, t][:, None] * Ws[r]).real
                        else:
                            ev = np.linalg.eigvalsh(sq[:, None] * Ws[r] * sq[None, :])
                        lp.append(ev.max())
                        lm.append(-ev.min())
                # same at round 0
                lp0, lm0 = [], []
                for r in np.nonzero(m)[0][::2]:
                    ev = np.linalg.eigvals(d_all[r, 0][:, None] * Ws[r]).real
                    lp0.append(ev.max())
                    lm0.append(-ev.min())
                row = dict(model=model, mode=mode, form=form, graph_class=cname, b_plus=bp, b_minus=bm,
                           Jbar=Jb, g=g, gamma=ga,
                           d_bal_mean=d_bal.mean(), d_bal_rms=np.sqrt((d_bal**2).mean()),
                           d_r0_rms=np.sqrt((d_r0**2).mean()), d_late_mean=d_late.mean(),
                           d_late_rms=np.sqrt((d_late**2).mean()),
                           Lam_bal=lam_dmft(d_bal, g, ga), Lam_r0=lam_dmft(d_r0, g, ga),
                           Lam_late=lam_dmft(d_late, g, ga),
                           homog_bal=g * (1 + abs(ga)) * np.sqrt((d_bal**2).mean()),
                           homog_late=g * (1 + abs(ga)) * np.sqrt((d_late**2).mean()),
                           meanfield_bal=Jb * d_bal.mean(), meanfield_late=Jb * d_late.mean(),
                           lam_plus_r0=np.mean(lp0), lam_minus_r0=np.mean(lm0),
                           lam_plus_r0_q=np.percentile(lp0, [10, 90]).tolist(),
                           lam_minus_r0_q=np.percentile(lm0, [10, 90]).tolist(),
                           frac_r0_unstable=float(np.mean(np.maximum(lp0, lm0) > 1)),
                           lam_plus_late=np.mean(lp), lam_minus_late=np.mean(lm),
                           lam_plus_late_q=np.percentile(lp, [10, 90]).tolist(),
                           lam_minus_late_q=np.percentile(lm, [10, 90]).tolist())
                rows.append(row)

            # replica amplification, M2 form only (spins; pairs of replicas of the same statement and graph)
            if form != "M2":
                continue
            phi2 = phi_maj(u) ** 2                                              # (R, T+1, N)
            key = c["stmt"] * 100 + c["graph"]
            # round 0 has no inbox and every community of a statement answers it independently, so its floor is
            # 2 Var(s_i(0)) estimated from the 40 communities of the statement (unbiased variance)
            n_st = np.bincount(c["stmt"])
            s0bar = np.stack([c["S"][c["stmt"] == k, 0].mean(0) for k in range(len(n_st))])        # (n_stmt, N)
            var0 = (n_st / (n_st - 1))[:, None] * (1 - s0bar**2)
            for cls in sorted(set(c["gclass"])):
                Dt, Ft = [], []
                for k in np.unique(key[c["gclass"] == cls]):
                    idx = np.nonzero(key == k)[0]
                    for ia in range(len(idx)):
                        for ib in range(ia + 1, len(idx)):
                            a, b = idx[ia], idx[ib]
                            Dt.append(((c["S"][a] - c["S"][b]).astype(float) ** 2).mean(1))       # (T+1,)
                            # floor at t from the fitted u at t-1; round 0 from var0
                            f = np.empty(T1)
                            f[0] = 2 * var0[c["stmt"][a]].mean()
                            f[1:] = (2 - phi2[a, :-1] - phi2[b, :-1]).mean(1)
                            Ft.append(f)
                Dt, Ft = np.array(Dt), np.array(Ft)
                D, F = Dt.mean(0), Ft.mean(0)
                n_pairs = len(Dt)
                Dse = Dt.std(0, ddof=1) / np.sqrt(n_pairs)
                for t in range(T1):
                    amp_rows.append(dict(model=model, mode=mode, graph_class=CLASS_SHORT[cls], t=t, D=D[t], D_se=Dse[t],
                                         D_floor=F[t], A=D[t] / F[t],
                                         Lam_hat=(D[t + 1] - F[t + 1]) / D[t] if t + 1 < T1 else np.nan))
        print(f"done {model} {mode}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "effective_coupling.csv", index=False)
    amp = pd.DataFrame(amp_rows)
    amp.to_csv(RESULTS / "effective_coupling_amplification.csv", index=False)

    fmt = lambda x: "inf" if not np.isfinite(x) else f"{x:.2f}"   # noqa: E731
    lines.append("Effective couplings (logit units) and stability, M2 (summing agent). Lam = claim2 (8), gamma = 1; "
                 "inf = no weak-coupling solution")
    lines.append(f"  {'cell':26s} {'graphs':18s} {'b+':>5s} {'b-':>6s} {'Jbar':>5s} {'g':>5s} {'gam':>4s} | "
                 f"{'d_bal':>5s} {'d_late':>6s} | {'Lam_bal':>7s} {'Lam_r0':>7s} {'Lam_late':>8s} | "
                 f"{'Jb<d>bal':>8s} {'Jb<d>late':>9s} | graph: lam+ lam- (r0)  lam+ lam- (late)")
    for _, r in df[df.form == "M2"].iterrows():
        lines.append(f"  {r.model + ' | ' + r['mode']:26s} {r.graph_class:18s} {r.b_plus:5.2f} {r.b_minus:+6.2f} "
                     f"{r.Jbar:5.2f} {r.g:5.2f} {r.gamma:4.2f} | {r.d_bal_rms:5.2f} {r.d_late_rms:6.2f} | "
                     f"{fmt(r.Lam_bal):>7s} {fmt(r.Lam_r0):>7s} {fmt(r.Lam_late):>8s} | {r.meanfield_bal:8.2f} "
                     f"{r.meanfield_late:9.2f} | {r.lam_plus_r0:5.2f} {r.lam_minus_r0:5.2f}      "
                     f"{r.lam_plus_late:5.2f} {r.lam_minus_late:5.2f}")
    lines.append("\nSame for M2avg (averaging agent), all graphs and by class:")
    for _, r in df[df.form == "M2avg"].iterrows():
        lines.append(f"  {r.model + ' | ' + r['mode']:26s} {r.graph_class:18s} {r.b_plus:5.2f} {r.b_minus:+6.2f} "
                     f"{r.Jbar:5.2f} {r.g:5.2f} {r.gamma:4.2f} | {r.d_bal_rms:5.2f} {r.d_late_rms:6.2f} | "
                     f"{fmt(r.Lam_bal):>7s} {fmt(r.Lam_r0):>7s} {fmt(r.Lam_late):>8s} | {r.meanfield_bal:8.2f} "
                     f"{r.meanfield_late:9.2f} | {r.lam_plus_r0:5.2f} {r.lam_minus_r0:5.2f}      "
                     f"{r.lam_plus_late:5.2f} {r.lam_minus_late:5.2f}")
    lines.append("\nReplica amplification A(t) = D(t)/D_floor(t), t = 0..8 (spins, 6 replica pairs per statement x graph)")
    for (model, mode, cls), g in amp.groupby(["model", "mode", "graph_class"]):
        lines.append(f"  {model + ' | ' + mode:26s} {cls:18s} A: " + " ".join(f"{x:5.2f}" for x in g.A)
                     + "   D(8) = %.3f, floor(8) = %.3f" % (g.D.iloc[-1], g.D_floor.iloc[-1]))
    lines.append("\nLambda_hat(t) = (D(t+1) - D_floor(t+1)) / D(t) for t = 0, 1, 2 (claim2 eq. 9; exact only for gamma = 0 "
                 "and small noise), and the finite-graph round-0 stability (fraction of communities with max(lam+, lam-) > 1)")
    r0 = df[df.form == "M2"].set_index(["model", "mode", "graph_class"])
    for (model, mode, cls), g in amp.groupby(["model", "mode", "graph_class"]):
        q = r0.loc[(model, mode, cls)]
        lines.append(f"  {model + ' | ' + mode:26s} {cls:18s} Lam_hat: " + " ".join(f"{x:5.2f}" for x in g.Lam_hat.iloc[:3])
                     + f"   | lam+ r0 {q.lam_plus_r0:.2f} [{q.lam_plus_r0_q[0]:.2f}, {q.lam_plus_r0_q[1]:.2f}], lam- r0 "
                     f"{q.lam_minus_r0:.2f} [{q.lam_minus_r0_q[0]:.2f}, {q.lam_minus_r0_q[1]:.2f}], unstable "
                     f"{q.frac_r0_unstable:.2f}")
    txt = "\n".join(lines)
    print(txt)
    (RESULTS / "effective_coupling_log.txt").write_text(txt + "\n")
    js = {"note": "see effective_coupling.py docstring", "rows": df.to_dict(orient="records")}
    (RESULTS / "effective_coupling.json").write_text(json.dumps(js, indent=1, default=float))


if __name__ == "__main__":
    main()
