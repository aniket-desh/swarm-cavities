"""Surrogate v2 of the claude-haiku-5-5 swarm agent: real-message gains and anchoring gated by the change of the inbox.

Why. The v1 surrogate (design_lib.py) drew i.i.d. noise around a static target and predicted c1 ~ 0.3 for the pilot
swarm; the pilot gave c1 = 0.68-0.80, 30-48% exactly repeated answers and no growth of the replica distance
(HANDOFF.md, afternoon session). The white-probe trajectories (field redrawn every round) show weak persistence. Both
are reproduced by one law in which the weight on the agent's own last answer depends on how much its inbox changed
since that answer (theory/quiet_regime.md section 1):

    f_A,i(t) = (1/K) sum_{j in A_i} s_j(t),   f_D,i(t) = (1/K) sum_{j in D_i} s_j(t)        (as in design_lib)
    u_i(t)   = (1/K) sum_{j in A_i u D_i} |s_j(t) - s_j(t-1)|       inbox change since i's last answer        (Q1)
    tau_i(t) = tanh(a_+ f_A,i - a_- f_D,i + a_p hpriv_i + c)        static target (the two-channel fit)       (Q2)
    rho(u)   = rho_inf + (rho_0 - rho_inf) exp(-u / u_0)            weight on the own last answer             (Q3)
    mu_i(t)  = (1 - rho(u_i)) (tau_i - s_i(t))                      intended move                             (Q4)
    with probability pi(mu) = pi_0 exp(-|mu| / mu_pi):   s_i(t+1) = s_i(t)                (exact repeat)      (Q5)
    otherwise:   s_i(t+1) = clip( s_i(t) + mu_i + (sig_0 + sig_1 |mu_i|) xi + b sig_J xi' ),                  (Q6)
                 xi, xi' ~ N(0, 1),  b = 1 with probability q_J (a rare large move), else 0

The first inbox (round 1) has no predecessor: u = infinity, so rho = rho_inf, and it uses its own gains (gains0),
because round-0 messages quote raw sensor reports and move readers more than later messages do.
Round 0 is s_i(0) = clip(tanh(A_P0 hpriv_i + C_0) + SIG_R0 xi), shared by the replicas of a run.

Gains are those of the target tau. A one-shot probe of an agent whose last answer is u0 (the transfer check, or
round 1 of a swarm) returns u0 + (1 - rho_inf)(tau - u0), so the gain a tanh fit of such answers reports is
(1 - rho_inf) times the target gain; `transfer_gains` and `fit_pilot_gains` fit the target gains in that form.

v1 is the special case rho = const, pi_0 = 0, sig_1 = 0, q_J = 0 with the noise added around tau, not the move.

What is fitted where (run `fit`):
    law_white (rho_0, rho_inf, u_0, sig_0, sig_1, pi_0, mu_pi; q_J = 0)
                       maximum likelihood on the white-probe transitions only (results/calibration/white.json,
                       rounds >= 2, K = 8 template messages)
    gains "transfer"   target gains from the real-message probes of experiments/05_swarm/transfer.py
    gains "pilot"      in-sample: round-1 response (gains0) and late fixed point (gains) of the pilot swarm
    law_swarm[K]       law_white with three numbers replaced by matching one-step statistics of the pilot
                       (in-sample): rho_0 (one value, from the own-answer weight in replica differences at small
                       u), and per K pi_0 (late repeat rate) and q_J (late moves of 0.15 or more; SIG_J fixed)
`check` simulates the pilot cells on their own graphs and evidence (experiments/05_swarm/cells.graph_for / game_for)
and compares c1, c2, P_delta, the repeat fraction, D(t) and m(t) with results/pilot-20261125-*.

Run:  uv run python experiments/04_swarm_design/surrogate_v2.py fit      (writes surrogate_v2_fit.json)
      uv run python experiments/04_swarm_design/surrogate_v2.py check    (prints the tables in surrogate_v2_out.txt)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "05_swarm"))
import design_lib as dl  # noqa: E402

FIT = HERE / "surrogate_v2_fit.json"
PILOT_TAG = "pilot-20261125"
A_P0, C_0, SIG_R0 = 0.50, 0.0, 0.017     # round 0 of the pilot: s = tanh(0.50 hpriv), residual s.d. 0.015-0.019
LAW_KEYS = ("rho0", "rho_inf", "u0", "sig0", "sig1", "pi0", "mu_pi")     # plus optional q_j, sig_j (default 0)
SIG_J, BIG = 0.25, 0.15                  # s.d. of a rare large move; moves of BIG or more count as large
LATE = slice(6, 11)                      # rounds counted as settled (mean |ds| has stopped falling by round 6)


# ------------------------------------------------------------------ the law (Q1)-(Q6)

def rho(u, law):
    """(Q3) weight on the own last answer; u = inf gives rho_inf."""
    return law["rho_inf"] + (law["rho0"] - law["rho_inf"]) * np.exp(-np.asarray(u, float) / law["u0"])


def target(fA, fD, hp, gains):
    """(Q2) with gains = dict(a_plus, a_minus, a_p, c)."""
    return np.tanh(gains["a_plus"] * fA - gains["a_minus"] * fD + gains["a_p"] * hp + gains["c"])


def step(s, tau, u, law, rng):
    """(Q4)-(Q6): one answer of each agent given its last answer s, target tau and inbox change u.
    Returns (new s, held mask)."""
    mu = (1 - rho(u, law)) * (tau - s)
    held = rng.random(np.shape(s)) < law["pi0"] * np.exp(-np.abs(mu) / law["mu_pi"])
    noise = (law["sig0"] + law["sig1"] * np.abs(mu)) * rng.standard_normal(np.shape(s))
    if law.get("q_j", 0.0) > 0:
        noise = noise + (rng.random(np.shape(s)) < law["q_j"]) * law["sig_j"] * rng.standard_normal(np.shape(s))
    return np.where(held, s, dl.clip(s + mu + noise)), held


def round0(hp, rng):
    return dl.clip(np.tanh(A_P0 * hp + C_0) + SIG_R0 * rng.standard_normal(len(hp)))


def simulate_v2(J, hp, gains, gains0, law, T, rng, replicas=2, init=None):
    """The surrogate on a signed graph J (rows receive, envs/graphs.py) with evidence fields hp.
    Returns S (replicas, T+1, N), raw sign convention (s > 0 leans HEADS). Replicas share round 0."""
    N = len(hp)
    K = int((J[0] != 0).sum())
    WA, WD, WU = (J > 0) / K, (J < 0) / K, (J != 0) / K
    init = round0(hp, rng) if init is None else init
    S = np.empty((replicas, T + 1, N))
    S[:, 0] = init
    for r in range(replicas):
        for t in range(T):
            s = S[r, t]
            u = np.full(N, np.inf) if t == 0 else WU @ np.abs(s - S[r, t - 1])            # (Q1)
            tau = target(WA @ s, WD @ s, hp, gains0 if t == 0 else gains)
            S[r, t + 1], _ = step(s, tau, u, law, rng)
    return S


def simulate_v1_on(J, hp, a_plus, a_minus, T, rng, replicas=2):
    """design_lib.simulate's update (v1: i.i.d. noise around the target, one-round memory kappa) on a given graph
    and evidence, for the side-by-side comparison in `check`."""
    K = int((J[0] != 0).sum())
    W = (a_plus * (J > 0) - a_minus * (J < 0)) / K
    s0 = np.tanh(dl.A_P * hp + dl.C0)
    kap = dl.kappa_for(np.mean(1 - s0**2))
    init = dl.clip(s0 + dl.SIG_XI * rng.standard_normal(len(hp)))
    S = np.empty((replicas, T + 1, len(hp)))
    S[:, 0] = init
    for r in range(replicas):
        for t in range(T):
            s = S[r, t]
            S[r, t + 1] = dl.clip(np.tanh(W @ s + dl.A_P * hp + dl.C0 + kap * (s - s0))
                                  + dl.SIG_XI * rng.standard_normal(len(hp)))
    return S


def kick_twin(S, J, hp, gains, law, i, t0, rng, second_order=False, gains0=None):
    """Kick-and-echo twin of one replica S (T+1, N) of simulate_v2 (theory/quiet_regime.md section 3).

    Readers of agent i see its round-t0 message mirrored (s_i -> -s_i); i's own history stays true. Replay rule
    of the engine: an agent whose prompt is unchanged returns its base answer; an agent whose prompt changed is
    resampled from (Q4)-(Q6) with fresh noise. A resampled agent's message text is new even if its number is not,
    so every reader of a resampled agent is resampled in the next round.
    Returns dict(D = kick size -2 s_i(t0), ds1 = {j: s_j^twin(t0+1) - s_j^base(t0+1)} for the readers j of i,
    echo = s_i^twin(t0+2) - s_i^base(t0+2) (exactly 0 if i reads none of its readers), and with second_order
    ds2 = {k: change at t0+2} for every resampled k != i). For t0 = 0 the readers' inbox is their first one
    (u = infinity, gains0).
    """
    K = int((J[0] != 0).sum())
    WA, WD, WU = (J > 0) / K, (J < 0) / K, (J != 0) / K
    pub = S[t0].copy()
    pub[i] = -S[t0, i]                                    # what the readers of i see
    readers = np.flatnonzero(J[:, i])
    tw1 = S[t0 + 1].copy()
    for j in readers:                                      # round t0+1: only readers of i have a changed prompt
        u = WU[j] @ np.abs(pub - S[t0 - 1]) if t0 > 0 else np.inf
        tau = target(WA[j] @ pub, WD[j] @ pub, hp[j], gains0 if (t0 == 0 and gains0 is not None) else gains)
        tw1[j] = step(S[t0, j], tau, u, law, rng)[0]
    out = {"D": -2 * S[t0, i], "ds1": {int(j): tw1[j] - S[t0 + 1, j] for j in readers}, "echo": 0.0, "ds2": {}}
    resampled = set(readers.tolist())
    for j in readers:
        resampled |= set(np.flatnonzero(J[:, j]).tolist())
    todo = sorted(resampled) if second_order else ([i] if i in resampled else [])
    for k in todo:                                         # round t0+2
        seen_prev = pub if J[k, i] != 0 else S[t0]        # what k saw of its senders at round t0
        u = WU[k] @ np.abs(tw1 - seen_prev)
        tau = target(WA[k] @ tw1, WD[k] @ tw1, hp[k], gains)
        new = step(tw1[k], tau, u, law, rng)[0]
        if k == i:
            out["echo"] = new - S[t0 + 2, i]
        else:
            out["ds2"][int(k)] = new - S[t0 + 2, k]
    return out


# ------------------------------------------------------------------ data

def load_white():
    """White-probe transitions with a predecessor inbox: (s_next, s_prev, hbar, hbar_prev), rounds >= 2,
    plus the trajectory index of each row and the raw arrays."""
    w = json.loads((ROOT / "results" / "calibration" / "white.json").read_text())
    s, hb = np.array(w["s"], float), np.array(w["hbar"], float)
    traj = np.repeat(np.arange(len(s)), hb.shape[1] - 1)
    return dict(sn=s[:, 2:].ravel(), sp=s[:, 1:-1].ravel(), h=hb[:, 1:].ravel(), hp=hb[:, :-1].ravel(), traj=traj,
                s=s, hbar=hb)


def pilot_cells():
    return json.loads((ROOT / "results" / PILOT_TAG / "manifest.json").read_text())["cells"]


def load_pilot_cell(c):
    """One pilot cell: raw states S (2, T+1, N) from p_carried (s > 0 leans HEADS), its graph, evidence and truth."""
    import cells as C
    d = ROOT / "results" / f"{PILOT_TAG}-{C.cell_id(c)}"
    J, game = C.graph_for(c), C.game_for(c)
    P = np.full((2, c["T"] + 1, c["N"]), np.nan)
    for line in open(d / "swarm_rounds.jsonl"):
        r = json.loads(line)
        P[r["replica"], r["round_t"]] = r["p_carried"]
    assert not np.isnan(P).any(), f"incomplete run {d.name}"
    assert np.allclose(game.h_priv, json.loads((d / "swarm_meta.json").read_text())["h_priv"]), "evidence mismatch"
    return 2 * P - 1, J, game.h_priv, game.truth


def _fit_target(fA, fD, hp, y, own=None, rho_inf=0.0):
    """Least squares for the target gains of (Q2). With `own` (the agent's last answer) the answers are one-shot
    responses own + (1 - rho_inf)(tau - own); without it they are fixed points y = tau."""
    def res(w):
        tau = np.tanh(w[0] * fA - w[1] * fD + w[2] * hp + w[3])
        return (tau if own is None else own + (1 - rho_inf) * (tau - own)) - y
    sol = least_squares(res, [1.0, 0.2, 0.4, 0.0])
    se = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac)) * (sol.fun**2).sum() / (len(y) - 4))
    return dict(a_plus=float(sol.x[0]), a_minus=float(sol.x[1]), a_p=float(sol.x[2]), c=float(sol.x[3]),
                a_plus_se=float(se[0]), a_minus_se=float(se[1]), resid_sd=float(sol.fun.std()))


def transfer_gains(rho_inf):
    """Real-message gains per (wording, K) from the transfer-check records, arm "real".
    one_shot: the published fit E[s] = tanh(a_+ f_A - a_- f_D + a_p hpriv + c) (eq. (1) of transfer.py; s.e. from
              its paired bootstrap in summary.json). These are the response gains of an agent reading a new inbox.
    target:   the same records fitted as u0 + (1 - rho_inf)(tau - u0), u0 = the probe's shown last answer."""
    summ = json.loads((ROOT / "results" / "transfer" / "summary.json").read_text())
    out = {}
    for wording in ("inverted", "label"):
        recs = json.loads((ROOT / "results" / "transfer" / f"{wording}.json").read_text())
        for K in (2, 4):
            g = summ[f"{wording}|K{K}"]
            rs = [r for r in recs if r["K"] == K and r["arm"] == "real" and r["s"] is not None]
            fA, fD, hp, y, own = (np.array([r[k] for r in rs]) for k in ("f_A", "f_D", "h_priv", "s", "u"))
            one = _fit_target(fA, fD, hp, y)
            assert np.allclose([one["a_plus"], one["a_minus"], one["a_p"]],
                               [g["real_a_plus"], g["real_a_minus"], g["real_a_p"]], atol=2e-3)
            one.update(a_plus_se=g["real_a_plus_se"], a_minus_se=g["real_a_minus_se"])
            out[f"{wording}|K{K}"] = {"one_shot": one, "target": _fit_target(fA, fD, hp, y, own, rho_inf)}
    return out


# ------------------------------------------------------------------ fits

def _unpack(th):
    return dict(zip(LAW_KEYS, th[:7]))


def white_target(h, wt):
    """Target of the white-probe agent (one evidence set, K = 8 identical template messages at field h):
    A tanh(beta h / A + c). A < 1 because uniform template inboxes saturate well below certainty (the answer
    stays inside about +-0.45 even at |h| = 0.9); plain tanh (A = 1) fits 85 log-likelihood units worse. The swarm
    target (Q2) keeps A = 1: swarm fields stay below 0.4, where the two forms agree."""
    return wt["A"] * np.tanh(wt["beta"] * np.asarray(h) / wt["A"] + wt["c"])


def white_nll(th, d):
    """-log likelihood of the white transitions under (Q3)-(Q6) with the target white_target;
    u = |hbar(t) - hbar(t-1)| because all K = 8 template messages state the same probability."""
    law = _unpack(th)
    tau = white_target(d["h"], dict(beta=th[7], c=th[8], A=th[9]))
    mu = (1 - rho(np.abs(d["h"] - d["hp"]), law)) * (tau - d["sp"])
    pi = np.clip(law["pi0"] * np.exp(-np.abs(mu) / law["mu_pi"]), 1e-9, 1 - 1e-9)
    sig = law["sig0"] + law["sig1"] * np.abs(mu)
    ds = d["sn"] - d["sp"]
    held = np.abs(ds) < 1e-9
    ll = np.where(held, np.log(pi), np.log1p(-pi) - 0.5 * ((ds - mu) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi))
    return -ll.sum()


WHITE_X0 = [0.6, 0.15, 0.2, 0.05, 0.2, 0.2, 0.07, 1.0, 0.4, 0.45]
WHITE_BOUNDS = [(0, 0.99), (0, 0.99), (0.01, 5), (0.005, 0.5), (0, 2), (0, 0.99), (0.005, 5), (0.1, 3), (-1, 1),
                (0.2, 1.0)]
WHITE_KEYS = LAW_KEYS + ("beta_white", "c_white", "A_white")


def fit_white(d, x0=WHITE_X0, bounds=WHITE_BOUNDS):
    return minimize(white_nll, x0, args=(d,), method="L-BFGS-B", bounds=bounds)


def fit_white_boot(d, n_boot=200, seed=0):
    """MLE and bootstrap s.e. over the 200 trajectories."""
    best = fit_white(d)
    rng = np.random.default_rng(seed)
    n_traj = int(d["traj"].max()) + 1
    rows = [np.flatnonzero(d["traj"] == k) for k in range(n_traj)]
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.integers(n_traj, size=n_traj)])
        bs.append(fit_white({k: d[k][idx] for k in ("sn", "sp", "h", "hp")}, best.x).x)
    return best.x, np.std(bs, axis=0, ddof=1), float(best.fun)


def fit_pilot_gains(rho_inf):
    """In-sample target gains of the pilot swarm (inverted wording), per K, pooled over its three gamma cells.
    gains0: round-1 answers as one-shot responses to the round-0 inbox, s(1) = s(0) + (1 - rho_inf)(tau - s(0)).
    gains:  fixed point of (Q4), s = tau, on the replica- and round-averaged settled states (rounds 6-10); also
            per gamma, because reciprocated ties could bias a cross-sectional fit (i's state sits in j's answer)."""
    out = {}
    for K in (2, 4):
        r1, fp = [], {}
        for c in pilot_cells():
            if c["K"] != K:
                continue
            S, J, hp, _ = load_pilot_cell(c)
            WA, WD = (J > 0) / K, (J < 0) / K
            Sd = np.round(S * 50) / 50                                    # displayed with two decimals of p
            for r in range(2):
                r1.append(np.c_[Sd[r, 0] @ WA.T, Sd[r, 0] @ WD.T, hp, S[r, 1], S[r, 0]])
            sbar = Sd[:, LATE].mean((0, 1))
            fp[c["gamma"]] = np.c_[sbar @ WA.T, sbar @ WD.T, hp, S[:, LATE].mean((0, 1))]
        r1 = np.vstack(r1)
        out[f"inverted|K{K}|gains0"] = _fit_target(*r1[:, :4].T, own=r1[:, 4], rho_inf=rho_inf)
        out[f"inverted|K{K}|gains"] = _fit_target(*np.vstack(list(fp.values())).T)
        for gam, rows in fp.items():
            out[f"inverted|K{K}|gains|gamma={gam:+.0f}"] = _fit_target(*rows.T)
    return out


def pilot_step_stats(Ss, Js):
    """One-step statistics of runs with a common K, used to set the swarm-specific law parameters:
    late repeat rate, fraction of late moves of BIG or more, s.d. of the other late moves, and the own-answer
    weight rho_eff of the replica-difference regression at u < 0.08 (with its s.e. and residual s.d.)."""
    ds = np.concatenate([np.diff(S[:, 5:], axis=1).ravel() for S in Ss])
    y, X, u = (np.concatenate(v) for v in zip(*[replica_diff_regression(S, J) for S, J in zip(Ss, Js)]))
    m = u < 0.08
    w, se = _ols(X[m], y[m])
    small = (np.abs(ds) > 1e-9) & (np.abs(ds) < BIG)
    return dict(repeat=float(np.mean(np.abs(ds) < 1e-9)), big=float(np.mean(np.abs(ds) >= BIG)),
                sd_small=float(ds[small].std()), rho_eff=float(w[0]), rho_eff_se=float(se[0]), b_A=float(w[1]),
                resid_sd=float((y[m] - X[m] @ w).std()))


STEP_SCALE = dict(repeat=0.011, big=0.004, rho_eff=0.019)      # binomial / regression s.e. of the pilot statistics


def fit_swarm_law(law_white, gains_pilot, n_rep=30):
    """Simulated moments: rho_0 (shared) and per K (pi_0, q_J) such that simulate_v2 on the pilot's graphs and
    evidence reproduces pilot_step_stats (repeat, big, rho_eff). Common random numbers make the objective smooth."""
    data, sims = {}, {}
    for K in (2, 4):
        runs = [load_pilot_cell(c) for c in pilot_cells() if c["K"] == K]
        data[K] = pilot_step_stats([r[0] for r in runs], [r[1] for r in runs])
        sims[K] = runs

    def laws(th):
        return {K: dict(law_white, rho0=th[0], pi0=th[1 + 2 * k], q_j=th[2 + 2 * k], sig_j=SIG_J) for k, K in enumerate((2, 4))}

    def sim_stats(th, K, seed=1):
        rng = np.random.default_rng(seed)
        g1, g0 = gains_pilot[f"inverted|K{K}|gains"], gains_pilot[f"inverted|K{K}|gains0"]
        st = [pilot_step_stats([simulate_v2(J, hp, g1, g0, laws(th)[K], S.shape[1] - 1, rng) for S, J, hp, _ in sims[K]],
                               [r[1] for r in sims[K]]) for _ in range(n_rep)]
        return {k: float(np.mean([s[k] for s in st])) for k in st[0]}

    def loss(th):
        if not (0 < th[0] < 0.99 and all(0 <= x < 0.9 for x in th[1:])):
            return 1e9
        return sum(((sim_stats(th, K)[k] - data[K][k]) / STEP_SCALE[k]) ** 2 for K in (2, 4) for k in STEP_SCALE)

    sol = minimize(loss, [0.8, 0.3, 0.07, 0.4, 0.03], method="Nelder-Mead", options=dict(xatol=2e-3, fatol=0.05, maxiter=400))
    return laws(sol.x), data, {K: sim_stats(sol.x, K, seed=2) for K in (2, 4)}, float(sol.fun)


def replica_diff_regression(S, J):
    """Rows of the regression that reads the own-answer weight off replica differences, free of agent offsets:
        delta_i(t+1) = rho_eff delta_i(t) + b_A dfA_i(t) + b_D dfD_i(t),   delta = s^a - s^b,   t = 1 .. T-1.
    Returns (y, X, u): y = delta(t+1), X = [delta(t), dfA(t), dfD(t)], and u = the inbox change (Q1) between rounds
    t-1 and t, averaged over the two replicas, for binning."""
    K = int((J[0] != 0).sum())
    WA, WD, WU = (J > 0) / K, (J < 0) / K, (J != 0) / K
    de = S[0] - S[1]
    u = (np.abs(np.diff(S, axis=1)) @ WU.T).mean(0)                      # (T, N): change between t and t+1
    y, X, uu = de[2:].ravel(), np.stack([de[1:-1], de[1:-1] @ WA.T, de[1:-1] @ WD.T], -1).reshape(-1, 3), u[:-1].ravel()
    return y, X, uu


def _ols(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ w
    return w, np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var() * len(y) / (len(y) - X.shape[1]))


# ------------------------------------------------------------------ observables

def observables(S, truth):
    """Thermal observables of one run in design_lib's definitions, plus the repeat fraction and the mean
    one-round change. S is raw; alignment with the truth only matters for m."""
    o = dl.thermal_observables(truth * S)
    ds = np.abs(np.diff(S, axis=1))
    return dict(c1=o["c1"], c2=o["c2"], P_delta=o["P_delta"], D_late=float(np.mean(o["D"][LATE])), D1=float(o["D"][1]),
                m1=float(o["m"][1]), m_late=float(np.mean(o["m"][LATE])), repeat_late=float(np.mean(ds[:, 5:] < 1e-9)),
                absds_1=float(ds[:, 0].mean()), absds_late=float(ds[:, 5:].mean()))


OBS = ("c1", "c2", "P_delta", "repeat_late", "absds_1", "absds_late", "D1", "D_late", "m1", "m_late")


# ------------------------------------------------------------------ commands

def _gline(k, g):
    return (f"  {k:34s} a+ {g['a_plus']:.2f} ({g['a_plus_se']:.2f})  a- {g['a_minus']:+.2f} ({g['a_minus_se']:.2f})  "
            f"a_p {g['a_p']:.3f}  c {g['c']:+.3f}  resid s.d. {g['resid_sd']:.3f}")


def cmd_fit():
    d = load_white()
    th, se, nll = fit_white_boot(d)
    fixed = lambda i, v: [b if k != i else (v, v) for k, b in enumerate(WHITE_BOUNDS)]
    # nested alternatives: no gating (rho, sigma, pi constant: u0, mu_pi huge, sig1 = 0), and a plain tanh target
    bc = [(0, .99), (0, .99), (5, 5), (.005, .5), (0, 0), (0, .99), (5, 5), (.1, 3), (-1, 1), (.2, 1)]
    const = fit_white(d, [0.2, 0.2, 5.0, 0.09, 0.0, 0.05, 5.0, 1.0, 0.4, 0.45], bc)
    plain = fit_white(d, [0.7, 0.2, 0.15, 0.05, 0.2, 0.2, 0.07, 0.75, 0.14, 1.0], fixed(9, 1.0))
    print("A. White-probe MLE of the law (Q3)-(Q6), n = %d transitions; bootstrap s.e. over the 200 trajectories" % len(d["sn"]))
    for k, v, e in zip(WHITE_KEYS, th, se):
        print(f"  {k:10s} {v:7.3f} ({e:.3f})")
    law_white = dict(zip(LAW_KEYS, map(float, th[:7])))
    print(f"  -log L = {nll:.1f};  without gating (constant rho, sigma, pi; 6 parameters): {const.fun:.1f}")
    print(f"  plain tanh target (A = 1): -log L = {plain.fun:.1f}, rho0 {plain.x[0]:.3f} rho_inf {plain.x[1]:.3f} u0 {plain.x[2]:.3f}")
    print("  rho(u) at u = 0, 0.05, 0.15, 0.3, 0.6:  adopted " + " ".join(f"{rho(u, law_white):.2f}" for u in (0, .05, .15, .3, .6))
          + "   plain tanh " + " ".join(f"{rho(u, _unpack(plain.x)):.2f}" for u in (0, .05, .15, .3, .6)))
    tg, pg = transfer_gains(law_white["rho_inf"]), fit_pilot_gains(law_white["rho_inf"])
    print("\nB. Gains (s.e.). one_shot = response gains of a tanh fit to answers to a new inbox; target = gains of tau (Q2)")
    for k, g in tg.items():
        print(_gline(f"transfer {k} one_shot", g["one_shot"]))
        print(_gline(f"transfer {k} target", g["target"]))
    for k, g in pg.items():
        print(_gline(f"pilot {k.replace('inverted|', '')} (target)", g))
    laws, data, sim, loss = fit_swarm_law(law_white, pg)
    print("\nC. Swarm-specific law parameters (in-sample, simulated moments on the pilot's one-step statistics)")
    for K in (2, 4):
        print(f"  K={K}: rho0 {laws[K]['rho0']:.3f}  pi0 {laws[K]['pi0']:.3f}  q_j {laws[K]['q_j']:.3f}  (sig_j = {SIG_J} fixed)")
        for k in ("repeat", "big", "rho_eff", "sd_small", "b_A", "resid_sd"):
            print(f"       {k:9s} pilot {data[K][k]:.3f}   surrogate {sim[K][k]:.3f}" + ("   (matched)" if k in STEP_SCALE else "   (not fitted)"))
    print(f"  loss {loss:.2f} (sum of squared standardized differences of the six matched statistics)")
    FIT.write_text(json.dumps({
        "law_white": law_white, "law_white_se": dict(zip(LAW_KEYS, map(float, se[:7]))),
        "white_target": {"beta": float(th[7]), "c": float(th[8]), "A": float(th[9])}, "white_nll": nll,
        "white_nll_const": float(const.fun), "white_nll_plain_tanh": float(plain.fun),
        "gains_transfer": tg, "gains_pilot": pg, "law_swarm": {str(K): laws[K] for K in laws},
        "pilot_step_stats": {str(K): data[K] for K in data}}, indent=1))


def load_fit():
    return json.loads(FIT.read_text())


def law_swarm(fit, K):
    return fit["law_swarm"][str(K)]


def variants(fit, K):
    """Parameter sets compared in `check`, as (law, gains, gains0).
    white law, transfer gains: nothing taken from the pilot.
    white law, pilot gains:    gains in-sample.
    v2, transfer gains:        the swarm law (three in-sample numbers) with the real-message gains.
    v2, pilot gains:           everything the pilot can fix is fixed by it (in-sample)."""
    tg = fit["gains_transfer"][f"inverted|K{K}"]["target"]
    g0, g1 = fit["gains_pilot"][f"inverted|K{K}|gains0"], fit["gains_pilot"][f"inverted|K{K}|gains"]
    lw, ls = fit["law_white"], law_swarm(fit, K)
    return {"white law, transfer gains": (lw, tg, tg), "white law, pilot gains": (lw, g1, g0),
            "v2, transfer gains": (ls, tg, tg), "v2, pilot gains": (ls, g1, g0)}


def white_stats(s, hb):
    """Summary statistics of white-probe trajectories s (M, T+1) driven by fields hb (M, T)."""
    sn, sp, h, hp = s[:, 2:].ravel(), s[:, 1:-1].ravel(), hb[:, 1:].ravel(), hb[:, :-1].ravel()
    X = np.c_[sp, h, np.ones(len(sp))]
    b, _ = _ols(np.c_[hb.ravel(), np.ones(hb.size)], s[:, 1:].ravel())
    res = s[:, 1:] - (b[0] * hb + b[1])                                  # residual of s(t) on the field it just read
    out = {"own-answer weight": _ols(X, sn)[0][0],
           "residual lag-1 autocorrelation": np.sum(res[:, 1:] * res[:, :-1]) / np.sqrt(np.sum(res[:, 1:] ** 2) * np.sum(res[:, :-1] ** 2)),
           "repeat fraction": np.mean(np.abs(sn - sp) < 1e-9)}
    u = np.abs(h - hp)
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 3)):
        m = (u >= lo) & (u < hi)
        w = _ols(X[m], sn[m])[0]
        out[f"own-answer weight, u in [{lo},{hi})"] = w[0]
        out[f"repeat fraction,   u in [{lo},{hi})"] = np.mean(np.abs(sn - sp)[m] < 1e-9)
        out[f"residual s.d.,     u in [{lo},{hi})"] = (sn[m] - X[m] @ w).std()
    return out


def cmd_check(n_sim=200, seed=0):
    fit = load_fit()
    law = fit["law_white"]
    rng = np.random.default_rng(seed)

    # ---- 1. white probes: data against the surrogate driven by the same field sequences
    d = load_white()
    s, hb = d["s"], d["hbar"]
    sims = []
    for _ in range(n_sim):
        z = np.empty_like(s)
        z[:, 0] = s[:, 0]
        for t in range(hb.shape[1]):
            u = np.full(len(s), np.inf) if t == 0 else np.abs(hb[:, t] - hb[:, t - 1])
            z[:, t + 1], _ = step(z[:, t], white_target(hb[:, t], fit["white_target"]), u, law, rng)
        sims.append(white_stats(z, hb))
    dat = white_stats(s, hb)
    print("1. White probes (200 trajectories, field redrawn each round, u = |hbar(t) - hbar(t-1)|):"
          f"\n   data | white-law surrogate driven by the same fields, mean (s.d. over {n_sim} reruns)")
    for k in dat:
        v = np.array([x[k] for x in sims])
        print(f"   {k:38s} {dat[k]:6.3f} | {v.mean():6.3f} ({v.std():.3f})")

    # ---- 2. pilot: own-answer weight in replica differences, data against the surrogate
    print("\n2. Pilot replica differences, delta(t+1) = rho_eff delta(t) + b_A dfA(t) + b_D dfD(t), by inbox change u (Q1):"
          "\n   pilot rho_eff (s.e.), b_A (s.e.) | white law, pilot gains | v2, pilot gains   (surrogate: mean of 20 reruns)")
    cells = pilot_cells()
    runs = {id(c): load_pilot_cell(c) for c in cells}
    for K in (2, 4):
        var = variants(fit, K)
        kc = [c for c in cells if c["K"] == K]
        y, X, u = (np.concatenate(v) for v in zip(*[replica_diff_regression(runs[id(c)][0], runs[id(c)][1]) for c in kc]))
        sim = {name: [[replica_diff_regression(simulate_v2(runs[id(c)][1], runs[id(c)][2], g1, g0, lw, c["T"], rng), runs[id(c)][1])
                       for c in kc] for _ in range(20)] for name, (lw, g1, g0) in var.items() if "pilot gains" in name}
        for lo, hi in ((0, 0.03), (0.03, 0.08), (0.08, 0.2), (0.2, 2)):
            m = (u >= lo) & (u < hi)
            w, se = _ols(X[m], y[m])
            line = f"   K={K} u in [{lo:.2f},{hi:.2f}) n={m.sum():4d}: {w[0]:.2f} ({se[0]:.2f}), {w[1]:.2f} ({se[1]:.2f})"
            for name, reps in sim.items():
                ws = []
                for rep in reps:
                    yy, XX, uu = (np.concatenate(v) for v in zip(*rep))
                    mm = (uu >= lo) & (uu < hi)
                    ws.append(_ols(XX[mm], yy[mm])[0])
                ws = np.mean(ws, 0)
                line += f" | {ws[0]:.2f}, {ws[1]:.2f}"
            print(line)

    # ---- 3. pilot thermal observables on the pilot's own graphs and evidence
    print(f"\n3. Pilot thermal observables on the pilot's own graphs and evidence: pilot | surrogate mean, s.d. and"
          f"\n   z = (pilot - mean) / s.d. over {n_sim} simulated replica pairs. late = rounds 6-10; repeat_late = fraction"
          "\n   of exactly repeated answers; absds = mean |s(t) - s(t-1)|; D = replica distance; m = truth-aligned mean.")
    zs = {}
    for c in cells:
        S, J, hp, truth = runs[id(c)]
        K, T = c["K"], c["T"]
        dat = observables(S, truth)
        tg1 = fit["gains_transfer"][f"inverted|K{K}"]["one_shot"]
        sims = {"v1 (design_lib law), transfer gains":
                [observables(simulate_v1_on(J, hp, tg1["a_plus"], tg1["a_minus"], T, rng), truth) for _ in range(n_sim)]}
        for name, (lw, g1, g0) in variants(fit, K).items():
            sims[name] = [observables(simulate_v2(J, hp, g1, g0, lw, T, rng), truth) for _ in range(n_sim)]
        print(f"   inverted K={K} gamma={c['gamma']:+.0f}")
        print("     " + " " * 36 + "".join(f"{k:>12s}" for k in OBS))
        print(f"     {'pilot':36s}" + "".join(f"{dat[k]:12.3f}" for k in OBS))
        for name, ss in sims.items():
            mean = {k: np.mean([x[k] for x in ss]) for k in OBS}
            sd = {k: np.std([x[k] for x in ss]) + 1e-12 for k in OBS}
            print(f"     {name:36s}" + "".join(f"{mean[k]:12.3f}" for k in OBS))
            print(f"     {'  s.d.':36s}" + "".join(f"{sd[k]:12.3f}" for k in OBS))
            print(f"     {'  z':36s}" + "".join(f"{np.clip((dat[k] - mean[k]) / sd[k], -99, 99):12.1f}" for k in OBS))
            zs.setdefault(name, []).append([np.clip((dat[k] - mean[k]) / sd[k], -99, 99) for k in OBS])
    print("\n   root-mean-square z over the six cells, per observable (z clipped at 99)")
    print("     " + " " * 36 + "".join(f"{k:>12s}" for k in OBS))
    for name, z in zs.items():
        print(f"     {name:36s}" + "".join(f"{v:12.1f}" for v in np.sqrt(np.mean(np.square(z), 0))))

    # ---- 4. gamma dependence of c2 at fixed K: the one reciprocity signature in the thermal observables
    print("\n4. c2(gamma=+1) - c2(gamma=-1) and c2(+1) - c2(0): pilot (one graph each) | v2, pilot gains: mean (s.d. over pairs)")
    for K in (2, 4):
        lw, g1, g0 = variants(fit, K)["v2, pilot gains"]
        c2 = {}
        for c in cells:
            if c["K"] == K:
                S, J, hp, truth = runs[id(c)]
                c2[c["gamma"]] = (observables(S, truth)["c2"],
                                  np.array([observables(simulate_v2(J, hp, g1, g0, lw, c["T"], rng), truth)["c2"] for _ in range(n_sim)]))
        for a, b in ((1.0, -1.0), (1.0, 0.0)):
            dsim = c2[a][1] - c2[b][1]
            print(f"   K={K} c2({a:+.0f}) - c2({b:+.0f}): pilot {c2[a][0] - c2[b][0]:+.3f} | {dsim.mean():+.3f} ({dsim.std():.3f})")

    # ---- 5. overlap excess dq(T) = q(same graph) - q(other graph, same evidence): the pinning by the couplings
    print("\n5. Overlap excess dq at rounds 6-10 (design_lib.overlap_excess, two graphs per evidence draw, gamma = 0):"
          "\n   v2, pilot gains: mean (s.d. over graphs); graphs for 3 s.e.")
    from envs.evidence_game import make_game
    from envs.graphs import fixed_indegree_signed_couplings
    for K in (2, 4):
        lw, g1, g0 = variants(fit, K)["v2, pilot gains"]
        dq = []
        for _ in range(n_sim):
            hp = make_game(64, mu=0.5, sigma=1.0, k=3, seed=int(rng.integers(2**31))).h_priv
            init = round0(hp, rng)
            pair = [simulate_v2(fixed_indegree_signed_couplings(64, K, 0.0, 0.5, rng), hp, g1, g0, lw, 10, rng, init=init)
                    for _ in range(2)]
            dq.extend(np.mean(x[LATE]) for x in dl.overlap_excess(pair))
        dq = np.array(dq)
        print(f"   K={K}: dq = {dq.mean():+.4f} ({dq.std():.4f}); {int(np.ceil((3 * dq.std() / abs(dq.mean())) ** 2))} graphs")

    # ---- 6. regime numbers quoted in theory/quiet_regime.md section 2
    print("\n6. Regime. Effective couplings from the one-shot gains (design_lib.effective), d = 0.78; critical d g from"
          "\n   max_theta |rho + (1 - rho) d g ((1+gamma) cos + i (1-gamma) sin)| = 1 (quiet_regime.md (B1)) at gamma_eff")
    th = np.linspace(0, 2 * np.pi, 20001)
    grid = np.linspace(0.01, 20, 8000)

    def crit(rho_, gam):
        return next(x for x in grid if np.abs(rho_ + (1 - rho_) * x * ((1 + gam) * np.cos(th) + 1j * (1 - gam) * np.sin(th))).max() > 1)

    rho_sw = law_swarm(fit, 2)["rho0"]
    for wording in ("inverted", "label"):
        for K in (2, 4):
            g1, gt = fit["gains_transfer"][f"{wording}|K{K}"]["one_shot"], fit["gains_transfer"][f"{wording}|K{K}"]["target"]
            for gam in (-1.0, 1.0):
                e, et = dl.effective(g1["a_plus"], g1["a_minus"], K, 0.5, gam), dl.effective(gt["a_plus"], gt["a_minus"], K, 0.5, gam)
                print(f"   {wording:8s} K={K} gamma={gam:+.0f}: Jbar {e['Jbar']:.2f}  g {e['g']:.3f}  gamma_eff {e['gamma_eff']:+.2f}  "
                      f"d g (1+|gamma_eff|) {0.78 * e['g'] * (1 + abs(e['gamma_eff'])):.2f} | target gains: d g {0.78 * et['g']:.2f}, "
                      f"critical d g {crit(0.0, e['gamma_eff']):.2f} (rho = 0), {crit(rho_sw, e['gamma_eff']):.2f} (rho = {rho_sw:.2f})")
    print("   P_delta against (1 - c2) / (1 - c1) in the pilot (identity for a stationary sequence):")
    for c in cells:
        o = observables(*[runs[id(c)][k] for k in (0, 3)])
        print(f"   K={c['K']} gamma={c['gamma']:+.0f}: P_delta {o['P_delta']:.2f}   (1-c2)/(1-c1) {(1 - o['c2']) / (1 - o['c1']):.2f}   1 + c1 {1 + o['c1']:.2f}")
    print("   Long runs of v2 (T = 30, 60 graphs per cell): replica distance D(10), D(30), and the largest real part of the"
          "\n   eigenvalues of d W (target gains, d from the final state)")
    for gname in ("transfer gains", "pilot gains"):
        for K in (2, 4):
            lw, g1, g0 = variants(fit, K)[f"v2, {gname}"]
            for gam in (-1.0, 0.0, 1.0):
                Ds, top = [], []
                for _ in range(60):
                    J = fixed_indegree_signed_couplings(64, K, gam, 0.5, rng)
                    hp = make_game(64, mu=0.5, sigma=1.0, k=3, seed=int(rng.integers(2**31))).h_priv
                    S = simulate_v2(J, hp, g1, g0, lw, 30, rng)
                    Ds.append(((S[0] - S[1]) ** 2).mean(1))
                    W = (g1["a_plus"] * (J > 0) - g1["a_minus"] * (J < 0)) / K
                    top.append(np.linalg.eigvals((1 - S[:, -1].mean(0) ** 2)[:, None] * W).real.max())
                Ds = np.array(Ds)
                print(f"   {gname:14s} K={K} gamma={gam:+.0f}: D(10) {Ds[:, 10].mean():.4f}  D(30) {Ds[:, 30].mean():.4f} "
                      f"(s.e. {Ds[:, 30].std() / np.sqrt(60):.4f})  max Re eig(d W) {np.mean(top):.2f}")


if __name__ == "__main__":
    {"fit": cmd_fit, "check": cmd_check}[sys.argv[1]]()
