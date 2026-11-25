"""Predictors for the swarm design (experiments/04_swarm_design/DESIGN.md).

The agent is the calibrated surrogate of experiments/03_calibration (two-channel, tanh), with two additions measured
there: sampling noise and a one-round memory of what the agent absorbed from its peers,

    x_i(t)   = (1/K) [ a_+ sum_{j in A_i} s_j(t) - a_- sum_{j in D_i} s_j(t) ] + a_p hpriv_i + c + kappa (s_i(t) - s0_i)
    s_i(t+1) = clip( tanh x_i(t) + sig_xi xi_i(t) ),      s0_i = tanh(a_p hpriv_i + c),   s_i(0) = clip(s0_i + sig_xi xi_i)

A_i / D_i are the agree / disagree in-neighbours (envs/graphs.py: J_ij = +1 / -1), so the coupling matrix is
W_ij = a_+/K for agree ties and -a_-/K for disagree ties (a_- < 0, the label wording, makes disagree ties attract).
The memory acts on the deviation from the evidence-only answer, so that the static probes (own last answer shown,
no effect) and the white-probe kernel R(lag) ~ R(1) (d kappa)^(lag-1) are both reproduced.

Two predictors:
    simulate()   the surrogate on actual fixed in-degree graphs (envs.graphs.fixed_indegree_signed_couplings),
                 synchronous or asynchronous, two replicas per graph sharing graph, evidence and round-0 answers
    dmft()       Gaussian single-site DMFT (theory/claim1.md eqs. (4)-(9)) with effective couplings
                     Jbar = p_+ a_+ - p_- a_-,   g^2 = (p_+ a_+^2 + p_- a_-^2)/K,   Gamma = r <a_ij a_ji>_recip / K
                 and the fixed in-degree noise covariance g^2 C - (Jbar^2/K) m m (theory/claim5.md section 2)

Observables (theory/claim2.md section 5, on replica pairs a, b; delta = s^a - s^b):
    D(t)      = <delta_i(t)^2>,  amplification A = D / D_floor with D_floor = 2 sig_xi^2
    c_k       = thermal lag-k correlation, sum delta(t) delta(t-k) / sqrt(sum delta(t)^2 sum delta(t-k)^2), t = 3..T
    P_delta   = sum (delta(t) - delta(t-2))^2 / sum (delta(t) - delta(t-1))^2     (period index of the fluctuations)
    P_raw     = A_2 / A_1 on the raw beliefs (README test 2), t = 3..T
    dq(t)     = q_same-graph - q_other-graph (same evidence)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.evidence_game import make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402

A_P, C0 = 0.42, -0.016          # evidence gain and offset of the label fit (surrogate.py)
SIG_XI = 0.10                   # within-cell s.d. of replicate probe calls (surrogate.py: 0.097)
D_KAPPA = 0.30                  # R(2)/R(1) of the white-probe kernel (0.149/0.501, s.e. 0.09)
CLIP = 0.98                     # stated probabilities stay inside (0.01, 0.99)

# (a_+, a_-) per unit normalized field.  "small": fits dominated by |field| <= 0.2; "large": agree-only /
# disagree-only inboxes spanning +-0.8 (gains design).  The inverted wording was only probed in the gains design;
# its small-field values are the measured ones scaled by the label ratio 0.91/0.68 = 1.33 (an extrapolation).
GAINS = {
    ("label", "small"): (0.91, -0.33),
    ("label", "large"): (0.68, -0.30),
    ("inverted", "large"): (0.99, 0.57),
    ("inverted", "small"): (1.32, 0.76),
}


def kappa_for(d_mean):
    """kappa such that d * kappa equals the measured memory ratio at the mean local slope d."""
    return D_KAPPA / d_mean


def effective(a_plus, a_minus, K, p_neg, gamma):
    """Effective block parameters of a fixed in-degree graph with n_r = round(|gamma| K) reciprocated ties,
    concordant for gamma >= 0 and discordant for gamma < 0 (theory/claim1.md eq. (13))."""
    r = round(abs(gamma) * K) / K
    pp = 1 - p_neg
    Jbar = pp * a_plus - p_neg * a_minus
    a2 = pp * a_plus**2 + p_neg * a_minus**2
    aa = a2 if gamma >= 0 else -a_plus * a_minus          # <a_ij a_ji> over reciprocated pairs
    return dict(Jbar=Jbar, g=np.sqrt(a2 / K), Gamma=r * aa / K, gamma_eff=r * aa / a2, r=r)


def clip(s):
    return np.clip(s, -CLIP, CLIP)


def simulate(N, K, gamma, p_neg, a_plus, a_minus, T=10, mode="sync", n_graphs=100, n_games=None, seed=0,
             kappa=None, sig_xi=SIG_XI, mu=0.5, sigma=1.0, graphs_per_game=2):
    """Surrogate on actual graphs. Each game (evidence draw) is run on `graphs_per_game` graphs, two replicas each.
    Returns per-graph observables (arrays over graphs) and the mean trajectories."""
    rng = np.random.default_rng(seed)
    n_games = n_games or n_graphs // graphs_per_game
    out = {k: [] for k in ("m", "Cd", "C2", "C0", "q", "D", "c1", "c2", "P_delta", "P_raw", "dq", "radius", "angle",
                           "d_mean", "mean_mode")}
    for gi in range(n_games):
        game = make_game(N, mu=mu, sigma=sigma, k=3, seed=int(rng.integers(2**31)), truth=1)
        hp = game.h_priv
        s0 = np.tanh(A_P * hp + C0)
        init = clip(s0 + sig_xi * rng.standard_normal(N))            # round-0 answers, shared by all runs of the game
        runs = []
        for g_ in range(graphs_per_game):
            J = fixed_indegree_signed_couplings(N, K, gamma, p_neg, rng)
            W = (a_plus * (J > 0) - a_minus * (J < 0)) / K
            kap = kappa if kappa is not None else kappa_for(np.mean(1 - s0**2))
            S = np.empty((2, T + 1, N)); S[:, 0] = init
            for rep in range(2):
                s = init.copy()
                for t in range(T):
                    if mode == "sync":
                        s = clip(np.tanh(W @ s + A_P * hp + C0 + kap * (s - s0)) + sig_xi * rng.standard_normal(N))
                    else:                                             # one agent at a time, fresh random order
                        for i in rng.permutation(N):
                            s[i] = clip(np.tanh(W[i] @ s + A_P * hp[i] + C0 + kap * (s[i] - s0[i]))
                                        + sig_xi * rng.standard_normal())
                    S[rep, t + 1] = s
            runs.append(S)
            for k, v in thermal_observables(S).items():
                out[k].append(v)
            # linear stability of the noise-free map at the replica-mean final state: delta s' = D (W + kappa) delta s
            d = 1 - np.clip(S[:, -1].mean(0), -CLIP, CLIP) ** 2
            ev = np.linalg.eigvals(d[:, None] * (W + kap * np.eye(N)))
            top = ev[np.argmax(np.abs(ev))]
            out["radius"].append(abs(top)); out["angle"].append(abs(np.degrees(np.angle(top))))
            out["d_mean"].append(d.mean())
            out["mean_mode"].append(d.mean() * (kap + W.sum(1).mean()))
        out["dq"].extend(overlap_excess(runs))
    return {k: np.array(v) for k, v in out.items()}


def thermal_observables(S):
    """Per-graph observables of one run S, shaped (2 replicas, T+1 rounds, N agents), in the definitions of the
    module docstring (theory/claim2.md section 5). Shared with the analysis of the real swarms
    (experiments/05_swarm/analyze.py), so predictions and data use identical formulas. NaN-free input assumed;
    the swarm analysis fills missing beliefs with the carried-forward state first."""
    T = S.shape[1] - 1
    de = S[0] - S[1]
    num = lambda k: np.sum(de[3:] * de[3 - k:T + 1 - k])
    den = lambda k: np.sqrt(np.sum(de[3:] ** 2) * np.sum(de[3 - k:T + 1 - k] ** 2))
    A1 = np.mean((S[:, 3:] - S[:, 2:-1]) ** 2); A2 = np.mean((S[:, 3:] - S[:, 1:-2]) ** 2)
    return dict(m=S.mean((0, 2)), Cd=(S**2).mean((0, 2)),
                C2=np.r_[np.nan, np.nan, (S[:, 2:] * S[:, :-2]).mean((0, 2))], C0=(S * S[:, :1]).mean((0, 2)),
                q=(S[0] * S[1]).mean(1), D=(de**2).mean(1), c1=num(1) / den(1), c2=num(2) / den(2),
                P_delta=np.sum((de[3:] - de[1:-2]) ** 2) / np.sum((de[3:] - de[2:-1]) ** 2), P_raw=A2 / A1)


def overlap_excess(runs):
    """dq(t) for each run in `runs` (runs sharing one evidence draw, each on its own graph, each (2, T+1, N)):
    the replica overlap on the same graph minus the mean cross-overlap with the runs on the other graphs."""
    out = []
    for a in range(len(runs)):
        others = [np.mean([(runs[a][x] * runs[b][y]).mean(1) for x in (0, 1) for y in (0, 1)], 0)
                  for b in range(len(runs)) if b != a]
        out.append((runs[a][0] * runs[a][1]).mean(1) - np.mean(others, 0))
    return out


def _cond_gauss(Kc, past, z):
    """eta(t) given eta(<t) for covariance Kc (pseudo-inverse: Kc is near-singular when states persist)."""
    t = Kc.shape[0] - 1
    if t == 0:
        return np.sqrt(max(Kc[0, 0], 0.0)) * z
    w = np.linalg.pinv(Kc[:t, :t], rcond=1e-10, hermitian=True) @ Kc[:t, t]
    return w @ past + np.sqrt(max(Kc[t, t] - Kc[:t, t] @ w, 0.0)) * z


def dmft(K, gamma, p_neg, a_plus, a_minus, T=10, M=20000, copies=4, seed=0, kappa=None, sig_xi=SIG_XI,
         mu=0.5, sigma=1.0, shared_couplings=True, fixed_indegree=True):
    """Gaussian single-site DMFT with nested thermal copies (theory/solver_spec.md 1.1 item 4), one population.

    Each of M sites has `copies` replicas sharing hpriv, the round-0 answer and (if shared_couplings) the frozen
    part of the cavity noise; shared_couplings=False gives the other-graph baseline of dq. Returns m, C, q, R and
    the thermal observables in the same definitions as simulate().
    """
    rng = np.random.default_rng(seed)
    e = effective(a_plus, a_minus, K, p_neg, gamma)
    Jbar, g2, Gam = e["Jbar"], e["g"] ** 2, e["Gamma"]
    mm_w = Jbar**2 / K if fixed_indegree else 0.0                    # claim5 section 2: g^2 C - (Jbar^2/K) m m
    hp = make_game(M, mu=mu, sigma=sigma, k=3, seed=seed, truth=1).h_priv
    s0 = np.tanh(A_P * hp + C0)
    kap = kappa if kappa is not None else kappa_for(np.mean(1 - s0**2))
    n = T + 1
    s = np.zeros((n, M, copies)); s[0] = clip(s0 + sig_xi * rng.standard_normal(M))[:, None]
    tan = np.zeros((n, T, M, copies))
    eta_f = np.zeros((T, M)); eta_th = np.zeros((T, M, copies))
    m = np.zeros(n); C = np.zeros((n, n)); q = np.zeros((n, n)); R = np.zeros((n, n))

    def moments(t):
        m[t] = s[t].mean()
        C[t, :t + 1] = C[:t + 1, t] = np.einsum("umc,mc->u", s[:t + 1], s[t]) / (M * copies)
        tot = s[:t + 1].sum(-1)
        q[t, :t + 1] = q[:t + 1, t] = (tot @ tot[t] - np.einsum("umc,mc->u", s[:t + 1], s[t])) / (M * copies * (copies - 1))

    moments(0)
    for t in range(T):
        mm = np.outer(m[:t + 1], m[:t + 1])
        Kf = g2 * q[:t + 1, :t + 1] - mm_w * mm if shared_couplings else np.zeros((t + 1, t + 1))
        Kth = g2 * C[:t + 1, :t + 1] - mm_w * mm - Kf
        eta_f[t] = _cond_gauss(Kf, eta_f[:t], rng.standard_normal(M))
        eta_th[t] = _cond_gauss(Kth, eta_th[:t].reshape(t, M * copies), rng.standard_normal(M * copies)).reshape(M, copies)
        h = Jbar * m[t] + eta_f[t][:, None] + eta_th[t] + Gam * np.tensordot(R[t, :t], s[:t], axes=1)   # claim1 (4)
        p = np.tanh(h + (A_P * hp + C0)[:, None] + kap * (s[t] - s0[:, None]))
        d = 1 - p**2
        dh = Gam * np.tensordot(R[t, :t], tan[:t], axes=1)
        dh[t] += 1.0
        tan[t + 1] = d * (dh + kap * tan[t])                         # tangent response (solver_spec 1.1 item 3)
        s[t + 1] = clip(p + sig_xi * rng.standard_normal((M, copies)))
        moments(t + 1)
        R[t + 1, :T] = tan[t + 1].mean((1, 2))
    # thermal observables from copy pairs (0, 1)
    de = s[:, :, 0] - s[:, :, 1]
    num = lambda k: np.sum(de[3:] * de[3 - k:T + 1 - k]); den = lambda k: np.sqrt(np.sum(de[3:] ** 2) * np.sum(de[3 - k:T + 1 - k] ** 2))
    A1 = np.mean((s[3:] - s[2:-1]) ** 2); A2 = np.mean((s[3:] - s[1:-2]) ** 2)
    # claim2 (8): Lam = max_omega g^2 <|R_i(omega)|^2>, per-site responses from two halves of the copies (unbiased)
    Ri = tan[T, ::-1]                                                # lags 1..T, (T, M, copies)
    h1, h2 = Ri[:, :, : copies // 2].mean(-1), Ri[:, :, copies // 2:].mean(-1)
    lam = {}
    for name, om in (("0", 0.0), ("pi/2", np.pi / 2), ("pi", np.pi)):
        ph = np.exp(-1j * om * np.arange(1, T + 1))
        lam[name] = float(g2 * np.mean(((ph @ h1) * np.conj(ph @ h2)).real))
    return dict(eff=e, kappa=kap, m=m, C=C, q=q, R=R, D=2 * (np.diag(C) - np.diag(q)), c1=num(1) / den(1),
                c2=num(2) / den(2), P_delta=np.sum((de[3:] - de[1:-2]) ** 2) / np.sum((de[3:] - de[2:-1]) ** 2),
                P_raw=A2 / A1, Lam=lam, d_mean=float((1 - s[T] ** 2).mean()))
