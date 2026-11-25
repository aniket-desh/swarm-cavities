"""Checks for theory/quiet_regime.md: the kick-and-echo formulas on swarms where the answer is known.

Kick: at round t0 the readers of agent i see its message mirrored, s_i -> -s_i (kick D_i = -2 s_i(t0)); i's own
history stays true. With W_jk = a_+/K for an agree tie (J_jk = +1) and -a_-/K for a disagree tie (J_jk = -1), and
d_j = 1 - s_j^2 the slope of the tanh target,

    first order   ds_j(t0+1) = d_j W_ji D_i                                          (K1)
    echo          ds_i(t0+2) = d_i sum_{j: J_ij != 0, J_ji != 0} W_ij d_j W_ji D_i     (K2)
    slopes        r(type) = sum ds_j D_i / sum D_i^2,   E = sum ds_i(t0+2) D_i / sum D_i^2
    cavity        E = Gamma <d_i> R,   Gamma = r <a_ij a_ji>_recip / K,  R = <d_j>    (K3)   (claim1.md eq. (6))
    on the graph  E = sum_types (n_type / K^2) a_ij a_ji D2_type,                     (K4)
                  D2_type = sum_i D_i^2 d_i <d_j>_{j recip. of type} / sum_i D_i^2
    factorization pred_i = d_i sum_{j recip.} c(J_ij) ds_j(t0+1),  slope F of ds_i(t0+2) on pred_i = 1     (K5)
                  with c(type) = sum ds_j d_j D_i / sum d_j^2 D_i^2 the first-order response per unit slope factor.
                  The raw version, pred_i = sum rbar(J_ij) ds_j with rbar = r(type), has slope <d_i>_w / <d_j>
                  below 1 even when responses compose exactly, because the estimators weight kicked agents by
                  D_i^2 = 4 s_i^2 and confident agents have small d_i = 1 - s_i^2.

Parts:
  A  engine conventions: envs.swarm with the deterministic MockBackend; the twin built from Swarm._context
     (and envs.kick.kick_twin when that module is present) against a direct numpy twin.
  B  (K1)-(K4) on the mock: per-agent formulas at a small kick, population slopes, sign pattern, finite mirror kick.
  C  (K5) and additivity over reciprocated ties, E(|gamma| = 0.5) = E(|gamma| = 1) / 2.
  D  anchoring moves the oscillatory boundary: spectral radius of rho + (1 - rho) d W against the formula.
  E  the same protocol on surrogate v2 (noise, exact repeats, replay rule): slopes, noise per echo, kicks needed.
  F  frozen predictions (one-shot real gains, slope factors from settled states) with propagated uncertainty.

Run: uv run python theory/checks/check_kick_echo.py [A B C D E F]   (default all; output in out_kick_echo.txt)
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "04_swarm_design"))
import design_lib as dl  # noqa: E402
import surrogate_v2 as sv  # noqa: E402
from envs.evidence_game import make_game  # noqa: E402
from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402
from envs.swarm import MockBackend, Swarm, SwarmConfig, _ReplicaState, run_swarm  # noqa: E402

N, T, P_NEG = 64, 10, 0.5
GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
slope = lambda y, x: float(np.sum(y * x) / np.sum(x * x))


def W_of(J, a_plus, a_minus):
    K = int((J[0] != 0).sum())
    return (a_plus * (J > 0) - a_minus * (J < 0)) / K


# ------------------------------------------------------------------ deterministic mock: s(t+1) = tanh(W s + hp)

def mock_run(W, hp, T=T):
    S = np.empty((T + 1, len(hp)))
    S[0] = np.tanh(hp)
    for t in range(T):
        S[t + 1] = np.tanh(W @ S[t] + hp)
    return S


def mock_twin(S, W, hp, i, t0, eta=1.0):
    """Exact twin of the mock for a kick eta * (-2 s_i(t0)) seen by the readers of i. Returns the kick and the
    changes (twin - base) at t0+1 and t0+2, each (N,)."""
    pub = S[t0].copy()
    D = -2 * eta * S[t0, i]
    pub[i] += D
    tw1 = np.tanh(W @ pub + hp)                      # W_ii = 0, so i itself is unchanged
    tw2 = np.tanh(W @ tw1 + hp)                      # i's round-(t0+1) message is its true one again
    return D, tw1 - S[t0 + 1], tw2 - S[t0 + 2]


def mock_cell(K, gamma, a_plus, a_minus, eta, n_graphs=20, seed=0, t0=8, mu=0.19, sigma=0.385):
    """Kick every agent of n_graphs mock swarms once. Evidence fields are scaled to a_p * hpriv (a_p ~ 0.385).
    Returns per-kick arrays and the per-agent formula values."""
    rng = np.random.default_rng([seed, K, int(10 * gamma) + 20])
    out = {k: [] for k in ("D", "echo", "echo_lin", "r_exact", "r_lin", "type", "pred_in", "d_i", "d_j_all", "d_read", "D_read")}
    for g in range(n_graphs):
        J = fixed_indegree_signed_couplings(N, K, gamma, P_NEG, rng)
        W = W_of(J, a_plus, a_minus)
        hp = make_game(N, mu=mu, sigma=sigma, k=3, seed=int(rng.integers(2**31))).h_priv
        S = mock_run(W, hp)
        d1, d2 = 1 - S[t0 + 1] ** 2, 1 - S[t0 + 2] ** 2
        out["d_j_all"].append(d1)
        for i in range(N):
            D, ds1, ds2 = mock_twin(S, W, hp, i, t0, eta)
            readers = np.flatnonzero(J[:, i])
            assert ds1[i] == 0.0 and np.all(ds1[np.setdiff1d(np.arange(N), readers)] == 0.0)
            recip = np.flatnonzero((J[i] != 0) & (J[:, i] != 0))
            out["D"].append(D)
            out["echo"].append(ds2[i])
            out["echo_lin"].append(d2[i] * np.sum(W[i, recip] * d1[recip] * W[recip, i]) * D)          # (K2)
            out["r_exact"].extend(ds1[readers] / D)
            out["r_lin"].extend(d1[readers] * W[readers, i])                                              # (K1)
            out["type"].extend(J[readers, i])
            out["d_read"].extend(d1[readers])
            out["D_read"].extend([D] * len(readers))
            out["pred_in"].append((J[i, recip], ds1[recip]))
            out["d_i"].append(d2[i])
    return {k: (np.array(v) if k not in ("pred_in", "d_j_all") else v) for k, v in out.items()}


def effective_Gamma(K, gamma, a_plus, a_minus):
    return dl.effective(a_plus, a_minus, K, P_NEG, gamma)["Gamma"]


# ------------------------------------------------------------------ Part A

def part_A():
    print("A. Engine conventions (envs.swarm, MockBackend deterministic, N=64, T=10, K=4, gamma=+1, t0=6)")
    K, gamma, t0 = 4, 1.0, 6
    a_plus, a_minus = 0.66, 0.22
    rng = np.random.default_rng(5)
    J = fixed_indegree_signed_couplings(N, K, gamma, P_NEG, rng)
    game = make_game(N, mu=0.19, sigma=0.385, k=3, seed=11)
    backend = MockBackend(beta=1.0, b_plus=a_plus / K, b_minus=a_minus / K)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SwarmConfig("mock-kick", T=T, H=1, replicas=1, results_root=Path(tmp))
        out = run_swarm(J, game, cfg, backend)
        sw = Swarm(J, game, cfg, backend)
        S_engine = out["s"][0]
        W, hp = W_of(J, a_plus, a_minus), game.h_priv
        S = mock_run(W, hp)
        print(f"   numpy map against envs.swarm run: max |difference| = {np.abs(S - S_engine).max():.1e}")

        def engine_twin(i):
            """The twin through Swarm._context: readers see the mirrored p_i, histories stay true."""
            st = _ReplicaState(N)
            for t in range(t0 + 1):
                st.p = out["p"][0, t].copy()
                st.msg = [f"My estimate for HEADS is {p:.2f}." for p in st.p]
                for k in range(N):
                    st.hist[k].append((float(st.p[k]), st.msg[k]))
            st.p[i] = 1 - st.p[i]
            tw = []
            for t in (t0 + 1, t0 + 2):
                res = asyncio.run(backend.respond(0, t, [sw._context(st, 0, t, k) for k in range(N)], cfg.H))
                for k, r in enumerate(res):
                    st.apply(k, r)
                tw.append(2 * st.p.copy() - 1)
            return np.array(tw)

        worst, worst_kick = 0.0, None
        for i in range(N):
            D, ds1, ds2 = mock_twin(S, W, hp, i, t0)
            tw = engine_twin(i)
            worst = max(worst, np.abs(tw[0] - S[t0 + 1] - ds1).max(), np.abs(tw[1] - S[t0 + 2] - ds2).max())
            assert tw[0][i] == S_engine[t0 + 1][i], "kicked agent changed at t0+1"
            assert set(np.flatnonzero(tw[0] != S_engine[t0 + 1])) <= set(np.flatnonzero(J[:, i])), "non-reader changed"
        print(f"   twin through Swarm._context against numpy twin, all 64 kicks: max |difference| = {worst:.1e}")
        print("   kicked agent unchanged at t0+1 and only its readers changed: True for all 64 kicks")
        try:
            from envs import kick as ek
            rows = ek.base_rows(Path(tmp) / "mock-kick")
            for i in (0, 17, 40):
                tw = asyncio.run(ek.kick_twin(sw, rows, 0, t0, [i]))
                D, ds1, ds2 = mock_twin(S, W, hp, i, t0)
                dp = 2 * (tw["twin_p"] - tw["base_p"])
                err = max(np.abs(dp[0] - ds1).max(), np.abs(dp[1] - ds2).max())
                # envs.kick shows the mirrored probability with two decimals, as the prompt does
                worst_kick = max(worst_kick or 0.0, err)
            print(f"   envs.kick.kick_twin against numpy twin (3 kicks): max |difference| = {worst_kick:.1e} "
                  "(its mirrored p is rounded to two decimals)")
        except Exception as e:                                # the module belongs to another work stream
            print(f"   envs.kick not checked ({type(e).__name__}: {e})")
    # gamma = 0: echo exactly zero
    J0 = fixed_indegree_signed_couplings(N, K, 0.0, P_NEG, rng)
    W0 = W_of(J0, a_plus, a_minus)
    S0 = mock_run(W0, hp)
    e0 = max(abs(mock_twin(S0, W0, hp, i, t0)[2][i]) for i in range(N))
    print(f"   gamma = 0: max |echo| over 64 kicks = {e0:.1e} (exactly zero: no reciprocated tie)")


# ------------------------------------------------------------------ Parts B, C

MOCK_GAINS = {"inverted-like": {2: (0.80, 0.15), 4: (0.66, 0.22)}, "label-like": {2: (0.54, -0.24), 4: (0.40, -0.21)}}


def part_B():
    print("\nB. Linear-response formulas on the mock (20 graphs x 64 kicks per cell, t0=8; gains as the real one-shot gains)")
    print("   per-agent check at a small kick (eta = 0.01): max relative error of (K1) and (K2) over all kicks")
    worst1 = worst2 = 0.0
    for name, gs in MOCK_GAINS.items():
        for K in (2, 4):
            for gamma in GAMMAS:
                c = mock_cell(K, gamma, *gs[K], eta=0.01, n_graphs=4)
                big = np.abs(c["D"]) > 1e-4
                worst1 = max(worst1, np.max(np.abs(c["r_exact"] / c["r_lin"] - 1)))
                if gamma != 0:
                    worst2 = max(worst2, np.max(np.abs(c["echo"][big] / c["echo_lin"][big] - 1)))
                else:
                    assert np.all(c["echo"] == 0)
    print(f"     (K1) first order {worst1:.1e}   (K2) echo {worst2:.1e}   (both O(eta); echo identically 0 at gamma = 0)")
    print("   population slopes. E_small: exact twin, eta=0.01; E_graph: (K2) summed on the graph; E_cav: (K3) with"
          "\n   <d_i> weighted by D^2 and R = <d_j>; E_mirror: exact twin at the full mirror kick (eta = 1)")
    print(f"   {'gains':14s} {'K':>2s} {'gamma':>5s} {'Gamma':>8s} {'E_small':>9s} {'E_graph':>9s} {'E_cav':>9s} {'E_mirror':>9s}"
          f" {'mirror/small':>12s} {'r_A':>7s} {'r_D':>7s}")
    for name, gs in MOCK_GAINS.items():
        for K in (2, 4):
            for gamma in GAMMAS:
                a_plus, a_minus = gs[K]
                c = mock_cell(K, gamma, a_plus, a_minus, eta=0.01)
                cm = mock_cell(K, gamma, a_plus, a_minus, eta=1.0)
                Gam = effective_Gamma(K, gamma, a_plus, a_minus)
                E_small, E_graph = slope(c["echo"], c["D"]), slope(c["echo_lin"], c["D"])
                d_w = np.sum(c["D"] ** 2 * c["d_i"]) / np.sum(c["D"] ** 2)
                E_cav = Gam * d_w * np.mean(np.concatenate(c["d_j_all"]))
                E_mir = slope(cm["echo"], cm["D"])
                # first-order slopes at the mirror kick, by edge type (each kick has its readers in order)
                rA = np.mean(cm["r_exact"][cm["type"] > 0]) if np.any(cm["type"] > 0) else np.nan
                rD = np.mean(cm["r_exact"][cm["type"] < 0]) if np.any(cm["type"] < 0) else np.nan
                ratio = E_mir / E_small if gamma != 0 else np.nan
                print(f"   {name:14s} {K:2d} {gamma:+5.1f} {Gam:+8.4f} {E_small:+9.4f} {E_graph:+9.4f} {E_cav:+9.4f} {E_mir:+9.4f}"
                      f" {ratio:12.2f} {rA:+7.3f} {rD:+7.3f}")


def part_C():
    print("\nC. Factorization (K5) and additivity on the mock (20 graphs x 64 kicks per cell)."
          "\n   F_raw: slope with type-mean first-order responses; <d_i>_w/<d_j>: its predicted value; F: with slope factors (K5)")
    print(f"   {'gains':14s} {'K':>2s} {'gamma':>5s} {'F_raw eta=.01':>13s} {'<d_i>_w/<d_j>':>13s} {'F eta=.01':>9s} {'F_raw mirror':>12s}"
          f" {'F mirror':>8s} {'E(gamma)/E(sign)':>17s}")
    for name, gs in MOCK_GAINS.items():
        for K in (2, 4):
            E, rows = {}, []
            for gamma in (-1.0, -0.5, 0.5, 1.0):
                res = []
                for eta in (0.01, 1.0):
                    c = mock_cell(K, gamma, *gs[K], eta=eta)
                    ds_read = c["r_exact"] * c["D_read"]
                    rbar = {ty: slope(ds_read[c["type"] == ty], c["D_read"][c["type"] == ty]) for ty in (1, -1)}
                    cty = {ty: slope(ds_read[c["type"] == ty], (c["d_read"] * c["D_read"])[c["type"] == ty]) for ty in (1, -1)}
                    pred = np.array([sum(rbar[int(ty)] * x for ty, x in zip(types, ds)) for types, ds in c["pred_in"]])
                    pred_d = c["d_i"] * np.array([sum(cty[int(ty)] * x for ty, x in zip(types, ds)) for types, ds in c["pred_in"]])
                    res += [slope(c["echo"], pred), slope(c["echo"], pred_d)]
                    if eta == 0.01:
                        bias = (np.sum(c["D"] ** 2 * c["d_i"]) / np.sum(c["D"] ** 2)) / (np.sum(c["D_read"] ** 2 * c["d_read"]) / np.sum(c["D_read"] ** 2))
                    else:
                        E[gamma] = slope(c["echo"], c["D"])
                rows.append((gamma, res, bias))
            for gamma, res, bias in rows:
                add = E[gamma] / E[np.sign(gamma)] if abs(gamma) == 0.5 else np.nan
                print(f"   {name:14s} {K:2d} {gamma:+5.1f} {res[0]:13.3f} {bias:13.3f} {res[1]:9.3f} {res[2]:12.3f} {res[3]:8.3f} {add:17.3f}")
    print("   (F = 1: responses compose; E(+-0.5)/E(+-1) = 0.5: echoes add over reciprocated ties)")


# ------------------------------------------------------------------ Part D

def part_D():
    print("\nD. Anchoring and the oscillatory boundary: delta(t+1) = rho delta(t) + (1 - rho) d W delta(t), Gaussian W with"
          "\n   variance g^2/N and reciprocity gamma (N=1500). Spectral radius: numerical | formula")
    Nn, rng = 1500, np.random.default_rng(0)
    from envs.graphs import gaussian_couplings
    print(f"   {'rho':>4s} {'gamma':>5s} {'d g':>5s} {'numerical':>9s} {'formula':>8s}   critical d g (formula)")
    for rho_ in (0.0, 0.7):
        for gamma in (-1.0, -0.5, 0.0, 1.0):
            for dg in (0.45, 1.5):
                Wm = gaussian_couplings([Nn], 0.0, dg, gamma, rng)
                ev = np.linalg.eigvals(rho_ * np.eye(Nn) + (1 - rho_) * Wm)
                # elliptic law: eigenvalues of W fill x^2/(1+gamma)^2 + y^2/(1-gamma)^2 <= (d g)^2
                th = np.linspace(0, 2 * np.pi, 20001)
                lam = dg * ((1 + gamma) * np.cos(th) + 1j * (1 - gamma) * np.sin(th))
                rad = np.abs(rho_ + (1 - rho_) * lam).max()
                # boundary: rightmost point d g (1+gamma) = 1, or for gamma < 0 the top of the ellipse,
                # rho^2 + (1-rho)^2 (d g)^2 (1-gamma)^2 = 1  ->  d g (1 + |gamma|) = sqrt((1+rho)/(1-rho))
                grid = np.linspace(0.01, 20, 4000)
                crit = next(x for x in grid if np.abs(rho_ + (1 - rho_) * x * ((1 + gamma) * np.cos(th) + 1j * (1 - gamma) * np.sin(th))).max() > 1)
                print(f"   {rho_:4.1f} {gamma:+5.1f} {dg:5.2f} {np.abs(ev).max():9.3f} {rad:8.3f}   {crit:.2f}")
    print("   gamma = -1: critical d g = sqrt((1+rho)/(1-rho)) / 2 = %.3f at rho = 0.7 (0.5 at rho = 0)" % (np.sqrt(1.7 / 0.3) / 2))


# ------------------------------------------------------------------ Part E: surrogate v2

def kick_cell(K, gamma, gains, gains0, law, n_graphs, t0s, rng, second_order=True, s_min=0.1):
    """Base runs of surrogate v2 on fresh graphs and games; one twin per (graph, replica, t0, agent with
    |s_i(t0)| >= s_min). Returns per-kick records."""
    recs = []
    for g in range(n_graphs):
        J = fixed_indegree_signed_couplings(N, K, gamma, P_NEG, rng)
        hp = make_game(N, mu=0.5, sigma=1.0, k=3, seed=int(rng.integers(2**31))).h_priv
        S = sv.simulate_v2(J, hp, gains, gains0, law, T, rng)
        Wg = W_of(J, gains["a_plus"], gains["a_minus"])
        for rep in range(2):
            for t0 in t0s:
                for i in range(N):
                    if abs(S[rep, t0, i]) < s_min:
                        continue
                    tw = sv.kick_twin(S[rep], J, hp, gains, law, i, t0, rng, second_order=second_order, gains0=gains0)
                    readers = list(tw["ds1"])
                    d1, d2 = 1 - S[rep, t0 + 1] ** 2, 1 - S[rep, t0 + 2] ** 2
                    rc = [j for j in readers if J[i, j] != 0]
                    tw.update(d1=d1, d2=d2, lin=d2[i] * sum(Wg[i, j] * d1[j] * Wg[j, i] for j in rc) * tw["D"],   # (K2)
                              i=i, g=g, J_ji={j: int(J[j, i]) for j in readers},
                              recip=[(j, int(J[i, j])) for j in readers if J[i, j] != 0],
                              open=[(k, [(j, int(J[k, j])) for j in readers if J[k, j] != 0]) for k in tw["ds2"]
                                    if k not in readers])
                    recs.append(tw)
    return recs


def kick_stats(recs, n_sub=200, rng=None):
    D = np.array([r["D"] for r in recs])
    echo = np.array([r["echo"] for r in recs])
    out = {"n": len(recs), "D2": float(np.mean(D**2))}
    rbar, cty = {}, {}
    for ty, name in ((1, "r_A"), (-1, "r_D")):
        v = np.array([(r["ds1"][j], r["D"], r["d1"][j]) for r in recs for j in r["ds1"] if r["J_ji"][j] == ty]).reshape(-1, 3)
        rbar[ty] = slope(v[:, 0], v[:, 1]) if len(v) else np.nan
        cty[ty] = slope(v[:, 0], v[:, 1] * v[:, 2]) if len(v) else np.nan       # response per unit slope factor
        out[name] = rbar[ty]
    vA = np.array([(r["ds1"][j], r["D"]) for r in recs for j in r["ds1"] if r["J_ji"][j] == 1]).reshape(-1, 2)
    if len(vA) > 30:                                                            # agree-tie response by kick size
        q1, q2 = np.quantile(np.abs(vA[:, 1]), [1 / 3, 2 / 3])
        out["r_A_small"] = slope(*vA[np.abs(vA[:, 1]) <= q1].T)
        out["r_A_large"] = slope(*vA[np.abs(vA[:, 1]) >= q2].T)
        out["D_terciles"] = (float(q1), float(q2))
    out["E"] = slope(echo, D)
    out["E_lin"] = slope(np.array([r["lin"] for r in recs]), D)                 # (K2) with the gains passed, no stickiness
    res = echo - out["E"] * D
    out["sig_y"] = float(res.std())
    out["zero_frac"] = float(np.mean(echo == 0))
    out["kurt"] = float(np.mean(res**4) / np.mean(res**2) ** 2) if np.any(res != 0) else np.nan
    out["se1"] = float(np.sqrt(np.sum(D**2 * res**2)) / np.sum(D**2) * np.sqrt(len(D)))     # s.e. of E times sqrt(n)
    out["n3"] = 9 * out["se1"] ** 2 / out["E"] ** 2 if out["E"] != 0 else np.inf           # kicks for |E| = 3 s.e.
    if rng is not None and len(D) >= 4 * n_sub:                                            # check of the s.e. formula
        Es = [slope(echo[ix], D[ix]) for ix in (rng.choice(len(D), n_sub, replace=False) for _ in range(400))]
        out["se_sub"] = float(np.std(Es) * np.sqrt(n_sub))
    pred = np.array([sum(rbar[ty] * r["ds1"][j] for j, ty in r["recip"]) for r in recs])
    pred_d = np.array([r["d2"][r["i"]] * sum(cty[ty] * r["ds1"][j] for j, ty in r["recip"]) for r in recs])
    if np.any(pred != 0):
        out["F_raw"] = slope(echo, pred)                                                   # raw factorization slope
        out["F"] = slope(echo, pred_d)                                                     # (K5), closed paths
        resF = echo - out["F"] * pred_d
        out["F_se1"] = float(np.sqrt(np.sum(pred_d**2 * resF**2)) / np.sum(pred_d**2) * np.sqrt(len(pred_d)))
    op = np.array([(r["ds2"][k], r["d2"][k] * sum(cty[ty] * r["ds1"][j] for j, ty in senders))
                   for r in recs for k, senders in r["open"]]).reshape(-1, 2)
    if len(op):
        out["F_open"] = slope(op[:, 0], op[:, 1])                                          # same regression, open paths
        resO = op[:, 0] - out["F_open"] * op[:, 1]
        out["F_open_se1"] = float(np.sqrt(np.sum(op[:, 1] ** 2 * resO**2)) / np.sum(op[:, 1] ** 2) * np.sqrt(len(recs)))
        out["n_open_per_kick"] = len(op) / len(recs)
    return out


def _erow(wording, gset, K, gamma, st, rho_inf):
    nF = (st["F_se1"] / 0.15) ** 2 if "F_se1" in st else np.nan
    att = st["E"] / ((1 - rho_inf) ** 2 * st["E_lin"]) if st["E_lin"] else np.nan
    se = st["se1"] / np.sqrt(st["n"])
    return (f"   {wording:8s} {gset:>1s} {K:2d} {gamma:+5.1f} {st['r_A']:+7.3f} {st['r_D']:+7.3f} {st['E']:+8.4f} {se:7.4f} {att:6.2f} {st['sig_y']:6.3f}"
            f" {st['kurt']:5.1f} {st['zero_frac']:5.2f} {st['D2']:6.2f} {st['n3']:6.0f} {st.get('F', np.nan):6.2f}"
            f" {st.get('F_open', np.nan):6.2f} {nF:5.0f} {st['se1']:6.3f}/{st.get('se_sub', np.nan):.3f}")


def part_E(n_graphs=32):
    fit = sv.load_fit()
    rho_inf = fit["law_white"]["rho_inf"]
    rng = np.random.default_rng(7)
    print("\nE. Kick protocol on surrogate v2 (swarm law; every agent whose prompt changed is resampled; settled t0 = 6, 8;"
          f"\n   {n_graphs} graphs x 2 replicas x 2 t0 x agents with |s_i| >= 0.1). E, r: slopes through the origin."
          "\n   se(E): its Monte Carlo s.e. here; E/lin: E over the linear-response value (K2) at one-shot gains (target"
          "\n   gains times (1 - rho_inf) per step);"
          "\n   sig_y: s.d. of one echo around E D; kurt: its kurtosis; zero: echoes that are exactly 0; n3: kicks for"
          "\n   |E| = 3 s.e.; F: factorization slope (K5) on closed paths, F_open: the same on open two-step paths;"
          "\n   nF: kicks for s.e.(F) = 0.15; se: sqrt(n) s.e.(E), sandwich formula / 400 random subsets of 200 kicks."
          "\n   Gains g: T = transfer (target form), P = pilot fit (inverted only).")
    hdr = (f"   {'wording':8s} {'g':>1s} {'K':>2s} {'gamma':>5s} {'r_A':>7s} {'r_D':>7s} {'E':>8s} {'se(E)':>7s} {'E/lin':>6s} {'sig_y':>6s} {'kurt':>5s} {'zero':>5s}"
           f" {'<D^2>':>6s} {'n3':>6s} {'F':>6s} {'F_open':>6s} {'nF':>5s} {'se':>12s}")
    print(hdr)
    table = {}
    for wording in ("inverted", "label"):
        for gset in ("T", "P"):
            if gset == "P" and wording == "label":
                continue
            for K in (2, 4):
                law = sv.law_swarm(fit, K)
                if gset == "T":
                    g1 = g0 = fit["gains_transfer"][f"{wording}|K{K}"]["target"]
                else:
                    g1, g0 = fit["gains_pilot"][f"inverted|K{K}|gains"], fit["gains_pilot"][f"inverted|K{K}|gains0"]
                for gamma in GAMMAS:
                    st = kick_stats(kick_cell(K, gamma, g1, g0, law, n_graphs, (6, 8), rng), rng=rng)
                    table[f"{wording}|{gset}|K{K}|g{gamma:+.1f}"] = st
                    print(_erow(wording, gset, K, gamma, st, rho_inf), flush=True)
    print("\n   Pooled over the |gamma| = 1 cells of both wordings (transfer gains), weighting by 1/s.e.^2:")
    for K in (2, 4):
        for key, sekey in (("F", "F_se1"), ("F_open", "F_open_se1")):
            v = [(table[f"{w}|T|K{K}|g{g:+.1f}"][key], table[f"{w}|T|K{K}|g{g:+.1f}"][sekey] / np.sqrt(table[f"{w}|T|K{K}|g{g:+.1f}"]["n"]))
                 for w in ("inverted", "label") for g in (-1.0, 1.0)]
            wts = np.array([1 / s**2 for _, s in v])
            print(f"     K={K} {key:6s} = {np.sum(wts * np.array([x for x, _ in v])) / wts.sum():.3f} +- {1 / np.sqrt(wts.sum()):.3f}")
    print("\n   Agree-tie first-order response by kick size (lowest and highest third of |D|), gamma = 0 cells:")
    for w in ("inverted", "label"):
        for K in (2, 4):
            st = table[f"{w}|T|K{K}|g+0.0"]
            print(f"     {w:8s} K={K}: |D| <= {st['D_terciles'][0]:.2f}: r_A = {st['r_A_small']:+.3f};  |D| >= {st['D_terciles'][1]:.2f}: r_A = {st['r_A_large']:+.3f};"
                  f"  ratio {st['r_A_small'] / st['r_A_large']:.2f}")
    print("\n   Early kicks (t0 = 0: the mirrored message is a round-0 sensor report; echo read at round 2), inverted, pilot gains."
          "\n   E/lin is printed against the settled formula with the later-round gains, so it is not an attenuation here.")
    print(hdr)
    for K in (2, 4):
        law = sv.law_swarm(fit, K)
        g1, g0 = fit["gains_pilot"][f"inverted|K{K}|gains"], fit["gains_pilot"][f"inverted|K{K}|gains0"]
        for gamma in (-1.0, 1.0):
            st = kick_stats(kick_cell(K, gamma, g1, g0, law, n_graphs, (0,), rng), rng=rng)
            table[f"inverted|P|K{K}|g{gamma:+.1f}|t0=0"] = st
            print(_erow("inverted", "P", K, gamma, st, rho_inf), flush=True)
    print("\n   Control: the same code with anchoring, repeats and noise switched off (rho = 0, pi_0 = 0, sigma = 0, one-shot"
          "\n   gains) is the mock of part B: E/E_lin must equal B's mirror/small ratio, F and F_open must be 1 up to the tanh"
          "\n   nonlinearity of a mirror kick, and sig_y must be the spread of slope factors only")
    law0 = dict(rho0=0.0, rho_inf=0.0, u0=1.0, sig0=0.0, sig1=0.0, pi0=0.0, mu_pi=1.0)
    for K, gamma in ((2, 1.0), (2, -1.0), (4, 1.0), (4, -1.0)):
        g = fit["gains_transfer"][f"inverted|K{K}"]["one_shot"]
        st = kick_stats(kick_cell(K, gamma, g, g, law0, 4, (8,), rng))
        print(f"   inverted K={K} gamma={gamma:+.0f}: E {st['E']:+.4f}, E_lin (K2) {st['E_lin']:+.4f}, ratio {st['E'] / st['E_lin']:.2f}; "
              f"F_raw {st['F_raw']:.3f}, F {st['F']:.3f}, F_open {st['F_open']:.3f}; sig_y {st['sig_y']:.3f}; "
              f"r_A small/large |D| {st['r_A_small'] / st['r_A_large']:.2f}")
    (Path(__file__).parent / "kick_surrogate_table.json").write_text(json.dumps(table, indent=1))
    return table


# ------------------------------------------------------------------ Part F: frozen predictions

def slope_factors(S, J):
    """Slope factors of a settled run S (R, T+1, N) for kicks at t0 = 6 and 8, weighted as the estimators weight:
    dbar_type[ty] = sum_i D_i^2 mean_{j reads i with J_ji = ty} d_j(t0+1) / sum_i D_i^2          (first order)
    D2[(tij, tji)] = sum_i D_i^2 d_i(t0+2) mean_{j recip. of that type} d_j(t0+1) / sum_i D_i^2   (echo, (K4))
    Returns dict of (numerator, denominator) so cells can be pooled."""
    acc = {}
    def add(key, num, den):
        a = acc.setdefault(key, [0.0, 0.0])
        a[0] += num
        a[1] += den
    for r in range(S.shape[0]):
        for t0 in (6, 8):
            D2w = 4 * S[r, t0] ** 2
            d1, d2 = 1 - S[r, t0 + 1] ** 2, 1 - S[r, t0 + 2] ** 2
            for i in range(S.shape[2]):
                for ty in (1, -1):
                    js = np.flatnonzero(J[:, i] == ty)
                    if len(js):
                        add(("r", ty), D2w[i] * d1[js].mean(), D2w[i])
                    for ty2 in (1, -1):                       # reciprocated ties: J_ij = ty, J_ji = ty2
                        js = np.flatnonzero((J[i] == ty) & (J[:, i] == ty2))
                        if len(js):
                            add(("E", ty, ty2), D2w[i] * d2[i] * d1[js].mean(), D2w[i])
            add(("D2",), np.sum(D2w), len(D2w))
    return acc


def lin_prediction(K, gamma, a_plus, a_minus, dE):
    """(K4): E = sum over reciprocated-tie types of (ties per agent / K^2) a_ij a_ji D2_type. With n_r = round(|gamma| K)
    ties per agent: gamma > 0 half agree-agree and half disagree-disagree (p_neg = 0.5), gamma < 0 all mixed."""
    n_r = round(abs(gamma) * K)
    if gamma > 0:
        return n_r / K**2 * (0.5 * a_plus**2 * dE[(1, 1)] + 0.5 * a_minus**2 * dE[(-1, -1)])
    if gamma < 0:
        return n_r / K**2 * (-a_plus * a_minus) * 0.5 * (dE[(1, -1)] + dE[(-1, 1)])
    return 0.0


def part_F(n_mc=20000, n_graphs=6):
    fit = sv.load_fit()
    rng = np.random.default_rng(11)
    print("\nF. Frozen predictions for settled kicks (t0 = 6, 8): linear response with the one-shot real-message gains")
    # slope factors: inverted from the pilot's own settled states; label from surrogate v2 with the transfer gains,
    # once with gains0 = gains and once with the first-inbox gains raised by the pilot's inverted ratio
    fac = {}
    for K in (2, 4):
        per_cell = {}
        for c in sv.pilot_cells():
            if c["K"] == K:
                S, J, hp, _ = sv.load_pilot_cell(c)
                per_cell[c["gamma"]] = slope_factors(S, J)
        fac[("inverted", K)] = per_cell
        law = sv.law_swarm(fit, K)
        g1 = fit["gains_transfer"][f"label|K{K}"]["target"]
        ratio = fit["gains_pilot"][f"inverted|K{K}|gains0"]["a_plus"] / fit["gains_transfer"][f"inverted|K{K}"]["target"]["a_plus"]
        for tag, g0 in (("label", g1), ("label, raised first-inbox gains", dict(g1, a_plus=g1["a_plus"] * ratio, a_minus=g1["a_minus"] * ratio))):
            per_cell = {}
            for gamma in (-1.0, 0.0, 1.0):
                accs = {}
                for g in range(n_graphs):
                    J = fixed_indegree_signed_couplings(N, K, gamma, P_NEG, rng)
                    hp = make_game(N, mu=0.5, sigma=1.0, k=3, seed=int(rng.integers(2**31))).h_priv
                    for key, (num, den) in slope_factors(sv.simulate_v2(J, hp, g1, g0, law, T, rng), J).items():
                        a = accs.setdefault(key, [0.0, 0.0]); a[0] += num; a[1] += den
                per_cell[gamma] = accs
            fac[(tag, K)] = per_cell
    val = lambda acc, key: acc[key][0] / acc[key][1] if key in acc else np.nan
    print("   Slope factors at the settled state (weights D_i^2). d_A, d_D: readers by edge type; D2: d_i d_j over reciprocated"
          "\n   ties, by type (AA, DD at gamma=+1; AD at gamma=-1); <D^2>: mean squared kick over all agents")
    summary = {}
    for (tag, K), per_cell in fac.items():
        dA = [val(a, ("r", 1)) for a in per_cell.values()]
        dD = [val(a, ("r", -1)) for a in per_cell.values()]
        AA, DD = val(per_cell[1.0], ("E", 1, 1)), val(per_cell[1.0], ("E", -1, -1))
        AD = np.nanmean([val(per_cell[-1.0], ("E", 1, -1)), val(per_cell[-1.0], ("E", -1, 1))])
        D2 = [val(a, ("D2",)) for a in per_cell.values()]
        summary[(tag, K)] = dict(dA=np.mean(dA), dD=np.mean(dD), AA=AA, DD=DD, AD=AD, D2=np.mean(D2),
                                 spread=float(np.std(dA + dD)))
        print(f"   {tag:32s} K={K}: d_A {np.mean(dA):.3f} d_D {np.mean(dD):.3f} (range {min(dA + dD):.3f}-{max(dA + dD):.3f})  "
              f"D2_AA {AA:.3f} D2_DD {DD:.3f} D2_AD {AD:.3f}  <D^2> {np.mean(D2):.2f}")
    # noise of one echo: two independent settled-state answers of the same agent, sig_y^2 = 2 <(s(t+1) - s(t))^2>_late
    sig_y = {}
    for K in (2, 4):
        ds = np.concatenate([np.diff(sv.load_pilot_cell(c)[0][:, 5:], axis=1).ravel() for c in sv.pilot_cells() if c["K"] == K])
        sig_y[K] = float(np.sqrt(2 * np.mean(ds**2)))
    print("\n   Predictions. r_A = d_A a_+/K, r_D = -d_D a_-/K, E from (K4); +- is 1 s.d. from the gain s.e. (transfer check)"
          "\n   and from the slope factors (6% each for inverted, 12% for label, where no swarm has been run)."
          f"\n   n3 = 9 sig_y^2 / (E^2 <D^2>), sig_y^2 = 2 <(s(t+1)-s(t))^2> of the pilot's rounds 6-10: sig_y = {sig_y[2]:.3f} (K=2), {sig_y[4]:.3f} (K=4).")
    print(f"   {'wording':8s} {'K':>2s} {'gamma':>5s} {'r_A':>15s} {'r_D':>15s} {'E':>17s} {'sign':>5s} {'n3':>7s} {'n3 at E-1sd':>11s}")
    pred = {}
    for wording in ("inverted", "label"):
        for K in (2, 4):
            g = fit["gains_transfer"][f"{wording}|K{K}"]["one_shot"]
            sm = summary[(wording, K)]
            rel = 0.06 if wording == "inverted" else 0.12
            ap = g["a_plus"] + g["a_plus_se"] * rng.standard_normal(n_mc)
            am = g["a_minus"] + g["a_minus_se"] * rng.standard_normal(n_mc)
            f1, f2 = 1 + rel * rng.standard_normal(n_mc), 1 + rel * rng.standard_normal(n_mc)
            rA, rD = sm["dA"] * f1 * ap / K, -sm["dD"] * f1 * am / K
            for gamma in GAMMAS:
                dE = {(1, 1): sm["AA"] * f1 * f2, (-1, -1): sm["DD"] * f1 * f2, (1, -1): sm["AD"] * f1 * f2, (-1, 1): sm["AD"] * f1 * f2}
                E = np.array(lin_prediction(K, gamma, ap, am, dE)) * np.ones(n_mc)
                Em, Es = float(E.mean()), float(E.std())
                n3 = 9 * sig_y[K] ** 2 / (Em**2 * sm["D2"]) if Em else np.nan
                n3lo = 9 * sig_y[K] ** 2 / ((abs(Em) - Es) ** 2 * sm["D2"]) if Em else np.nan
                sign = "0" if gamma == 0 else ("+" if Em > 0 else "-")
                pred[f"{wording}|K{K}|g{gamma:+.1f}"] = dict(r_A=float(rA.mean()), r_A_sd=float(rA.std()), r_D=float(rD.mean()),
                                                           r_D_sd=float(rD.std()), E=Em, E_sd=Es, sign=sign, n3=n3)
                print(f"   {wording:8s} {K:2d} {gamma:+5.1f} {rA.mean():+7.3f} +- {rA.std():.3f} {rD.mean():+7.3f} +- {rD.std():.3f}"
                      f" {Em:+8.4f} +- {Es:.4f} {sign:>5s} {n3:7.0f} {n3lo:11.0f}")
    (Path(__file__).parent / "kick_predictions.json").write_text(json.dumps(
        {"definition": "settled mirror kicks at t0 in {6, 8}; slopes through the origin; linear response, one-shot real gains",
         "sig_y": sig_y, "predictions": pred,
         "slope_factors": {f"{k[0]}|K{k[1]}": v for k, v in summary.items()}}, indent=1))
    return pred


if __name__ == "__main__":
    parts = sys.argv[1:] or list("ABCDEF")
    for name in parts:
        globals()[f"part_{name}"]()
