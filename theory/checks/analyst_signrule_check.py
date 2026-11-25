"""De-biased sign-rule test (theory/analyst_signrule.md): the law, its checks, and the frozen numbers.

Design: Qwen analysts, N = 256, K = 16, T = 20, R = 2 replicas per graph, three trust worlds at p_neg = 1/2. The model
returns the half log-odds x of BUY; the experimenter samples the call from tanh(x - chat), chat = the bias measured in
the calibration cell. The note is written by the model from the same prompt, after the sampled call.

Law (analyst_signrule.md (S1)-(S3)); indices are rounds, X_i(t) is the logit behind the round-t decision:

    X_i(t+1) = (1/K) sum_{j ally of i}  [uA sigma_j(t) + vA sr_j(t)]
             + (1/K) sum_{j faded for i}[uD sigma_j(t) + vD sr_j(t)]
             + a0 h0_i + nu xi_i(t+1) + kappa sigma_i(t) + c + eps e_i(t+1)                               (S1)
    sr_i(t) = tanh X_i(t)             raw belief: what the model holds and writes its note from          (S2)
    st_i(t) = tanh(X_i(t) - chat)     sampling belief: what the experimenter draws the call from
    sigma_i(t) = +1 iff U_i(t) < (1 + st_i(t)) / 2                                                        (S3)

uA, uD: pull of x toward an ally's / a faded colleague's CALL per unit normalized field (calls-only law M1 of the
calibration: uA = a_plus, uD = -a_minus, vA = vD = 0). vA, vD: pull toward the sender's RAW BELIEF as carried by its
note (leak law M2 of the calibration: uA = a_plus - b_LA, vA = b_LA, uD = -a_minus - b_LD, vD = b_LD). eps: residual
of the calibration fit, white in time (pilot: 4% of its variance is per-analyst, lag-1 autocorrelation 0); a fraction
eps_shared of its variance is common to the replicas of a swarm. gA, gD: extra pull per unit GROUP MEAN of calls (the
averaging model M4), i.e. per-reader call pulls uA + gA K / n_allies and uD + gD K / n_faded.

Parts (run: uv run python theory/checks/analyst_signrule_check.py [A B C D E M K S F]):
  A  calibration cells P0 and P1 refitted from the probes (M1, M2, joint bootstrap) -> analyst_signrule_calib.json
  B  code and identity checks: de-biased sampling is the frozen law at c - chat (exact); the kick replay equals
     analyst_lib.kick_twins; the echo function needs the sampling belief; in-swarm regressions recover the law
  C  central predictions per cell, law and world at the design size, with the leading-order reaction coefficient
     and the K = infinity solver
  D  law variants and the residual-bias table for the chosen cell
  E  calibration s.e. propagated through the simulation (bootstrap draws, common random numbers)
  M  response of the mean belief to a common bias, per world
  K  kick twins
  S  sizing: run-to-run s.d., replicas, run splits, asynchronous control
  F  collect analyst_signrule_predictions.json
"""
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import analyst_lib as al  # noqa: E402

K, N, T, R, BURN = 16, 256, 20, 2, 4            # the design; BURN: statistics use rounds BURN+1 .. T
W3 = ("mutual", "oneway", "antagonistic")
CELL = "P0"                                     # the calibration cell the design uses (analyst_signrule.md section 1)
CALIB = HERE / "analyst_signrule_calib.json"
SEED = 20261125


class Tee:
    def __init__(self, part):
        self.f = open(HERE / f"analyst_signrule_out_{part}.txt", "w")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def save(part, res):
    json.dump(res, open(HERE / f"analyst_signrule_part_{part}.json", "w"), indent=1, default=float)


# ----------------------------------------------------------------------------------------------------------------
# Law, simulation, kick replay
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Law:
    uA: float                      # pull toward an ally's call
    uD: float                      # pull toward a faded colleague's call (calls-only law: -a_minus)
    nu: float                      # logit s.d. of the news
    a0: float                      # logit s.d. of the standing note
    kappa: float                   # weight of the analyst's own last call
    c: float                       # the model's bias (intercept of the law)
    chat: float                    # the experimenter's de-bias constant
    vA: float = 0.0                # pull toward an ally's raw belief (through its note)
    vD: float = 0.0                # pull toward a faded colleague's raw belief
    eps: float = 0.0               # residual logit s.d., white
    eps_shared: float = 0.0        # fraction of the residual variance shared by replicas
    gA: float = 0.0                # extra pull per unit ally group mean (averaging)
    gD: float = 0.0                # extra pull per unit faded group mean


def law_M1(p, chat, **kw):
    """Calls-only law from an M1 fit (a_plus, a_minus, nu_05, a0, kappa, c, resid_sd)."""
    return Law(uA=p["a_plus"], uD=-p["a_minus"], nu=p["nu_05"], a0=p["a0"], kappa=p["kappa"], c=p["c"], chat=chat,
               eps=p["resid_sd"], **kw)


def law_M2(p, chat, **kw):
    """Leak law from an M2 fit: x = a+ f_A - a- f_D + b_LA L_A + b_LD L_D + ..., L = sum_group (s_j - sigma_j) / K,
    i.e. call pulls a+ - b_LA and -a- - b_LD, belief pulls b_LA and b_LD."""
    bD = p.get("b_LD", 0.0)                                     # M2A: ally leak only
    return Law(uA=p["a_plus"] - p["b_LA"], vA=p["b_LA"], uD=-p["a_minus"] - bD, vD=bD, nu=p["nu_05"],
               a0=p["a0"], kappa=p["kappa"], c=p["c"], chat=chat, eps=p["resid_sd"], **kw)


def weights(J, L: Law):
    """CA[i, j]: weight of sigma_j in X_i; CB[i, j]: weight of sr_j in X_i (rows = readers)."""
    A, D = (J > 0).astype(float), (J < 0).astype(float)
    uA = L.uA + L.gA * K / np.maximum(A.sum(1), 1)              # per-reader call pulls (constant unless gA, gD != 0)
    uD = L.uD + L.gD * K / np.maximum(D.sum(1), 1)
    return (uA[:, None] * A + uD[:, None] * D) / K, (L.vA * A + L.vD * D) / K


def draw_inputs(n, T_, R_, rng):
    """Notes h0 (n,), news xi (T+1, 1, n), call uniforms U (T+1, R, n), residuals: shared (T+1, 1, n), own (T+1, R, n)."""
    return dict(h0=rng.standard_normal(n), xi=rng.standard_normal((T_ + 1, 1, n)), U=rng.random((T_ + 1, R_, n)),
                e_sh=rng.standard_normal((T_ + 1, 1, n)), e_own=rng.standard_normal((T_ + 1, R_, n)))


def _priv(L, inp, t):
    e = np.sqrt(L.eps_shared) * inp["e_sh"][t] + np.sqrt(1 - L.eps_shared) * inp["e_own"][t]
    return L.a0 * inp["h0"] + L.nu * inp["xi"][t] + L.c + L.eps * e


def simulate(J, L: Law, T_, inp):
    """(S1)-(S3), synchronous. Returns X, s (sampling belief), sr (raw belief), sig with shape (T+1, R, N)."""
    CA, CB = weights(J, L)
    R_, n = inp["U"].shape[1:]
    X = np.empty((T_ + 1, R_, n))
    sig = np.empty_like(X)
    X[0] = _priv(L, inp, 0)
    for t in range(T_ + 1):
        sig[t] = np.where(inp["U"][t] < (1 + np.tanh(X[t] - L.chat)) / 2, 1.0, -1.0)                 # (S3)
        if t < T_:
            X[t + 1] = sig[t] @ CA.T + np.tanh(X[t]) @ CB.T + _priv(L, inp, t + 1) + L.kappa * sig[t]   # (S1)
    return dict(X=X, s=np.tanh(X - L.chat), sr=np.tanh(X), sig=sig)


def kick_twins(J, L: Law, base, inp, rep, t0, agents, regenerate=True):
    """Twin b: the readers of agents[b] see its round-t0 call flipped, D = -2 sigma(t0); its raw belief, its own
    memory and all random numbers are the base run's. Rounds t0+1, t0+2 are replayed.
    regenerate=True: every reader whose prompt changed writes its note from its new belief (the belief route
    vA, vD is open). False: a reader's note is rewritten only if its call flipped (belief route closed for the rest).
    Returns D (B,), dX1, dX2 (B, N): logit changes at t0+1, t0+2."""
    CA, CB = weights(J, L)
    agents = np.asarray(agents)
    B = len(agents)
    sig0, sr0 = base["sig"][t0, rep], base["sr"][t0, rep]
    seen = np.repeat(sig0[None], B, 0)
    seen[np.arange(B), agents] *= -1
    p1 = _priv(L, inp, t0 + 1)[0 if inp["e_own"].shape[1] == 1 else rep]
    p2 = _priv(L, inp, t0 + 2)[rep]
    X1 = seen @ CA.T + sr0 @ CB.T + p1 + L.kappa * sig0
    sig1 = np.where(inp["U"][t0 + 1, rep] < (1 + np.tanh(X1 - L.chat)) / 2, 1.0, -1.0)
    sr1 = np.tanh(X1)
    if not regenerate:
        sr1 = np.where(sig1 != base["sig"][t0 + 1, rep], sr1, base["sr"][t0 + 1, rep])
    X2 = sig1 @ CA.T + sr1 @ CB.T + p2 + L.kappa * sig1
    return dict(D=-2 * sig0[agents], dX1=X1 - base["X"][t0 + 1, rep], dX2=X2 - base["X"][t0 + 2, rep],
                dsig1=sig1 - base["sig"][t0 + 1, rep])


def run_stats(o):
    """Statistics of one run, all on the sampling belief s = tanh(X - chat) (analyst_lib.run_stats)."""
    st = al.run_stats(o["s"], o["sig"], burn=BURN)
    out = dict(m=st["m"], q=st["q"], d=st["d"], rho1=st["rho"][0], rho2=st["rho"][1], rho3=st["rho"][2],
               rho4=st["rho"][3], X2=st["rho"][1] - st["rho"][0] ** 2, mr=o["sr"][BURN:].mean(),
               dr=(1 - o["sr"][BURN:] ** 2).mean(), xvar=st["x_var"])
    if "c" in st:
        out.update(c1=st["c"][0], c2=st["c"][1], Xc2=st["c"][1] - st["c"][0] ** 2, q_ab=st["q_ab"])
    return out


_GRAPHS = {}


def graph(world, g, n=N):
    """Graph number g of a world: a fixed pool, so that laws and parameter draws are compared on the same graphs."""
    key = (world, g, n)
    if key not in _GRAPHS:
        _GRAPHS[key] = al.trust_graph(n, K, world, np.random.default_rng([SEED, W3.index(world), g, n]))
    return _GRAPHS[key]


def runs(L: Law, world, G, R_=R, T_=T, n=N, tag=0, g0=0):
    """G runs (graph g, its own notes, news, uniforms; common random numbers across laws). Dict of arrays (G,)."""
    acc = []
    for g in range(g0, g0 + G):
        inp = draw_inputs(n, T_, R_, np.random.default_rng([SEED, 7, W3.index(world), g, n, tag]))
        acc.append(run_stats(simulate(graph(world, g, n), L, T_, inp)))
    return {k: np.array([a[k] for a in acc]) for k in acc[0]}


def ms(x):
    """mean, s.e. of the mean, run-to-run s.d."""
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), float(x.std(ddof=1))


def reaction_lo(L: Law, world, d, dr):
    """Leading-order reaction coefficient (S6): the echo is X2 = G d with
        tau_g = u_g d + v_g dr                  what a group-g sender's message returns per unit of its logit
        G = (uA tau_A + uD tau_D) / 2K          mutual (concordant ties: ally-ally and faded-faded)
        G = (uA tau_D + uD tau_A) / 2K          antagonistic graph (discordant ties: ally-faded)
        G = 0                                   one-way
    d = <1 - st^2>, dr = <1 - sr^2>. Calls-only law: (a+^2 + a-^2) d / 2K and -a+ a- d / K, i.e. Gamma d of (A4)."""
    tA, tD = L.uA * d + L.vA * dr, L.uD * d + L.vD * dr
    return {"mutual": (L.uA * tA + L.uD * tD) / (2 * K), "oneway": 0.0,
            "antagonistic": (L.uA * tD + L.uD * tA) / (2 * K)}[world]


def calib():
    return json.load(open(CALIB))


def cell_laws(cell=None, cal=None):
    """The two laws of a cell at its point estimates, de-biased with chat = the cell's M1 intercept."""
    c = (cal or calib())["cells"][cell or CELL]
    return {"M1": law_M1(c["M1"]["est"], c["chat"]), "M2": law_M2(c["M2"]["est"], c["chat"])}


# ----------------------------------------------------------------------------------------------------------------
def part_A():
    """Refit the calibration cells from the probe records (results/, outside git) and store estimates, s.e. and
    bootstrap draws, so that every later part runs from analyst_signrule_calib.json alone."""
    import importlib.util
    out = Tee("A")
    spec = importlib.util.spec_from_file_location("an08b", ROOT / "experiments" / "08b-analyst-calibration-2" / "analyze.py")
    an = importlib.util.module_from_spec(spec)
    sys.modules["an08b"] = an
    spec.loader.exec_module(an)
    A8 = an.AN08
    out("Part A. Calibration cells, refitted from results/analyst-calib2-20261125 (1,000 base probes per cell, K = 16,")
    out("'max' wording). M1: x = a+ f_A - a- f_D + nu xi + a0 h0 + kappa sigma_own + c. M2 = M1 + b_LA L_A + b_LD L_D,")
    out("L = sum_group (s_j - sigma_j)/16 with s_j the belief of the pilot analyst who wrote the note.")
    out("M2A = M1 + b_LA L_A (ally leak only). Bootstrap over base probes, B = 400, the same resample for the three")
    out("models of a cell. s.e. in brackets.\n")
    B = 400
    res = {"cells": {}, "source": "results/analyst-calib2-20261125/{cell}_K16.jsonl, design_K16.json", "B": B}
    summ = json.load(open(ROOT / "experiments" / "08b-analyst-calibration-2" / "summary.json"))["cells"]
    for cell in ("P0", "P1", "T0", "F3"):
        rows, _ = an.load_cell(cell, 16)
        y = np.array([r["x"] for r in rows])
        Xs = {m: np.array([A8._cols(r, m) for r in rows]) for m in ("M1", "M2")}
        Xs["M2A"] = Xs["M2"][:, :-1]

        def fit(m, w=None):
            if m != "M2A":
                return A8.fit_arrays(Xs[m], y, m, w)
            co = A8.ols(Xs[m], y, w)
            f = dict(zip(A8.M1_NAMES + ["b_LA"], co))
            ww = np.ones_like(y) if w is None else w
            f["resid_sd"] = float(np.sqrt((ww * (y - Xs[m] @ co) ** 2).sum() / (ww.sum() - Xs[m].shape[1])))
            return f

        est = {m: fit(m) for m in Xs}
        rng = np.random.default_rng(0)
        draws = {m: [] for m in Xs}
        for _ in range(B):
            w = np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y)).astype(float)
            for m in draws:
                draws[m].append(fit(m, w))
        rec = {"n": len(y), "chat": float(est["M1"]["c"])}
        for m in draws:
            rec[m] = {"est": {k: float(v) for k, v in est[m].items()},
                      "se": {k: float(np.std([d[k] for d in draws[m]], ddof=1)) for k in est[m]},
                      "draws": [{k: round(float(v), 5) for k, v in d.items()} for d in draws[m]]}
            out(f"  {cell} {m}: " + "  ".join(f"{k} {rec[m]['est'][k]:+.3f} ({rec[m]['se'][k]:.3f})" for k in est[m]))
        dev = max(abs(est["M1"][k] - summ[cell][k]["est"]) for k in A8.M1_NAMES)
        out(f"     largest difference of the M1 estimates from experiments/08b summary.json: {dev:.1e}")
        e2 = est["M2"]
        out(f"     M2 as pulls: ally call uA = {e2['a_plus'] - e2['b_LA']:+.3f}, ally belief vA = {e2['b_LA']:+.3f}, "
            f"faded call uD = {-e2['a_minus'] - e2['b_LD']:+.3f}, faded belief vD = {e2['b_LD']:+.3f}")
        res["cells"][cell] = rec
    out("\n  T0 (template notes: no sender belief in the text) and F3 (faded notes not shown) are placebos for the belief")
    out("  pulls: a nonzero b there is not a response to note content.")
    out("\n  Identification of the belief pulls. corr(f, L) in the probe design, and M2 with the group size (n_allies - 8)/16")
    out("  added as a regressor (L_D carries the number of faded colleagues, because every pool sender is bearish):")
    for cell in ("P0", "P1", "T0", "F3"):
        rows, _ = an.load_cell(cell, 16)
        y = np.array([r["x"] for r in rows])
        X = np.array([A8._cols(r, "M2") + [(r["n_allies"] - 8) / 16] for r in rows])
        co = A8.ols(X, y)
        rng = np.random.default_rng(1)
        dr = np.array([A8.ols(X[i], y[i]) for i in (rng.integers(len(y), size=len(y)) for _ in range(B))])
        se = dr.std(0, ddof=1)
        res["cells"][cell]["M2_nA"] = dict(b_LA=[float(co[7]), float(se[7])], b_LD=[float(co[8]), float(se[8])],
                                           n_allies=[float(co[9]), float(se[9])], c=[float(co[6]), float(se[6])])
        out(f"    {cell}: corr(f_A, L_A) {np.corrcoef(X[:, 0], X[:, 7])[0, 1]:+.2f}  corr(f_D, L_D) {np.corrcoef(-X[:, 1], X[:, 8])[0, 1]:+.2f}   "
            f"b_LA {co[7]:+.3f} ({se[7]:.3f})  b_LD {co[8]:+.3f} ({se[8]:.3f})  n_allies term {co[9]:+.3f} ({se[9]:.3f})  c {co[6]:+.3f} ({se[6]:.3f})")
    out("\n  Residual of the in-swarm M1 regression in the pilot (results/analyst-pilot-20261125-*, N = 256, T = 10, mild")
    out("  wording, x on f_A, f_D, news, note, own call): s.d., share of its variance that is a per-analyst constant")
    out("  (variance of analyst means minus the within-analyst variance / T), and lag-1 autocorrelation.")
    from engine.ledger import read_jsonl
    res["pilot_residual"] = {}
    for w in W3:
        d = ROOT / "results" / f"analyst-pilot-20261125-{w}"
        inp = np.load(d / "inputs.npz")
        J = inp["J"].astype(float)
        recs = list(read_jsonl(d / "analyst_calls.jsonl"))
        Tp, n = max(r["round_t"] for r in recs), J.shape[0]
        x, sg = np.zeros((Tp + 1, n)), np.zeros((Tp + 1, n))
        for r in recs:
            x[r["round_t"], r["agent_id"]], sg[r["round_t"], r["agent_id"]] = r["x"], r["call"]
        Am, Dm = (J > 0) / K, (J < 0) / K
        Xp = np.concatenate([np.stack([Am @ sg[t - 1], -(Dm @ sg[t - 1]), inp["xi_real"][t], inp["h0_real"], sg[t - 1],
                                       np.ones(n)], 1) for t in range(1, Tp + 1)])
        yp = x[1:].ravel()
        e = (yp - Xp @ np.linalg.lstsq(Xp, yp, rcond=None)[0]).reshape(Tp, n)
        vq = e.mean(0).var() - e.var(0, ddof=1).mean() / Tp
        ac = float((e[1:] * e[:-1]).mean() / e.var())
        res["pilot_residual"][w] = dict(sd=float(e.std()), quenched_fraction=float(vq / e.var()), lag1=ac)
        out(f"    {w:13s} s.d. {e.std():.3f}  per-analyst share {vq / e.var():.2f}  lag-1 autocorrelation {ac:+.3f}")
    pool = json.loads((ROOT / "experiments" / "08b-analyst-calibration-2" / "note_pool.json").read_text())["rows"]
    c, s = np.array([r[0] for r in pool]), np.array([r[2] for r in pool])
    res["pool"] = {"n_buy": int((c == 1).sum()), "n_sell": int((c == -1).sum()), "s_given_buy": float(s[c == 1].mean()),
                   "s_given_sell": float(s[c == -1].mean())}
    out(f"\n  Note pool (pilot, calls sampled from the raw belief): {res['pool']['n_buy']} BUY notes written at mean raw belief "
        f"{res['pool']['s_given_buy']:+.3f}, {res['pool']['n_sell']} SELL notes at {res['pool']['s_given_sell']:+.3f}.")
    json.dump(res, open(CALIB, "w"), default=float)
    print("wrote", CALIB)


# ----------------------------------------------------------------------------------------------------------------
def inswarm_fit(o, J, L, leak=False):
    """The in-swarm regression (analyst_scenario.md section 1) on the de-biased logit X - chat: columns f_A, -f_D,
    own call, 1 (+ L_A, L_D with leak=True, L = sum_group (sr_j - sigma_j) / K). News, notes and residual are
    orthogonal to these and left in the residual. Returns a_plus, a_minus, kappa, c (+ b_LA, b_LD)."""
    A, D = (J > 0).astype(float) / K, (J < 0).astype(float) / K
    sig, sr, X = o["sig"][BURN:-1], o["sr"][BURN:-1], o["X"][BURN + 1:] - L.chat
    cols = [sig @ A.T, -(sig @ D.T), sig, np.ones_like(sig)]
    if leak:
        cols += [(sr - sig) @ A.T, (sr - sig) @ D.T]
    co = np.linalg.lstsq(np.stack([c.ravel() for c in cols], 1), X.ravel(), rcond=None)[0]
    return dict(zip(["a_plus", "a_minus", "kappa", "c", "b_LA", "b_LD"], co))


def part_B():
    out = Tee("B")
    res = {}
    cal = calib()
    laws = cell_laws(cal=cal)
    L1, L2 = laws["M1"], laws["M2"]
    out("Part B. Code and identity checks (cell", CELL + ").")

    out("\nB1. De-biased sampling is the frozen law (A1)-(A3) with bias c - chat. Calls-only law, no residual, same graph,")
    out("    notes, news and uniforms; this script's simulate against analyst_lib.simulate(Agent(c = c - chat)).")
    for w in W3:
        for chat in (L1.c, L1.c + 0.1):
            J = graph(w, 0)
            inp = draw_inputs(N, T, R, np.random.default_rng(1))
            La = replace(L1, eps=0.0, chat=chat)
            o = simulate(J, La, T, inp)
            P = al.Agent(La.uA, -La.uD, nu=La.nu, a0=La.a0, c=La.c - chat, kappa=La.kappa)
            ref = al.simulate(al.couplings(J, P), P, T, inp)
            ds, dsig = np.abs(o["s"] - ref["s"]).max(), int((o["sig"] != ref["sig"]).sum())
            out(f"    {w:13s} chat = {chat:+.3f}: max |s - s_ref| = {ds:.1e}, calls that differ: {dsig} of {o['sig'].size}")
            res[f"B1|{w}|{chat:+.3f}"] = [float(ds), dsig]
            assert ds < 1e-12 and dsig == 0

    out("\nB2. Kick replay against analyst_lib.kick_twins (calls-only law, no residual): largest difference of the belief")
    out("    changes at t0+1 and t0+2 over 256 twins.")
    for w in W3:
        J = graph(w, 1)
        inp = draw_inputs(N, T, R, np.random.default_rng(2))
        La = replace(L1, eps=0.0)
        o = simulate(J, La, T, inp)
        P = al.Agent(La.uA, -La.uD, nu=La.nu, a0=La.a0, c=La.c - La.chat, kappa=La.kappa)
        base = al.simulate(al.couplings(J, P), P, T, inp)
        tw = kick_twins(J, La, o, inp, 1, 9, np.arange(N))
        ref = al.kick_twins(al.couplings(J, P), J, P, base, inp, 1, 9, np.arange(N))
        d1 = np.abs(np.tanh(o["X"][10, 1] + tw["dX1"] - La.chat) - o["s"][10, 1] - ref["ds1"]).max()
        d2 = np.abs(np.tanh(o["X"][11, 1] + tw["dX2"] - La.chat) - o["s"][11, 1] - ref["ds2"]).max()
        out(f"    {w:13s} {d1:.1e}, {d2:.1e}")
        res[f"B2|{w}"] = [float(d1), float(d2)]
        assert max(d1, d2) < 1e-12

    out("\nB3. The echo function needs the sampling belief. zeta = sigma - tanh(X - chat) has zero mean given the past;")
    out("    sigma - tanh X (the raw belief) does not. rho_1, rho_2, X_2 from 100 runs of the calls-only law, with the")
    out("    estimator fed (a) the sampling belief, (b) the raw belief. Mean (s.e.).")
    for w in W3:
        a, b = [], []
        for g in range(100):
            inp = draw_inputs(N, T, R, np.random.default_rng([SEED, 31, g]))
            o = simulate(graph(w, g), L1, T, inp)
            ra = al.echo_function(o["s"], o["sig"], burn=BURN)
            rb = al.echo_function(o["sr"], o["sig"], burn=BURN)
            a.append([ra[0], ra[1], ra[1] - ra[0] ** 2])
            b.append([rb[0], rb[1], rb[1] - rb[0] ** 2])
        a, b = np.array(a), np.array(b)
        f = lambda z: "  ".join(f"{n} {ms(z[:, k])[0]:+.4f} ({ms(z[:, k])[1]:.4f})" for k, n in enumerate(("rho1", "rho2", "X2")))
        out(f"    {w:13s} (a) {f(a)}\n    {'':13s} (b) {f(b)}")
        res[f"B3|{w}"] = {"sampling": a.mean(0).tolist(), "raw": b.mean(0).tolist()}

    out("\nB4. What the in-swarm regressions return (one-way world, 20 runs pooled by averaging the coefficients).")
    out("    Law M1 simulated: the calls-only regression must return (a+, a-, kappa, c - chat). Law M2 simulated: the")
    out("    regression with L_A, L_D must return the M2 coefficients, and the calls-only regression returns the")
    out("    effective gains a_eff = u + v beta with beta the slope of the sender's raw belief on its call.")
    for name, L in (("M1", L1), ("M2", L2)):
        f1, f2, beta, nb = [], [], [], []
        for g in range(20):
            inp = draw_inputs(N, T, R, np.random.default_rng([SEED, 41, g]))
            J = graph("oneway", g)
            o = simulate(J, L, T, inp)
            f1.append(list(inswarm_fit(o, J, L).values()))
            f2.append(list(inswarm_fit(o, J, L, leak=True).values()))
            sg, sr = o["sig"][BURN:], o["sr"][BURN:]
            beta.append(np.cov(sg.ravel(), sr.ravel())[0, 1] / sg.var())
            nb.append([sr[sg > 0].mean(), sr[sg < 0].mean(), (sr[sg > 0] < 0).mean(), (sr[sg < 0] > 0).mean()])
        f1, f2, beta = np.mean(f1, 0), np.mean(f2, 0), float(np.mean(beta))
        truth = dict(a_plus=L.uA + L.vA, a_minus=-(L.uD + L.vD), kappa=L.kappa, b_LA=L.vA, b_LD=L.vD)
        out(f"    law {name}: truth a+ {truth['a_plus']:+.3f} a- {truth['a_minus']:+.3f} kappa {truth['kappa']:+.3f} b_LA {truth['b_LA']:+.3f} "
            f"b_LD {truth['b_LD']:+.3f};  beta = {beta:.3f}")
        out(f"       calls-only regression: a+ {f1[0]:+.3f} a- {f1[1]:+.3f} kappa {f1[2]:+.3f} c {f1[3]:+.3f}"
            f"   (u + v beta: a+ {L.uA + L.vA * beta:+.3f}, a- {-(L.uD + L.vD * beta):+.3f})")
        out(f"       with L_A, L_D:         a+ {f2[0]:+.3f} a- {f2[1]:+.3f} kappa {f2[2]:+.3f} c {f2[3]:+.3f} b_LA {f2[4]:+.3f} b_LD {f2[5]:+.3f}")
        nb = np.mean(nb, 0)
        out(f"       notes: mean raw belief of BUY writers {nb[0]:+.3f}, of SELL writers {nb[1]:+.3f}; share of BUY notes written at"
            f" raw P(BUY) < 1/2: {nb[2]:.2f}; share of SELL notes written at raw P(BUY) > 1/2: {nb[3]:.2f}")
        res[f"B4|{name}"] = dict(truth=truth, beta=beta, calls_only=f1.tolist(), with_leak=f2.tolist(), notes=nb.tolist())
        assert abs(f2[0] - truth["a_plus"]) < 0.05 and abs(f2[1] - truth["a_minus"]) < 0.05 and abs(f2[4] - L.vA) < 0.05
    save("B", res)


# ----------------------------------------------------------------------------------------------------------------
KEYS = ("m", "mr", "q", "d", "dr", "rho1", "rho2", "X2", "rho4", "c1", "c2", "Xc2")


def part_C():
    out = Tee("C")
    res = {}
    cal = calib()
    G = 300
    out(f"Part C. Central predictions at the design size: N = {N}, K = {K}, T = {T}, R = {R}, rounds {BURN + 1}..{T}, {G} runs")
    out("per entry (run = one graph with its notes, news and uniforms). Entries: mean (s.e. of the mean over runs).")
    out("m, q, d = mean, mean square, mean slope 1 - s^2 of the sampling belief; mr, dr the same for the raw belief;")
    out("rho_tau = echo function; X2 = rho2 - rho1^2; c1, c2 = lag correlations of replica differences of the sampling")
    out("belief, residual independent between replicas; Xc2 = c2 - c1^2 (c2s, Xc2s: residual fully shared by replicas).")
    out("lo = leading order G d with G of reaction_lo and the simulated d, dr.")
    for cell in ("P0", "P1"):
        laws = cell_laws(cell, cal)
        e = cal["cells"][cell]["M1"]["est"]
        out(f"\n--- cell {cell}: chat = {cal['cells'][cell]['chat']:+.4f};  M1 a+ {e['a_plus']:+.3f} a- {e['a_minus']:+.3f} nu {e['nu_05']:.3f} "
            f"a0 {e['a0']:.3f} kappa {e['kappa']:+.3f} resid {e['resid_sd']:.3f};  M2 pulls uA {laws['M2'].uA:+.3f} vA {laws['M2'].vA:+.3f} "
            f"uD {laws['M2'].uD:+.3f} vD {laws['M2'].vD:+.3f} c {laws['M2'].c:+.3f}")
        variants = {"M1": laws["M1"], "M2": laws["M2"], "M1, no residual (08b basis)": replace(laws["M1"], eps=0.0)}
        for name, L in variants.items():
            for w in W3:
                r = runs(L, w, G)
                rs = runs(replace(L, eps_shared=1.0), w, G) if L.eps else r
                rec = {k: ms(r[k]) for k in KEYS}
                rec["c2s"], rec["Xc2s"] = ms(rs["c2"]), ms(rs["Xc2"])
                rec["lo"] = reaction_lo(L, w, rec["d"][0], rec["dr"][0]) * rec["d"][0]
                res[f"{cell}|{name}|{w}"] = rec
                out(f"  {name:28s} {w:13s} " + " ".join(f"{k} {rec[k][0]:+.4f}({rec[k][1]:.4f})" for k in KEYS + ("c2s", "Xc2s"))
                    + f"  lo {rec['lo']:+.4f}")
        # K = infinity solver, calls-only law: the residual and the news enter the map as one white field
        L = laws["M1"]
        for eps in (L.eps, 0.0):
            P = al.Agent(L.uA, -L.uD, nu=float(np.hypot(L.nu, eps)), a0=L.a0, c=L.c - L.chat, kappa=L.kappa)
            row = []
            for w in W3:
                d = al.dmft(P, w, T=T, M=400000, seed=5, burn=BURN)
                rec = dict(q=float(d["q"][BURN + 1:].mean()), rho1=float(d["rho"][0]), rho2=float(d["rho"][1]),
                           X2=float(d["rho"][1] - d["rho"][0] ** 2))
                res[f"{cell}|solver eps={eps:.3f}|{w}"] = rec
                row.append(f"{w} q {rec['q']:.4f} rho1 {rec['rho1']:+.4f} rho2 {rec['rho2']:+.4f} X2 {rec['X2']:+.4f}")
            out(f"  solver (K = inf, M1, residual {eps:.3f}): " + ";  ".join(row))
    save("C", res)


# ----------------------------------------------------------------------------------------------------------------
def part_D():
    out = Tee("D")
    res = {}
    cal = calib()
    laws = cell_laws(cal=cal)
    L1, L2 = laws["M1"], laws["M2"]
    G = 100
    out(f"Part D. Law variants, cell {CELL}, design size, {G} runs per entry on the graphs and random numbers of part C.")
    out("Entries: mean (s.e.). dc = c - chat is the bias left after de-biasing.\n")
    cellM4 = {"P0": dict(a_plus=0.505, a_minus=-0.457, g_A=0.654, g_D=0.154),
              "P1": dict(a_plus=0.523, a_minus=-0.423, g_A=0.563, g_D=0.157)}[CELL]     # M4 fits (08 analyze.py, model M4)
    variants = {"M1 (central)": L1, "M2 (note-channel)": L2}
    for dc in (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15):
        variants[f"M1, dc = {dc:+.2f}"] = replace(L1, c=L1.c + dc)
    variants.update({
        "M2A (ally leak only, refitted)": law_M2(cal["cells"][CELL]["M2A"]["est"], L1.chat),
        "M2, chat at its own fixed point": None,
        "M1, averaging (M4 gains)": replace(L1, uA=cellM4["a_plus"], gA=cellM4["g_A"], uD=-cellM4["a_minus"], gD=-cellM4["g_D"]),
        "M1, pilot-like transfer (a+ x0.76, a- x1.85)": replace(L1, uA=0.76 * L1.uA, uD=1.85 * L1.uD),
        "M1, a- = 0": replace(L1, uD=0.0),
        "M1, a- = +0.15 (leans against)": replace(L1, uD=-0.15),
        "M1, no memory": replace(L1, kappa=0.0),
        "M1, residual 0.33 (pilot in-swarm)": replace(L1, eps=0.33),
        "M1, news x 1.5": replace(L1, nu=1.5 * L1.nu),
    })
    # chat for which the M2 law leaves no mean: fixed point of chat = c + (vA + vD)/2 <tanh X> + (uA + uD)/2 m at m = 0,
    # found by bisection on the simulated one-way mean belief
    lo_, hi_ = L2.chat - 0.2, L2.chat + 0.2
    for _ in range(12):
        mid = (lo_ + hi_) / 2
        m_mid = runs(replace(L2, chat=mid), "oneway", 24)["m"].mean()
        lo_, hi_ = (mid, hi_) if m_mid > 0 else (lo_, mid)
    variants["M2, chat at its own fixed point"] = replace(L2, chat=(lo_ + hi_) / 2)
    out(f"  (M2 law: the de-bias constant that leaves m = 0 in the one-way world is {(lo_ + hi_) / 2:+.3f}; the design's is {L2.chat:+.3f})\n")
    res["M2_chat_fixed_point"] = (lo_ + hi_) / 2
    keys = ("m", "q", "rho1", "rho2", "X2", "c2", "Xc2")
    for name, L in variants.items():
        for w in W3:
            r = runs(L, w, G)
            rec = {k: ms(r[k]) for k in keys}
            res[f"{name}|{w}"] = rec
            out(f"  {name:46s} {w:13s} " + " ".join(f"{k} {rec[k][0]:+.4f}({rec[k][1]:.4f})" for k in keys))
    save("D", res)


# ----------------------------------------------------------------------------------------------------------------
def part_E():
    out = Tee("E")
    res = {}
    cal = calib()
    c = cal["cells"][CELL]
    nd, G = 200, 32
    out(f"Part E. Calibration uncertainty propagated: {nd} bootstrap draws of the cell-{CELL} fit (M1 and M2 from the same")
    out(f"resample), each simulated on the same {G} graphs with the same random numbers (design size). chat is the design")
    out("constant and stays at the point estimate, so a draw's c - chat is the bias left by the calibration's error.")
    out("Entries: value at the point estimate, s.d. over draws (= propagated calibration s.e.), 2.5% and 97.5% of draws.\n")
    keys = ("m", "q", "rho1", "rho2", "X2", "c2", "Xc2")
    for name, mk, fit in (("M1", law_M1, "M1"), ("M2", law_M2, "M2")):
        for w in W3:
            est = runs(mk(c[fit]["est"], c["chat"]), w, G)
            dr = np.array([[runs(mk(d, c["chat"]), w, G)[k].mean() for k in keys] for d in c[fit]["draws"][:nd]])
            rec = {k: dict(est=float(est[k].mean()), sd=float(dr[:, i].std(ddof=1)), lo=float(np.percentile(dr[:, i], 2.5)),
                           hi=float(np.percentile(dr[:, i], 97.5))) for i, k in enumerate(keys)}
            res[f"{name}|{w}"] = rec
            out(f"  {name} {w:13s} " + "  ".join(f"{k} {rec[k]['est']:+.4f} sd {rec[k]['sd']:.4f} [{rec[k]['lo']:+.4f}, {rec[k]['hi']:+.4f}]" for k in keys))
    save("E", res)


# ----------------------------------------------------------------------------------------------------------------
def part_M():
    out = Tee("M")
    res = {}
    laws = cell_laws()
    G = 100
    out(f"Part M. Response of the mean belief to a common bias, cell {CELL}, design size, {G} runs per entry.")
    out("The experimenter shifts the de-bias constant: calls sampled from tanh(X - chat + b). m = mean sampling belief,")
    out("rounds 5..20. Entries: mean (s.e.) [run-to-run s.d.]. slope = [m(+b) - m(-b)] / 2b from the paired runs")
    out("(same graphs and random numbers), with its s.e. and the s.d. of one +-b pair of runs.\n")
    for name, L in laws.items():
        for w in W3:
            row, mb = [], {}
            for b in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
                r = runs(replace(L, chat=L.chat - b), w, G)
                mb[b] = r["m"]
                row.append(f"b {b:+.2f}: {ms(r['m'])[0]:+.3f} ({ms(r['m'])[1]:.3f}) [{ms(r['m'])[2]:.3f}]")
                res[f"{name}|{w}|b{b:+.2f}"] = dict(m=ms(r["m"]), q=ms(r["q"]), X2=ms(r["X2"]))
            out(f"  {name} {w:13s} " + "  ".join(row))
            for b in (0.05, 0.1):
                sl = (mb[b] - mb[-b]) / (2 * b)
                # unpaired version: a +b run and a -b run on different graphs
                sl_u = (mb[b][: G // 2] - mb[-b][G // 2:]) / (2 * b)
                res[f"{name}|{w}|slope{b}"] = dict(paired=ms(sl), unpaired=ms(sl_u))
                out(f"     slope at b = {b}: {ms(sl)[0]:.2f} ({ms(sl)[1]:.2f}), s.d. per pair of runs: same graph {ms(sl)[2]:.2f}, different graphs {ms(sl_u)[2]:.2f}")
    # single-site estimate: chi / (1 - Jbar chi), chi = d / (1 - kappa d) in the one-way world
    L = laws["M1"]
    r = runs(L, "oneway", 50)
    d = r["d"].mean()
    chi = d / (1 - L.kappa * d)
    Jbar = (L.uA + L.uD) / 2
    out(f"\n  One-way, calls-only law, single-site estimate: d = {d:.3f}, chi = d / (1 - kappa d) = {chi:.3f}, Jbar = (uA + uD)/2 = {Jbar:.3f}, "
        f"chi / (1 - Jbar chi) = {chi / (1 - Jbar * chi):.2f}")
    res["oneway_single_site"] = dict(d=float(d), chi=float(chi), Jbar=float(Jbar), slope=float(chi / (1 - Jbar * chi)))
    save("M", res)


# ----------------------------------------------------------------------------------------------------------------
def part_K():
    out = Tee("K")
    res = {}
    laws = cell_laws()
    Gk, t0s = 8, (8, 12)
    out(f"Part K. Kick twins, cell {CELL}, N = {N}. {Gk} graphs x {R} replicas x t0 in {t0s}, every analyst kicked once per")
    out(f"(replica, t0): {Gk * R * len(t0s) * N} twins per entry. Kick D = -2 sigma_i(t0) as seen by i's readers; all random numbers")
    out("(call uniforms, residuals) are the base run's. Per unit kick: r_A, r_D = sampling-belief change of a reader that")
    out("has i as ally / as faded, at t0+1; w_A, w_D the same in logit (exactly uA/K and uD/K); E = sampling-belief")
    out("change of i at t0+2 (s.e.); E_x the same in logit; lo = G d of reaction_lo; sd = s.d. of one twin's E;")
    out("n3 = twins for E = 3 s.e. 'notes kept' = a reader's note is rewritten only when its call flips. The last two")
    out("ratios compare E with the first-order forms of (A18): K r_A r_D (discordant ties), (K/2)(r_A^2 + r_D^2) (concordant).\n")
    for name, L in laws.items():
        for w in W3:
            acc = {k: [] for k in ("rA", "rD", "wA", "wD", "E", "Ex", "Ek", "d", "dr")}
            for g in range(Gk):
                J = graph(w, g)
                inp = draw_inputs(N, T, R, np.random.default_rng([SEED, 51, g]))
                o = simulate(J, L, T, inp)
                ag = np.arange(N)
                ally, faded = (J[:, ag] > 0).T, (J[:, ag] < 0).T               # [b, j]: reader j has agents[b] as ally / faded
                for rep in range(R):
                    for t0 in t0s:
                        tw = kick_twins(J, L, o, inp, rep, t0, ag)
                        tk = kick_twins(J, L, o, inp, rep, t0, ag, regenerate=False)
                        D = tw["D"]
                        ds1 = np.tanh(o["X"][t0 + 1, rep] + tw["dX1"] - L.chat) - o["s"][t0 + 1, rep]
                        E = lambda t_: (np.tanh(o["X"][t0 + 2, rep] + t_["dX2"][ag, ag] - L.chat) - o["s"][t0 + 2, rep]) * D / 4
                        acc["rA"].append((ds1 * D[:, None])[ally] / 4)
                        acc["rD"].append((ds1 * D[:, None])[faded] / 4)
                        acc["wA"].append((tw["dX1"] * D[:, None])[ally] / 4)
                        acc["wD"].append((tw["dX1"] * D[:, None])[faded] / 4)
                        acc["E"].append(E(tw))
                        acc["Ek"].append(E(tk))
                        acc["Ex"].append(tw["dX2"][ag, ag] * D / 4)
                acc["d"].append([(1 - o["s"][BURN:] ** 2).mean()])
                acc["dr"].append([(1 - o["sr"][BURN:] ** 2).mean()])
            a = {k: np.concatenate(v) for k, v in acc.items()}
            Em, sd = a["E"].mean(), a["E"].std()
            d, dr = a["d"].mean(), a["dr"].mean()
            rec = dict(rA=float(a["rA"].mean()), rD=float(a["rD"].mean()), wA=float(a["wA"].mean()), wD=float(a["wD"].mean()),
                       E=float(Em), E_se=float(sd / np.sqrt(len(a["E"]))), E_x=float(a["Ex"].mean()), sd=float(sd),
                       E_notes_kept=float(a["Ek"].mean()), E_notes_kept_se=float(a["Ek"].std() / np.sqrt(len(a["Ek"]))),
                       sd_notes_kept=float(a["Ek"].std()),
                       E_over_KrArD=float(Em / (K * a["rA"].mean() * a["rD"].mean())),
                       E_over_half_K_r2=float(Em / (K / 2 * (a["rA"].mean() ** 2 + a["rD"].mean() ** 2))),
                       lo=float(reaction_lo(L, w, d, dr) * d), d=float(d), dr=float(dr),
                       max_abs=float(np.abs(a["E"]).max()), frac_zero=float((a["E"] == 0).mean()))
            # twins for 3 s.e. when rewriting notes adds logit noise s_text at the kicked analyst: per unit kick the
            # belief noise is s_text d / 2
            rec["n3"] = {f"{st:.2f}": float(9 * (sd**2 + (st * d / 2) ** 2) / Em**2) if abs(Em) > 1e-9 else None
                         for st in (0.0, 0.05, 0.1, 0.2)}
            res[f"{name}|{w}"] = rec
            out(f"  {name} {w:13s} r_A {rec['rA']:+.4f} r_D {rec['rD']:+.4f}  w_A {rec['wA']:+.4f} w_D {rec['wD']:+.4f}  E {rec['E']:+.4f} ({rec['E_se']:.4f})  "
                f"E_x {rec['E_x']:+.4f}  lo {rec['lo']:+.4f}  sd {rec['sd']:.4f}  notes kept: E {rec['E_notes_kept']:+.4f} ({rec['E_notes_kept_se']:.4f}) sd {rec['sd_notes_kept']:.4f}  "
                f"E/(K rA rD) {rec['E_over_KrArD']:.2f}  E/((K/2)(rA^2 + rD^2)) {rec['E_over_half_K_r2']:.2f}"
                + (f"  max |E| {rec['max_abs']:.1e}" if w == "oneway" else
                   "  n3 at text noise 0 / 0.05 / 0.1 / 0.2: " + " / ".join(f"{rec['n3'][k]:.0f}" for k in rec["n3"])))
    save("K", res)


# ----------------------------------------------------------------------------------------------------------------
def part_S():
    out = Tee("S")
    res = {}
    laws = cell_laws()
    G = 300
    keys = ("rho1", "rho2", "X2", "c2", "Xc2", "m", "q")
    out(f"Part S. Sizing, cell {CELL}. Run-to-run s.d. of each statistic at N = {N}, T = {T}, rounds {BURN + 1}..{T}, from {G} runs")
    out("(R = 2) and 150 runs (R = 4). Entries: mean [s.d. of one run].\n")
    sd = {}
    for name, L in laws.items():
        for w in W3:
            for R_ in (2, 4):
                r = runs(L, w, G if R_ == 2 else 150, R_=R_)
                rec = {k: [ms(r[k])[0], ms(r[k])[2]] for k in keys}
                res[f"{name}|{w}|R{R_}"] = rec
                sd[(name, w, R_)] = rec
                out(f"  {name} {w:13s} R = {R_}: " + "  ".join(f"{k} {rec[k][0]:+.4f} [{rec[k][1]:.4f}]" for k in keys))
    out("\nComparisons in units of their s.e., for a split (mutual, one-way, antagonistic) of runs at R = 2. Difference of")
    out("world means; s.e. = sqrt(sd_1^2 / n_1 + sd_2^2 / n_2). Law M1 / law M2.")
    splits = [(4, 4, 4), (3, 4, 5), (2, 5, 5), (2, 4, 6), (1, 5, 6)]
    for sp in splits:
        n = dict(zip(W3, sp))
        row = []
        for stat in ("X2", "Xc2"):
            for a, b in (("mutual", "oneway"), ("antagonistic", "oneway"), ("mutual", "antagonistic")):
                z = []
                for name in laws:
                    A_, B_ = sd[(name, a, 2)][stat], sd[(name, b, 2)][stat]
                    z.append((A_[0] - B_[0]) / np.sqrt(A_[1] ** 2 / n[a] + B_[1] ** 2 / n[b]))
                res[f"split{sp}|{stat}|{a}-{b}"] = z
                row.append(f"{stat} {a[:3]}-{b[:3]} {z[0]:.1f} / {z[1]:.1f}")
        # one-way null: s.e. of X2 with n runs
        se0 = sd[("M1", "oneway", 2)]["X2"][1] / np.sqrt(n["oneway"])
        out(f"  split {sp}: " + "   ".join(row) + f"   s.e. of one-way X2: {se0:.4f}")
    out("\nThe same with R = 4 replicas on every graph (twice the calls per run), split (2, 5, 5):")
    n = dict(zip(W3, (2, 5, 5)))
    for stat in ("X2", "Xc2"):
        for a, b in (("mutual", "oneway"), ("antagonistic", "oneway")):
            z = []
            for name in laws:
                A_, B_ = sd[(name, a, 4)][stat], sd[(name, b, 4)][stat]
                z.append((A_[0] - B_[0]) / np.sqrt(A_[1] ** 2 / n[a] + B_[1] ** 2 / n[b]))
            res[f"R4 split(2, 5, 5)|{stat}|{a}-{b}"] = z
            out(f"  {stat} {a}-{b}: {z[0]:.1f} / {z[1]:.1f}")

    out("\nAsynchronous control (analyst_lib.simulate, mode 'async'; calls-only law, residual and news as one white field):")
    out("echo function rho_1..rho_3, 24 runs per world, mean [s.d. of one run]; and the synchronous values on the same law.")
    L = laws["M1"]
    P = al.Agent(L.uA, -L.uD, nu=float(np.hypot(L.nu, L.eps)), a0=L.a0, c=L.c - L.chat, kappa=L.kappa)
    for mode in ("async", "sync"):
        for w in W3:
            rho = []
            for g in range(24):
                rng = np.random.default_rng([SEED, 61, g])
                J = graph(w, g)
                o = al.simulate(al.couplings(J, P), P, T, al.draw_inputs(N, T, R, rng), mode=mode, rng=rng)
                rho.append(al.echo_function(o["s"], o["sig"], burn=BURN)[:3])
            rho = np.array(rho)
            res[f"{mode}|{w}"] = dict(mean=rho.mean(0).tolist(), sd=rho.std(0, ddof=1).tolist())
            out(f"  {mode:5s} {w:13s} " + "  ".join(f"rho{k + 1} {rho[:, k].mean():+.4f} [{rho[:, k].std(ddof=1):.4f}]" for k in range(3)))
    save("S", res)


# ----------------------------------------------------------------------------------------------------------------
def part_F():
    import subprocess
    cal = calib()
    parts = {p: json.load(open(HERE / f"analyst_signrule_part_{p}.json")) for p in "BCDEMKS"
             if (HERE / f"analyst_signrule_part_{p}.json").exists()}
    c = cal["cells"][CELL]
    laws = cell_laws(cal=cal)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    E, C, Kk, S = parts["E"], parts["C"], parts["K"], parts["S"]
    pred = {}
    for w in W3:
        pred[w] = {}
        for law in ("M1", "M2"):
            e = E[f"{law}|{w}"]
            cc = C[f"{CELL}|{law}|{w}"]
            pred[w][law] = {
                **{k: dict(value=cc[k][0], mc_se=cc[k][1], calib_sd=e[k]["sd"]) for k in ("rho1", "rho2", "X2", "c2", "Xc2", "m", "q")},
                "c2_residual_shared": cc["c2s"][0], "Xc2_residual_shared": cc["Xc2s"][0],
                "kick": {k: Kk[f"{law}|{w}"][k] for k in ("rA", "rD", "wA", "wD", "E", "E_se", "E_x", "sd", "E_notes_kept",
                                                         "sd_notes_kept", "n3")},
                "run_sd_R2": {k: S[f"{law}|{w}|R2"][k][1] for k in ("rho2", "X2", "c2", "Xc2", "m", "q")}}
    frozen = dict(
        note="theory/analyst_signrule.md", frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), parent_commit=commit,
        code="theory/checks/analyst_signrule_check.py (uses analyst_lib.py)",
        status="frozen before any de-biased analyst swarm data exist",
        design=dict(N=N, K=K, T=T, R=R, burn=BURN, p_neg=0.5, worlds=dict(mutual=1.0, oneway=0.0, antagonistic=-1.0),
                    cell=CELL, wording="max", variant="default prompt (question 'close up', BUY listed first, ordinary fading)",
                    chat=c["chat"], call="sigma = +1 iff u < (1 + tanh(x - chat)) / 2, u stored per (replica, round, analyst)",
                    statistics="all on the sampling belief s = tanh(x - chat); zeta = sigma - s"),
        law="X_i(t+1) = (1/K) sum_allies [uA sigma_j(t) + vA sr_j(t)] + (1/K) sum_faded [uD sigma_j(t) + vD sr_j(t)] + a0 h0_i "
            "+ nu xi_i(t+1) + kappa sigma_i(t) + c + eps e_i(t+1); sr = tanh X; s = tanh(X - chat); sigma ~ s",
        laws={k: vars(v) for k, v in laws.items()},
        calibration={m: dict(est=c[m]["est"], se=c[m]["se"]) for m in ("M1", "M2")},
        sign_rule="sign of X2 (and of the kick echo E) = sign of G; G = (uA tau_A + uD tau_D)/2K mutual, 0 one-way, "
                  "(uA tau_D + uD tau_A)/2K antagonistic graph; tau_g = u_g d + v_g dr; calls-only: -a_plus a_minus d / K",
        predictions=pred, parts=parts)
    json.dump(frozen, open(HERE / "analyst_signrule_predictions.json", "w"), indent=1, default=float)
    print("wrote", HERE / "analyst_signrule_predictions.json")


if __name__ == "__main__":
    fns = {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E, "M": part_M, "K": part_K, "S": part_S, "F": part_F}
    for p in sys.argv[1:] or list(fns):
        t = time.time()
        fns[p]()
        print(f"[part {p}: {time.time() - t:.0f} s]", flush=True)
