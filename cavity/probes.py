"""Single-agent probe harness: the agent the swarm runs (envs/prompts.py), fed synthetic inboxes.

Field convention. An inbox of K messages has agree-group states s_j (J_ij = +1) and disagree-group states
(J_ij = -1), with s = 2p - 1 from each displayed probability. The nominal field and its normalized form are

    h = sum_agree s_j - sum_disagree s_j,      hbar = h / K = f_A - f_D,
    f_A = sum_agree s_j / K,                   f_D = sum_disagree s_j / K,

so hbar is the normalized field (1/K) sum_j w_ij s_j of theory/claim1.md eq. (10), and every slope below is
per unit of hbar. For the mock agent of envs/swarm.MockBackend (gains b_+-), the response is
tanh(beta (b_+ K f_A - b_- K f_D) + h_priv + kappa u), so the true answers are known: conformity slope
beta_resp = beta b K, channel slopes a_+- = beta b_+- K, and eta^2_comp = 0 when b_+ = b_-.

Probe sets (each prints a cost projection with engine.BudgetGuard.project before running):
  (i)   sufficiency_design  claim1.md section 4.1: 5 field levels x 6 compositions x 2 evidence levels x
                            2 own-history states x n samples; analysed by eta2_comp.
  (ii)  conformity_design   uniform inboxes over a range of hbar; fit_conformity gives beta_resp.
  (iii) gains_design        agree-only against disagree-only inboxes; fit_channel_gains gives a_+, a_-.
  (iv)  run_white_probe     T-round trajectories of one agent with white field probes theta(t);
                            white_probe_kernel gives R(t, t') = <s(t) theta(t')> / eps^2 (solver_spec.md eq. (S1)).
  (v)   fit_surrogate       logistic-family F(h_priv, own-history summary, f_A, f_D) with held-out error.

Synthetic messages are short templated sentences consistent with each stated probability. Real swarm
messages carry more content (arguments, counts of sensors), so a probe passes only for this delivery; the
sufficiency test with messages taken from swarm transcripts is a later step.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from envs.evidence_game import Evidence, evidence_for_field
from envs.prompts import AgentContext, InboxMsg, make_request

COMPOSITIONS = ("uniform", "polarized", "few_strong", "route_A", "route_B", "shuffled")
S_POLAR = 0.8   # |s| of the polarized peers (p = 0.9 / 0.1)
S_MAX = 0.96    # largest |s| a synthetic peer may state (p = 0.98)

_TEXT = {
    "even": ["My sensor reports are split, so I can't tell either way.",
             "I see no clear lean toward HEADS or TAILS.",
             "It looks like a coin flip to me from what I have."],
    "weak": ["My reports lean slightly toward {S}.", "I think {S} is a bit more likely.",
             "The signal is weak, but it points to {S}."],
    "mid": ["My sensors favor {S}.", "I think {S} is more likely than not.", "Most of what I see points to {S}."],
    "strong": ["My sensors point clearly to {S}.", "I'm fairly confident it landed {S}.",
               "The evidence I have strongly favors {S}."],
}
HISTORY_STATES = {+1: (0.70, "My sensors favor HEADS."), -1: (0.30, "My sensors favor TAILS.")}


def peer_text(p: float, rng: np.random.Generator) -> str:
    """A short public message consistent with stated probability p of HEADS."""
    d = abs(p - 0.5)
    key = "even" if d < 0.03 else "weak" if d < 0.15 else "mid" if d < 0.3 else "strong"
    return str(rng.choice(_TEXT[key])).format(S="HEADS" if p > 0.5 else "TAILS")


# ---------------------------------------------------------------- synthetic inboxes

def synth_states(hbar: float, comp: str, K: int) -> tuple[list[float], list[float]]:
    """(agree states, disagree states) with (sum_agree - sum_disagree) / K = hbar exactly (before display
    rounding). Compositions of claim1.md section 4.1 at fixed field h = K hbar:

      uniform     all K peers in the agree group at s = hbar
      polarized   K-1 agree peers at +-S_POLAR, majority on the side of h, plus one filler peer that
                  absorbs the remainder (|filler| <= S_POLAR)
      few_strong  the field carried by n = max(2, ceil(|h| / S_MAX)) agree peers at h / n, the rest at 0
      route_A     K/2 agree peers at h / (K/2) (trusted, same sign as h); K/2 disagree peers at 0
      route_B     K/2 disagree peers at -h / (K/2) (distrusted, opposite view); K/2 agree peers at 0
      shuffled    uniform; the message order is permuted when the texts are attached (a null)
    route_A and route_B have the same group sizes, so their contrast isolates b_+ against b_-.
    """
    h = K * hbar
    if comp in ("uniform", "shuffled"):
        a, d = [hbar] * K, []
    elif comp == "polarized":
        n = K - 1
        dd = [x for x in range(-n, n + 1) if (x - n) % 2 == 0]           # n_+ - n_- has the parity of n
        diff = min(dd, key=lambda x: (abs(h / S_POLAR - x), -x))
        n_pos = (n + diff) // 2
        a = [S_POLAR] * n_pos + [-S_POLAR] * (n - n_pos) + [h - S_POLAR * diff]
        d = []
    elif comp == "few_strong":
        n = max(2, math.ceil(abs(h) / S_MAX - 1e-12))
        a, d = [h / n] * n + [0.0] * (K - n), []
    elif comp in ("route_A", "route_B"):
        m = K // 2
        sign = 1.0 if comp == "route_A" else -1.0
        carry, rest = [sign * h / m] * m, [0.0] * (K - m)
        a, d = (carry, rest) if comp == "route_A" else (rest, carry)
    else:
        raise ValueError(comp)
    if max(map(abs, a + d), default=0) > S_MAX + 1e-9:
        raise ValueError(f"hbar={hbar} is too large for composition {comp} at K={K}")
    return a, d


def to_msgs(states, rng: np.random.Generator, permute: bool = False) -> tuple[InboxMsg, ...]:
    """Attach displayed probabilities (2 decimals, as the swarm shows them) and templated texts."""
    probs = [round((1 + s) / 2, 2) for s in states]
    msgs = [InboxMsg(p, peer_text(p, rng)) for p in probs]
    if permute and len(msgs) > 1:
        order = rng.permutation(len(msgs))
        while np.all(order == np.arange(len(msgs))):
            order = rng.permutation(len(msgs))
        msgs = [msgs[i] for i in order]
    return tuple(msgs)


def realized_fields(ctx: AgentContext, K: int) -> tuple[float, float]:
    """(f_A, f_D) from the displayed probabilities."""
    return (sum(2 * m.prob - 1 for m in ctx.agree) / K, sum(2 * m.prob - 1 for m in ctx.disagree) / K)


def history_summary(ctx: AgentContext) -> float:
    """Own-history summary u: mean s of the shown own answers (0 if none)."""
    return float(np.mean([2 * p - 1 for p, _ in ctx.history])) if ctx.history else 0.0


# ---------------------------------------------------------------- probe sets

@dataclass
class Probe:
    ctx: AgentContext
    design: dict = field(default_factory=dict)


def _key(x) -> int:
    return int(round(float(x) * 1000)) % 2**31


def _evidence(level: float, k: int, seed: int) -> Evidence:
    return evidence_for_field(level, k, np.random.default_rng([seed, 0xE71D, _key(level)]))


def _probe(pid, session, ev: Evidence, hist_state, agree, disagree, K, design) -> Probe:
    hist = (HISTORY_STATES[hist_state],) if hist_state is not None else ()
    ctx = AgentContext(agent_id=pid, session=session, round_t=1, h_priv=ev.h_priv,
                       evidence=ev.text(), history=hist, agree=agree, disagree=disagree)
    fA, fD = realized_fields(ctx, K)
    return Probe(ctx, {**design, "K": K, "h_priv": ev.h_priv, "u": history_summary(ctx), "f_A": fA, "f_D": fD,
                       "hbar_realized": fA - fD})


def sufficiency_design(K: int = 8, unit: float = 0.1, levels=(-2, -1, 0, 1, 2), ev_levels=(0.4, 1.4),
                       hist_states=(+1, -1), n: int = 25, k_clues: int = 3, seed: int = 0) -> list[Probe]:
    """(i) claim1.md section 4.1. Field levels hbar = level * unit (unit 0.1 at K = 8 puts the top level's
    two "few strong" peers at s = 0.8). hist_states=(None,) for a memoryless agent (H = 0)."""
    out, idx = [], 0
    for ev_level in ev_levels:
        ev = _evidence(ev_level, k_clues, seed)
        for hs in hist_states:
            for lev in levels:
                for smp in range(n):
                    for comp in COMPOSITIONS:
                        # uniform and shuffled share texts: same rng stream, shuffled permutes the order
                        trng = np.random.default_rng([seed, 0x5AFF, _key(ev_level), (hs or 0) + 1, lev + 100, smp])
                        a, d = synth_states(lev * unit, comp, K)
                        agree = to_msgs(a, trng, permute=(comp == "shuffled"))
                        disagree = to_msgs(d, trng)
                        pid = f"suff|ev{ev_level}|hist{hs}|h{lev}|{comp}|{smp}"
                        out.append(_probe(pid, 10_000 + idx, ev, hs, agree, disagree, K,
                                          {"set": "sufficiency", "level": lev, "hbar": lev * unit, "comp": comp,
                                           "ev_level": ev_level, "hist": hs, "sample": smp}))
                        idx += 1
    return out


def conformity_design(K: int = 8, hbars=tuple(np.round(np.linspace(-0.8, 0.8, 9), 3)), ev_levels=(0.4,),
                      hist_states=(+1,), n: int = 20, k_clues: int = 3, seed: int = 0) -> list[Probe]:
    """(ii) Uniform (all-agree) inboxes at each hbar, for the conformity curve."""
    out, idx = [], 0
    for ev_level in ev_levels:
        ev = _evidence(ev_level, k_clues, seed)
        for hs in hist_states:
            for hb in hbars:
                for smp in range(n):
                    trng = np.random.default_rng([seed, 0xC0F, _key(ev_level), (hs or 0) + 1, _key(hb), smp])
                    a, _ = synth_states(hb, "uniform", K)
                    pid = f"conf|ev{ev_level}|hist{hs}|hbar{hb}|{smp}"
                    out.append(_probe(pid, 20_000 + idx, ev, hs, to_msgs(a, trng), (), K,
                                      {"set": "conformity", "hbar": float(hb), "ev_level": ev_level, "hist": hs,
                                       "sample": smp}))
                    idx += 1
    return out


def gains_design(K: int = 8, xs=(-0.8, -0.4, 0.0, 0.4, 0.8), ev_levels=(0.4,), hist_states=(+1,), n: int = 20,
                 k_clues: int = 3, seed: int = 0) -> list[Probe]:
    """(iii) All K peers at state x, either all in the agree group (f_A = x) or all in the disagree group
    (f_D = x). An agent that moves away from distrusted sources has a negative response to x in the
    disagree arm, i.e. a_- > 0."""
    out, idx = [], 0
    for ev_level in ev_levels:
        ev = _evidence(ev_level, k_clues, seed)
        for hs in hist_states:
            for arm in ("agree", "disagree"):
                for x in xs:
                    for smp in range(n):
                        trng = np.random.default_rng([seed, 0x6A1, _key(ev_level), (hs or 0) + 1,
                                                      int(arm == "agree"), _key(x), smp])
                        msgs = to_msgs([x] * K, trng)
                        agree, disagree = (msgs, ()) if arm == "agree" else ((), msgs)
                        pid = f"gain|ev{ev_level}|hist{hs}|{arm}|x{x}|{smp}"
                        out.append(_probe(pid, 30_000 + idx, ev, hs, agree, disagree, K,
                                          {"set": "gains", "arm": arm, "x": float(x), "ev_level": ev_level, "hist": hs,
                                           "sample": smp}))
                        idx += 1
    return out


# ---------------------------------------------------------------- cost projection and running

def project_cost(guard, contexts: list[AgentContext], H: int, model: str = "claude-haiku-5-5",
                 batch: bool = False, out_tokens: int = 450, chars_per_token: float = 3.5,
                 n_calls: int | None = None) -> float:
    """Print and return the projected cost via engine.BudgetGuard.project (CLAUDE.md guardrail 1c).

    Prompt tokens are estimated from the longest rendered request at ``chars_per_token``, all billed as
    uncached input, which overstates cost when the rules + evidence prefix is cached. out_tokens includes
    Haiku's thinking at effort low.
    """
    lens = [len(r.system_rules) + len(r.evidence) + len(r.dynamic) for r in (make_request(c, H) for c in contexts)]
    in_tok = int(max(lens) / chars_per_token) + 1
    return guard.project(n_calls or len(contexts), in_tokens=in_tok, cached_tokens=0, out_tokens=out_tokens,
                         model=model, batch=batch)


def _records(probes: list[Probe], responses) -> list[dict]:
    out = []
    for pr, r in zip(probes, responses):
        ok = r.status == "ok"
        out.append({**pr.design, "id": pr.ctx.agent_id, "status": r.status, "reason": r.reason, "cost_usd": r.cost_usd,
                    "p": r.prob if ok else np.nan, "s": 2 * r.prob - 1 if ok else np.nan,
                    "message": r.message})
    return out


async def run_probe_set(probes: list[Probe], backend, guard, H: int = 1, model: str = "claude-haiku-5-5",
                        batch: bool = False, out_tokens: int = 450, seed: int = 0) -> list[dict]:
    """Print the cost projection, then run every probe once (Runner key: seed, round 0, probe id)."""
    project_cost(guard, [p.ctx for p in probes], H, model, batch, out_tokens)
    res = await backend.respond(seed, 0, [p.ctx for p in probes], H)
    return _records(probes, res)


async def run_white_probe(backend, guard, M: int = 200, T: int = 8, K: int = 8, eps: float = 0.15,
                          bg_mean: float = 0.0, bg_sd: float = 0.3, ev_level: float = 0.4, H: int = 1,
                          k_clues: int = 3, seed: int = 0, model: str = "claude-haiku-5-5",
                          batch: bool = False, out_tokens: int = 450) -> dict:
    """(iv) M independent trajectories of one agent over rounds 0..T (solver_spec.md section 2).

    Round 0 has no inbox. For t = 0..T-1 the round-(t+1) inbox is uniform (all agree) at
    hbar(t) = h_bg(t) + theta(t), h_bg ~ N(bg_mean, bg_sd^2) and the white probe theta ~ N(0, eps^2), both
    i.i.d. over (trajectory, round) and known to the analysis; each peer is clipped to |s| <= S_MAX. The
    agent sees its own last H answers, so the measured R includes the path through its own memory.
    Returns s (M, T+1) and, shaped (M, T), the delivered field hbar (after clipping and display rounding)
    and the delivered probe theta = hbar - clip(h_bg).
    """
    rng = np.random.default_rng([seed, 0x7E1])
    ev = _evidence(ev_level, k_clues, seed)
    h_bg = bg_mean + bg_sd * rng.standard_normal((M, T))
    theta = eps * rng.standard_normal((M, T))
    target = np.clip(h_bg + theta, -S_MAX, S_MAX)
    longest = AgentContext(agent_id="x", session=40_000 + M, round_t=T, h_priv=ev.h_priv, evidence=ev.text(),
                           history=((0.5, "x" * 120),) * H, agree=to_msgs([S_MAX] * K, rng))
    project_cost(guard, [longest], H, model, batch, out_tokens, n_calls=M * (T + 1))
    s = np.full((M, T + 1), np.nan)
    delivered = np.zeros((M, T))
    hist = [[] for _ in range(M)]
    for t in range(T + 1):
        ctxs = []
        for m in range(M):
            agree = ()
            if t > 0:
                agree = to_msgs([target[m, t - 1]] * K, np.random.default_rng([seed, 0x7E2, m, t]))
                delivered[m, t - 1] = sum(2 * x.prob - 1 for x in agree) / K
            ctxs.append(AgentContext(agent_id=f"traj{m}", session=40_000 + m, round_t=t, h_priv=ev.h_priv,
                                     evidence=ev.text(), history=tuple(hist[m][-H:]) if H else (), agree=agree))
        res = await backend.respond(seed, t, ctxs, H)
        for m, r in enumerate(res):
            if r.status == "ok":
                s[m, t] = 2 * r.prob - 1
                hist[m].append((r.prob, r.message))
            else:  # carry the previous answer forward in the agent's history, record NaN
                hist[m].append(hist[m][-1] if hist[m] else (0.5, ""))
    return {"s": s, "theta": delivered - np.clip(h_bg, -S_MAX, S_MAX), "theta_nominal": theta, "hbar": delivered,
            "h_bg": h_bg, "eps": eps, "h_priv": ev.h_priv, "K": K}


# ---------------------------------------------------------------- analysis

def _arr(records, key):
    return np.array([r[key] for r in records], dtype=float)


def eta2_comp(records: list[dict]) -> dict:
    """(i) Composition variance ratio of claim1.md section 4.1, pooling the evidence and history cells.

        eta2 = mean_h Var_c[sbar(h, c)] / Var_h[sbar(h, .)],

    Var over the six compositions (ddof 0) and over the five field levels of the composition-averaged
    mean. Finite samples inflate Var_c by about mean_c se(h, c)^2 (C-1)/C; ``eta2_corrected`` subtracts
    that. Also returned: each composition's contrast with uniform per level (with s.e.), the largest
    |contrast| as a fraction of the range of sbar over levels (pass criterion < 0.2), the shuffled-vs-uniform
    noise floor, and eta2 within each (evidence, history) cell.
    """
    recs = [r for r in records if r.get("set") == "sufficiency" and np.isfinite(r["s"])]
    n_missing = sum(1 for r in records if r.get("set") == "sufficiency" and not np.isfinite(r["s"]))

    def table(rs):
        """sbar(h, c) = mean over (evidence, history) cells of the cell means (the design is balanced), and
        its s.e. from the within-cell spread only: the spread between evidence/history cells is common to
        all compositions and does not inflate Var_c."""
        levels = sorted({r["level"] for r in rs})
        cells_ = sorted({(r["ev_level"], r["hist"]) for r in rs}, key=str)
        sb = np.full((len(levels), len(COMPOSITIONS)), np.nan)
        se = np.full_like(sb, np.nan)
        for a, lv in enumerate(levels):
            for b, c in enumerate(COMPOSITIONS):
                means, var_of_mean = [], []
                for cl in cells_:
                    x = np.array([r["s"] for r in rs if r["level"] == lv and r["comp"] == c
                                  and (r["ev_level"], r["hist"]) == cl])
                    if x.size:
                        means.append(x.mean())
                        var_of_mean.append(x.var(ddof=1) / x.size if x.size > 1 else np.nan)
                if means:
                    sb[a, b] = np.mean(means)
                    se[a, b] = np.sqrt(np.sum(var_of_mean)) / len(means)
        return levels, sb, se

    def ratio(sb, se):
        C = sb.shape[1]
        num = np.nanmean(np.nanvar(sb, axis=1))
        den = np.nanvar(np.nanmean(sb, axis=1))
        noise = np.nanmean(se**2) * (C - 1) / C if np.isfinite(se).any() else 0.0
        return num / den, max(num - noise, 0.0) / den

    levels, sb, se = table(recs)
    eta2, eta2_c = ratio(sb, se)
    rng_s = np.nanmax(np.nanmean(sb, 1)) - np.nanmin(np.nanmean(sb, 1))
    u = COMPOSITIONS.index("uniform")
    contrast = sb - sb[:, [u]]
    contrast_se = np.sqrt(se**2 + se[:, [u]] ** 2)
    cells = {}
    for key in sorted({(r["ev_level"], r["hist"]) for r in recs}, key=str):
        _, sbc, sec = table([r for r in recs if (r["ev_level"], r["hist"]) == key])
        cells[str(key)] = ratio(sbc, sec)[0]
    return {"levels": levels, "compositions": COMPOSITIONS, "sbar": sb, "se": se, "eta2": eta2, "eta2_corrected": eta2_c,
            "range_over_levels": rng_s, "contrast_vs_uniform": contrast, "contrast_se": contrast_se,
            "max_contrast_frac": np.nanmax(np.abs(contrast)) / rng_s,
            "noise_floor_shuffled": contrast[:, COMPOSITIONS.index("shuffled")],
            "route_A_minus_B": sb[:, COMPOSITIONS.index("route_A")] - sb[:, COMPOSITIONS.index("route_B")],
            "eta2_by_cell": cells, "n_used": len(recs), "n_missing": n_missing}


def _fit_tanh(X: np.ndarray, y: np.ndarray, groups: np.ndarray, x0) -> tuple[np.ndarray, dict, np.ndarray]:
    """Least squares for E[s] = tanh(X @ w + c_group); returns (w, {group: c}, s.e. of w from the Jacobian)."""
    ug, gi = np.unique(groups, return_inverse=True)
    nw = X.shape[1]

    def resid(th):
        return np.tanh(X @ th[:nw] + th[nw:][gi]) - y

    th0 = np.concatenate([np.asarray(x0, float), np.zeros(len(ug))])
    sol = least_squares(resid, th0)
    dof = max(len(y) - len(th0), 1)
    s2 = (sol.fun**2).sum() / dof
    try:
        se = np.sqrt(np.diag(np.linalg.inv(sol.jac.T @ sol.jac) * s2))[:nw]
    except np.linalg.LinAlgError:
        se = np.full(nw, np.nan)
    return sol.x[:nw], dict(zip([str(g) for g in ug], sol.x[nw:])), se


def _ok(records, set_name):
    return [r for r in records if r.get("set") == set_name and np.isfinite(r["s"])]


def fit_conformity(records: list[dict]) -> dict:
    """(ii) E[s] = tanh(beta_resp * hbar + c_cell), one offset per (evidence, history) cell. Also returns
    the per-level means for plotting."""
    rs = _ok(records, "conformity")
    X = _arr(rs, "hbar_realized")[:, None]
    groups = np.array([f"{r['ev_level']}|{r['hist']}" for r in rs])
    w, c, se = _fit_tanh(X, _arr(rs, "s"), groups, [1.0])
    hb = sorted({r["hbar"] for r in rs})
    means = [float(np.mean([r["s"] for r in rs if r["hbar"] == h])) for h in hb]
    return {"beta_resp": float(w[0]), "beta_resp_se": float(se[0]), "offsets": c, "hbar": hb, "sbar": means,
            "n_used": len(rs)}


def fit_channel_gains(records: list[dict]) -> dict:
    """(iii) E[s] = tanh(a_+ f_A - a_- f_D + c_cell). In claim1.md eq. (13) units, b_-/b_+ = a_-/a_+;
    a_- < 0 means a message from a distrusted source still pulls toward that source."""
    rs = _ok(records, "gains")
    X = np.stack([_arr(rs, "f_A"), -_arr(rs, "f_D")], axis=1)
    groups = np.array([f"{r['ev_level']}|{r['hist']}" for r in rs])
    w, c, se = _fit_tanh(X, _arr(rs, "s"), groups, [1.0, 1.0])
    return {"a_plus": float(w[0]), "a_minus": float(w[1]), "a_plus_se": float(se[0]), "a_minus_se": float(se[1]),
            "ratio_minus_over_plus": float(w[1] / w[0]) if w[0] else np.nan, "offsets": c, "n_used": len(rs)}


def white_probe_kernel(out: dict) -> dict:
    """(iv) R(t, t') = <s(t) theta(t')> / eps^2 for t' < t (solver_spec.md eq. (S1)), with its s.e.

    s(t) is the round-t answer and theta(t') the probe in the field delivered with the round-(t'+1) inbox,
    so R(t, t') = d s(t) / d hbar(t') is zero for t' >= t (claim1.md eq. (3)). eps^2 is the realized probe
    variance. Also returns lag averages R(lag) = mean_t R(t, t - lag) over the available t.
    """
    s, th = out["s"], out["theta"]
    T1 = s.shape[1]
    eps2 = float(np.nanvar(th))
    R = np.zeros((T1, T1))
    se = np.zeros((T1, T1))
    for t in range(1, T1):
        for tp in range(t):
            x = s[:, t] * th[:, tp]
            x = x[np.isfinite(x)]
            R[t, tp] = x.mean() / eps2
            se[t, tp] = x.std(ddof=1) / np.sqrt(x.size) / eps2
    lags = np.arange(1, T1)
    lag_mean = np.array([np.mean([R[t, t - L] for t in range(L, T1)]) for L in lags])
    lag_se = np.array([np.sqrt(np.mean([se[t, t - L] ** 2 for t in range(L, T1)]) / (T1 - L)) for L in lags])
    return {"R": R, "se": se, "lags": lags, "R_lag": lag_mean, "R_lag_se": lag_se, "eps2": eps2}


def fit_surrogate(records: list[dict], holdout_frac: float = 0.2, seed: int = 0, two_channel: bool = True) -> dict:
    """(v) Parametric surrogate F(h_priv, u, f_A, f_D) in the logistic family of Physics of Agents' fitted
    rule, written in the spin form of the mock agent:

        E[s] = tanh(a_A f_A - a_D f_D + a_p h_priv + a_u u + c)        (two_channel)
        E[s] = tanh(a_h hbar + a_p h_priv + a_u u + c)                 (single channel)

    u = mean own shown answer (``history_summary``). Fitted on all probe records with a finite response,
    holding out a random ``holdout_frac`` of design cells (all samples of a cell go to the same side, so
    the held-out error measures generalization to unseen conditions). Reports the held-out RMSE on
    samples, the RMSE between predicted and observed held-out cell means, and the sampling floor of the
    latter, sqrt(mean se^2) of the held-out cell means. The residual s.d. is the surrogate's noise law
    (homoscedastic, additive in s).
    """
    rs = [r for r in records if np.isfinite(r["s"])]
    cell = np.array([r["id"].rsplit("|", 1)[0] for r in rs])
    uc = np.unique(cell)
    rng = np.random.default_rng(seed)
    test_cells = set(rng.choice(uc, max(1, int(round(holdout_frac * len(uc)))), replace=False))
    test = np.array([c in test_cells for c in cell])
    fA, fD, hp, u, y = (_arr(rs, k) for k in ("f_A", "f_D", "h_priv", "u", "s"))
    X = np.stack([fA, -fD, hp, u], 1) if two_channel else np.stack([fA - fD, hp, u], 1)
    names = ["a_A", "a_D", "a_p", "a_u"] if two_channel else ["a_h", "a_p", "a_u"]
    w, c, se = _fit_tanh(X[~test], y[~test], np.zeros((~test).sum()), np.ones(X.shape[1]))
    c0 = next(iter(c.values()))
    pred = np.tanh(X @ w + c0)
    cm_obs, cm_pred, cm_se = [], [], []
    for cc in sorted(test_cells):
        m = cell == cc
        cm_obs.append(y[m].mean())
        cm_pred.append(pred[m].mean())
        cm_se.append(y[m].std(ddof=1) / np.sqrt(m.sum()) if m.sum() > 1 else 0.0)
    cm_obs, cm_pred, cm_se = map(np.array, (cm_obs, cm_pred, cm_se))
    return {"params": dict(zip(names, map(float, w))), "params_se": dict(zip(names, map(float, se))), "c": float(c0),
            "rmse_heldout_samples": float(np.sqrt(np.mean((y[test] - pred[test]) ** 2))),
            "rmse_heldout_cellmeans": float(np.sqrt(np.mean((cm_obs - cm_pred) ** 2))),
            "cellmean_sampling_floor": float(np.sqrt(np.mean(cm_se**2))),
            "residual_sd_train": float((y[~test] - pred[~test]).std()), "n_train": int((~test).sum()),
            "n_test": int(test.sum()), "n_test_cells": len(test_cells)}


def run(coro):
    """Convenience for scripts: asyncio.run."""
    return asyncio.run(coro)
