"""Analyst scenario (theory/analyst_scenario.md): agent law, simulation on actual trust graphs, single-site DMFT.

Agent law (analyst_scenario.md eqs. (A1)-(A3)). sigma_i(t) = +1 (BUY) or -1 (SELL) is the call that colleagues read,
s_i(t) = 2 P_i(BUY at round t) - 1 the belief read from the decision token's logprobs, x_i the half log-odds:

    x_i(t)       = sum_j W_ij y_j(t) + a0 h0_i + nu xi_i(t+1) + c + kappa sigma_i(t)                    (A1)
    s_i(t+1)     = tanh x_i(t)                       i.e. P(BUY) = 1 / (1 + exp(-2 x))                   (A2)
    sigma_i(t+1) = +1 with probability (1 + s_i(t+1)) / 2      ("sampled" call, temperature 1)           (A3)
                   or sign s_i(t+1)                             ("greedy" call)
    W_ij = +a_plus / K for an ally (J_ij = +1), -a_minus / K for a faded colleague (J_ij = -1), 0 otherwise
    y_j  = (1 - leak) sigma_j + leak s_j             leak = 0: readers get the call only (default)
    round 0:  x_i(-1) = a0 h0_i + nu xi_i(0) + c     (no inbox yet)

h0_i ~ N(0, 1) is the standing note (quenched), xi_i(t) ~ N(0, 1) the fresh news (white, known to the experimenter),
so nu = a_n * sd(news) and a0 = a_0 * sd(note) are logit standard deviations. a_plus, a_minus are logit gains per
unit normalized field f = (1/K) sum sigma_j, so a full ally bloc (K/2 colleagues, all BUY) moves x by a_plus / 2.

Effective couplings (claim1.md eq. (13) with gains, p_neg = 1/2, fixed in-degree K):
    Jbar = (a_plus - a_minus) / 2,   g^2 = (a_plus^2 + a_minus^2) / (2 K),
    Gamma = N <W_ij W_ji> = g^2 (mutual), 0 (one-way), -a_plus a_minus / K (antagonistic),  gamma_eff = Gamma / g^2.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.graphs import fixed_indegree_signed_couplings  # noqa: E402
from theory import dmft_solver as ds  # noqa: E402

WORLDS = {"mutual": +1.0, "oneway": 0.0, "antagonistic": -1.0}     # graph reciprocity of each trust world


@dataclass(frozen=True)
class Agent:
    a_plus: float                  # ally gain (logit per unit normalized field)
    a_minus: float                 # fade gain (positive = leans against faded colleagues)
    nu: float = 0.5                # logit s.d. of the fresh private news, a_n * sd(xi)
    a0: float = 0.0                # logit s.d. of the standing private note, a_0 * sd(h0)
    c: float = 0.0                 # common bias toward BUY (logit)
    kappa: float = 0.0             # logit weight of the agent's own last call (memory)
    call: str = "sampled"          # "sampled" | "greedy"
    leak: float = 0.0              # fraction of the sender's belief (not only its call) that the note transmits
    K: int = 16


def effective(P: Agent, world: str) -> dict:
    """Jbar, g, Gamma, gamma_eff of the trust world (module docstring)."""
    g2 = (P.a_plus**2 + P.a_minus**2) / (2 * P.K)
    Gam = {"mutual": g2, "oneway": 0.0, "antagonistic": -P.a_plus * P.a_minus / P.K}[world]
    return dict(Jbar=(P.a_plus - P.a_minus) / 2, g=np.sqrt(g2), g2=g2, Gamma=Gam, gamma_eff=Gam / g2 if g2 else 0.0)


def beta_eff(nu: float, call: str = "sampled") -> float:
    """Slope at 0 of phi(u) = E[sigma | u] (the mean call as a function of the field without news).
    sampled: phi(u) = <tanh(u + nu z)>;  greedy: phi(u) = erf(u / (nu sqrt 2)), slope sqrt(2/pi) / nu."""
    if call == "greedy":
        return np.sqrt(2 / np.pi) / nu
    z, w = np.polynomial.hermite_e.hermegauss(80)
    return float((w / w.sum()) @ (1 - np.tanh(nu * z) ** 2))


def bethe_gain(K: int, ratio: float = 1.0) -> float:
    """Ally gain a_plus at the spin-glass boundary of the mutual world on a K-regular tree, logistic sampled calls,
    no news, no standing note:  (K - 1) [tanh^2(a_plus / K) + tanh^2(a_minus / K)] / 2 = 1,  a_minus = ratio a_plus.
    (K -> infinity: (a_plus^2 + a_minus^2) / (2 K) = 1, i.e. g = 1.)"""
    from scipy.optimize import brentq
    f = lambda a: (K - 1) * (np.tanh(a / K) ** 2 + np.tanh(ratio * a / K) ** 2) / 2 - 1
    return float(brentq(f, 0.1, 50.0))


# ----------------------------------------------------------------------------------------------------------------
# Direct simulation on an actual graph
# ----------------------------------------------------------------------------------------------------------------

def trust_graph(N: int, K: int, world: str, rng) -> np.ndarray:
    """J in {-1, 0, +1}, rows = readers: envs.graphs.fixed_indegree_signed_couplings at p_neg = 1/2."""
    return fixed_indegree_signed_couplings(N, K, WORLDS[world], 0.5, rng)


def couplings(J: np.ndarray, P: Agent) -> sp.csr_matrix:
    W = sp.csr_matrix(J, dtype=float)                           # sparse first: no dense float copy at large N
    W.data = np.where(W.data > 0, P.a_plus, -P.a_minus) / P.K
    return W


def draw_inputs(N, T, R, rng, share_news=True):
    """Standing notes h0 (N,), news xi (T+1, R or 1, N), call uniforms U (T+1, R, N)."""
    return dict(h0=rng.standard_normal(N), xi=rng.standard_normal((T + 1, 1 if share_news else R, N)),
                U=rng.random((T + 1, R, N)))


def _call(s, U, P):
    return np.where(U < (1 + s) / 2, 1.0, -1.0) if P.call == "sampled" else np.where(s >= 0, 1.0, -1.0)


def simulate(W, P: Agent, T: int, inp: dict, mode: str = "sync", rng=None) -> dict:
    """Eqs. (A1)-(A3) on the graph W for R replicas that share the graph, the notes and (if xi has a singleton
    replica axis) the news. Returns s, sig with shape (T+1, R, N).

    mode "async": from round 1 on, agents update one at a time in a fresh random order per round and read the latest
    calls of their colleagues, including those already made in this round (envs/swarm.py, the asynchronous control).
    """
    h0, xi, U = inp["h0"], inp["xi"], inp["U"]
    R, N = U.shape[1:]
    s, sig = np.empty((T + 1, R, N)), np.empty((T + 1, R, N))
    s[0] = np.tanh(P.a0 * h0 + P.nu * xi[0] + P.c)
    sig[0] = _call(s[0], U[0], P)
    Wt = W.T.tocsr()
    for t in range(T):
        priv = P.a0 * h0 + P.nu * xi[t + 1] + P.c
        if mode == "sync":
            y = sig[t] if P.leak == 0 else (1 - P.leak) * sig[t] + P.leak * s[t]
            s[t + 1] = np.tanh(y @ Wt + priv + P.kappa * sig[t])
            sig[t + 1] = _call(s[t + 1], U[t + 1], P)
        else:
            Wd = W.toarray() if t == 0 else Wd
            cur_sig, cur_s = sig[t].copy(), s[t].copy()
            priv = np.broadcast_to(priv, (R, N))
            for i in rng.permutation(N):
                y = cur_sig if P.leak == 0 else (1 - P.leak) * cur_sig + P.leak * cur_s
                si = np.tanh(y @ Wd[i] + priv[:, i] + P.kappa * sig[t][:, i])
                cur_s[:, i] = si
                cur_sig[:, i] = _call(si, U[t + 1][:, i], P)
            s[t + 1], sig[t + 1] = cur_s, cur_sig
    return dict(s=s, sig=sig)


def kick_twins(W, J, P: Agent, base: dict, inp: dict, rep: int, t0: int, agents, crn: bool = True, rng=None):
    """Kick twins of replica `rep` (analyst_scenario.md section 6): in twin b the readers of agent agents[b] see its
    round-t0 call flipped (kick D = -2 sigma_i(t0)); the agent's own memory keeps the true call. Rounds t0+1 and
    t0+2 are replayed with the base run's news. crn=True reuses the base run's call uniforms (an agent whose
    belief did not change repeats its call exactly; a changed belief flips the call with probability |ds|/2).
    crn=False redraws the call of every agent whose belief changed (independent resampling of changed prompts).
    Returns D (B,), ds1, ds2 (B, N): belief changes at t0+1 and t0+2, and dsig1 (B, N).
    """
    agents = np.asarray(agents)
    B, N = len(agents), W.shape[0]
    Wt = W.T.tocsr()
    sig0, s0 = base["sig"][t0, rep], base["s"][t0, rep]
    xi = inp["xi"][:, 0 if inp["xi"].shape[1] == 1 else rep]
    seen = np.repeat(sig0[None], B, 0)
    seen[np.arange(B), agents] *= -1
    D = -2 * sig0[agents]
    out = []
    own = np.repeat(sig0[None], B, 0)
    for k in (1, 2):
        s_new = np.tanh(seen @ Wt + P.a0 * inp["h0"] + P.nu * xi[t0 + k] + P.c + P.kappa * own)
        s_base = base["s"][t0 + k, rep]
        U = np.repeat(inp["U"][t0 + k, rep][None], B, 0)
        if not crn:
            changed = np.abs(s_new - s_base) > 1e-12
            U = np.where(changed, rng.random((B, N)), U)
        sig_new = _call(s_new, U, P)
        out.append((s_new - s_base, sig_new - base["sig"][t0 + k, rep]))
        seen = own = sig_new
    return dict(D=D, ds1=out[0][0], dsig1=out[0][1], ds2=out[1][0], agents=agents)


# ----------------------------------------------------------------------------------------------------------------
# Observables of a run (analyst_scenario.md section 4). s, sig have shape (T+1, R, N).
# ----------------------------------------------------------------------------------------------------------------

def _centre_loo(y):
    """Subtract from each replica the per-agent mean over time of the OTHER replicas (leave one out), which removes
    quenched offsets without correlating the centred belief with the same replica's innovations. One replica: no
    centring."""
    R = y.shape[1]
    if R < 2:
        return y
    tot = y.mean(0)                                            # (R, N) time means
    return y - ((tot.sum(0, keepdims=True) - tot) / (R - 1))[None]


def echo_function(s, sig, burn=4, L=6, logit=False):
    """rho_tau = sum zeta_i(t) s_i(t+tau) / sum zeta_i(t)^2,  zeta = sigma - s the call innovation      (O2)
    (logit=True: the same with x_i(t+tau-1) = atanh s_i(t+tau) in place of s_i(t+tau)). Returns lags 1..L."""
    z = (sig - s)[burn:]
    y = np.arctanh(np.clip(s[burn:], -1 + 1e-12, 1 - 1e-12)) if logit else s[burn:]
    y = _centre_loo(y)
    return np.array([(z[:-k] * y[k:]).sum() / (z[:-k] ** 2).sum() for k in range(1, L + 1)])


def diff_autocorr(s, burn=4, L=6):
    """c_tau of replica differences delta = s^a - s^b (pairs (0,1), (2,3), ...), lags 1..L, and D = <delta^2>.  (O3)"""
    d = (s[burn:, 0::2] - s[burn:, 1::2])
    c = np.array([(d[:-k] * d[k:]).sum() / np.sqrt((d[:-k] ** 2).sum() * (d[k:] ** 2).sum()) for k in range(1, L + 1)])
    return c, float((d**2).mean())


def spectrum3(c):
    """S(omega) = 1 + 2 sum_tau c_tau cos(omega tau) at omega = 0, pi/2, pi, from lags 1..len(c)."""
    k = np.arange(1, len(c) + 1)
    return np.array([1 + 2 * (c * np.cos(om * k)).sum() for om in (0.0, np.pi / 2, np.pi)])


def chi_sg(sig, burn=4):
    """Spin-glass susceptibility of the calls from R >= 4 replicas that share graph, notes and news:            (O4)
        c_ij(k) = thermal covariance of sigma_i(t+k) and sigma_j(t), estimated from one replica pair (a, b) as the
                  time average of Delta_i(t+k) Delta_j(t), Delta = (sigma^a - sigma^b) / sqrt(2);
        chi_k   = (1/N) sum_{i,j} c_ij(k)^2, the square taken as the product of the estimates from two disjoint
                  replica pairs, so that sampling noise adds no bias;
        amplification  A = (chi_0 + chi_1) / chi_self,  chi_self = (1/N) sum_i c_ii(0)^2.
    On a tree with Lam per generation, chi_0 = chi_self / (1 - Lam^2) and chi_1 = chi_self Lam / (1 - Lam^2), so
    A = 1 / (1 - Lam). Returns (A, chi_0 / chi_self, chi_1 / chi_self)."""
    R, N = sig.shape[1:]
    D = [(sig[burn:, 2 * p] - sig[burn:, 2 * p + 1]) / np.sqrt(2) for p in range(2)]
    c0 = [d.T @ d / d.shape[0] for d in D]
    c1 = [d[1:].T @ d[:-1] / (d.shape[0] - 1) for d in D]
    self_ = (np.diag(c0[0]) * np.diag(c0[1])).sum() / N
    chi0, chi1 = (c0[0] * c0[1]).sum() / N, (c1[0] * c1[1]).sum() / N
    return (chi0 + chi1) / self_, chi0 / self_, chi1 / self_


def run_stats(s, sig, burn=4, L=6):
    """The scalar observables of one run."""
    x = np.arctanh(np.clip(s, -1 + 1e-12, 1 - 1e-12))
    rho, lam = echo_function(s, sig, burn, L), echo_function(s, sig, burn, L, logit=True)
    out = dict(m=s[burn:].mean(), q=(s[burn:] ** 2).mean(), d=(1 - s[burn:] ** 2).mean(),
               x_var=x[burn:].var(), x_kurt=_kurt(x[burn:]),
               rho=rho, lam=lam,
               csig2=_lag_corr(sig[burn:], 2), csig1=_lag_corr(sig[burn:], 1))
    if s.shape[1] >= 2:
        c, D = diff_autocorr(s, burn, L)
        out.update(c=c, D=D, S=spectrum3(c[:4]), q_ab=(s[burn:, 0::2] * s[burn:, 1::2]).mean())
    return out


def _kurt(x):
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2 - 3)


def _lag_corr(a, k):
    """Connected lag-k autocorrelation of calls, per-agent means removed."""
    a = a - a.mean(0, keepdims=True)
    return float((a[:-k] * a[k:]).sum() / np.sqrt((a[:-k] ** 2).sum() * (a[k:] ** 2).sum()))


# ----------------------------------------------------------------------------------------------------------------
# Single-site DMFT: claim1.md eqs. (4)-(8) for the law (A1)-(A3), through theory/dmft_solver.forward_pass
# ----------------------------------------------------------------------------------------------------------------

class AnalystMap:
    """Single-site map of (A1)-(A3) for dmft_solver.forward_pass. The solver's state is the call sigma; its field
    h(t) = Jbar m(t) + eta(t) + Gamma sum_t' R(t, t') sigma(t') already contains the gains (beta = 1). The belief
    tanh x is logged per round in self.beliefs. Response by the score function (solver_spec.md 1.1 item 3):
        sampled:  d/dh log P(sigma | x) = sigma - tanh x          (conditional on the news draw)
        greedy:   P(sigma = +1 | u) = Phi(u / nu), u the field without news; score = sigma phi(u/nu) / (nu Phi(sigma u/nu))
    """
    frozen, memoryless, response = False, False, "score"

    def __init__(self, P: Agent, copies: int = 1):
        if P.leak:
            raise ValueError("the single-site solver transmits calls only (leak = 0)")
        self.P, self.beliefs, self.copies = P, [], copies

    def _emit(self, u, rng):
        P = self.P
        # thermal copies of a site are replicas of one swarm: they share the standing note and the news
        xi = np.repeat(rng.standard_normal(len(u) // self.copies), self.copies)
        p = np.tanh(u + P.nu * xi)
        self._u, self._p = u, p
        self.beliefs.append(p)
        return _call(p, rng.random(u.shape), P)

    def init(self, hpriv, rng):
        return self._emit(self.P.a0 * hpriv + self.P.c, rng)

    def step(self, hpriv, hist, h, rng):
        return self._emit(h + self.P.a0 * hpriv + self.P.c + self.P.kappa * hist[-1], rng)

    def score(self, hpriv, hist, h, s_next):
        if self.P.call == "sampled":
            return s_next - self._p
        zz = self._u / self.P.nu
        return s_next * np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi) / (self.P.nu * np.clip(ndtr(s_next * zz), 1e-300, None))

    def mean_slope(self, hpriv, hist, h):
        if self.P.call == "sampled":
            return 1 - self._p**2
        return 2 * np.exp(-(self._u / self.P.nu) ** 2 / 2) / np.sqrt(2 * np.pi) / self.P.nu


def dmft(P: Agent, world: str, T: int, M: int, seed: int = 0, copies: int = 1, L: int = 6, burn: int = 4,
         reaction: bool = True, jbar: float | None = None) -> dict:
    """Single-site process for one trust world. reaction=False drops the reaction term (the reciprocity-blind
    closure, which is also what the one-way world obeys). jbar overrides the mean coupling (jbar = 0 isolates the
    effect of reciprocity from herding when a_plus != a_minus). copies > 1 needs something shared between the
    copies (nu > 0 or a0 > 0); with nothing frozen use copies = 1.

    Returns m(t), q(t) = <s^2>, C_sig(t, t'), R(t, t'), rho (echo function, lags 1..L, rounds > burn), Lam(omega)
    at omega = 0, pi/2, pi from the last round's kernel, and the solver output.
    Not included: the fixed in-degree correction -(Jbar^2 / K) m m to the noise covariance (claim5.md), below
    Jbar^2 m^2 / (K g^2) of the noise variance, which is under 1% for every case in the note.
    """
    e = effective(P, world)
    if jbar is not None:
        e["Jbar"] = jbar
    blocks = ds.Blocks([1.0], e["Jbar"], e["g"], e["gamma_eff"])
    mp = AnalystMap(P, copies)
    out = ds.forward_pass(blocks, [mp], 0.0, 1.0, T, M, np.random.default_rng(seed), copies=copies,
                          variant=None if reaction else "no_reaction")
    b = np.array(mp.beliefs)                                    # (T+1, M*copies)
    sig = out["s"][0]
    z = sig - b
    rho = np.array([(z[burn:-k] * b[burn + k:]).sum() / (z[burn:-k] ** 2).sum() for k in range(1, L + 1)])
    R = out["R"][0]
    lags = np.arange(1, T + 1)
    if copies == 1:                                             # statistically identical sites: Lam = g^2 |R(omega)|^2
        Rw = np.array([(R[T, T - 1::-1] * np.exp(-1j * om * lags)).sum() for om in (0, np.pi / 2, np.pi)])
        Lam = e["g2"] * np.abs(Rw) ** 2
    else:                                                       # claim2 (8): g^2 <|R_i(omega)|^2> over sites
        st = out["sites"][0]
        rows = [(st["R1"][:, T, T - 1::-1], st["R2"][:, T, T - 1::-1])]
        Lam = np.array([ds.stability_lambda(blocks, rows, omegas=(om,)) for om in (0, np.pi / 2, np.pi)])
    x = np.arctanh(np.clip(b[burn:], -1 + 1e-12, 1 - 1e-12))
    return dict(m=out["m"][0], q=(b**2).mean(1), d=(1 - b**2).mean(1), C=out["C"][0], R=R, rho=rho, Lam=Lam,
                beliefs=b, sig=sig, eff=e, q_ab=out["q"][0] if copies > 1 else None, x_kurt=_kurt(x))
