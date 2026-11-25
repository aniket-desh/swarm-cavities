"""DMFT / cavity solver for agent swarms, as specified in theory/solver_spec.md.

Equations live in theory/claim1.md (single-site process), claim2.md (stability), claim3.md (two-site cavity, PR) and
claim4.md (zealots). Comments of the form "claim1 (4)" name the equation implemented on that line.

Mode (a), known single-site map:
    forward_pass           Monte Carlo over trajectories, one causal pass (spec 1.1), optionally nested copies (item 4)
    quadrature_gamma0      exact Gauss-Hermite integration for gamma = 0 and memoryless deterministic maps (spec 1.2)
    linear_kernels         exact kernel recursion for the linear map (spec A2, inputs of C1/C2)
    static_fixed_point     static cavity equations, claim2 (5)-(6) / claim4 (2)-(4)
Mode (b), black-box map:
    BlackBoxMap            wraps a batched agent(hpriv, history, field) -> states; forward_pass then uses the white probe
    fit_surrogate          logistic response surface from single-agent probes, returned as a known map for mode (a)
    live_pool              population dynamics of M trajectories on a random directed graph (spec 2, item 2)
Mode (c), two-site PR:
    site_moments, four_point, psi_equal_time, pr_over_n, mean_mode_transient, pr_from_sites
Mode (d), zealots:
    zealot_thresholds, branches (static); zealot_transient, forward_threshold_dynamic, hysteresis_dynamic (dynamic)

Conventions (spec section 0). Rounds t = 0..T, arrays over time have length n = T + 1, and s(t+1) = F(hpriv,
s(<=t), h(t)). J_ij is the influence of j on i (envs/graphs.py), so h = J s. Block parameters rho[a], Jbar[a, b],
g[a, b], gamma[a, b] (gamma symmetric): mean Jbar/N, variance g^2/N, Corr(J_ij, J_ji) = gamma. C_a(t, t') is the
full second moment <s s>, R_a(t, t') = <ds_a(t)/dh_a(t')> vanishes for t' >= t, Gam_ab = rho_b gamma_ab g_ab g_ba.

A single-site map is an object with
    init(hpriv, rng) -> s(0)
    step(hpriv, hist, h, rng) -> s(t+1)            hist = s(0..t), shape (t+1, B); everything vectorized over B
and optionally
    step_with_grad(hpriv, hist, h, rng) -> (s(t+1), dF/dh, dF/dhist or None)      deterministic, differentiable
    score(hpriv, hist, h, s_next) -> d log P(s_next | .)/dh,  mean_slope(...) -> d E[s_next]/dh     known noise law
which select the tangent or the score-function response estimator. The attribute `response` names which one
("tangent", "score" or None); a map with None (a black box) needs the white probe whenever R is required.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar


# ----------------------------------------------------------------------------------------------------------------
# Block parameters
# ----------------------------------------------------------------------------------------------------------------

class Blocks:
    """Population fractions and block moments of the couplings (claim1 (2))."""

    def __init__(self, rho, Jbar, g, gamma):
        self.rho = np.atleast_1d(np.asarray(rho, float))
        P = len(self.rho)
        as_block = lambda x: np.full((P, P), float(x)) if np.ndim(x) == 0 else np.asarray(x, float).reshape(P, P)
        self.Jbar, self.g, self.gamma = as_block(Jbar), as_block(g), as_block(gamma)
        if not np.allclose(self.gamma, self.gamma.T):
            raise ValueError("gamma_ab must be symmetric: Corr(J_ij, J_ji) is one number per pair of populations")
        self.P = P
        # claim1 (6): Gam_ab = rho_b gamma_ab g_ab g_ba, the coefficient of the reaction term
        self.Gam = self.rho[None, :] * self.gamma * self.g * self.g.T
        # claim1 (5): K_a(t, t') = sum_b rho_b g_ab^2 C_b(t, t'); noise_w[a, b] = rho_b g_ab^2
        self.noise_w = self.rho[None, :] * self.g**2


# ----------------------------------------------------------------------------------------------------------------
# Single-site maps
# ----------------------------------------------------------------------------------------------------------------

class IsingAgent:
    """Mock agent (spec section 0; claim1 section 6), with a context-window memory of its own states.

        u(t)     = lam u(t-1) + (1 - lam) s(t),  u(-1) = 0          i.e. u(t) = (1-lam) sum_tau lam^(t-tau) s(tau)
        x(t)     = beta h(t) + w_priv hpriv + bias + kappa u(t)
        s(t+1)   = tanh(x(t)) [+ out_noise * xi]                     deterministic (tangent response)
                   or +-1 with P(+1) = (1 + tanh x(t)) / 2            stochastic (score-function response)
        s(0)     = tanh(w_priv hpriv + bias), or its +-1 sample

    w_priv, bias and out_noise exist so that a fitted surrogate (fit_surrogate) is an instance of this class; the mock
    agent of the spec has w_priv = 1, bias = 0, out_noise = 0.
    """

    frozen = False

    def __init__(self, beta, kappa=0.0, lam=0.0, w_priv=1.0, bias=0.0, stochastic=False, out_noise=0.0):
        self.beta, self.kappa, self.lam = float(beta), float(kappa), float(lam)
        self.w_priv, self.bias, self.out_noise = float(w_priv), float(bias), float(out_noise)
        self.stochastic = bool(stochastic)
        self.response = "score" if stochastic else "tangent"
        self.memoryless = self.kappa == 0.0

    def _mem_weights(self, n):
        return (1.0 - self.lam) * self.lam ** np.arange(n - 1, -1, -1.0)      # weight of s(tau) in u(t), tau = 0..t

    def _x(self, hpriv, hist, h):
        x = self.beta * h + self.w_priv * hpriv + self.bias
        if not self.memoryless:
            x = x + self.kappa * (self._mem_weights(len(hist)) @ hist)
        return x

    def _emit(self, p, rng):
        if self.stochastic:
            return np.where(rng.random(p.shape) < (1 + p) / 2, 1.0, -1.0)
        if self.out_noise > 0:
            return p + self.out_noise * rng.standard_normal(p.shape)
        return p

    def init(self, hpriv, rng):
        return self._emit(np.tanh(self.w_priv * hpriv + self.bias), rng)

    def step(self, hpriv, hist, h, rng):
        return self._emit(np.tanh(self._x(hpriv, hist, h)), rng)

    def step_with_grad(self, hpriv, hist, h, rng):
        p = np.tanh(self._x(hpriv, hist, h))
        d = 1.0 - p**2
        dhist = None if self.memoryless else self.kappa * self._mem_weights(len(hist))[:, None] * d[None, :]
        return self._emit(p, rng), self.beta * d, dhist

    def score(self, hpriv, hist, h, s_next):
        # d/dh log P(s | x) for P(s) = (1 + s tanh x)/2, s = +-1:  beta (s - tanh x)
        return self.beta * (s_next - np.tanh(self._x(hpriv, hist, h)))

    def mean_slope(self, hpriv, hist, h):
        return self.beta * (1.0 - np.tanh(self._x(hpriv, hist, h)) ** 2)


class LinearAgent:
    """Linear map of spec A2: x(t+1) = a h(t) + noise * xi(t), xi ~ N(0, 1) white, x(0) = 0."""

    response, memoryless, frozen = "tangent", True, False

    def __init__(self, a, noise=1.0):
        self.a, self.noise = float(a), float(noise)

    def init(self, hpriv, rng):
        return np.zeros_like(hpriv)

    def step(self, hpriv, hist, h, rng):
        return self.a * h + self.noise * rng.standard_normal(h.shape)

    def step_with_grad(self, hpriv, hist, h, rng):
        return self.step(hpriv, hist, h, rng), np.full(h.shape, self.a), None


class Frozen:
    """Zealots (spec 1.1 item 2): s(t) = value, never updated, R = 0.

    With release_at = t_r the population is pinned for s(0..t_r) and from then on follows the map `then`, with its
    pinned states in its own history. This converts zealots back to benign agents in the hysteresis protocol.
    """

    frozen = True

    def __init__(self, value=-1.0, release_at=None, then=None):
        self.value, self.release_at, self.then = float(value), release_at, then
        self.response = None if then is None else then.response
        self.memoryless = True if then is None else then.memoryless

    def is_frozen(self, t):
        """True if s(t+1) is pinned."""
        return self.release_at is None or t < self.release_at

    def init(self, hpriv, rng):
        return np.full(np.shape(hpriv), self.value)

    def step(self, *args):
        return self.then.step(*args)

    def step_with_grad(self, *args):
        return self.then.step_with_grad(*args)

    def score(self, *args):
        return self.then.score(*args)

    def mean_slope(self, *args):
        return self.then.mean_slope(*args)


def _frozen_at(mp, t):
    return getattr(mp, "frozen", False) and mp.is_frozen(t)


class BlackBoxMap:
    """A batched black-box agent (spec section 2): agent(hpriv, history, field) -> states.

    hpriv (B,), history (t, B) = the agent's own past states s(0..t-1), field (B,): returns s(t), shape (B,).
    s(0) is the reply to an empty history and zero field. The solver calls it once per round with every trajectory of
    every population that shares this agent, so a backend that issues one LLM request per row can run them
    concurrently inside that call (e.g. asyncio.run(asyncio.gather(...))) without the solver knowing. Budget is
    counted in queries (rows): M x (T + 1) per population for one pass.
    """

    response, memoryless, frozen = None, False, False

    def __init__(self, agent):
        self.agent, self.n_calls, self.n_queries = agent, 0, 0

    def __call__(self, hpriv, history, field):
        self.n_calls += 1
        self.n_queries += len(hpriv)
        out = np.asarray(self.agent(hpriv, history, field), float)
        if out.shape != np.shape(hpriv):
            raise ValueError(f"black box returned shape {out.shape}, expected {np.shape(hpriv)}")
        return out

    def init(self, hpriv, rng):
        return self(hpriv, np.zeros((0, len(hpriv))), np.zeros(len(hpriv)))

    def step(self, hpriv, hist, h, rng):
        return self(hpriv, hist, h)


def as_black_box(mp, rng):
    """Hide a known map behind the black-box interface (for validating mode (b) on the mock agent)."""
    def agent(hpriv, history, field):
        return mp.init(hpriv, rng) if len(history) == 0 else mp.step(hpriv, history, field, rng)
    return agent


# ----------------------------------------------------------------------------------------------------------------
# Mode (a): Monte Carlo forward pass (spec 1.1)
# ----------------------------------------------------------------------------------------------------------------

PINV_RTOL = 1e-10


def _causal_gaussian(K, past, z):
    """Draw eta(t) given eta(<t) for a zero-mean Gaussian process with covariance K[:t+1, :t+1] (spec 1.1 item 1).

    Conditional mean w . eta(<t), w = K[<t,<t]^+ K[<t,t]; conditional variance K[t,t] - K[t,<t] w, clipped at zero.
    The pseudo-inverse tolerates the singular K of frozen states, where a Cholesky update fails. Its cutoff is
    rtol = 1e-10 relative to the largest eigenvalue: once the states stop moving, the smallest eigenvalues of K are
    rounding noise, and inverting them (numpy's default cutoff is ~1e-15) amplifies that noise into the sampled field.
    Discarding directions with eigenvalue below rtol * max changes the sampled covariances by at most ~sqrt(rtol).
    """
    t = K.shape[0] - 1
    if t == 0:
        return np.sqrt(max(K[0, 0], 0.0)) * z
    w = np.linalg.pinv(K[:t, :t], rtol=PINV_RTOL, hermitian=True) @ K[:t, t]
    return w @ past + np.sqrt(max(K[t, t] - K[:t, t] @ w, 0.0)) * z


def forward_pass(blocks, maps, mu, sigma, T, M, rng, *, copies=1, frozen_noise=True, probe_eps=0.0,
                 response="auto", variant=None):
    """Single-site process of claim1 (4)-(8) by Monte Carlo over trajectories, in one causal forward pass.

    blocks: Blocks; maps: one single-site map per population; mu, sigma: private fields hpriv ~ N(mu_a, sigma_a^2).
    M sites per population, each with `copies` thermal copies (spec 1.1 item 4). With copies = 1 every trajectory
    is its own site and the whole noise has covariance K_a. With copies > 1 the copies of a site share hpriv and the
    frozen part of the noise, eta = eta_f + eta_th,
        Cov eta_f  = sum_b rho_b g_ab^2 q_b,   Cov eta_th = sum_b rho_b g_ab^2 (C_b - q_b)      (claim1 (9))
    with q_b(t, t') the cross-copy correlation; the reaction term is applied per copy. frozen_noise = False gives the
    copies independent noise (independent couplings, shared hpriv and mean field), the baseline of Delta q.

    probe_eps > 0 adds a white probe theta(t) ~ N(0, eps^2) to every field and returns R_probe (spec (S1)).
    response: "auto" (tangent, else score, else probe, per population; spec 1.1 item 3), or "tangent" / "score" /
    "probe" to force the estimator whose R enters the reaction term and is returned as R, or "none" to skip R
    (allowed only when no population needs it, i.e. gamma = 0; saves the O(T^2 M) tangent storage).
    variant: None, or one of the wrong forms used to show that the tests resolve them (claim1 section 6):
        "connected_noise"   <ds ds> instead of <s s> in claim1 (5)
        "shifted_reaction"  sum_t' R(t, t'+1) s(t') instead of sum_t' R(t, t') s(t') in claim1 (4)
        "no_reaction"       reaction term dropped

    Returns a dict with m (P, n), C (P, n, n), R (P, n, n), R_probe, Lam (claim2 (8) from per-agent responses at the
    last round; NaN where they are not available), trajectories s[a] (n, M*copies), hpriv[a], theta[a]; with
    copies > 1 also q (P, n, n) and sites[a] = per-site mu (M, n), v (M, n, n), R (M, n, n) and the same from two
    disjoint halves of the copies (R1, R2, v1, v2), for unbiased site moments (spec section 3 step 1).
    """
    P, n = blocks.P, T + 1
    mu, sigma = np.broadcast_to(np.asarray(mu, float), (P,)), np.broadcast_to(np.asarray(sigma, float), (P,))
    nested = copies > 1
    Mt = M * copies
    site = np.repeat(np.arange(M), copies)
    shift = 1 if variant == "shifted_reaction" else 0
    Gam = np.zeros_like(blocks.Gam) if variant == "no_reaction" else blocks.Gam
    always_frozen = [getattr(mp, "frozen", False) and mp.release_at is None for mp in maps]

    # which response estimator gives R of each population (spec 1.1 item 3: the first that applies)
    kinds = []
    for a, mp in enumerate(maps):
        k = mp.response if response == "auto" else response
        if response == "auto" and k is None and probe_eps > 0:
            k = "probe"
        if always_frozen[a] or k == "none":
            k = None
        if k == "probe" and probe_eps <= 0:
            raise ValueError("response='probe' needs probe_eps > 0")
        if k in ("tangent", "score") and mp.response != k:
            raise ValueError(f"population {a}: map does not provide the {k} estimator")
        if k is None and not always_frozen[a] and np.any(Gam[:, a] != 0):
            raise ValueError(f"population {a} needs R for the reaction term but has no response estimator "
                             "(a black box needs probe_eps > 0; response='none' is only for gamma = 0)")
        kinds.append(k)

    hpriv = [np.repeat(mu[a] + sigma[a] * rng.standard_normal(M), copies) for a in range(P)]
    s = [np.zeros((n, Mt)) for _ in range(P)]
    theta = [probe_eps * rng.standard_normal((T, Mt)) if (probe_eps > 0 and not always_frozen[a]) else None
             for a in range(P)]
    tan = [np.zeros((n, T, Mt)) if kinds[a] == "tangent" else None for a in range(P)]   # tan[t, t'] = ds(t)/dh(t')
    score = [np.zeros((T, Mt)) if kinds[a] == "score" else None for a in range(P)]
    slope = [np.zeros((T, Mt)) if kinds[a] == "score" else None for a in range(P)]
    eta_f = [np.zeros((T, M)) for _ in range(P)]
    eta_th = [np.zeros((T, Mt)) for _ in range(P)] if nested else None
    m = np.zeros((P, n)); C = np.zeros((P, n, n)); R = np.zeros((P, n, n))
    q = np.zeros((P, n, n)) if nested else None
    R_probe = np.zeros((P, n, n)) if probe_eps > 0 else None

    def update_moments(t):          # claim1 (8): row t of m, C (and of the cross-copy correlation q, claim1 (9))
        for a in range(P):
            m[a, t] = s[a][t].mean()
            C[a, t, : t + 1] = C[a, : t + 1, t] = s[a][: t + 1] @ s[a][t] / Mt
            if nested:
                X = s[a][: t + 1].reshape(t + 1, M, copies)
                tot = X.sum(-1)
                # sum over copy pairs alpha != beta of s^alpha(t) s^beta(t'), per site, then averaged
                row = (tot @ tot[t] - np.einsum("umc,mc->u", X, X[t])) / (M * copies * (copies - 1))
                q[a, t, : t + 1] = q[a, : t + 1, t] = row

    # round 0: hpriv and s(0); black boxes shared by several populations get one call
    _apply_maps(maps, list(range(P)), lambda a: (hpriv[a], None, None), s, 0, rng, init=True)
    update_moments(0)

    for t in range(T):
        live = [a for a in range(P) if not _frozen_at(maps[a], t)]
        fields, dh = {}, {}
        for a in range(P):
            if always_frozen[a]:
                continue
            # claim1 (5): K_a(t, t') = sum_b rho_b g_ab^2 C_b(t, t'), the FULL second moment
            Cn = C - m[:, :, None] * m[:, None, :] if variant == "connected_noise" else C
            K = np.einsum("b,bij->ij", blocks.noise_w[a], Cn[:, : t + 1, : t + 1])
            if nested:
                Kf = np.einsum("b,bij->ij", blocks.noise_w[a], q[:, : t + 1, : t + 1]) if frozen_noise else 0 * K
                eta_f[a][t] = _causal_gaussian(Kf, eta_f[a][:t], rng.standard_normal(M))
                eta_th[a][t] = _causal_gaussian(K - Kf, eta_th[a][:t], rng.standard_normal(Mt))
                eta = eta_f[a][t][site] + eta_th[a][t]
            else:
                eta_f[a][t] = _causal_gaussian(K, eta_f[a][:t], rng.standard_normal(M))
                eta = eta_f[a][t]
            # claim1 (4): h_a(t) = sum_b rho_b Jbar_ab m_b(t) + eta_a(t) + sum_b Gam_ab sum_{t'<t} R_b(t, t') s_a(t')
            h = (blocks.rho * blocks.Jbar[a]) @ m[:, t] + eta
            if theta[a] is not None:
                h = h + theta[a][t]
            d = None
            if tan[a] is not None:
                d = np.zeros((T, Mt))
                d[t] = 1.0                                   # dh(t)/dtheta(t'): the direct part
            for b in range(P):
                if Gam[a, b] == 0.0 or t <= shift:
                    continue
                Rb = R[b, t, shift:t]                        # shift = 1 only in the wrong "shifted_reaction" form
                h = h + Gam[a, b] * (Rb @ s[a][: t - shift])
                if d is not None:                            # the path through the agent's own reaction term
                    d += Gam[a, b] * np.tensordot(Rb, tan[a][: t - shift], axes=1)
            fields[a], dh[a] = h, d

        # claim1 (7): s_a(t+1) = F_a(hpriv, s_a(<=t), h_a(t)); pinned populations keep their state
        for a in range(P):
            if a not in live:
                s[a][t + 1] = s[a][t]
        grads = _apply_maps(maps, live, lambda a: (hpriv[a], s[a][: t + 1], fields[a]), s, t + 1, rng,
                            tangent=[tan[a] is not None for a in range(P)])
        for a in live:
            if tan[a] is not None:
                # tangent recursion (spec 1.1 item 3): ds(t+1)/dh(t') = dF/dh dh(t)/dh(t') + sum_u dF/ds(u) ds(u)/dh(t')
                dFdh, dFdhist = grads[a]
                tan[a][t + 1] = dFdh * dh[a]
                if dFdhist is not None:
                    tan[a][t + 1] += np.einsum("um,uvm->vm", dFdhist, tan[a][: t + 1])
            elif score[a] is not None:
                args = (hpriv[a], s[a][: t + 1], fields[a])
                score[a][t] = maps[a].score(*args, s[a][t + 1])
                slope[a][t] = maps[a].mean_slope(*args)

        update_moments(t + 1)
        for a in range(P):
            if theta[a] is not None:
                # white probe (S1): R(t+1, t') = <s(t+1) theta(t')> / eps^2
                R_probe[a, t + 1, : t + 1] = theta[a][: t + 1] @ s[a][t + 1] / Mt / probe_eps**2
            if kinds[a] == "tangent":
                R[a, t + 1, : t + 1] = tan[a][t + 1, : t + 1].mean(-1)
            elif kinds[a] == "score":
                # score function: R(t+1, t') = <s(t+1) d_h log P(s(t'+1) | .)>, t' < t; lag 1 = <dE[s(t+1)]/dh(t)>
                R[a, t + 1, :t] = score[a][:t] @ s[a][t + 1] / Mt
                R[a, t + 1, t] = slope[a][t].mean()
            elif kinds[a] == "probe":
                R[a, t + 1, : t + 1] = R_probe[a, t + 1, : t + 1]

    out = dict(m=m, C=C, R=R, R_probe=R_probe, s=s, hpriv=hpriv, theta=theta, q=q, kinds=kinds)
    # Lam (claim2 (8)) needs per-agent responses: a trajectory's tangent for deterministic maps, copy averages
    # (from disjoint halves) for nested passes; with a stochastic or black-box population and no copies it is NaN.
    rows = [None] * P
    if nested:
        out["sites"] = [None if always_frozen[a] else _site_quantities(s[a], tan[a], score[a], slope[a], M, copies, T)
                        for a in range(P)]
        for a in range(P):
            st = out["sites"][a]
            if st is not None and "R1" in st and kinds[a] in ("tangent", "score"):
                rows[a] = (st["R1"][:, T, T - 1::-1], st["R2"][:, T, T - 1::-1])
    else:
        for a in range(P):
            if kinds[a] == "tangent":
                rows[a] = (tan[a][T, ::-1].T, None)          # (Mt, T): lags 1..T
    ok = all(rows[a] is not None or always_frozen[a] for a in range(P))
    out["Lam"] = stability_lambda(blocks, rows) if ok else np.nan
    return out


def _apply_maps(maps, pops, args_of, s, t_out, rng, init=False, tangent=None):
    """Write s[a][t_out] for each population in pops. Black boxes shared by several populations are called once per
    round with the trajectories of all of them concatenated (spec section 2: one call per round)."""
    grads, groups = {}, {}
    for a in pops:
        if isinstance(maps[a], BlackBoxMap):
            groups.setdefault(id(maps[a]), []).append(a)
            continue
        hp, hist, h = args_of(a)
        if init:
            s[a][0] = maps[a].init(hp, rng)
        elif tangent[a]:
            s[a][t_out], dFdh, dFdhist = maps[a].step_with_grad(hp, hist, h, rng)
            grads[a] = (dFdh, dFdhist)
        else:
            s[a][t_out] = maps[a].step(hp, hist, h, rng)
    for members in groups.values():
        box = maps[members[0]]
        parts = [args_of(a) for a in members]
        hp = np.concatenate([p[0] for p in parts])
        if init:
            out = box.init(hp, rng)
        else:
            out = box.step(hp, np.concatenate([p[1] for p in parts], axis=1), np.concatenate([p[2] for p in parts]), rng)
        for a, piece in zip(members, np.split(out, np.cumsum([len(p[0]) for p in parts])[:-1])):
            s[a][t_out] = piece
    return grads


def _site_quantities(s, tan, score, slope, M, copies, T):
    """Per-site thermal averages over the copies (spec 1.1 item 4): mu_i(t), v_i(t, t') and R_i(t, t'), and the same
    from two disjoint halves of the copies (R1, R2, v1, v2) when copies >= 4."""
    n = T + 1
    S = s.reshape(n, M, copies)

    def response(sl):
        Rr = np.zeros((M, n, n))
        if tan is not None:
            Rr[:, :, :T] = tan.reshape(n, T, M, copies)[..., sl].mean(-1).transpose(2, 0, 1)
        elif score is not None:
            Ss, Sc, Sl = S[..., sl], score.reshape(T, M, copies)[..., sl], slope.reshape(T, M, copies)[..., sl]
            for t in range(1, n):
                Rr[:, t, : t - 1] = np.einsum("mc,umc->mu", Ss[t], Sc[: t - 1]) / Ss.shape[-1]
                Rr[:, t, t - 1] = Sl[t - 1].mean(-1)
        return Rr

    def cov(sl):
        X = S[..., sl]
        Xc = X - X.mean(-1, keepdims=True)
        return np.einsum("tmc,umc->mtu", Xc, Xc) / (X.shape[-1] - 1)

    out = dict(mu=S.mean(-1).T, v=cov(slice(None)), R=response(slice(None)))
    half = copies // 2
    if half >= 2:
        h1, h2 = slice(0, half), slice(half, 2 * half)
        out.update(R1=response(h1), R2=response(h2), v1=cov(h1), v2=cov(h2))
    return out


def stability_lambda(blocks, rows, omegas=(0.0, np.pi / 2, np.pi)):
    """Lam = max_omega spectral radius of M_ab(omega) = <|R_a(omega)|^2> rho_b g_ab^2   (claim2 (8), multi-population).

    rows[a] = (R, R2): per-agent dressed responses at lags 1..L, shape (agents, L); R2 an independent estimate for
    the same agents (other half of the copies), so that <|R|^2> = <Re R conj(R2)> carries no estimation bias, or
    None. rows[a] = None for populations that never respond. The maximum is over omega in {0, pi/2, pi}, where it
    lies for memoryless agents (claim2 section 2); evaluated at the last round, it assumes a near-stationary kernel.
    """
    best = 0.0
    for om in omegas:
        A = np.zeros(blocks.P)
        for a, r in enumerate(rows):
            if r is None:
                continue
            ph = np.exp(-1j * om * np.arange(1, r[0].shape[1] + 1))
            f1 = r[0] @ ph
            f2 = f1 if r[1] is None else r[1] @ ph
            A[a] = np.mean((f1 * np.conj(f2)).real)
        best = max(best, np.abs(np.linalg.eigvals(A[:, None] * blocks.noise_w)).max())
    return float(best)


# ----------------------------------------------------------------------------------------------------------------
# Mode (a): exact Gaussian integration for gamma = 0, memoryless deterministic maps (spec 1.2)
# ----------------------------------------------------------------------------------------------------------------

def quadrature_gamma0(blocks, maps, mu, sigma, T, n_nodes=60):
    """m_a(t), C_a(t, t') by Gauss-Hermite quadrature (spec 1.2). Requires gamma = 0 (no R in the single-site process)
    and memoryless deterministic maps, so that s(t+1) = F(M_a(t) + eta_a(t), hpriv) with Gaussian eta_a(t):

        m(t+1)       = int Dz Dh phi_t(z, h)
        C(t+1, t'+1) = int Dz Du Dh phi_t(z, h) phi_t'(c z + sqrt(1-c^2) u, h),   c = K(t,t') / sqrt(K(t,t) K(t',t'))
        C(t+1, 0)    = int Dz Dh phi_t(z, h) s0(h),     phi_t(z, h) = F(sum_b rho_b Jbar_ab m_b(t) + sqrt(K_a(t,t)) z, h)

    with K_a = sum_b rho_b g_ab^2 C_b (claim1 (5)). Frozen populations (zealots) have m = value, C = value^2.
    Returns m (P, n), C (P, n, n) and Lam = spectral radius of <F'^2>_a rho_b g_ab^2 at the last round (claim2 (8)).
    """
    if np.any(blocks.Gam != 0):
        raise ValueError("quadrature path needs gamma = 0 (otherwise R enters the single-site process)")
    for mp in maps:
        if getattr(mp, "frozen", False):
            if mp.release_at is not None:
                raise ValueError("quadrature path does not handle released zealots")
        elif not (isinstance(mp, IsingAgent) and mp.memoryless and not mp.stochastic and mp.out_noise == 0.0):
            raise ValueError("quadrature path needs memoryless deterministic maps")
    P, n = blocks.P, T + 1
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    mu, sigma = np.broadcast_to(np.asarray(mu, float), (P,)), np.broadcast_to(np.asarray(sigma, float), (P,))
    hp = [mu[a] + sigma[a] * x if sigma[a] > 0 else np.array([mu[a]]) for a in range(P)]
    wh = [w if sigma[a] > 0 else np.ones(1) for a in range(P)]
    frozen = [getattr(mp, "frozen", False) for mp in maps]
    m = np.zeros((P, n)); C = np.zeros((P, n, n))
    s0 = [maps[a].init(hp[a], None) for a in range(P)]
    for a in range(P):
        if frozen[a]:
            m[a], C[a] = maps[a].value, maps[a].value ** 2
        else:
            m[a, 0], C[a, 0, 0] = wh[a] @ s0[a], wh[a] @ s0[a] ** 2

    def F(a, h, hpr):
        h, hpr = np.broadcast_arrays(h, hpr)
        val, slope_, _ = maps[a].step_with_grad(hpr.ravel(), None, h.ravel(), None)
        return val.reshape(h.shape), slope_.reshape(h.shape)

    for t in range(T):
        Mf = lambda u: (blocks.rho * blocks.Jbar) @ m[:, u]                              # mean field M_a(u)
        K = np.einsum("ab,bij->aij", blocks.noise_w, C[:, : t + 1, : t + 1])
        sd = lambda a, u: np.sqrt(max(K[a, u, u], 0.0))
        for a in range(P):
            if frozen[a]:
                continue
            phi_t, _ = F(a, Mf(t)[a] + sd(a, t) * x[:, None], hp[a][None, :])            # [z, h]
            m[a, t + 1] = w @ phi_t @ wh[a]
            C[a, t + 1, t + 1] = w @ phi_t**2 @ wh[a]
            C[a, t + 1, 0] = C[a, 0, t + 1] = (w @ phi_t) @ (wh[a] * s0[a])
            for tp in range(t):
                c = np.clip(K[a, t, tp] / (sd(a, t) * sd(a, tp)), -1, 1) if sd(a, t) * sd(a, tp) > 0 else 0.0
                zp = c * x[:, None, None] + np.sqrt(1 - c * c) * x[None, :, None]           # [z, u, h]
                phi_tp, _ = F(a, Mf(tp)[a] + sd(a, tp) * zp, hp[a][None, None, :])
                C[a, t + 1, tp + 1] = C[a, tp + 1, t + 1] = np.einsum("z,u,h,zh,zuh->", w, w, wh[a], phi_t, phi_tp)
    # Lam at the last round: gamma = 0 and memoryless, so |R_i(omega)| = F'_i at every omega
    K = np.einsum("ab,bij->aij", blocks.noise_w, C[:, :T, :T])
    A = np.zeros(P)
    for a in range(P):
        if not frozen[a]:
            Mf_last = (blocks.rho * blocks.Jbar) @ m[:, T - 1]
            _, d = F(a, Mf_last[a] + np.sqrt(max(K[a, T - 1, T - 1], 0)) * x[:, None], hp[a][None, :])
            A[a] = w @ d**2 @ wh[a]
    return dict(m=m, C=C, Lam=float(np.abs(np.linalg.eigvals(A[:, None] * blocks.noise_w)).max()))


# ----------------------------------------------------------------------------------------------------------------
# Mode (a): linear map, exact kernel recursion (spec A2; inputs of the mode (c) tests C1, C2)
# ----------------------------------------------------------------------------------------------------------------

def linear_kernels(a, g, gamma, noise, n, rho_b=1.0, weights=None):
    """Single-site DMFT of x(t+1) = a_k h(t) + noise_k xi(t), x(0) = 0, for site types k (one updating population,
    a fraction 1 - rho_b of frozen agents at x = 0). Exact, no sampling.

    Response (tangent recursion of spec 1.1 with dF/dh = a_k):
        R_k(t+1, t') = a_k [delta_{t t'} + Gam sum_u Rbar(t, u) R_k(u, t')],   Gam = rho_b gamma g^2   (claim1 (4), (6))
    Covariance: x_k = R_k (eta + (noise_k / a_k) xi), so
        v_k = R_k (rho_b g^2 Cbar + (noise_k / a_k)^2 1) R_k^T,   Cbar = <v_k>                          (claim1 (5))
    solved row by row. Returns R (K, n, n), v (K, n, n), weights (K,).
    """
    a, noise = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(noise, float))
    Kt = len(a)
    w = np.full(Kt, 1.0 / Kt) if weights is None else np.asarray(weights, float)
    Gam = rho_b * gamma * g**2
    R = np.zeros((Kt, n, n)); Rbar = np.zeros((n, n))
    for t in range(n - 1):
        for k in range(Kt):
            R[k, t + 1] = a[k] * ((np.arange(n) == t) + Gam * Rbar[t] @ R[k])
        Rbar[t + 1] = w @ R[:, t + 1]
    v = np.zeros((Kt, n, n)); Cbar = np.zeros((n, n))
    for t in range(1, n):
        for k in range(Kt):
            src = rho_b * g**2 * Cbar + (noise[k] / a[k]) ** 2 * np.eye(n)
            row = R[k, t] @ src @ R[k, : t + 1].T
            v[k, t, : t + 1] = v[k, : t + 1, t] = row
        Cbar[t, : t + 1] = Cbar[: t + 1, t] = w @ v[:, t, : t + 1]
    return R, v, w


# ----------------------------------------------------------------------------------------------------------------
# Mode (a)/(d): static cavity equations (claim2 (5)-(6), claim4 (2)-(4))
# ----------------------------------------------------------------------------------------------------------------

class UnstableFixedPoint(ValueError):
    """A fixed point with Lam >= 1 lies past the bulk instability and describes nothing (claim2 section 2)."""


def _site_newton(a, k, s, phi, dphi, iters=200):
    """Solve s = phi(a + k s) elementwise by Newton from s; starting from the previous iterate selects the branch
    continuously (spec 4 step 1)."""
    for _ in range(iters):
        y = a + k * s
        f, df = phi(y) - s, k * dphi(y) - 1.0
        s_new = np.clip(s - f / np.where(np.abs(df) < 1e-9, -1e-9, df), -1.0, 1.0)
        if np.max(np.abs(s_new - s)) < 1e-13:
            return s_new
        s = s_new
    return s


def _tanh_slope(y):
    return 1.0 - np.tanh(y) ** 2


def static_fixed_point(beta, g, gamma, mu, sigma, Jbar=0.0, rho=0.0, *, m_init=0.0, m_clamp=None, warm=None,
                       phi=np.tanh, dphi=_tanh_slope, n_z=64, n_h=48, damp=0.5, tol=1e-10, max_iter=4000,
                       stability=True, allow_unstable=False):
    """Static single-agent problem for benign agents with a fraction rho of zealots at -1 (claim4 (2)-(4); claim2
    (5)-(6) for rho = 0, Jbar = 0):

        s   = phi(beta [Jbar((1-rho) m - rho) + g sqrt(Q) z + Gam chi s] + hpriv),  z ~ N(0,1), hpriv ~ N(mu, sigma^2)
        m   = <s>,  q = <s^2>,  Q = (1-rho) q + rho,  Gam = (1-rho) gamma g^2
        chi = <d / (1 - d Gam chi)>,  d = beta phi'(.)

    on a Gauss-Hermite grid in (z, hpriv), the implicit equation solved by Newton, (m, q, chi) updated with damping
    `damp` until the change is below `tol`. m_init selects the branch. With m_clamp the mean in the field is held at
    m_clamp and only (q, chi) are solved; m_out = <s> is then the mean-field map f(m_clamp) (spec 4 step 2).
    Lam (claim2 (8), at the most unstable frequency) and Lam_static = (1-rho) g^2 <chi_i^2> (spec 4 step 4; equal
    to Lam for gamma >= 0) are reported, and a fixed point with Lam >= 1 raises UnstableFixedPoint unless
    allow_unstable. `warm` = (q, chi, s) from a previous call speeds up scans.
    """
    z, wz = np.polynomial.hermite_e.hermegauss(n_z)
    e, we = np.polynomial.hermite_e.hermegauss(n_h if sigma > 0 else 1)
    wz, we = wz / wz.sum(), we / we.sum()
    Z, E = np.meshgrid(z, e, indexing="ij")
    W = np.outer(wz, we)
    hp = mu + sigma * E
    m = m_init if m_clamp is None else m_clamp
    if warm is not None:
        q, chi, s = warm[0], warm[1], warm[2].copy()
    else:
        q, chi, s = max(m * m, 0.3), 0.0, np.full_like(Z, m)
    Gam = (1 - rho) * gamma * g**2                                         # claim4 (1): no echo through zealots
    err = np.inf
    for _ in range(max_iter):
        Q = max((1 - rho) * q + rho, 0.0)                                  # claim4 (3)
        a = beta * (Jbar * ((1 - rho) * m - rho) + g * np.sqrt(Q) * Z) + hp
        s = _site_newton(a, beta * Gam * chi, s, phi, dphi)                # claim4 (2)
        d = beta * dphi(a + beta * Gam * chi * s)
        chi_i = d / (1 - d * Gam * chi)                                    # claim4 (4), claim2 (6)
        m_new, q_new, chi_new = (W * s).sum(), (W * s**2).sum(), (W * chi_i).sum()
        err = abs(q_new - q) + abs(chi_new - chi) + (abs(m_new - m) if m_clamp is None else 0.0)
        if m_clamp is None:
            m = (1 - damp) * m + damp * m_new
        q = (1 - damp) * q + damp * q_new
        chi = (1 - damp) * chi + damp * chi_new
        if err < tol:
            break
    out = dict(m=m, m_out=m_new, q=q, chi=chi, s=s, d=d, chi_i=chi_i, W=W, converged=err < tol, warm=(q, chi, s))
    if stability:
        out["Lam_static"] = (1 - rho) * g**2 * (W * chi_i**2).sum()
        out["Lam"] = _lambda_static(d, W, g, gamma, 1 - rho)
        if m_clamp is None and out["Lam"] >= 1 and not allow_unstable:
            raise UnstableFixedPoint(f"Lam = {out['Lam']:.3f} >= 1")
    return out


def _lambda_static(d, W, g, gamma, rho_b):
    """claim2 (8) for memoryless agents at a fixed point: at the most unstable frequency (0 or pi for gamma > 0,
    pi/2 for gamma < 0) |R_i| = d_i / (1 - |gamma| g^2 k d_i), with k = rho_b <d_i / (1 - |gamma| g^2 k d_i)> the
    smallest non-negative root, and Lam = rho_b g^2 <|R_i|^2>. Returns inf if no root exists (already unstable)."""
    ag = abs(gamma) * g**2
    if ag == 0.0:
        return rho_b * g**2 * (W * d**2).sum()
    f = lambda k: rho_b * (W * d / (1.0 - ag * k * d)).sum() - k
    ks = np.linspace(0.0, 1.0 / (ag * d.max()), 4001)[:-1]
    neg = np.nonzero(np.array([f(k) for k in ks]) < 0)[0]
    if len(neg) == 0:
        return np.inf
    k = brentq(f, ks[neg[0] - 1], ks[neg[0]], xtol=1e-15)
    return rho_b * g**2 * (W * (d / (1.0 - ag * k * d)) ** 2).sum()


# ----------------------------------------------------------------------------------------------------------------
# Mode (d): zealot fixed points, saddle-nodes, hysteresis (spec section 4)
# ----------------------------------------------------------------------------------------------------------------

class _MeanMap:
    """f(m; rho) - m at clamped mean m (claim4 section 2), with warm starts across calls."""

    def __init__(self, beta, g, gamma, mu, sigma, Jbar, **kw):
        self.args, self.kw, self.warm = (beta, g, gamma, mu, sigma, Jbar), kw, None

    def resid(self, m, rho):
        o = static_fixed_point(*self.args, rho, m_clamp=m, warm=self.warm, stability=False, **self.kw)
        self.warm = o["warm"]
        return o["m_out"] - m


def branches(beta, g, gamma, mu, sigma, Jbar, rho, n_grid=161, with_lambda=False, **kw):
    """All solutions m of the mean equation f(m; rho) = m at zealot fraction rho (spec 4 step 2): roots of f(m) - m
    on a grid of m in (-1, 1), refined by bisection. A pair of roots closer than the grid spacing is caught by
    refining each interior extremum of the grid values. One root: no bistability; three: the outer two are stable.
    with_lambda=True returns (m, Lam) pairs and raises UnstableFixedPoint if any Lam >= 1 (spec 4 step 4)."""
    F = _MeanMap(beta, g, gamma, mu, sigma, Jbar, **kw)
    ms = np.linspace(-0.999, 0.999, n_grid)
    Fs = np.array([F.resid(m, rho) for m in ms])
    root = lambda lo, hi: brentq(lambda m: F.resid(m, rho), lo, hi, xtol=1e-12)
    roots = [root(ms[i], ms[i + 1]) for i in np.nonzero(np.sign(Fs[:-1]) != np.sign(Fs[1:]))[0]]
    for i in range(1, n_grid - 1):
        for sgn in (1.0, -1.0):          # a local min with F > 0 (sgn = 1) or a local max with F < 0 (sgn = -1)
            if sgn * (Fs[i] - Fs[i - 1]) < 0 < sgn * (Fs[i + 1] - Fs[i]) and sgn * Fs[i] > 0:
                r = minimize_scalar(lambda m: sgn * F.resid(m, rho), bounds=(ms[i - 1], ms[i + 1]),
                                    method="bounded", options=dict(xatol=1e-10))
                if r.fun < 0:            # the extremum crosses zero: two roots between grid points
                    roots += [root(ms[i - 1], r.x), root(r.x, ms[i + 1])]
    roots = sorted(roots)
    if not with_lambda:
        return roots
    out = []
    for m in roots:
        Lam = static_fixed_point(beta, g, gamma, mu, sigma, Jbar, rho, m_clamp=m, **kw)["Lam"]
        if Lam >= 1:
            raise UnstableFixedPoint(f"Lam = {Lam:.3f} >= 1 at m = {m:.4f}")
        out.append((m, Lam))
    return out


def zealot_thresholds(beta, g, gamma, mu, sigma, Jbar, n_m=99, rho_range=(-0.2, 0.95), **kw):
    """Forward and backward critical zealot fractions (claim4 (5); spec 4 step 3).

    Along the fixed-point line f(m; rho) = m, rho is a single-valued function rho(m) (f decreases with rho through the
    mean field -Jbar rho (1 + m)). The saddle-nodes, where f = m and df/dm = 1, are the extrema of rho(m): rho_up is
    the local maximum (the benign-majority branch disappears as rho rises past it) and rho_down the local minimum
    (the captured branch disappears as rho falls below it), set to 0 when the captured state survives the removal of
    every zealot. A monotone rho(m) means there is no bistable window, and None is returned. This is the condition of
    the spec's bisection on the number of roots of branches(), computed without a grid in m limiting how close two
    merging roots can be resolved. Extrema are located on a grid of n_m values of m and refined.

    Returns dict(rho_up, m_up, rho_down, m_down, Lam_up, Lam_down, rho_of_m) or None. Raises UnstableFixedPoint if
    Lam >= 1 at a threshold (spec 4 step 4).
    """
    F = _MeanMap(beta, g, gamma, mu, sigma, Jbar, **kw)
    lo, hi = rho_range

    def rho_of_m(m):
        if F.resid(m, lo) < 0:
            return -np.inf
        if F.resid(m, hi) > 0:
            return np.inf
        return brentq(lambda r: F.resid(m, r), lo, hi, xtol=1e-13)

    ms = np.linspace(-0.98, 0.98, n_m)
    rs = np.array([rho_of_m(m) for m in ms])
    rc = np.clip(rs, -1e9, 1e9)
    dr = np.sign(np.diff(rc))
    maxima = [i for i in range(1, n_m - 1) if dr[i - 1] > 0 and dr[i] < 0]
    minima = [i for i in range(1, n_m - 1) if dr[i - 1] < 0 and dr[i] > 0]
    if not maxima:
        return None
    if len(maxima) > 1 or len(minima) > 1:
        raise RuntimeError(f"rho(m) has {len(maxima)} maxima and {len(minima)} minima: more than one window")
    i = maxima[0]
    r = minimize_scalar(lambda m: -rho_of_m(m), bounds=(ms[i - 1], ms[i + 1]), method="bounded",
                        options=dict(xatol=1e-9))
    rho_up, m_up = -r.fun, r.x
    j = minima[0] if minima else None
    if j is None or rc[j] <= 0:
        # the captured branch survives at rho = 0: its m is the root of rho(m) = 0 left of the minimum
        k = [k for k in range(j if j is not None else i) if rc[k] > 0 >= rc[k + 1]]
        if not k:
            raise RuntimeError("could not locate the captured branch at rho = 0")
        rho_down, m_down = 0.0, brentq(lambda m: F.resid(m, 0.0), ms[k[0]], ms[k[0] + 1], xtol=1e-12)
    else:
        r = minimize_scalar(rho_of_m, bounds=(ms[j - 1], ms[j + 1]), method="bounded", options=dict(xatol=1e-9))
        rho_down, m_down = r.fun, r.x
    out = dict(rho_up=rho_up, m_up=m_up, rho_down=rho_down, m_down=m_down, rho_of_m=(ms, rs))
    for tag, rr, mm in (("up", rho_up, m_up), ("down", rho_down, m_down)):
        fp = static_fixed_point(beta, g, gamma, mu, sigma, Jbar, rr, m_clamp=mm, **kw)
        out["Lam_" + tag] = fp["Lam"]
        if fp["Lam"] >= 1:
            raise UnstableFixedPoint(f"Lam = {fp['Lam']:.3f} >= 1 at rho_{tag}")
    return out


def zealot_transient(benign, mu, sigma, Jbar, g, gamma, rho, T, M, rng, **kw):
    """Benign population plus a fraction rho of zealots at -1, all blocks equal (claim4 (1)), by the mode (a) or (b)
    forward pass. Returns the forward_pass output; m_B(t) = m[0]."""
    bl = Blocks([1 - rho, rho], Jbar, g, gamma)
    return forward_pass(bl, [benign, Frozen(-1.0)], [mu, 0.0], [sigma, 0.0], T, M, rng, **kw)


def forward_threshold_dynamic(benign, mu, sigma, Jbar, g, gamma, T, M, seed=0, lo=0.0, hi=0.5, tol=1e-3, **kw):
    """Horizon-T forward threshold (spec section 4, dynamic solver): the smallest rho with m_B(T) < 0 from the benign
    initial condition, by bisection with common random numbers (the same seed at every rho)."""
    mB = lambda r: zealot_transient(benign, mu, sigma, Jbar, g, gamma, r, T, M, np.random.default_rng(seed),
                                    **kw)["m"][0, T]
    if mB(lo) < 0 or mB(hi) > 0:
        raise ValueError("threshold not bracketed")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if mB(mid) < 0 else (mid, hi)
    return 0.5 * (lo + hi)


def hysteresis_dynamic(benign, mu, sigma, Jbar, g, gamma, rho_capture, rhos_back, T_capture, T_back, M, seed=0, **kw):
    """Hysteresis protocol (spec section 4): run at rho_capture until captured (m_B(T_capture) < 0), then continue
    the same pool at each lower rho, converting a fraction rho_capture - rho of zealots back to benign agents that
    start at -1, and record m_B after T_back more rounds. All blocks are equal, so the converted agents form a third
    population, frozen until T_capture and benign afterwards, and the capture phase is the same for every rho (common
    random numbers). Returns dict(rhos, m_B, rho_back = the largest rho with m_B > 0, or None)."""
    T = T_capture + T_back
    mB = []
    for r in rhos_back:
        bl = Blocks([1 - rho_capture, r, rho_capture - r], Jbar, g, gamma)
        maps = [benign, Frozen(-1.0), Frozen(-1.0, release_at=T_capture, then=benign)]
        o = forward_pass(bl, maps, [mu, 0.0, mu], [sigma, 0.0, sigma], T, M, np.random.default_rng(seed), **kw)
        if o["m"][0, T_capture] >= 0:
            raise ValueError(f"not captured at rho = {rho_capture} after {T_capture} rounds")
        w = np.array([1 - rho_capture, rho_capture - r])
        mB.append(float((w[0] * o["m"][0, T] + w[1] * o["m"][2, T]) / w.sum()))
    up = [r for r, x in zip(rhos_back, mB) if x > 0]
    return dict(rhos=np.asarray(rhos_back), m_B=np.array(mB), rho_back=max(up) if up else None)


# ----------------------------------------------------------------------------------------------------------------
# Mode (b): black-box map, surrogate route and live route (spec section 2)
# ----------------------------------------------------------------------------------------------------------------

def probe_design(n, mu, sigma, h_scale, rng, history=False, binary=False):
    """Single-agent probe inputs: hpriv ~ N(mu, sigma^2), field h uniform on +-2.5 h_scale, and (if history) the
    agent's previous state, uniform in [-1, 1] or +-1."""
    hp = mu + sigma * rng.standard_normal(n)
    h = h_scale * rng.uniform(-2.5, 2.5, n)
    s_last = (rng.choice([-1.0, 1.0], n) if binary else rng.uniform(-1, 1, n)) if history else None
    return hp, h, s_last


def fit_surrogate(agent, hpriv, h, s_last=None, family="auto", holdout=0.2, rng=None):
    """Surrogate route (spec section 2, item 1): fit F(hpriv, s_last, h) = tanh(a h + b hpriv + c s_last + d) from
    single-agent probes and return it as an IsingAgent (memory weight c on the last own state, lam = 0), so that mode
    (a) runs on it with the tangent or score estimator.

    agent: batched black box agent(hpriv, history, field). All probes go in one call, each with a history of length 1
    (s_last, or 0 without the history feature), so the probe cost is len(hpriv) queries. Noise law:
        "binary": outputs are +-1 with P(+1) = (1 + tanh x)/2, fitted by maximum likelihood;
        "tanh":   continuous outputs, least squares on tanh x, plus additive Gaussian noise with the residual s.d.
                  (taken as zero if below 1e-3 of the spread of the outputs).
    "auto" picks "binary" if every output is +-1. Held-out probes (fraction `holdout`) measure the misfit: r.m.s. of
    y - E[s | x], reported with the value a correct model would give (its noise floor).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    box = agent if isinstance(agent, BlackBoxMap) else BlackBoxMap(agent)
    n = len(hpriv)
    y = box(hpriv, (np.zeros(n) if s_last is None else s_last)[None, :], h)
    binary = bool(np.all(np.isin(y, (-1.0, 1.0)))) if family == "auto" else family == "binary"
    X = np.stack([h, hpriv] + ([s_last] if s_last is not None else []) + [np.ones(n)], 1)
    test = rng.random(n) < holdout
    Xtr, ytr = X[~test], y[~test]
    if binary:
        # P(y = +1) = sigmoid(2x), so the negative log-likelihood of y = +-1 is log(1 + exp(-2 y x))
        nll = lambda c: np.logaddexp(0.0, -2.0 * ytr * (Xtr @ c)).mean()
        jac = lambda c: ((-2.0 * ytr / (1.0 + np.exp(2.0 * ytr * (Xtr @ c))))[:, None] * Xtr).mean(0)
        coef = minimize(nll, np.zeros(X.shape[1]), jac=jac, method="BFGS", options=dict(gtol=1e-10)).x
        noise_sd = 0.0
    else:
        coef = least_squares(lambda c: np.tanh(Xtr @ c) - ytr, np.r_[1.0, np.zeros(X.shape[1] - 1)],
                             xtol=1e-15, ftol=1e-15, gtol=1e-15).x
        noise_sd = float(np.std(ytr - np.tanh(Xtr @ coef)))
        if noise_sd < 1e-3 * np.std(ytr):
            noise_sd = 0.0
    sur = IsingAgent(beta=coef[0], kappa=coef[2] if s_last is not None else 0.0, lam=0.0, w_priv=coef[1],
                     bias=coef[-1], stochastic=binary, out_noise=noise_sd)
    pred = np.tanh(X[test] @ coef)
    info = dict(coef=coef, n_queries=box.n_queries, family="binary" if binary else "tanh",
                heldout_rms=float(np.sqrt(np.mean((y[test] - pred) ** 2))),
                heldout_noise_floor=float(np.sqrt(np.mean(1 - pred**2))) if binary else noise_sd)
    return sur, info


def live_pool(agent, mu, sigma, T, M, K, rng, b=1.0, p_plus=1.0):
    """Live route (spec section 2, item 2; claim5 section 1): M trajectories on a random directed graph, each reading
    K randomly chosen others of the pool with signed weights b * (+-1, + with probability p_plus), fixed for the run:
        h_i(t) = b sum_k eps_ik s_{p_ik}(t)
    One agent call per round over the whole pool, M x (T+1) queries. No Gaussian or scalar-field assumption; the
    matching dense parameters are Jbar = K b (2 p_plus - 1), g^2 = K b^2 (claim1 (13)). For gamma = 0 only: shared
    ancestry is not controlled once K^T > M, so check by doubling M. Returns m (n,), C (n, n), s (n, M), counts."""
    box = agent if isinstance(agent, BlackBoxMap) else BlackBoxMap(agent)
    nbr = rng.integers(0, M - 1, (M, K))
    nbr += nbr >= np.arange(M)[:, None]                                   # nobody reads itself
    wts = b * np.where(rng.random((M, K)) < p_plus, 1.0, -1.0)
    hp = mu + sigma * rng.standard_normal(M)
    s = np.zeros((T + 1, M))
    s[0] = box(hp, np.zeros((0, M)), np.zeros(M))
    for t in range(T):
        s[t + 1] = box(hp, s[: t + 1], (wts * s[t][nbr]).sum(1))
    return dict(m=s.mean(1), C=s @ s.T / M, s=s, n_calls=box.n_calls, n_queries=box.n_queries)


# ----------------------------------------------------------------------------------------------------------------
# Mode (c): two-site cavity and participation ratio (spec section 3; claim3)
# ----------------------------------------------------------------------------------------------------------------

def _outer_moment(w, A, B):
    K, n = A.shape[0], A.shape[-1]
    return np.einsum("k,ka,kb->ab", w, A.reshape(K, -1), B.reshape(K, -1)).reshape((n,) * 4)


def site_moments(R=None, v=None, w=None, halves=None):
    """Site moments <R (x) R>, <R (x) v>, <v (x) v> with four free time indices (spec section 3 step 1).

    R, v: (K, n, n) per site or site type, weights w (default uniform). halves = (R1, R2, v1, v2): estimates from two
    disjoint halves of each site's thermal copies; the moments are then symmetrized cross products such as
    (R1 (x) R2 + R2 (x) R1) / 2, so the estimation noise of R_i, v_i does not inflate them.
    """
    if halves is not None:
        R1, R2, v1, v2 = halves
        w = np.full(R1.shape[0], 1.0 / R1.shape[0]) if w is None else np.asarray(w, float)
        sym = lambda A, B: 0.5 * (_outer_moment(w, A, B) + _outer_moment(w, B, A))
        return sym(R1, R2), 0.5 * (_outer_moment(w, R1, v2) + _outer_moment(w, R2, v1)), sym(v1, v2)
    R, v = np.asarray(R, float), np.asarray(v, float)
    w = np.full(R.shape[0], 1.0 / R.shape[0]) if w is None else np.asarray(w, float)
    return _outer_moment(w, R, R), _outer_moment(w, R, v), _outer_moment(w, v, v)


def four_point(MRR, MRv, Mvv, g, gamma, rho_b=1.0):
    """Two-site four-point functions (claim3 (5)-(12)), one updating population (fraction rho_b) plus frozen ones.

    Index placement follows theory/checks/two_site.py::four_point, the reference named by the spec. Arrays are
    [t, t', u, u'] for a pair (0, 1) of updating sites, normalized by N_b = rho_b N:
        SS = N_b <Sigma_01 (x) Sigma_01>,  XX = N_b <chi_01 (x) chi_01>,  XXt = N_b <chi_01 (x) chi_10>,
        SX = N_b <Sigma_01 (x) chi_01>,    SXt = N_b <Sigma_01 (x) chi_10>.
    R is strictly causal, so iterating from zero terminates; at most n sweeps (spec section 3 step 2).
    """
    n = MRR.shape[0]
    G2, tau = rho_b * g**2, rho_b * gamma * g**2        # N_b Var(J_01), N_b Cov(J_01, J_10)
    I2 = np.einsum("pq,rs->pqrs", np.eye(n), np.eye(n))
    MvR = MRv.transpose(2, 3, 0, 1)                      # <v[p,q] R[p',q']>
    E = lambda spec, *ops: np.einsum(spec, *ops, optimize=True)
    sym = lambda A: A + A.transpose(2, 3, 0, 1)
    SS, XX, XXt, SX, SXt = (np.zeros((n,) * 4) for _ in range(5))
    for sweep in range(n + 1):
        KK = G2 * I2 + G2**2 * (MRR + XX)               # claim3 (7)  N <K_01 (x) K_01>
        KKt = tau * I2 + tau**2 * (MRR + XXt)           # claim3 (8)  N <K_01 (x) K_10>
        cc = G2**2 * (Mvv + SS)                         # claim3 (9)  N <c_01 (x) c_01>
        cK = G2 * tau * (MvR + SX)                      # claim3 (10) N <c_01 (x) K_01>
        cKt = G2 * tau * (MvR + SXt)                    # claim3 (11) N <c_01 (x) K_10>
        # claim3 (5): chi_01 = R_0 K_01 R_1, chi_10 = R_1 K_10 R_0
        XX_new = E("tpux,qrys,pqxy->trus", MRR, MRR, KK)
        XXt_new = E("tpys,qrux,pqxy->trus", MRR, MRR, KKt)
        # claim3 (6): Sigma_01 = T1 + T2 + T3, T1 = R_0 c R_1^T, T2 = R_0 K_01 v_1, T3 = v_0 K_10^T R_1^T
        T1T1 = E("tpux,rqsy,pqxy->trus", MRR, MRR, cc)
        T2T2 = E("tpux,qrys,pqxy->trus", MRR, Mvv, KK)
        T3T3 = E("tquy,rpsx,pqxy->trus", Mvv, MRR, KK)
        T1T2 = E("tpux,rqys,pqxy->trus", MRR, MRv, cK)
        T1T3 = E("tpuy,rqsx,pqxy->trus", MRv, MRR, cKt)
        T2T3 = E("tpuy,sxqr,pqxy->trus", MRv, MRv, KKt)
        SS_new = T1T1 + T2T2 + T3T3 + sym(T1T2) + sym(T1T3) + sym(T2T3)
        SX_new = (E("tpux,rqys,pqxy->trus", MRR, MRR, cK) + E("tpux,ysqr,pqxy->trus", MRR, MRv, KK)
                  + E("uxtq,rpys,xypq->trus", MRv, MRR, KKt))
        SXt_new = (E("tpys,rqux,pqxy->trus", MRR, MRR, cKt) + E("tpys,uxqr,pqxy->trus", MRR, MRv, KKt)
                   + E("ystq,rpux,pqxy->trus", MRv, MRR, KK))
        new = (SS_new, XX_new, XXt_new, SX_new, SXt_new)
        delta = max(np.abs(x - y).max() for x, y in zip(new, (SS, XX, XXt, SX, SXt)))
        SS, XX, XXt, SX, SXt = new
        if delta <= 1e-13 * max(1.0, np.abs(SS).max()):
            return dict(SS=SS, XX=XX, XXt=XXt, SX=SX, SXt=SXt, sweeps=sweep + 1)
    raise RuntimeError("two-site system did not terminate within n sweeps: is R strictly causal?")


def psi_equal_time(r2, v2, g, rho_b=1.0):
    """Equal-time recursion for gamma = 0 and memoryless maps (claim3 (4)), with G2 = rho_b g^2 for frozen agents:
        psi_b(t+1) = G2^2 <r_i(t)^2>^2 [<v_i(t)^2> + psi_b(t)],   psi_b(0) = 0,
    r2[t] = <r_i(t)^2> with r_i(t) = d mu_i(t+1)/dh_i(t), v2[t] = <v_i(t, t)^2>. Returns psi_b (n,)."""
    G2 = rho_b * g**2
    psi = np.zeros(len(v2))
    for t in range(len(v2) - 1):
        psi[t + 1] = G2**2 * r2[t] ** 2 * (v2[t] + psi[t])
    return psi


def pr_over_n(vbar, v2, psi_b, Lambda1=None, N_B=None):
    """PR(t)/N_B = vbar(t)^2 / (<v_i(t)^2> + psi_b(t) + Lambda1(t)^2 / N_B)   (claim3 (13), (16); spec 3 step 3).
    psi_b = N_B <Sigma_ij(t,t)^2>, as returned by four_point and psi_equal_time; no further factor of rho_b."""
    extra = 0.0 if Lambda1 is None else np.asarray(Lambda1) ** 2 / N_B
    return np.asarray(vbar) ** 2 / (np.asarray(v2) + np.asarray(psi_b) + extra)


def mean_mode_transient(vbar, lam, kappa2, N_B, var0=0.0):
    """Mean-mode recursion (spec 3 step 3; claim3 section 5):
        Var dm(t+1) = lam(t)^2 Var dm(t) + vbar(t+1) / N_B,    Lambda1(t+1) = kappa(t)^2 N_B Var dm(t)
    with lam(t) = Jbar rho_B <r_i(t)> and kappa(t)^2 = (Jbar rho_B)^2 <r_i(t)^2>, all arrays over t (n,). Returns
    N_B Var(m_B)(t) and Lambda1(t). Its stationary limit is claim3 (16): N_B Var = vbar / (1 - lam^2),
    Lambda1 = kappa^2 vbar / (1 - lam^2)."""
    n = len(vbar)
    var, Lam1 = np.zeros(n), np.zeros(n)
    var[0] = var0
    for t in range(n - 1):
        var[t + 1] = lam[t] ** 2 * var[t] + vbar[t + 1] / N_B
        Lam1[t + 1] = kappa2[t] * N_B * var[t]
    return N_B * var, Lam1


def pr_from_sites(sites, g, gamma, rho_b=1.0, equal_time=False):
    """PR(t)/N_B without the mean mode, from nested single-site output (forward_pass with copies >= 4).

    sites = forward_pass(...)["sites"][a] of the updating population. All squared site averages use disjoint halves
    of the copies. equal_time=True uses claim3 (4) (gamma = 0, memoryless maps), otherwise the general system
    claim3 (5)-(13). Returns dict(pr, psi, vbar, v2)."""
    vd = np.einsum("ktt->kt", sites["v"])
    vd1, vd2 = np.einsum("ktt->kt", sites["v1"]), np.einsum("ktt->kt", sites["v2"])
    vbar, v2 = vd.mean(0), (vd1 * vd2).mean(0)
    if equal_time:
        n = vd.shape[1]
        r1 = np.array([sites["R1"][:, t + 1, t] for t in range(n - 1)])
        r2 = np.array([sites["R2"][:, t + 1, t] for t in range(n - 1)])
        psi = psi_equal_time(np.r_[(r1 * r2).mean(1), 0.0], v2, g, rho_b)
    else:
        MRR, MRv, Mvv = site_moments(halves=(sites["R1"], sites["R2"], sites["v1"], sites["v2"]))
        psi = np.einsum("tttt->t", four_point(MRR, MRv, Mvv, g, gamma, rho_b)["SS"])
    return dict(pr=pr_over_n(vbar, v2, psi), psi=psi, vbar=vbar, v2=v2)
