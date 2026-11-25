"""Toy swarm used to check the formulas in theory/claim*.md.

Model (theory/claim1.md, eq. (1)):
    h_i(t)   = sum_j J_ij s_j(t)
    u_i(t)   = lam * u_i(t-1) + (1 - lam) * s_i(t)          (context-window memory, u_i(-1) = 0)
    x_i(t)   = beta_a * h_i(t) + hpriv_i + kappa_a * u_i(t)
    s_i(t+1) = tanh(x_i(t))                                  (deterministic mock Ising agent)
            or +-1 with P(+1) = (1 + tanh(x_i(t))) / 2       (stochastic variant)
    s_i(0)   = tanh(hpriv_i)   (or its +-1 sample)

Couplings, i in population a, j in population b:
    J_ij = Jbar_ab / N + (g_ab / sqrt(N)) z_ij,  Corr(z_ij, z_ji) = gamma_ab,  J_ii = 0.

Two functions matter:
    simulate(...)  direct N-agent simulation
    dmft_mc(...)   single-site DMFT, eqs. (4)-(8) of claim1.md, solved in one causal forward pass by
                   Monte Carlo over M single-site trajectories (Eissfeller-Opper in discrete time).
"""

import numpy as np


def make_labels(N, rho):
    """Population label of each agent; sizes round(rho_a N) with the remainder in the last population."""
    sizes = [int(round(r * N)) for r in rho[:-1]]
    sizes.append(N - sum(sizes))
    return np.repeat(np.arange(len(rho)), sizes)


def sample_J(rng, labels, Jbar, g, gamma):
    """Dense couplings with block means Jbar/N, block variances g^2/N and pair correlation gamma (symmetric)."""
    N = len(labels)
    Jbar, g, gamma = (np.asarray(a, float) for a in (Jbar, g, gamma))
    assert np.allclose(gamma, gamma.T), "gamma_ab must be symmetric"
    X = rng.standard_normal((N, N))
    Xu, Xl = np.triu(X, 1), np.tril(X, -1)
    G = gamma[np.ix_(labels, labels)]
    # upper triangle: z_ij = X_ij ; lower triangle: z_ji = gamma X_ij + sqrt(1 - gamma^2) X_ji
    Z = Xu + np.tril(G * Xu.T + np.sqrt(1.0 - G**2) * Xl, -1)
    J = Jbar[np.ix_(labels, labels)] / N + g[np.ix_(labels, labels)] / np.sqrt(N) * Z
    np.fill_diagonal(J, 0.0)
    return J


def simulate(rng, J, labels, hpriv, beta, kappa, lam, T, stochastic=False, s0=None, theta=None, frozen=None):
    """Direct simulation. Returns s with shape (T+1, N).

    theta:  optional (T, N) extra field added to h(t) (for measuring responses).
    frozen: optional boolean mask of agents whose state never changes (zealots); they keep s0.
    """
    N = len(labels)
    beta, kappa = np.asarray(beta, float)[labels], np.asarray(kappa, float)[labels]
    s = np.empty((T + 1, N))
    if s0 is None:
        p = np.tanh(hpriv)
        s0 = np.where(rng.random(N) < (1 + p) / 2, 1.0, -1.0) if stochastic else p
    s[0] = s0
    u = np.zeros(N)
    for t in range(T):
        u = lam * u + (1 - lam) * s[t]
        h = J @ s[t]
        if theta is not None:
            h = h + theta[t]
        p = np.tanh(beta * h + hpriv + kappa * u)
        s[t + 1] = np.where(rng.random(N) < (1 + p) / 2, 1.0, -1.0) if stochastic else p
        if frozen is not None:
            s[t + 1, frozen] = s[0, frozen]
    return s


def dmft_mc(rng, rho, Jbar, g, gamma, mu, sig, beta, kappa, lam, T, M, stochastic=False,
            cov="full", echo_shift=0, reaction=True, frozen_pops=(), probe_eps=0.0):
    """Single-site DMFT by Monte Carlo, one causal forward pass (claim1.md eqs. (4)-(8)).

    Returns m[a, t], C[a, t, t'] = <s(t) s(t')>, R[a, t, t'] = <d s(t) / d h(t')>  (zero unless t' < t).

    Options that reproduce the spec's two errors, for comparison only:
        cov="connected"  uses <ds ds> instead of <s s> in the noise covariance.
        echo_shift=1     uses sum_{t'} R(t, t'+1) s(t') instead of sum_{t'} R(t, t') s(t').
    frozen_pops: populations whose agents never update (zealots): s(t) = s(0), R = 0.
    probe_eps:   if > 0, adds an independent white probe theta(t) ~ N(0, probe_eps^2) to every field, returned as
                 'theta', for the black-box response estimator R(t,t') = <s(t) theta(t')> / probe_eps^2 (solver_spec.md).
    """
    P = len(rho)
    rho, Jbar, g, gamma = (np.asarray(a, float) for a in (rho, Jbar, g, gamma))
    Gam = rho[None, :] * gamma * g * g.T          # Gam[a, b] = rho_b gamma_ab g_ab g_ba   (eq. (6))
    hpriv = [mu[a] + sig[a] * rng.standard_normal(M) for a in range(P)]
    s = np.zeros((P, T + 1, M))
    x = np.zeros((P, T, M))
    for a in range(P):
        p = np.tanh(hpriv[a])
        s[a, 0] = np.where(rng.random(M) < (1 + p) / 2, 1.0, -1.0) if stochastic else p
    u = np.zeros((P, M))
    tan = np.zeros((P, T + 1, T, M))               # tan[a, t, t'] = d s(t) / d h(t') per trajectory
    dudh = np.zeros((P, T, M))                     # d u(t) / d h(t')
    eta = np.zeros((P, T, M))                      # cavity noise eta_a(t) per trajectory
    z = rng.standard_normal((P, T, M))
    theta = probe_eps * rng.standard_normal((P, T, M)) if probe_eps > 0 else np.zeros((P, T, M))
    m = np.zeros((P, T + 1)); C = np.zeros((P, T + 1, T + 1)); R = np.zeros((P, T + 1, T + 1))
    m[:, 0] = s[:, 0].mean(-1)
    C[:, 0, 0] = (s[:, 0] ** 2).mean(-1)
    for t in range(T):
        Cuse = C - m[:, :, None] * m[:, None, :] if cov == "connected" else C
        for a in range(P):
            # eq. (5): <eta_a(t) eta_a(t')> = sum_b rho_b g_ab^2 C_b(t, t'), sampled causally
            # (conditional Gaussian on the past noise; pinv tolerates a degenerate covariance, e.g. frozen s)
            K = np.einsum("b,bij->ij", rho * g[a] ** 2, Cuse[:, : t + 1, : t + 1])
            w = np.linalg.pinv(K[:t, :t], hermitian=True) @ K[:t, t] if t else np.zeros(0)
            eta[a, t] = w @ eta[a, :t] + np.sqrt(max(K[t, t] - K[:t, t] @ w, 0.0)) * z[a, t]
            # eq. (4): mean field + noise + Onsager reaction term
            h = (rho * Jbar[a]) @ m[:, t] + eta[a, t] + theta[a, t]
            dh = np.zeros((T, M)); dh[t] = 1.0     # d h(t) / d h_ext(t')
            if reaction:
                for b in range(P):
                    if Gam[a, b] == 0.0:
                        continue
                    for tp in range(echo_shift, t):
                        w = Gam[a, b] * R[b, t, tp]
                        # correct: R(t, t') s(t').  spec (echo_shift=1): R(t, t'+1) s(t'), i.e. R(t, tp) s(tp-1)
                        h = h + w * s[a, tp - echo_shift]
                        dh += w * tan[a, tp - echo_shift]
            u[a] = lam * u[a] + (1 - lam) * s[a, t]
            dudh[a] = lam * dudh[a] + (1 - lam) * tan[a, t]
            x[a, t] = beta[a] * h + hpriv[a] + kappa[a] * u[a]
            p = np.tanh(x[a, t])
            if a in frozen_pops:
                s[a, t + 1] = s[a, 0]
                continue
            if stochastic:
                s[a, t + 1] = np.where(rng.random(M) < (1 + p) / 2, 1.0, -1.0)
            else:
                s[a, t + 1] = p
                tan[a, t + 1] = (1 - p**2) * (beta[a] * dh + kappa[a] * dudh[a])
        for a in range(P):
            m[a, t + 1] = s[a, t + 1].mean()
            C[a, t + 1, : t + 2] = (s[a, t + 1] * s[a, : t + 2]).mean(-1)
            C[a, : t + 2, t + 1] = C[a, t + 1, : t + 2]
            if a in frozen_pops:
                continue
            if stochastic:
                # score-function estimator: R(t+1, t') = beta <s(t+1) (s(t'+1) - tanh x(t'))>, t' <= t
                R[a, t + 1, : t + 1] = beta[a] * (s[a, t + 1] * (s[a, 1 : t + 2] - np.tanh(x[a, : t + 1]))).mean(-1)
                R[a, t + 1, t] = beta[a] * (1 - np.tanh(x[a, t]) ** 2).mean()   # exact lag-1 value, less noise
            else:
                R[a, t + 1, :T] = tan[a, t + 1].mean(-1)
    return dict(m=m, C=C, R=R, s=s, hpriv=hpriv, eta=eta, x=x, theta=theta)


def empirical_moments(s, labels, P):
    """m[a, t] and C[a, t, t'] = (1/N_a) sum_i s_i(t) s_i(t') from one simulated trajectory s[t, i]."""
    m = np.stack([s[:, labels == a].mean(1) for a in range(P)])
    C = np.stack([s[:, labels == a] @ s[:, labels == a].T / (labels == a).sum() for a in range(P)])
    return m, C
