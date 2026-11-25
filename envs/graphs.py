"""Coupling matrices for P populations, plus a log of what each realization actually is.

Convention (same as docs/spec.md Part 1): J[i, j] is the influence of agent j on agent i, so the
field on i is h_i = sum_j J_ij s_j, i.e. ``h = J @ s``. Rows are receivers, columns senders.
Physics of Agents stores the transpose (J[i, j] = edge sender i -> receiver j, see
analysis/poa_data.py); feed ``J.T`` to its inbox code.

Populations are contiguous blocks: ``sizes = [N_1, ..., N_P]`` and agent i belongs to population
``population_labels(sizes)[i]``. Block (a, b) means receiver in a, sender in b.
"""

from __future__ import annotations

import numpy as np


def population_labels(sizes) -> np.ndarray:
    """(N,) int population index for contiguous blocks of the given sizes."""
    return np.repeat(np.arange(len(sizes)), sizes)


def _as_block(x, P: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = np.full((P, P), float(x))
    if x.shape != (P, P):
        raise ValueError(f"{name} must be a scalar or ({P}, {P}), got shape {x.shape}")
    return x


def gaussian_couplings(sizes, Jbar, g, gamma, rng: np.random.Generator) -> np.ndarray:
    """Dense Gaussian J (N x N, zero diagonal) with block-structured moments.

    For i in population a, j in population b (i != j):
        E[J_ij] = Jbar[a, b] / N,   Var(J_ij) = g[a, b]^2 / N,   Corr(J_ij, J_ji) = gamma[a, b].

    Construction. Draw Z with i.i.d. N(0, 1) entries and mix each entry with its transpose,
        Zt_ij = (Z_ij + c_ab Z_ji) / sqrt(1 + c_ab^2),   c = gamma / (1 + sqrt(1 - gamma^2)),
    so Var(Zt_ij) = 1 and Cov(Zt_ij, Zt_ji) = 2c / (1 + c^2) = gamma exactly, for any gamma in [-1, 1]
    (c = 1 gives Zt symmetric, c = -1 antisymmetric). This needs c_ab = c_ba, hence gamma symmetric.
    The same line covers within-block pairs (both entries in block (a, a)) and cross-block pairs
    (Zt_ij in block (a, b), Zt_ji in block (b, a)). Then J_ij = Jbar_ab / N + g_ab Zt_ij / sqrt(N),
    so Cov(J_ij, J_ji) = gamma_ab g_ab g_ba / N, which is the Onsager-term prefactor in theory/.
    """
    sizes = [int(n) for n in sizes]
    P, N = len(sizes), int(sum(sizes))
    Jbar, g, gamma = (_as_block(x, P, nm) for x, nm in ((Jbar, "Jbar"), (g, "g"), (gamma, "gamma")))
    if not np.allclose(gamma, gamma.T):
        raise ValueError("gamma must be symmetric in (a, b): Corr(J_ij, J_ji) is one number per pair")
    if np.any(np.abs(gamma) > 1) or np.any(g < 0):
        raise ValueError("need |gamma| <= 1 and g >= 0")

    lab = population_labels(sizes)
    blk = lambda M: M[lab[:, None], lab[None, :]]          # expand P x P to N x N
    c = blk(gamma / (1.0 + np.sqrt(1.0 - gamma**2)))
    Z = rng.standard_normal((N, N))
    Zt = (Z + c * Z.T) / np.sqrt(1.0 + c**2)
    J = blk(Jbar) / N + blk(g) * Zt / np.sqrt(N)
    np.fill_diagonal(J, 0.0)
    return J


def sparse_signed_couplings(N: int, k: float, gamma: float, p_neg: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Directed signed graph J in {-1, 0, +1} (int8, zero diagonal), mean in-degree k.

    Reciprocity here means the same thing as for the dense ensemble: gamma = Corr(J_ij, J_ji), the
    Pearson correlation over all ordered pairs i != j (non-edges count as 0). That is the quantity
    the cavity equations see. Network-science edge reciprocity r = P(J_ji != 0 | J_ij != 0) and
    sign agreement on reciprocated pairs are reported by ``spectral_log`` alongside it.

    Model. Write J_ij = A_ij sigma_ij with edge density p = k / (N - 1) and P(sigma = -1) = p_neg,
    so mu = E[J] = p mu_s with mu_s = 1 - 2 p_neg, and Var(J) = p - p^2 mu_s^2. Each unordered pair
    is reciprocated with probability p r, single in either direction with probability p (1 - r)
    each, and empty otherwise, so every ordered pair is an edge with probability p.
      * gamma >= gamma_0: reciprocated pairs are concordant (sigma_ji = sigma_ij, both negative with
        probability p_neg). Then E[J_ij J_ji] = p r and
            gamma = (r - p mu_s^2) / (1 - p mu_s^2).
      * gamma < gamma_0: reciprocated pairs are discordant ((+,-) or (-,+) with probability 1/2),
        and single edges are negative with probability (p_neg - r/2) / (1 - r) so that the overall
        negative fraction stays p_neg. Then E[J_ij J_ji] = -p r and
            gamma = -(r + p mu_s^2) / (1 - p mu_s^2).
    Both branches meet at r = 0, gamma_0 = -p mu_s^2 / (1 - p mu_s^2), so r = |gamma - gamma_0| (1 - p mu_s^2).
    Feasible range: r <= 1 for gamma > gamma_0, and r <= 2 min(p_neg, 1 - p_neg) for gamma < gamma_0
    (with p_neg != 1/2, strongly negative gamma would change the sign balance).
    The target holds in expectation; the realized value is in ``spectral_log(J)``.
    """
    p = k / (N - 1)
    if not (0 < p <= 0.5):
        raise ValueError(f"need 0 < k/(N-1) <= 0.5, got {p:.3g}")
    a = p * (1.0 - 2.0 * p_neg) ** 2                       # p mu_s^2
    gamma0 = -a / (1.0 - a)
    r = abs(gamma - gamma0) * (1.0 - a)
    concordant = gamma >= gamma0
    r_max = 1.0 if concordant else 2.0 * min(p_neg, 1.0 - p_neg)
    if r > r_max + 1e-12:
        raise ValueError(f"gamma={gamma} infeasible for p_neg={p_neg}, k={k}, N={N} "
                         f"(needs edge reciprocity {r:.3f} > {r_max:.3f})")

    # One uniform per unordered pair over the upper triangle, as in sample_J_num_edges_symmetric of
    # third_party/physics-of-agents/lib/datagen/graph.py (Apache-2.0), extended to directed pairs.
    iu, ju = np.triu_indices(N, k=1)
    u = rng.random(iu.size)
    both = u < p * r
    fwd = (u >= p * r) & (u < p * r + p * (1 - r))        # only i <- j, i.e. J[i, j] != 0 with i < j
    bwd = (u >= p * r + p * (1 - r)) & (u < p * r + 2 * p * (1 - r))

    def signs(n, q):
        return np.where(rng.random(n) < q, -1, 1).astype(np.int8)

    if concordant:
        p_neg_single = p_neg
    else:                                                  # keep the overall negative fraction at p_neg
        p_neg_single = (p_neg - r / 2) / (1 - r) if r < 1 else 0.5
    J = np.zeros((N, N), dtype=np.int8)
    J[iu[fwd], ju[fwd]] = signs(fwd.sum(), p_neg_single)
    J[ju[bwd], iu[bwd]] = signs(bwd.sum(), p_neg_single)
    nb = int(both.sum())
    if concordant:
        s = signs(nb, p_neg)
        J[iu[both], ju[both]] = s
        J[ju[both], iu[both]] = s
    else:
        s = signs(nb, 0.5)
        J[iu[both], ju[both]] = s
        J[ju[both], iu[both]] = -s
    return J


def _pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of paired samples; nan if either side is constant."""
    sx, sy = x.std(), y.std()
    if x.size < 2 or sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def spectral_log(J: np.ndarray, labels=None, eigs: bool = True) -> dict:
    """What a realized coupling matrix actually is, for logging next to every experiment.

    Per block (a, b) (receiver a, sender b), over off-diagonal entries:
      N_mean = N * mean(J_ij)    (estimates Jbar_ab)
      N_var  = N * var(J_ij)     (estimates g_ab^2)
      gamma  = Corr(J_ij, J_ji) over ordered pairs i in a, j in b (estimates gamma_ab; for a != b
               this pairs block (a, b) with block (b, a))
      in_degree = mean number of nonzero J_ij per receiver i in a from senders in b
      frac_neg  = fraction of nonzero entries that are negative
      edge_reciprocity = P(J_ji != 0 | J_ij != 0), sign_agreement = P(sign J_ji = sign J_ij | both != 0)
    plus the eigenvalues, spectral radius and largest real part of J (which set linear stability).
    Values are Python floats / nested lists except ``eigvals`` (complex ndarray). ``eigs=False``
    skips the O(N^3) eigendecomposition.
    """
    J = np.asarray(J, dtype=float)
    N = J.shape[0]
    lab = np.zeros(N, dtype=int) if labels is None else np.asarray(labels)
    P = int(lab.max()) + 1
    off = ~np.eye(N, dtype=bool)
    keys = ("N_mean", "N_var", "gamma", "in_degree", "frac_neg", "edge_reciprocity", "sign_agreement")
    out = {k: np.full((P, P), np.nan) for k in keys}
    for a in range(P):
        for b in range(P):
            m = (lab[:, None] == a) & (lab[None, :] == b) & off
            ii, jj = np.nonzero(m)
            x, y = J[ii, jj], J[jj, ii]                     # (J_ij, J_ji) for every ordered pair
            nz = x != 0
            out["N_mean"][a, b] = N * x.mean()
            out["N_var"][a, b] = N * x.var()
            out["gamma"][a, b] = _pair_corr(x, y)
            out["in_degree"][a, b] = nz.sum() / max((lab == a).sum(), 1)
            if nz.any():
                out["frac_neg"][a, b] = (x[nz] < 0).mean()
                out["edge_reciprocity"][a, b] = (y[nz] != 0).mean()
                both = nz & (y != 0)
                if both.any():
                    out["sign_agreement"][a, b] = (np.sign(x[both]) == np.sign(y[both])).mean()
    ii, jj = np.nonzero(off)
    log = {k: v.tolist() for k, v in out.items()}
    log.update(
        N=N, P=P, sizes=np.bincount(lab, minlength=P).tolist(),
        gamma_all=_pair_corr(J[ii, jj], J[jj, ii]),
        mean_in_degree=float((J[off] != 0).sum() / N),
    )
    if eigs:
        ev = np.linalg.eigvalsh(J).astype(complex) if np.array_equal(J, J.T) else np.linalg.eigvals(J)
        log.update(spectral_radius=float(np.abs(ev).max()), max_real_eig=float(ev.real.max()), eigvals=ev)
    return log
