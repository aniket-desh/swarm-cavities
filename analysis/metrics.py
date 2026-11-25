"""Order parameters and diagnostics on belief arrays.

Shapes: ``s`` is (R replicas, T+1 rounds, N agents); ``labels`` is an optional (N,) int array of
population indices (default: one population). Means over agents and replicas ignore NaN (a missing
belief); the covariance / participation-ratio functions need complete data and raise otherwise.

Notation follows docs/spec.md Part 1: m_a(t) = <s_a(t)>, C_a(t, t') = <s_a(t) s_a(t')> (the full
correlation, which is what enters the cavity-noise covariance, see docs/literature.md (d) Claim 1),
Sigma_ij(t) = Cov_replicas(s_i(t), s_j(t)), PR(t) = (Tr Sigma)^2 / Tr(Sigma^2).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _labels(N: int, labels) -> tuple[np.ndarray, int]:
    lab = np.zeros(N, dtype=int) if labels is None else np.asarray(labels, dtype=int)
    return lab, int(lab.max()) + 1


# ---------------------------------------------------------------- means and two-time correlations

def population_means(s: np.ndarray, labels=None, per_replica: bool = False) -> np.ndarray:
    """m_a(t): mean belief of population a. Returns (P, T+1), or (R, P, T+1) if per_replica."""
    lab, P = _labels(s.shape[2], labels)
    m = np.stack([np.nanmean(s[:, :, lab == a], axis=2) for a in range(P)], axis=1)   # (R, P, T+1)
    return m if per_replica else np.nanmean(m, axis=0)


def two_time_correlation(s: np.ndarray, labels=None, connected: str | None = None) -> np.ndarray:
    """C_a(t, t') averaged over agents in a and replicas. Returns (P, T+1, T+1).

    connected=None:          C_a(t, t') = <s_i(t) s_i(t')>                   (full correlation)
    connected="population":  <s_i(t) s_i(t')> - m_a(t) m_a(t')               (the spec's <ds ds>)
    connected="agent":       <(s_i(t) - sbar_i(t)) (s_i(t') - sbar_i(t'))>, sbar_i(t) the replica mean,
                             with the R/(R-1) correction: the within-disorder (thermal) fluctuation,
                             meaningful only when replicas share J and fields.
    """
    lab, P = _labels(s.shape[2], labels)
    x = s
    if connected == "agent":
        R = s.shape[0]
        x = (s - np.nanmean(s, axis=0, keepdims=True)) * np.sqrt(R / (R - 1))
    C = np.stack([np.nanmean(np.einsum("rti,rui->rtui", x[:, :, lab == a], x[:, :, lab == a]), axis=(0, 3))
                  for a in range(P)])
    if connected == "population":
        m = population_means(s, labels)
        C = C - m[:, :, None] * m[:, None, :]
    elif connected not in (None, "agent"):
        raise ValueError(f"connected must be None, 'population' or 'agent', got {connected!r}")
    return C


# ---------------------------------------------------------------- covariance and participation ratio

def ensemble_covariance(s: np.ndarray) -> np.ndarray:
    """Sigma(t): N x N covariance across replicas at each round, (T+1, N, N), denominator R-1."""
    if not np.isfinite(s).all():
        raise ValueError("ensemble_covariance needs complete data (no NaN); drop or impute first")
    x = s - s.mean(axis=0, keepdims=True)
    return np.einsum("rti,rtj->tij", x, x) / (s.shape[0] - 1)


def participation_ratio(Sigma: np.ndarray) -> np.ndarray:
    """Plug-in PR = (Tr Sigma)^2 / Tr(Sigma^2) for one (N, N) or a stack (..., N, N) of covariances.

    From R replicas this is biased low by O(N/R): Tr(Sigma_hat^2) picks up sampling noise in every
    off-diagonal entry. Use ``participation_ratio_corrected`` on the raw beliefs instead.
    """
    tr = np.trace(Sigma, axis1=-2, axis2=-1)
    return tr**2 / np.einsum("...ij,...ji->...", Sigma, Sigma)


def _trace_moments_ustat(X: np.ndarray) -> tuple[float, float]:
    """Unbiased estimates of (Tr Sigma)^2 and Tr(Sigma^2) from R i.i.d. rows X (R, N), any distribution.

    Both are U-statistics over ordered quadruples of distinct replicas (q, r, u, v). With d = x_q - x_r
    and e = x_u - x_v independent with mean 0 and covariance 2 Sigma,
        E[(d . e)^2] / 4 = Tr(Sigma^2),     E[|d|^2 |e|^2] / 4 = (Tr Sigma)^2.
    Averaged over all quadruples they reduce to three invariants of the centered Gram matrix
    G = Y Y^T (Y = X - mean): a1 = (Tr G)^2, a2 = ||G||_F^2, a3 = sum_r G_rr^2. With D = R(R-2)(R-3),
        Tr(Sigma^2)^ = [ a1/(R-1) + (R-2) a2 - R a3 ] / D                  (Himeno & Yamada 2014, JMVA)
        (Tr Sigma)^2^ = [ ((R-2) - 1/(R-1)) a1 + 2 a2/(R-1) - R a3 ] / D
    (coefficients checked against the brute-force quadruple average in tests/test_metrics.py).
    """
    R = X.shape[0]
    if R < 4:
        raise ValueError("need at least 4 replicas")
    Y = X - X.mean(axis=0)
    G = Y @ Y.T
    a1, a2, a3 = np.trace(G) ** 2, (G**2).sum(), (np.diag(G) ** 2).sum()
    D = R * (R - 2) * (R - 3)
    tr2 = (a1 / (R - 1) + (R - 2) * a2 - R * a3) / D
    trsq = (((R - 2) - 1 / (R - 1)) * a1 + 2 * a2 / (R - 1) - R * a3) / D
    return trsq, tr2


def _trace_moments_gaussian(X: np.ndarray) -> tuple[float, float]:
    """Same targets assuming Gaussian rows. With n = R-1, (R-1) Sigma_hat ~ Wishart(Sigma, n) gives
        E[Tr(Sigma_hat^2)] = (1 + 1/n) Tr(Sigma^2) + (Tr Sigma)^2 / n,
        E[(Tr Sigma_hat)^2] = (Tr Sigma)^2 + 2 Tr(Sigma^2) / n,
    (so a bare R/(R-1) rescaling of Tr(Sigma_hat^2) is not enough), and solving the pair,
        Tr(Sigma^2)^  = n^2 / ((n-1)(n+2)) * [ Tr(Sigma_hat^2) - (Tr Sigma_hat)^2 / n ],
        (Tr Sigma)^2^ = (Tr Sigma_hat)^2 - 2 Tr(Sigma^2)^ / n.
    Bounded beliefs are not Gaussian (for +-1 spins Var is fixed by the mean), so the default is
    the distribution-free U-statistic; this one is kept as a cross-check.
    """
    n = X.shape[0] - 1
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / n
    A, B = np.trace(S) ** 2, (S**2).sum()
    tr2 = n**2 / ((n - 1) * (n + 2)) * (B - A / n)
    return A - 2 * tr2 / n, tr2


def participation_ratio_corrected(s: np.ndarray, method: str = "ustat") -> np.ndarray:
    """Finite-replica corrected PR(t) = (Tr Sigma)^2^ / Tr(Sigma^2)^ for each round. Returns (T+1,).

    Numerator and denominator are each unbiased (see _trace_moments_ustat); their ratio still has an
    O(1/R) bias from fluctuations, against O(N/R) for the plug-in estimate.
    """
    if not np.isfinite(s).all():
        raise ValueError("participation_ratio_corrected needs complete data (no NaN)")
    f = {"ustat": _trace_moments_ustat, "gaussian": _trace_moments_gaussian}[method]
    out = []
    for t in range(s.shape[1]):
        trsq, tr2 = f(s[:, t, :])
        out.append(trsq / tr2)
    return np.array(out)


def bootstrap_ci(stat, s: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap over replicas (axis 0): returns (estimate, lo, hi) of ``stat(s)``.

    Resampling with replacement duplicates replicas, and a duplicated pair looks like two replicas
    that agree perfectly, which biases statistics built from distinct replica pairs (the corrected
    PR, the replica overlap). For those use ``subsample_ci``: with independent Gaussian agents
    (N=32, R=60) the bootstrap interval for the corrected PR is centred ~11 below the estimate and
    covered the true PR in 0 of 100 trials, while the subsampling interval covered it in 100 of 100.
    """
    rng = rng or np.random.default_rng()
    R = s.shape[0]
    est = np.asarray(stat(s))
    boots = np.stack([np.asarray(stat(s[rng.integers(0, R, R)])) for _ in range(n_boot)])
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return est, lo, hi


def subsample_ci(stat, s: np.ndarray, frac: float = 0.5, n_sub: int = 1000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsampling CI over replicas without replacement (Politis-Romano), for statistics that need
    distinct replicas. Draws b = frac * R replicas, and rescales the spread of stat(sub) - stat(s) by
    sqrt(b / R) assuming a root-R estimator: interval est - q_hi * sqrt(b/R) ... est - q_lo * sqrt(b/R).
    """
    rng = rng or np.random.default_rng()
    R = s.shape[0]
    b = max(4, int(round(frac * R)))
    est = np.asarray(stat(s))
    dev = np.stack([np.asarray(stat(s[rng.choice(R, b, replace=False)])) - est for _ in range(n_sub)])
    q_lo, q_hi = np.nanquantile(dev, [alpha / 2, 1 - alpha / 2], axis=0)
    scale = np.sqrt(b / (R - b))
    return est, est - q_hi * scale, est - q_lo * scale


# ---------------------------------------------------------------- freezing: replica overlap

def replica_overlap(s: np.ndarray, labels=None) -> np.ndarray:
    """q_a(t) = mean_{i in a} mean_{replicas r != r'} s_i^r(t) s_i^r'(t). Returns (P, T+1).

    All replicas in ``s`` must share the disorder (J, private fields, statement) and differ only in
    sampling. Uses sum_{r != r'} x_r x_r' = (sum_r x_r)^2 - sum_r x_r^2 per agent.
    """
    lab, P = _labels(s.shape[2], labels)
    tot = np.nansum(s, axis=0)
    sq = np.nansum(s**2, axis=0)
    n = np.isfinite(s).sum(axis=0)
    q_i = (tot**2 - sq) / (n * (n - 1))                   # (T+1, N), per agent
    return np.stack([np.nanmean(q_i[:, lab == a], axis=1) for a in range(P)])


def overlap_excess(s: np.ndarray, baseline, labels=None) -> np.ndarray:
    """Freezing order parameter q(t) - q_field(t), returns (P, T+1).

    A quenched private field alone holds each agent near its own sign, so replicas agree (q > 0) even
    with no coupling; that part says nothing about echo-chamber locking (docs/literature.md (d) Claim 2).
    ``baseline`` is either the field-only overlap (scalar or array broadcastable to (P, T+1)) or an
    uncoupled (g = 0) control belief array (R0, T+1, N) with the same fields, whose overlap is used.
    """
    q = replica_overlap(s, labels)
    b = np.asarray(baseline, dtype=float)
    if b.ndim == 3:
        b = replica_overlap(b, labels)
    return q - b


# ---------------------------------------------------------------- relaxation and early warning

def autocorrelation_from_C(C: np.ndarray, t0: int = 0) -> np.ndarray:
    """Time-translation-averaged normalized autocorrelation a(tau) = <C(t+tau, t)>_t / <C(t, t)>_t
    over t >= t0, for one (T+1, T+1) two-time matrix. Returns a(0..T-t0). Use a connected C:
    the full C of agents in a quenched field has a plateau, and a(tau) would never decay.
    """
    Tp = C.shape[0]
    diag = np.mean([C[t, t] for t in range(t0, Tp)])
    return np.array([np.mean([C[t + tau, t] for t in range(t0, Tp - tau)]) for tau in range(Tp - t0)]) / diag


def correlation_time(a: np.ndarray, method: str = "integrated") -> float:
    """Correlation time from a(tau) with a(0) = 1.

    integrated: tau_int = 1/2 + sum_{tau>=1} a(tau), summed up to the first non-positive a. For
                a = exp(-tau / tau_c) this equals coth(1 / (2 tau_c)) / 2 ~ tau_c.
    exp:        least-squares slope of log a(tau) through the origin over lags with a > 0,
                tau_c = -1 / slope (exact for a pure exponential).
    With T ~ 8-10 rounds, a correlation time beyond a few rounds is a lower bound, not a measurement.
    """
    a = np.asarray(a, dtype=float)
    stop = np.argmax(a[1:] <= 0) + 1 if np.any(a[1:] <= 0) else len(a)
    if method == "integrated":
        return float(0.5 + a[1:stop].sum())
    if method == "exp":
        tau = np.arange(1, stop)
        if tau.size == 0:
            return 0.0
        slope = (tau * np.log(a[1:stop])).sum() / (tau**2).sum()
        return float(-1.0 / slope) if slope < 0 else float("inf")
    raise ValueError(method)


def ensemble_ews(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble early-warning baselines from per-replica means m (R, T+1) (e.g. one population of
    ``population_means(..., per_replica=True)``): Var_r m(t), shape (T+1,), and the lag-1
    autocorrelation Corr_r(m(t), m(t+1)), shape (T,). Compare these with PR(t), not raw values.
    """
    v = np.nanvar(m, axis=0, ddof=1)
    x = m - np.nanmean(m, axis=0)
    c = np.nanmean(x[:, :-1] * x[:, 1:], axis=0) / np.sqrt(np.nanmean(x[:, :-1] ** 2, axis=0)
                                                            * np.nanmean(x[:, 1:] ** 2, axis=0))
    return v, c


def rolling_ews(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Single-run early-warning baselines on a time series x (T+1,) (Scheffer et al. 2009): variance
    and lag-1 autocorrelation over trailing windows of ``window`` points, detrended by the window
    mean. Entry t uses x[t-window+1 .. t]; the first window-1 entries are NaN.
    """
    x = np.asarray(x, dtype=float)
    var = np.full(x.shape, np.nan)
    ac1 = np.full(x.shape, np.nan)
    for t in range(window - 1, len(x)):
        w = x[t - window + 1: t + 1] - x[t - window + 1: t + 1].mean()
        var[t] = (w**2).mean()
        den = (w**2).sum()
        ac1[t] = (w[:-1] * w[1:]).sum() / den if den > 0 else np.nan
    return var, ac1


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC = P(score_pos > score_neg) + P(tie)/2, via the Mann-Whitney rank sum."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- one-step prediction accuracy

def flip_class_balanced_accuracy(pred: np.ndarray, s_next: np.ndarray, s_prev: np.ndarray,
                                 mask: np.ndarray | None = None) -> float:
    """Flip- and class-balanced one-step accuracy (%), from fcba in
    third_party/physics-of-agents/res/utils.py (Apache-2.0).

    Spins in {-1, +1}, 0 = unparsed (excluded). Transitions split into four groups, (flip vs stay) x
    (s_next = +1 vs -1), and the score is the unweighted mean of accuracy over the non-empty groups,
    so always predicting "stay" scores ~50 even though most transitions are stays. Arrays share any
    shape; ``mask`` (broadcastable) selects transitions to score. PoA builds s_prev from s(0) and
    the observed s(1..T-1), i.e. one-step prediction from the true previous state.
    """
    pred, s_next, s_prev = (np.asarray(a) for a in (pred, s_next, s_prev))
    ok = (s_next != 0) & (s_prev != 0)
    if mask is not None:
        ok &= np.broadcast_to(mask, ok.shape)
    correct = pred == s_next
    groups = [ok & (s_next != s_prev) & (s_next == v) for v in (1, -1)]
    groups += [ok & (s_next == s_prev) & (s_next == v) for v in (1, -1)]
    return 100 * float(np.mean([correct[gr].mean() for gr in groups if gr.any()]))
