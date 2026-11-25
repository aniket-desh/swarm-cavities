"""analysis/metrics.py on synthetic data with known answers."""

import itertools

import numpy as np
import pytest

from analysis import metrics as M


def test_ustat_trace_moments_equal_bruteforce_quadruple_average():
    """The closed forms are exact rewrites of the average over ordered distinct quadruples."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 3)) ** 3                # non-Gaussian on purpose
    u1 = u2 = 0.0
    quads = list(itertools.permutations(range(7), 4))
    for q, r, u, v in quads:
        d, e = X[q] - X[r], X[u] - X[v]
        u2 += (d @ e) ** 2 / 4
        u1 += (d @ d) * (e @ e) / 4
    trsq, tr2 = M._trace_moments_ustat(X)
    assert trsq == pytest.approx(u1 / len(quads), rel=1e-10)
    assert tr2 == pytest.approx(u2 / len(quads), rel=1e-10)


@pytest.mark.parametrize("method", ["ustat", "gaussian"])
def test_trace_moments_unbiased_for_gaussian_known_sigma(method):
    rng = np.random.default_rng(1)
    N, R = 10, 12
    A = rng.standard_normal((N, N))
    Sig = A @ A.T / N + np.eye(N)
    L = np.linalg.cholesky(Sig)
    est = np.array([getattr(M, f"_trace_moments_{method}")(rng.standard_normal((R, N)) @ L.T)
                    for _ in range(20000)])
    for k, target in enumerate([np.trace(Sig) ** 2, np.trace(Sig @ Sig)]):
        se = est[:, k].std() / np.sqrt(len(est))
        assert abs(est[:, k].mean() - target) < 4 * se


def test_pr_independent_gaussian_agents_is_N_after_correction():
    rng = np.random.default_rng(2)
    N, R = 32, 50
    s = rng.standard_normal((R, 3, N))
    plug = M.participation_ratio(M.ensemble_covariance(s))
    assert np.all(plug < 0.75 * N)                      # plug-in ~ N / (1 + N/R) ~ 19.5
    pr = np.array([M.participation_ratio_corrected(rng.standard_normal((R, 1, N)))[0] for _ in range(300)])
    assert pr.mean() == pytest.approx(N, rel=0.02)


def test_pr_independent_pm1_spins_ustat_unbiased_gaussian_not():
    """Bounded beliefs: the Gaussian-moment correction overshoots (~33.5 for N=32, R=40)."""
    rng = np.random.default_rng(3)
    N, R = 32, 40
    pr_u = [M.participation_ratio_corrected(rng.choice([-1.0, 1.0], (R, 1, N)))[0] for _ in range(400)]
    assert np.mean(pr_u) == pytest.approx(N, rel=0.015)
    pr_g = [M.participation_ratio_corrected(rng.choice([-1.0, 1.0], (R, 1, N)), "gaussian")[0] for _ in range(400)]
    assert np.mean(pr_g) > 1.025 * N


def test_pr_rank1_plus_noise_analytic():
    """Sigma = I + c u u^T: Tr = N + c, Tr(Sigma^2) = N + 2c + c^2."""
    rng = np.random.default_rng(4)
    N, c, R = 32, 10.0, 800
    u = rng.standard_normal(N)
    u /= np.linalg.norm(u)
    Sig = np.eye(N) + c * np.outer(u, u)
    pr_true = (N + c) ** 2 / (N + 2 * c + c**2)
    assert M.participation_ratio(Sig) == pytest.approx(pr_true)
    L = np.linalg.cholesky(Sig)
    pr = [M.participation_ratio_corrected(rng.standard_normal((R, 1, N)) @ L.T)[0] for _ in range(100)]
    assert np.mean(pr) == pytest.approx(pr_true, rel=0.02)


def test_subsample_ci_covers_pr_independent_agents():
    rng = np.random.default_rng(5)
    N, R = 16, 40
    f = lambda x: M.participation_ratio_corrected(x)[0]
    hits = 0
    for _ in range(30):
        _, lo, hi = M.subsample_ci(f, rng.standard_normal((R, 1, N)), n_sub=200, rng=rng)
        hits += lo <= N <= hi
    assert hits >= 26


def test_means_and_two_time_correlations_hand_example():
    # 2 replicas, 2 rounds, 3 agents; agents 0,1 in population 0, agent 2 in population 1.
    s = np.array([[[1, 1, -1], [1, -1, -1]],
                  [[1, -1, 1], [-1, -1, 1]]], dtype=float)
    lab = np.array([0, 0, 1])
    m = M.population_means(s, lab)
    np.testing.assert_allclose(m, [[0.5, -0.5], [0.0, 0.0]])
    C = M.two_time_correlation(s, lab)
    # pop 0, C(0,1) = mean of s(0)s(1) over (r,i) = (1*1 + 1*-1 + 1*-1 + -1*-1)/4 = 0
    np.testing.assert_allclose(C[0], [[1, 0], [0, 1]])
    np.testing.assert_allclose(C[1], [[1, 1], [1, 1]])
    Cp = M.two_time_correlation(s, lab, connected="population")
    np.testing.assert_allclose(Cp[0], [[0.75, 0.25], [0.25, 0.75]])
    Ca = M.two_time_correlation(s, lab, connected="agent")
    np.testing.assert_allclose(Ca[1], [[2, 2], [2, 2]])        # agent 2: +-1 across 2 replicas, R/(R-1)=2


def test_ensemble_covariance_matches_numpy():
    s = np.random.default_rng(6).standard_normal((20, 2, 5))
    np.testing.assert_allclose(M.ensemble_covariance(s)[1], np.cov(s[:, 1, :].T))


def test_overlap_field_only_and_excess():
    """Independent spins in agent-specific fields: q = mean_i tanh(h_i)^2, and the excess over a
    g = 0 control with the same fields is ~0."""
    rng = np.random.default_rng(7)
    N, R, T1 = 200, 30, 3
    h = rng.normal(0, 1, N)
    p_up = (1 + np.tanh(h)) / 2
    draw = lambda R: np.where(rng.random((R, T1, N)) < p_up, 1.0, -1.0)
    s, ctrl = draw(R), draw(R)
    q = M.replica_overlap(s)
    assert q.shape == (1, T1)
    np.testing.assert_allclose(q[0], np.mean(np.tanh(h) ** 2), atol=0.03)
    np.testing.assert_allclose(M.overlap_excess(s, ctrl)[0], 0.0, atol=0.04)
    np.testing.assert_allclose(M.overlap_excess(s, np.mean(np.tanh(h) ** 2))[0], 0.0, atol=0.03)


def test_correlation_time_ar1():
    phi, T1, R = 0.7, 12, 4000
    rng = np.random.default_rng(8)
    x = np.zeros((R, T1, 1))
    x[:, 0, 0] = rng.standard_normal(R)                  # stationary start
    for t in range(1, T1):
        x[:, t, 0] = phi * x[:, t - 1, 0] + np.sqrt(1 - phi**2) * rng.standard_normal(R)
    a = M.autocorrelation_from_C(M.two_time_correlation(x)[0])
    np.testing.assert_allclose(a[:5], phi ** np.arange(5), atol=0.02)
    tau_c = -1 / np.log(phi)
    assert M.correlation_time(phi ** np.arange(T1), "exp") == pytest.approx(tau_c)
    assert M.correlation_time(phi ** np.arange(200), "integrated") == \
        pytest.approx(0.5 / np.tanh(1 / (2 * tau_c)), rel=1e-6)
    assert M.correlation_time(a, "exp") == pytest.approx(tau_c, rel=0.1)


def test_ews_baselines():
    var, ac1 = M.rolling_ews(np.array([0.0, 1.0, 0.0, 1.0, 0.0]), window=3)
    assert np.isnan(var[:2]).all()
    np.testing.assert_allclose(var[2:], 2 / 9)
    # window [0,1,0] - 1/3 -> [-1/3, 2/3, -1/3]: sum products = -4/9, sum squares = 6/9
    np.testing.assert_allclose(ac1[2], -2 / 3)
    rng = np.random.default_rng(9)
    m = np.cumsum(rng.standard_normal((5000, 4)), axis=1)  # random walk: Var(m_t) = t + 1
    v, c = M.ensemble_ews(m)
    np.testing.assert_allclose(v, [1, 2, 3, 4], rtol=0.06)
    np.testing.assert_allclose(c, np.sqrt(np.arange(1, 4) / np.arange(2, 5)), atol=0.03)


def test_roc_auc_matches_pairwise_count():
    rng = np.random.default_rng(10)
    sc = rng.integers(0, 5, 60).astype(float)            # many ties
    y = rng.random(60) < 0.4
    pos, neg = sc[y], sc[~y]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert M.roc_auc(sc, y) == pytest.approx(brute)
    assert M.roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0


def test_flip_class_balanced_accuracy_hand_example():
    s_prev = np.array([1, 1, 1, -1, -1, -1, 1, 0])
    s_next = np.array([1, 1, -1, -1, 1, -1, 1, 1])
    stay = s_prev.copy()                                   # persistence predictor
    # groups: flip->+1 {4}: 0/1; flip->-1 {2}: 0/1; stay +1 {0,1,6}: 3/3; stay -1 {3,5}: 2/2; idx 7 undefined
    assert M.flip_class_balanced_accuracy(stay, s_next, s_prev) == pytest.approx(50.0)
    assert M.flip_class_balanced_accuracy(s_next, s_next, s_prev) == pytest.approx(100.0)
