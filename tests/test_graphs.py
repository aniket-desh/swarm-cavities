"""envs/graphs.py: realized coupling moments and the elliptic law."""

import numpy as np
import pytest

from envs.graphs import gaussian_couplings, population_labels, sparse_signed_couplings, spectral_log


def test_gaussian_block_moments_match_targets_N2000():
    # Three populations, asymmetric Jbar and g, and gamma covering -1, 0, 1 and in-between values.
    sizes = [1200, 500, 300]
    N = sum(sizes)
    Jbar = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0], [2.0, 1.0, 0.0]])
    g = np.array([[1.0, 0.5, 2.0], [1.5, 0.8, 1.0], [0.3, 1.0, 1.2]])
    gamma = np.array([[0.5, -1.0, 0.0], [-1.0, 1.0, 0.3], [0.0, 0.3, -0.6]])
    J = gaussian_couplings(sizes, Jbar, g, gamma, np.random.default_rng(0))
    assert np.all(np.diag(J) == 0)
    log = spectral_log(J, population_labels(sizes), eigs=False)
    for a in range(3):
        for b in range(3):
            n = sizes[a] * sizes[b] - (sizes[a] if a == b else 0)     # off-diagonal entries in block
            infl = np.sqrt(2.0)                                       # pairs within a block are correlated
            se_mean = g[a, b] * np.sqrt(N) / np.sqrt(n) * infl        # SE of N * mean
            se_var = g[a, b] ** 2 * np.sqrt(2.0 / n) * infl           # SE of N * var
            se_corr = (1 - gamma[a, b] ** 2) / np.sqrt(n / 2) + 1e-9  # exact at |gamma| = 1
            assert abs(log["N_mean"][a][b] - Jbar[a, b]) < 5 * se_mean, (a, b)
            assert abs(log["N_var"][a][b] - g[a, b] ** 2) < 5 * se_var, (a, b)
            assert abs(log["gamma"][a][b] - gamma[a, b]) < 5 * se_corr, (a, b)


def test_gaussian_gamma_must_be_symmetric():
    with pytest.raises(ValueError):
        gaussian_couplings([10, 10], 0.0, 1.0, np.array([[0.0, 0.5], [0.2, 0.0]]), np.random.default_rng(0))


@pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.5, 1.0])
def test_elliptic_law(gamma):
    """One population, Jbar = 0: eigenvalues fill the ellipse with semi-axes a = g(1+gamma),
    b = g(1-gamma) uniformly. For a uniform ellipse E[x^2] = a^2/4 and E[y^2] = b^2/4 (for gamma = 1
    this is the semicircle on [-2g, 2g] with variance g^2), and the support edges are at a and b."""
    g, N = 1.3, 1000
    J = gaussian_couplings([N], 0.0, g, gamma, np.random.default_rng(1))
    log = spectral_log(J)
    ev = log["eigvals"]
    a, b = g * (1 + gamma), g * (1 - gamma)
    assert np.mean(ev.real**2) == pytest.approx(a**2 / 4, rel=0.03)
    assert np.mean(ev.imag**2) == pytest.approx(b**2 / 4, rel=0.03, abs=1e-8)
    assert np.abs(ev.real).max() == pytest.approx(a, rel=0.05)
    assert np.abs(ev.imag).max() == pytest.approx(b, rel=0.05, abs=1e-6)
    inside = (ev.real / a) ** 2 + (ev.imag / max(b, 1e-12)) ** 2 <= 1.05 if b > 0 else np.abs(ev.real) <= 1.05 * a
    assert inside.mean() > 0.995
    assert log["spectral_radius"] == pytest.approx(max(a, b), rel=0.05)


@pytest.mark.parametrize("gamma,p_neg", [(-0.5, 0.5), (0.0, 0.3), (0.5, 0.2), (1.0, 0.3)])
def test_sparse_signed_degree_sign_and_reciprocity(gamma, p_neg):
    N, k = 2000, 4.0
    J = sparse_signed_couplings(N, k, gamma, p_neg, np.random.default_rng(2))
    assert J.dtype == np.int8 and np.all(np.diag(J) == 0) and set(np.unique(J)) <= {-1, 0, 1}
    log = spectral_log(J, eigs=False)
    n_edges = N * k
    assert log["mean_in_degree"] == pytest.approx(k, abs=5 * np.sqrt(n_edges) / N)
    assert log["frac_neg"][0][0] == pytest.approx(p_neg, abs=5 * np.sqrt(p_neg * (1 - p_neg) / n_edges))
    # Corr over ordered pairs ~ (fraction of reciprocated edges): SE ~ sqrt(r(1-r) / (N k / 2)) + exact at 1
    r = abs(gamma)
    assert log["gamma_all"] == pytest.approx(gamma, abs=5 * np.sqrt(max(r * (1 - r), 1e-4) / (n_edges / 2)))


def test_sparse_signed_infeasible_negative_gamma():
    with pytest.raises(ValueError):
        sparse_signed_couplings(200, 4, -0.8, 0.1, np.random.default_rng(0))
