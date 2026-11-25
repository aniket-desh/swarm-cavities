"""Checks for claim5.md.

Part 1 (directed sparse graph, exact solution). N agents, each reading exactly K others (no reciprocal edges by
    construction of independent draws at large N), signs +1 with probability p_plus, weight b:
        P(s_i(t+1) = +1) = (1 + tanh(b sum_k eps_ik s_k(t) + hpriv_i)) / 2.
    Compared: simulation; the exact recursion for m(t) (eq. (1)); population dynamics over trajectories for
    C(t, t') (eq. (2)); and the dense Gaussian DMFT of claim1.md with Jbar = K b (2 p_plus - 1), g^2 = K b^2.
Part 2 (symmetric sparse graph, loop corrections to the local susceptibility). Physics-of-Agents-like graphs:
    N = 32, 56 undirected signed edges (mean degree 3.5). chi = (1/d - J)^-1 exactly, on the tree approximation
    (eq. (3)), and with the triangle correction (eq. (4)).

Run: uv run python theory/checks/check_claim5_sparse.py
"""
import numpy as np
from math import comb
from toy import dmft_mc

np.set_printoptions(precision=4, suppress=True, linewidth=150)
rng = np.random.default_rng(0)

# ---------------- Part 1
K, p_plus, b, mu, sig, T = 3, 0.8, 0.9, 0.1, 0.2, 8
N, n_run, M = 200_000, 5, 1_000_000
e, we = np.polynomial.hermite_e.hermegauss(48); we = we / we.sum()

m_sim, C_sim = [], []
for r in range(n_run):
    nbr = rng.integers(0, N, (N, K)); eps = np.where(rng.random((N, K)) < p_plus, 1.0, -1.0)
    hp = mu + sig * rng.standard_normal(N)
    s = np.empty((T + 1, N)); s[0] = np.where(rng.random(N) < (1 + np.tanh(hp)) / 2, 1.0, -1.0)
    for t in range(T):
        h = b * (eps * s[t][nbr]).sum(1)
        s[t + 1] = np.where(rng.random(N) < (1 + np.tanh(h + hp)) / 2, 1.0, -1.0)
    m_sim.append(s.mean(1)); C_sim.append(s @ s.T / N)
m_sem = np.std(m_sim, 0, ddof=1).max() / np.sqrt(n_run)
m_sim, C_sim = np.mean(m_sim, 0), np.mean(C_sim, 0)

# exact one-time recursion, eq. (1): each input contributes eps s = +1 with probability pi(t)
m_ex = np.zeros(T + 1); m_ex[0] = we @ np.tanh(mu + sig * e)
for t in range(T):
    pi = p_plus * (1 + m_ex[t]) / 2 + (1 - p_plus) * (1 - m_ex[t]) / 2
    m_ex[t + 1] = sum(comb(K, n) * pi**n * (1 - pi) ** (K - n) * (we @ np.tanh(b * (2 * n - K) + mu + sig * e))
                      for n in range(K + 1))

# population dynamics over trajectories, eq. (2): generation d holds trajectories valid up to round d
pool = np.where(rng.random((M, 1)) < (1 + np.tanh(mu + sig * rng.standard_normal((M, 1)))) / 2, 1.0, -1.0)
for d in range(1, T + 1):
    par = rng.integers(0, M, (M, K)); eps = np.where(rng.random((M, K)) < p_plus, 1.0, -1.0)
    hp = mu + sig * rng.standard_normal(M)
    new = np.empty((M, d + 1)); new[:, 0] = np.where(rng.random(M) < (1 + np.tanh(hp)) / 2, 1.0, -1.0)
    for t in range(d):
        h = b * (eps * pool[par, t]).sum(1)
        new[:, t + 1] = np.where(rng.random(M) < (1 + np.tanh(h + hp)) / 2, 1.0, -1.0)
    pool = new
C_pd = pool.T @ pool / M

# dense Gaussian DMFT with matched first two moments of the couplings
Jbar, g = K * b * (2 * p_plus - 1), np.sqrt(K) * b
dm = dmft_mc(np.random.default_rng(1), [1.0], [[Jbar]], [[g]], [[0.0]], [mu], [sig], [1.0], [0.0], 0.0, T, M, stochastic=True)

print(f"Part 1: directed, K={K}, p_plus={p_plus}, b={b}; simulation N={N} x {n_run} runs")
print("t                      ", np.arange(T + 1))
print("m(t)  simulation       ", m_sim)
print("m(t)  exact eq.(1)     ", m_ex)
print("m(t)  Gaussian DMFT    ", dm["m"][0])
print("C(t,t-1) simulation    ", np.diag(C_sim, 1))
print("C(t,t-1) pop. dynamics ", np.diag(C_pd, 1))
print("C(t,t-1) Gaussian DMFT ", np.diag(dm["C"][0], 1))
print("simulation s.e. of m (max over t): %.4f" % m_sem)
print("max |error| in m:  exact %.4f   Gaussian DMFT %.4f;   in C(t,t-1):  pop. dyn. %.4f   Gaussian DMFT %.4f" % (
    np.abs(m_ex - m_sim).max(), np.abs(dm["m"][0] - m_sim).max(),
    np.abs(np.diag(C_pd, 1) - np.diag(C_sim, 1)).max(), np.abs(np.diag(dm["C"][0], 1) - np.diag(C_sim, 1)).max()))

# ---------------- Part 2
Nn, n_pairs, d, n_graphs = 32, 56, 0.2, 400
iu, ju = np.triu_indices(Nn, 1)
err_tree, err_tri, n_tri, shift = [], [], [], []
for gidx in range(n_graphs):
    # same ensemble as sample_J_num_edges_symmetric in third_party/physics-of-agents/lib/datagen/graph.py
    pick = rng.choice(len(iu), n_pairs, replace=False)
    J = np.zeros((Nn, Nn)); sg = rng.choice([-1.0, 1.0], n_pairs)
    J[iu[pick], ju[pick]] = sg; J[ju[pick], iu[pick]] = sg
    A = (J != 0)
    chi = np.diag(np.linalg.inv(np.eye(Nn) / d - J))
    # tree (Bethe) cavity, eq. (3): G_{i->j} = 1 / (1/d - sum_{k in di \ j} J_ik J_ki G_{k->i})
    G = np.where(A, d, 0.0)                       # G[i, j] = G_{i -> j}
    for it in range(500):
        tot = (A * G.T).sum(1)                    # sum_k G_{k->i} over neighbours k of i   (J_ik J_ki = 1)
        G_new = np.where(A, 1.0 / (1.0 / d - (tot[:, None] - G.T)), 0.0)
        if np.abs(G_new - G).max() < 1e-13:
            break
        G = G_new
    chi_tree = 1.0 / (1.0 / d - (A * G.T).sum(1))
    # triangle correction, eq. (4): chi_i^2 sum over ordered neighbour pairs (j, k) with j ~ k of J_ij G_j J_jk G_k J_ki
    tri = np.einsum("ij,j,jk,k,ki->i", J, chi_tree, J, chi_tree, J)
    chi_tri = chi_tree + chi_tree**2 * tri
    deg = A.sum(1) > 0
    err_tree.append(np.abs(chi_tree - chi)[deg] / chi[deg]); err_tri.append(np.abs(chi_tri - chi)[deg] / chi[deg])
    n_tri.append(np.trace(np.linalg.matrix_power(A.astype(float), 3)) / 6)
    shift.append((chi[deg] / d - 1))
print(f"\nPart 2: N={Nn}, {n_pairs} undirected signed edges, gain d={d}, {n_graphs} graphs")
print("mean number of triangles per graph: %.2f" % np.mean(n_tri))
print("mean relative effect of the network on chi_ii (chi/d - 1): %.4f" % np.mean(np.concatenate(shift)))
print("relative error of chi_ii:  tree  mean %.5f  95th pct %.5f   |  tree + triangles  mean %.5f  95th pct %.5f" % (
    np.mean(np.concatenate(err_tree)), np.quantile(np.concatenate(err_tree), 0.95),
    np.mean(np.concatenate(err_tri)), np.quantile(np.concatenate(err_tri), 0.95)))
