"""Gate (a): theory/dmft_solver.py against every reference table of theory/solver_spec.md.

Tolerances are the spec's (section 1.3, 2, 3, 4 and the gate rule of section 5). Where a reference number in the spec
turned out to be an artifact of the prototype, the test checks the corrected quantity and its docstring says why:
D1 rho_down (grid-resolution bias of the prototype's root counting), D3 m_c (the prototype reports the upper root
just below the saddle-node, not the merge point) and the A3 probe errors (the prototype averages over 64 entries, 28
of them zero by causality). Tests marked slow use M = 4e5 trajectories or large-N simulations; they run by default
(gate (a) is the whole file), and `-m "not slow"` deselects them.
"""

from math import comb

import numpy as np
import pytest

from envs.graphs import gaussian_couplings, population_labels
from theory import dmft_solver as ds


def gate_exact(ref):
    """Spec section 5: against an exact or quadrature reference, |solver - ref| <= 0.01 max(|ref|, 0.2)."""
    return 0.01 * np.maximum(np.abs(ref), 0.2)


def gate_sim(ref, se):
    """Spec section 5: against a simulation reference, |solver - sim| <= 0.01 max(|sim|, 0.2) + 3 s.e."""
    return 0.01 * np.maximum(np.abs(ref), 0.2) + 3 * se


# ---------------------------------------------------------------------------------------------------------------
# A1: gamma = 0, memoryless mock agent (beta 1.2, g 1, Jbar 0.5, mu 0.3, sigma 0.5)
# ---------------------------------------------------------------------------------------------------------------

A1_BLOCKS = dict(rho=[1.0], Jbar=0.5, g=1.0, gamma=0.0)
A1_M = np.array([0.24353, 0.30679, 0.30917, 0.30332, 0.29962, 0.29802, 0.29744])
A1_CTT = np.array([0.21708, 0.34969, 0.39606, 0.40780, 0.40992, 0.41001, 0.40984])
A1_CT1 = np.array([0.20523, 0.32231, 0.36965, 0.38829, 0.39681, 0.40148])
A1_CT0 = np.array([0.21708, 0.20523, 0.19619, 0.19229, 0.19093, 0.19051, 0.19040])


def a1_rows(m, C):
    return m, np.diag(C), np.diag(C, 1), C[:, 0]


def a1_quadrature(agent=None):
    return ds.quadrature_gamma0(ds.Blocks(**A1_BLOCKS), [agent or ds.IsingAgent(1.2)], [0.3], [0.5], 6)


def test_A1_quadrature():
    o = a1_quadrature()
    for got, ref in zip(a1_rows(o["m"][0], o["C"][0]), (A1_M, A1_CTT, A1_CT1, A1_CT0)):
        np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)


@pytest.mark.slow
def test_A1_monte_carlo():
    o = ds.forward_pass(ds.Blocks(**A1_BLOCKS), [ds.IsingAgent(1.2)], [0.3], [0.5], 6, 400_000, np.random.default_rng(3))
    for got, ref in zip(a1_rows(o["m"][0], o["C"][0]), (A1_M, A1_CTT, A1_CT1, A1_CT0)):
        np.testing.assert_allclose(got, ref, atol=0.003, rtol=0)
    # Lam from per-trajectory tangents agrees with the quadrature value (claim2 (8) at gamma = 0)
    assert o["Lam"] == pytest.approx(a1_quadrature()["Lam"], abs=0.005)


# ---------------------------------------------------------------------------------------------------------------
# A2: linear map x(t+1) = a h(t) + xi(t), closed forms
# ---------------------------------------------------------------------------------------------------------------

def catalan(k):
    return comb(2 * k, k) // (k + 1)


def test_A2_kernel_recursion():
    a, g, gam = 0.55, 1.0, 0.6
    R, v, _ = ds.linear_kernels([a], g, gam, [1.0], 12)
    closed = [a ** (2 * k + 1) * (gam * g**2) ** k * catalan(k) for k in range(5)]
    for t0 in (0, 2):      # the response is time-translation invariant
        np.testing.assert_allclose(R[0, [t0 + 1, t0 + 3, t0 + 5, t0 + 7, t0 + 9], t0], closed, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(R[0, [t0 + 2, t0 + 4, t0 + 6], t0], 0.0)
    np.testing.assert_allclose(R[0, [1, 3, 5, 7], 0], [0.55000, 0.09983, 0.03624, 0.01644], atol=5e-6, rtol=0)
    xg = (a * g) ** 2
    vd = np.diag(v[0])
    np.testing.assert_allclose(vd[1:4], [1, 1 + xg, 1 + xg + xg**2 * (1 + gam**2)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(vd[1:6], [1, 1.30250, 1.42695, 1.49449, 1.53433], atol=5e-6, rtol=0)
    # stationary variances
    R, v, _ = ds.linear_kernels([0.8], 1.0, 0.0, [1.0], 200)
    assert v[0, -1, -1] == pytest.approx(1 / (1 - 0.64), abs=1e-6)                  # 2.77778
    R, v, _ = ds.linear_kernels([0.45], 1.0, 1.0, [1.0], 400)
    x = 0.45**2
    assert v[0, -1, -1] == pytest.approx((1 - np.sqrt(1 - 4 * x)) / (2 * x), abs=1e-6)  # 1.39286


@pytest.mark.slow
def test_A2_monte_carlo():
    rng = np.random.default_rng(2)
    o = ds.forward_pass(ds.Blocks([1.0], 0.0, 1.0, 0.6), [ds.LinearAgent(0.55)], [0.0], [0.0], 8, 400_000, rng)
    np.testing.assert_allclose(o["R"][0, [1, 3, 5, 7], 0], [0.55, 0.0998324, 0.0362392, 0.0164425], rtol=0.01)
    np.testing.assert_allclose(np.diag(o["C"][0])[1:6], [1, 1.30250, 1.42695, 1.49449, 1.53433], rtol=0.01)
    # stationary variances; at T = 30 the exact recursion is within 2e-5 of the limit for both cases
    o = ds.forward_pass(ds.Blocks([1.0], 0.0, 1.0, 0.0), [ds.LinearAgent(0.8)], [0.0], [0.0], 30, 400_000, rng,
                        response="none")
    assert o["C"][0, -1, -1] == pytest.approx(2.77778, rel=0.01)
    o = ds.forward_pass(ds.Blocks([1.0], 0.0, 1.0, 1.0), [ds.LinearAgent(0.45)], [0.0], [0.0], 30, 400_000, rng)
    assert o["C"][0, -1, -1] == pytest.approx(1.39286, rel=0.01)


# ---------------------------------------------------------------------------------------------------------------
# A3: response estimators (memory, gamma 0.5)
# ---------------------------------------------------------------------------------------------------------------

A3_BLOCKS = dict(rho=[1.0], Jbar=0.3, g=1.0, gamma=0.5)


def a3_agent():
    return ds.IsingAgent(1.0, kappa=0.3, lam=0.5)


@pytest.fixture(scope="module")
def a3_tangent():
    return ds.forward_pass(ds.Blocks(**A3_BLOCKS), [a3_agent()], [0.2], [0.5], 8, 400_000, np.random.default_rng(4))


@pytest.mark.slow
def test_A3_tangent(a3_tangent):
    np.testing.assert_allclose(a3_tangent["R"][0, 8, 3:8], [0.0509, 0.0399, 0.0924, 0.0440, 0.4284], atol=0.003, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize("route", ["known map, tangent R in the reaction term", "black box, probe R in the reaction term"])
def test_A3_white_probe(a3_tangent, route):
    """White probe with eps = 0.3 against the unprobed tangent kernel. The spec's r.m.s. errors (0.0086, 0.0416,
    0.0821 at M = 4e5, 2000, 500) are the prototype's metric, which averages over rows t = 1..8 and all 8 columns, 28
    of which vanish by causality; that metric is used here, required within 20% of the quoted values (40 seeds at
    small M). The per-entry error over the 36 causal entries must also match (S1): s.d. <= sqrt(C)/(eps sqrt(M))."""
    ref, eps = a3_tangent["R"][0], 0.3
    C88 = a3_tangent["C"][0, 8, 8]
    causal = np.tri(9, 9, -1).astype(bool)
    for M, n_seeds, quoted in [(400_000, 1, 0.0086), (2000, 40, 0.0416), (500, 40, 0.0821)]:
        e64, e36, r87 = [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(50 + seed)
            if route.startswith("known"):
                o = ds.forward_pass(ds.Blocks(**A3_BLOCKS), [a3_agent()], [0.2], [0.5], 8, M, rng, probe_eps=eps,
                                    response="tangent")
            else:
                box = ds.BlackBoxMap(ds.as_black_box(a3_agent(), np.random.default_rng(1000 + seed)))
                o = ds.forward_pass(ds.Blocks(**A3_BLOCKS), [box], [0.2], [0.5], 8, M, rng, probe_eps=eps)
                assert o["kinds"] == ["probe"] and box.n_calls == 9
            E = o["R_probe"][0] - ref
            e64.append(E[1:, :8]); e36.append(E[causal]); r87.append(o["R_probe"][0, 8, 7])
        rms64 = np.sqrt(np.mean(np.square(e64)))
        rms36 = np.sqrt(np.mean(np.square(e36)))
        sd_S1 = np.sqrt(C88) / (eps * np.sqrt(M))
        print(f"{route}: M={M} rms (prototype metric) {rms64:.4f} vs quoted {quoted}; per causal entry {rms36:.4f}, "
              f"(S1) bound {sd_S1:.4f}; R(8,7) {np.mean(r87):.4f}")
        assert 0.8 * quoted <= rms64 <= 1.2 * quoted
        if M < 10_000:      # variance-dominated: the per-entry error is the probe noise of (S1)
            assert 0.8 * sd_S1 <= rms36 <= 1.05 * sd_S1
        else:               # bias-dominated: the O(eps^2) shift of the operating point, R(8,7) reads 0.4395
            assert abs(r87[0] - 0.4395) <= 3 * sd_S1


def test_black_box_without_probe_refuses_reaction_term():
    box = ds.BlackBoxMap(ds.as_black_box(a3_agent(), np.random.default_rng(0)))
    with pytest.raises(ValueError, match="probe_eps"):
        ds.forward_pass(ds.Blocks(**A3_BLOCKS), [box], [0.2], [0.5], 4, 100, np.random.default_rng(0))


# ---------------------------------------------------------------------------------------------------------------
# A4: static fixed point at small g
# ---------------------------------------------------------------------------------------------------------------

def test_A4_static():
    fp = ds.static_fixed_point(1.0, 0.2, 0.5, 0.5, 0.3, m_init=0.5)
    assert fp["converged"]
    assert fp["m"] == pytest.approx(0.43441, abs=2e-4)
    assert fp["q"] == pytest.approx(0.24839, abs=2e-4)
    assert fp["chi"] == pytest.approx(0.76088, abs=2e-4)
    # the test resolves the g^2 term: the uncoupled value 0.43292 is 7 tolerances away, second order 0.43453 is not
    assert abs(fp["m"] - 0.43292) > 7 * 2e-4 and abs(fp["m"] - 0.43453) < 2e-4


@pytest.mark.slow
def test_A4_dynamic_long_time_limit():
    fp = ds.static_fixed_point(1.0, 0.2, 0.5, 0.5, 0.3, m_init=0.5)
    o = ds.forward_pass(ds.Blocks([1.0], 0.0, 0.2, 0.5), [ds.IsingAgent(1.0)], [0.5], [0.3], 25, 200_000,
                        np.random.default_rng(7))
    assert o["m"][0, -1] == pytest.approx(fp["m"], rel=0.01)
    assert o["C"][0, -1, -1] == pytest.approx(fp["q"], rel=0.01)
    assert o["Lam"] == pytest.approx(fp["Lam"], rel=0.05)


# ---------------------------------------------------------------------------------------------------------------
# A5: two populations, everything switched on, against an N = 4000 simulation
# ---------------------------------------------------------------------------------------------------------------

A5_BLOCKS = dict(rho=[0.6, 0.4], Jbar=[[0.5, -0.3], [0.8, 0.2]], g=[[1.0, 0.6], [0.8, 1.2]],
                 gamma=[[0.6, 0.5], [0.5, -0.4]])
A5_MU, A5_SIG, A5_T = [0.3, -0.2], [0.5, 0.3], [2, 4, 6, 8, 10]
A5_SIM_M = np.array([[0.3679, 0.3743, 0.3677, 0.3603, 0.3541], [-0.0456, -0.0363, -0.0336, -0.0358, -0.0375]])
A5_SIM_C = np.array([[0.4763, 0.5831, 0.6286, 0.6515, 0.6645], [0.1808, 0.2077, 0.2203, 0.2266, 0.2301]])
A5_KICK1 = np.array([0.712, 0.0, -0.051, 0.005, 0.019])           # spec section 1.3, population 1, lags 1..5
A5_KICK0_SIM = np.array([0.500, 0.068, 0.129, 0.060, 0.066])      # claim1.md section 6, population 0 (memory)
KICK_TOL = 0.01      # the spec gives no tolerance; well below the 0.03 by which the wrong forms must miss


def a5_maps(stochastic=False):
    return [ds.IsingAgent(1.2, kappa=0.4, lam=0.5, stochastic=stochastic), ds.IsingAgent(0.9, stochastic=stochastic)]


def a5_errors(o):
    em = np.abs(o["m"][:, A5_T] - A5_SIM_M)
    eC = np.abs(np.diagonal(o["C"], axis1=1, axis2=2)[:, A5_T] - A5_SIM_C)
    ek = np.abs(o["R"][1, 4:9, 3] - A5_KICK1)
    return em, eC, ek


@pytest.fixture(scope="module")
def a5_runs():
    return {v: ds.forward_pass(ds.Blocks(**A5_BLOCKS), a5_maps(), A5_MU, A5_SIG, 10, 400_000,
                               np.random.default_rng(1), variant=v)
            for v in (None, "connected_noise", "shifted_reaction")}


@pytest.mark.slow
def test_A5_deterministic(a5_runs):
    o = a5_runs[None]
    em, eC, ek = a5_errors(o)
    print(f"A5: max |m - sim| {em.max():.4f}, max |C(t,t) - sim| {eC.max():.4f}, kick {ek.max():.4f}, Lam {o['Lam']:.3f}")
    assert np.all(em <= gate_sim(A5_SIM_M, 0.0028))
    assert np.all(eC <= gate_sim(A5_SIM_C, 0.0016))
    assert np.all(ek <= KICK_TOL)
    assert o["R"][1, 5, 3] == 0.0                     # memoryless population: no response at lag 2 (two-round echo)
    assert np.abs(o["R"][0, 4:9, 3] - A5_KICK0_SIM).max() <= KICK_TOL


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["connected_noise", "shifted_reaction"])
def test_A5_wrong_forms_fail(a5_runs, variant):
    """The connected covariance and the shifted reaction index must fail A5 with errors above 0.03. Both fail the
    m / C(t,t) gate; the shifted index also misses the kick test by 0.056. The connected covariance misses the kick
    test by 0.027 (it fails KICK_TOL, but not by more than 0.03): its 0.03-level failure is in m (0.034)."""
    em, eC, ek = a5_errors(a5_runs[variant])
    print(f"{variant}: max |m - sim| {em.max():.4f}, max |C(t,t) - sim| {eC.max():.4f}, kick {ek.max():.4f}")
    assert np.any(em > gate_sim(A5_SIM_M, 0.0028)) or np.any(eC > gate_sim(A5_SIM_C, 0.0016))
    assert ek.max() > KICK_TOL
    assert max(em.max(), eC.max(), ek.max()) > 0.03


def simulate_a5(stochastic, N=4000, n_dis=40, T=10, seed=0):
    """Direct simulation of the A5 swarm (claim1 (1)-(2)) with envs.graphs couplings: m, C and their standard errors."""
    sizes = [int(0.6 * N), N - int(0.6 * N)]
    lab = population_labels(sizes)
    mu, sig = np.array(A5_MU)[lab], np.array(A5_SIG)[lab]
    beta, kappa, lam = np.array([1.2, 0.9])[lab], np.array([0.4, 0.0])[lab], 0.5
    rng = np.random.default_rng(seed)
    emit = (lambda p: np.where(rng.random(N) < (1 + p) / 2, 1.0, -1.0)) if stochastic else (lambda p: p)
    ms, Cs = [], []
    for _ in range(n_dis):
        J = gaussian_couplings(sizes, A5_BLOCKS["Jbar"], A5_BLOCKS["g"], A5_BLOCKS["gamma"], rng)
        hp = mu + sig * rng.standard_normal(N)
        S, u = [emit(np.tanh(hp))], np.zeros(N)
        for _ in range(T):
            u = lam * u + (1 - lam) * S[-1]
            S.append(emit(np.tanh(beta * (J @ S[-1]) + hp + kappa * u)))
        S = np.array(S)
        ms.append([S[:, lab == a].mean(1) for a in range(2)])
        Cs.append([S[:, lab == a] @ S[:, lab == a].T / (lab == a).sum() for a in range(2)])
    ms, Cs = np.array(ms), np.array(Cs)
    return ms.mean(0), ms.std(0, ddof=1) / np.sqrt(n_dis), Cs.mean(0), Cs.std(0, ddof=1) / np.sqrt(n_dis)


@pytest.mark.slow
def test_A5_stochastic():
    """+-1 agents, score-function response. m against the simulation of check_claim1_dmft.py --stochastic (s.e.
    0.005, out_claim1_stoch.txt); C(t, t-2), which the spec quotes no numbers for, against a fresh direct simulation
    at the same size (N = 4000, 40 disorder samples) with its measured standard errors."""
    o = ds.forward_pass(ds.Blocks(**A5_BLOCKS), a5_maps(True), A5_MU, A5_SIG, 10, 400_000, np.random.default_rng(1))
    sim_m = np.array([[0.2751, 0.289, 0.2947, 0.301, 0.2976], [-0.0539, -0.0534, -0.048, -0.0522, -0.0495]])
    em = np.abs(o["m"][:, A5_T] - sim_m)
    _, _, C_sim, C_se = simulate_a5(True)
    lag2, lag2_se = (np.diagonal(x, offset=2, axis1=1, axis2=2) for x in (C_sim, C_se))
    e2 = np.abs(np.diagonal(o["C"], offset=2, axis1=1, axis2=2) - lag2)
    print(f"A5 stochastic: max |m - sim| {em.max():.4f}, max |C(t,t-2) - sim| {e2.max():.4f} (s.e. <= {lag2_se.max():.4f})")
    assert np.all(em <= gate_sim(sim_m, 0.005))
    assert np.all(e2 <= gate_sim(lag2, lag2_se))


# ---------------------------------------------------------------------------------------------------------------
# Nested copies (spec 1.1 item 4): the split eta = eta_f + eta_th must reproduce the full process
# ---------------------------------------------------------------------------------------------------------------

def test_nested_copies_reproduce_plain_pass():
    bl = ds.Blocks(**A5_BLOCKS)
    plain = ds.forward_pass(bl, a5_maps(True), A5_MU, A5_SIG, 6, 200_000, np.random.default_rng(5))
    nest = ds.forward_pass(bl, a5_maps(True), A5_MU, A5_SIG, 6, 25_000, np.random.default_rng(6), copies=8)
    assert np.abs(nest["m"] - plain["m"]).max() < 0.01
    assert np.abs(nest["C"] - plain["C"]).max() < 0.01
    # deterministic map: the copies of a site coincide, so q = C; with independent couplings q < C
    det = ds.forward_pass(bl, a5_maps(False), A5_MU, A5_SIG, 4, 2000, np.random.default_rng(0), copies=4)
    np.testing.assert_allclose(det["q"], det["C"], atol=1e-12)
    ind = ds.forward_pass(bl, a5_maps(True), A5_MU, A5_SIG, 4, 20_000, np.random.default_rng(0), copies=4,
                          frozen_noise=False)
    dq = np.diagonal(nest["q"][:, :5, :5], axis1=1, axis2=2) - np.diagonal(ind["q"], axis1=1, axis2=2)
    assert np.all(dq[:, 1:] > 0)            # Delta q > 0 once the couplings have acted (claim2 section 5)


# ---------------------------------------------------------------------------------------------------------------
# Mode (b): black-box map
# ---------------------------------------------------------------------------------------------------------------

def test_black_box_gaussian_delivery_reproduces_A1():
    """Live route with Gaussian field delivery on the mock agent hidden behind the black-box interface must reproduce
    A1 to within the Monte Carlo error at the M used (spec: <~0.02 at M = 2000). For a deterministic agent it gives
    the same numbers as the known map, so its error is the Monte Carlo error of mode (a); the test checks that this
    error is unbiased and falls as 1/sqrt(M) (40 seeds at M = 2000 and 8000), and reports its size at M = 2000
    (r.m.s. 0.021 for the worst entry, m(4), where the spec's "<~0.02" is approximate)."""
    ref = a1_quadrature()
    bl = ds.Blocks(**A1_BLOCKS)
    rms = {}
    for M in (2000, 8000):
        devs = []
        for seed in range(40):
            box = ds.BlackBoxMap(ds.as_black_box(ds.IsingAgent(1.2), np.random.default_rng(99)))
            o = ds.forward_pass(bl, [box], [0.3], [0.5], 6, M, np.random.default_rng(seed))
            assert box.n_calls == 7 and box.n_queries == 7 * M
            if seed < 3:
                known = ds.forward_pass(bl, [ds.IsingAgent(1.2)], [0.3], [0.5], 6, M, np.random.default_rng(seed))
                np.testing.assert_array_equal(o["C"], known["C"])
            devs.append(np.r_[o["m"][0] - ref["m"][0], (o["C"][0] - ref["C"][0])[np.triu_indices(7)]])
        devs = np.array(devs)
        rms[M] = np.sqrt((devs**2).mean(0))
        assert np.all(np.abs(devs.mean(0)) <= 4 * devs.std(0) / np.sqrt(len(devs)))     # no bias
    ratio = rms[2000][1:].mean() / rms[8000][1:].mean()
    print(f"black box A1: worst-entry r.m.s. {rms[2000].max():.4f} at M=2000, {rms[8000].max():.4f} at M=8000; "
          f"mean ratio {ratio:.2f} (1/sqrt(M) predicts 2)")
    assert 1.6 <= ratio <= 2.5
    assert rms[2000].max() <= 0.025


def test_black_box_one_call_per_round():
    """Two populations sharing one agent: one call per round over all trajectories, same result as separate maps."""
    agent = ds.IsingAgent(1.0)
    bl = ds.Blocks([0.5, 0.5], [[0.4, 0.2], [0.1, 0.3]], 0.8, 0.0)
    box = ds.BlackBoxMap(ds.as_black_box(agent, np.random.default_rng(0)))
    o = ds.forward_pass(bl, [box, box], [0.2, -0.1], [0.4, 0.3], 5, 1000, np.random.default_rng(1))
    known = ds.forward_pass(bl, [agent, agent], [0.2, -0.1], [0.4, 0.3], 5, 1000, np.random.default_rng(1))
    assert box.n_calls == 6 and box.n_queries == 6 * 2000
    np.testing.assert_array_equal(o["C"], known["C"])


def test_surrogate_route_reproduces_A1():
    """Surrogate fitted on 600 probes of the black-box mock agent with a logistic family in (h, hpriv), then mode (a):
    A1 to 1% (quadrature, so the only error is the calibration)."""
    rng = np.random.default_rng(0)
    hp, h, _ = ds.probe_design(600, 0.3, 0.5, h_scale=0.65, rng=rng)
    sur, info = ds.fit_surrogate(ds.as_black_box(ds.IsingAgent(1.2), rng), hp, h, rng=rng)
    assert info["n_queries"] == 600 and info["family"] == "tanh"
    o = a1_quadrature(sur)
    for got, ref in zip(a1_rows(o["m"][0], o["C"][0]), (A1_M, A1_CTT, A1_CT1, A1_CT0)):
        np.testing.assert_allclose(got, ref, rtol=0.01)


def test_surrogate_binary_noise_law():
    """Stochastic +-1 black box with own-history dependence: the binary family recovers the response surface
    (h, hpriv, last state, bias) within 4 standard errors of its maximum-likelihood fit."""
    rng = np.random.default_rng(1)
    true = np.array([1.2, 1.0, 0.3, 0.0])
    hp, h, sl = ds.probe_design(600, 0.3, 0.5, h_scale=0.65, rng=rng, history=True, binary=True)
    box = ds.as_black_box(ds.IsingAgent(1.2, kappa=0.3, lam=0.0, stochastic=True), rng)
    sur, info = ds.fit_surrogate(box, hp, h, sl, rng=rng)
    assert info["family"] == "binary" and sur.stochastic and sur.response == "score"
    X = np.stack([h, hp, sl, np.ones(600)], 1)
    p = 1 / (1 + np.exp(-2 * X @ true))
    fisher = ((4 * p * (1 - p))[:, None, None] * X[:, :, None] * X[:, None, :]).sum(0)   # logit = 2 x.c
    se = np.sqrt(np.diag(np.linalg.inv(fisher))) / np.sqrt(0.8)                          # 80% of probes fit
    assert np.all(np.abs(info["coef"] - true) <= 4 * se)


def test_live_pool_matches_exact_directed_solution():
    """Pool route (population dynamics on a random directed graph, claim5 section 1) against the exact recursion
    claim5 (1) for K = 3, p_plus = 0.8, b = 0.9: m(t) to 0.01 at M = 1e6 (seed-to-seed s.d. about 0.003)."""
    K, p_plus, b, mu, sig, T = 3, 0.8, 0.9, 0.1, 0.2, 8
    e, we = np.polynomial.hermite_e.hermegauss(48)
    we = we / we.sum()
    m_ex = np.zeros(T + 1)
    m_ex[0] = we @ np.tanh(mu + sig * e)
    for t in range(T):
        pi = p_plus * (1 + m_ex[t]) / 2 + (1 - p_plus) * (1 - m_ex[t]) / 2
        m_ex[t + 1] = sum(comb(K, n) * pi**n * (1 - pi) ** (K - n) * (we @ np.tanh(b * (2 * n - K) + mu + sig * e))
                          for n in range(K + 1))
    box = ds.BlackBoxMap(ds.as_black_box(ds.IsingAgent(1.0, stochastic=True), np.random.default_rng(100)))
    o = ds.live_pool(box, mu, sig, T, 1_000_000, K, np.random.default_rng(0), b=b, p_plus=p_plus)
    assert o["n_calls"] == T + 1
    assert np.abs(o["m"] - m_ex).max() <= 0.01


# ---------------------------------------------------------------------------------------------------------------
# Mode (c): two-site PR predictor
# ---------------------------------------------------------------------------------------------------------------

def linear_pr(a, gamma, n, rho_b=1.0):
    R, v, w = ds.linear_kernels([a], 1.0, gamma, [1.0], n, rho_b=rho_b)
    psi = np.einsum("tttt->t", ds.four_point(*ds.site_moments(R, v, w), 1.0, gamma, rho_b)["SS"])
    vd = np.einsum("ktt->kt", v)
    with np.errstate(invalid="ignore"):
        return ds.pr_over_n(w @ vd, w @ vd**2, psi), w @ vd


def test_C1_closed_form():
    R, v, _ = ds.linear_kernels([0.8], 1.0, 0.0, [1.0], 200)
    vd = np.einsum("ktt->kt", v)[0]
    psi = ds.psi_equal_time(np.full(200, 0.8**2), vd**2, 1.0)
    with np.errstate(invalid="ignore"):                    # PR(0) = 0/0 for x(0) = 0
        pr = ds.pr_over_n(vd, vd**2, psi)
    assert pr[-1] == pytest.approx(0.59040, abs=1e-4)
    assert vd[-1] == pytest.approx(2.77778, abs=1e-4)
    # gamma = 0, memoryless: the equal-time recursion and the general system agree (spec section 3 step 4)
    R, v, w = ds.linear_kernels([0.8], 1.0, 0.0, [1.0], 12)
    vd = np.einsum("ktt->kt", v)[0]
    general = np.einsum("tttt->t", ds.four_point(*ds.site_moments(R, v, w), 1.0, 0.0)["SS"])
    np.testing.assert_allclose(general, ds.psi_equal_time(np.full(12, 0.64), vd**2, 1.0), atol=1e-12)


def test_C2_transient_with_reciprocity():
    ref = [1, 0.94882, 0.90100, 0.86145, 0.83289]
    for gamma in (0.6, -0.6):
        pr, _ = linear_pr(0.55, gamma, 12)
        np.testing.assert_allclose(pr[1:6], ref, atol=1e-4, rtol=0)
        assert pr[11] == pytest.approx(0.76436, abs=1e-4)
    pr, vbar = linear_pr(0.75, 0.5, 12, rho_b=0.6)
    assert vbar[11] == pytest.approx(1.6811, abs=1e-4)
    assert pr[11] == pytest.approx(0.7429, abs=1e-4)


@pytest.mark.slow
def test_C3_nonlinear_stochastic_gamma0():
    """+-1 agents, beta 1.2, g 1, sigma 0.5: per-site thermal quantities from nested copies (2e5 sites x 16 copies,
    site moments from disjoint halves), psi and PR/N to 1% by the equal-time recursion; the general system on the
    same inputs agrees with it to 1%."""
    o = ds.forward_pass(ds.Blocks([1.0], 0.0, 1.0, 0.0), [ds.IsingAgent(1.2, stochastic=True)], [0.0], [0.5], 8,
                        200_000, np.random.default_rng(0), copies=16)
    et = ds.pr_from_sites(o["sites"][0], 1.0, 0.0, equal_time=True)
    gen = ds.pr_from_sites(o["sites"][0], 1.0, 0.0)
    ts = [1, 2, 4, 8]
    print("C3 psi", et["psi"][ts], "general", gen["psi"][ts], "PR/N", et["pr"][ts], "general", gen["pr"][ts])
    np.testing.assert_allclose(et["psi"][ts], [0.1053, 0.1286, 0.1370, 0.1382], rtol=0.01)
    np.testing.assert_allclose(et["pr"][ts], [0.8544, 0.8419, 0.8390, 0.8387], rtol=0.01)
    np.testing.assert_allclose(gen["psi"][1:], et["psi"][1:], rtol=0.01)
    np.testing.assert_allclose(gen["pr"], et["pr"], rtol=0.01)


def test_C4_mean_mode():
    """+-1 agents with zealots, g = 0, N = 128: the g = 0 fixed point gives mu_i = tanh(x_i), v_i = 1 - mu_i^2,
    r_i = beta v_i; the mean-mode recursion run to stationarity gives N_B Var(m_B) and PR/N_B (claim3 (16))."""
    beta, N = 1.5, 128
    for nz, pr_ref, var_ref in [(0, 0.875, 0.14), (13, 0.897, 0.31), (16, 0.905, 0.39)]:
        rho, NB = nz / N, N - nz
        fp = ds.static_fixed_point(beta, 0.0, 0.0, 0.3, 0.2, 1.0, rho, m_init=1.0)
        W, v = fp["W"], 1 - fp["s"] ** 2
        vbar, v2 = (W * v).sum(), (W * v**2).sum()
        lam, k2 = (1 - rho) * beta * vbar, (1 - rho) ** 2 * beta**2 * v2
        n = 400
        NV, L1 = ds.mean_mode_transient(np.full(n, vbar), np.full(n, lam), np.full(n, k2), NB)
        assert NV[-1] == pytest.approx(vbar / (1 - lam**2), rel=1e-10)       # stationary limit of the recursion
        assert ds.pr_over_n(vbar, v2, 0.0, L1[-1], NB) == pytest.approx(pr_ref, rel=0.01)
        assert abs(NV[-1] - var_ref) <= 0.01 * var_ref + 0.005                # reference quoted to two decimals


# ---------------------------------------------------------------------------------------------------------------
# Mode (d): zealots (beta 1.5, Jbar 1, mu 0.3 throughout)
# ---------------------------------------------------------------------------------------------------------------

def test_D1_closed_form():
    """g = 0, sigma = 0. The forward saddle-node matches the closed form. For the backward one the spec quotes the
    prototype's 0.13234, but the saddle-node equations beta (1-rho)(1-m^2) = 1, atanh m = beta((1-rho) m - rho) + mu
    on the captured branch give 0.132028; the prototype's bisection on the number of roots on a 61-point grid in m
    cannot see two roots inside one grid cell and is biased upward by 3e-4. The test uses the closed form."""
    sn = ds.zealot_thresholds(1.5, 0.0, 0.0, 0.3, 0.0, 1.0)
    assert sn["rho_up"] == pytest.approx(0.22704, abs=3e-4)
    assert sn["m_up"] == pytest.approx(0.37082, abs=3e-4)
    b, mu = 1.5, 0.3
    m_of = lambda r: -np.sqrt(1 - 1 / (b * (1 - r)))
    from scipy.optimize import brentq
    rho_down = brentq(lambda r: np.arctanh(m_of(r)) - (b * ((1 - r) * m_of(r) - r) + mu), 0.0, 1 - 1 / b - 1e-12)
    assert sn["rho_down"] == pytest.approx(rho_down, abs=3e-4)
    assert sn["m_down"] == pytest.approx(m_of(rho_down), abs=3e-4)


def mean_map_slope(g, gamma, rho, m, h=1e-4):
    F = ds._MeanMap(1.5, g, gamma, 0.3, 0.2, 1.0)
    return (F.resid(m + h, rho) - F.resid(m - h, rho)) / (2 * h) + 1.0, F.resid(m, rho)


D3_CASES = [  # (g, gamma, rho_up, rho_down) from solver_spec.md D2, D3, D3b
    (0.0, 0.0, 0.2202, 0.1539),
    (0.3, -0.5, 0.2039, 0.1938), (0.3, 0.0, 0.2108, 0.1792), (0.3, 0.5, 0.2189, 0.1567), (0.3, 1.0, 0.2278, 0.1244),
    (0.45, 0.0, 0.2023, 0.1966), (0.45, 0.5, 0.2134, 0.1708), (0.45, 1.0, 0.2261, 0.1220),
]


@pytest.mark.slow
@pytest.mark.parametrize("g,gamma,rho_up,rho_down", D3_CASES)
def test_D2_D3_saddle_nodes(g, gamma, rho_up, rho_down):
    """Tolerance 1% on both thresholds. The spec's m_c column (0.223, 0.275, 0.307, 0.336 at g = 0.3) is not
    tested: it is the prototype's upper root just below the saddle-node (at g = 0, sigma = 0.2 its value 0.363 gives
    df/dm = 0.982), so instead the returned m_c is checked against the saddle-node conditions f(m) = m, df/dm = 1."""
    sn = ds.zealot_thresholds(1.5, g, gamma, 0.3, 0.2, 1.0)
    print(f"g={g} gamma={gamma}: rho_up {sn['rho_up']:.5f} (spec {rho_up}), rho_down {sn['rho_down']:.5f} "
          f"(spec {rho_down}), m_up {sn['m_up']:.4f}, Lam {sn['Lam_up']:.3f}")
    assert sn["rho_up"] == pytest.approx(rho_up, rel=0.01)
    assert sn["rho_down"] == pytest.approx(rho_down, rel=0.01)
    for rr, mm in ((sn["rho_up"], sn["m_up"]), (sn["rho_down"], sn["m_down"])):
        slope, resid = mean_map_slope(g, gamma, rr, mm)
        assert abs(resid) < 1e-8 and abs(slope - 1) < 1e-3


@pytest.mark.slow
def test_D3_root_count_definition():
    """The spec defines the thresholds by the number of roots of f(m) = m (step 3): three just inside the window,
    one just outside."""
    sn = ds.zealot_thresholds(1.5, 0.3, 0.5, 0.3, 0.2, 1.0)
    count = lambda r: len(ds.branches(1.5, 0.3, 0.5, 0.3, 0.2, 1.0, r))
    assert count(sn["rho_up"] - 2e-4) == 3 and count(sn["rho_up"] + 2e-4) == 1
    assert count(sn["rho_down"] + 2e-4) == 3 and count(sn["rho_down"] - 2e-4) == 1


@pytest.mark.slow
def test_D3b_no_window():
    assert ds.zealot_thresholds(1.5, 0.45, -0.5, 0.3, 0.2, 1.0) is None


@pytest.mark.slow
def test_D4_transient():
    """m_B(10) from the forward pass with a frozen population, averaged over three seeds at M = 4e5. The references
    are the prototype's M = 2e5 values; at rho = 0.20 the prototype itself gives 0.5007-0.5047 across seeds."""
    refs = {0.10: 0.809, 0.20: 0.505, 0.24: -0.151, 0.28: -0.556, 0.32: -0.624}
    for rho, ref in refs.items():
        mB = np.mean([ds.zealot_transient(ds.IsingAgent(1.5), 0.3, 0.2, 1.0, 0.3, 0.5, rho, 10, 400_000,
                                          np.random.default_rng(seed))["m"][0, 10] for seed in range(3)])
        print(f"D4 rho={rho}: m_B(10) {mB:+.4f} (spec {ref:+.3f})")
        assert abs(mB - ref) <= (0.02 if rho == 0.24 else 0.005)


@pytest.mark.slow
def test_dynamic_threshold_and_hysteresis():
    """Horizon-10 forward threshold lies between 0.20 and 0.24 (the D4 signs; claim4 section 5 says about 0.23). At
    g = 0 the dynamic backward threshold after 80 rounds sits at or just below the static rho_down = 0.1539 (a finite
    run overshoots the saddle-node), within one step of 0.005."""
    th = ds.forward_threshold_dynamic(ds.IsingAgent(1.5), 0.3, 0.2, 1.0, 0.3, 0.5, 10, 100_000, lo=0.20, hi=0.24)
    assert 0.20 < th < 0.24
    hy = ds.hysteresis_dynamic(ds.IsingAgent(1.5), 0.3, 0.2, 1.0, 0.0, 0.0, 0.3, np.arange(0.13, 0.171, 0.005), 20, 80,
                               20_000, response="none")
    print(f"horizon-10 forward threshold {th:.4f}; dynamic backward threshold {hy['rho_back']:.3f}")
    assert 0.1539 - 0.005 - 1e-9 <= hy["rho_back"] <= 0.1539
