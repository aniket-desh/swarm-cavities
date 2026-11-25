"""Checks and frozen numbers for theory/analyst_scenario.md (law and notation in analyst_lib.py).

Parts:
  A  Gaussian closure at K = 16: the single-site DMFT (K = infinity) against simulations of the same law on actual
     trust graphs at N = 256, 1024, 4096 (K = 16) and at K = 64.
  B  Stability: exact results of the mutual world (chi = 1, Lam = g^2, <sigma(t) sigma(t+2)> = <s^2>), the tree form
     (K - 1) tanh^2(a / K) against the spin-glass susceptibility of a long simulation, Lam(omega) per world from the
     solver, the boundary gain against news strength, the herding (mean-field) margin.
  C  Signatures per world: echo function, lag correlations of replica differences, spectrum at 0, pi/2, pi,
     correlation time, spin-glass susceptibility; synchronous, with memory, asynchronous; N dependence.
  D  Kick twins: first-order responses, echo, noise per kick, kicks needed.
  E  Sizing: run-to-run spread of every statistic at N = 256 and 1024, T = 20.
  G  Location of the boundary on K = 16 graphs from long simulations (amplification against gain).
  F  Collect everything into analyst_predictions.json.

Run: uv run python theory/checks/check_analyst.py [A B C D E G F]    (output: out_analyst_<part>.txt, part JSONs)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyst_lib as al  # noqa: E402
from analyst_lib import Agent  # noqa: E402

K = 16
REF = dict(nu=0.5, a0=0.3)                       # reference design: news logit s.d. 0.5, standing-note logit s.d. 0.3
W3 = ("mutual", "oneway", "antagonistic")
OUT = {}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def fmt(x, n=3):
    return np.array2string(np.asarray(x, float), precision=n, floatmode="fixed", separator=", ", max_line_width=200)


def sim_runs(P, world, N, T, R, n_graphs, seed, mode="sync", graphs=None, burn=4, pair=False, Kdeg=K):
    """n_graphs independent runs (graph, notes, news, calls); returns list of run_stats dicts."""
    rng = np.random.default_rng(seed)
    runs = []
    for gi in range(n_graphs):
        J = graphs[gi] if graphs is not None else al.fixed_indegree_signed_couplings(N, Kdeg, al.WORLDS[world], 0.5, rng)
        Pk = P if Kdeg == P.K else al.replace(P, K=Kdeg)
        W = al.couplings(J, Pk)
        inp = al.draw_inputs(N, T, R, rng)
        o = al.simulate(W, Pk, T, inp, mode=mode, rng=rng)
        st = al.run_stats(o["s"], o["sig"], burn=burn)
        st["m_t"], st["q_t"] = o["s"].mean((1, 2)), (o["s"] ** 2).mean((1, 2))
        st["id2"] = float((o["sig"][burn:-2] * o["sig"][burn + 2:]).mean())      # <sigma(t) sigma(t+2)>, full moment
        if pair:                                                 # needs R >= 4
            st["A"] = np.array(al.chi_sg(o["sig"], burn=burn))   # (amplification, equal-time part, lag-1 part)
        runs.append(st)
    return runs


def agg(runs, key):
    x = np.array([r[key] for r in runs], float)
    return x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.full(x.shape[1:], np.nan), \
        x.std(0, ddof=1) if len(x) > 1 else np.full(x.shape[1:], np.nan)


# ----------------------------------------------------------------------------------------------------------------
def part_A():
    out = Tee(HERE / "out_analyst_A.txt")
    out("Part A. Gaussian closure at K = 16: DMFT (K = infinity) against the same law on actual graphs.")
    out("Late-round averages (rounds 5..24), R = 4 replicas per graph sharing notes and news; +- is the s.e. over graphs.")
    out("columns: q = <s^2>, q_ab = replica overlap, rho2, rho4 = echo function at lags 2, 4, m = <s>.\n")
    cells = {"quiet a=(2,2)": Agent(2, 2, **REF), "near a=(4,4)": Agent(4, 4, **REF),
             "tilted a=(4,3), c=0.15": Agent(4, 3, c=0.15, **REF)}
    T, res = 24, {}
    sizes = [(256, 16, 8), (1024, 16, 4), (4096, 16, 2), (4096, 64, 2)]
    for name, P in cells.items():
        out(f"--- {name}:  Jbar = {al.effective(P, 'mutual')['Jbar']:.2f}, g^2 = {al.effective(P, 'mutual')['g2']:.3f}, "
            f"gamma_eff(antagonistic) = {al.effective(P, 'antagonistic')['gamma_eff']:.2f}")
        for w in W3:
            d = al.dmft(P, w, T=T, M=25000, copies=8, seed=11, burn=4)
            dm = dict(q=d["q"][5:].mean(), q_ab=np.mean([d["q_ab"][t, t] for t in range(5, T + 1)]), rho2=d["rho"][1],
                      rho4=d["rho"][3], m=d["m"][5:].mean())
            out(f"  {w:13s} DMFT            q {dm['q']:.4f}  q_ab {dm['q_ab']:.4f}  rho2 {dm['rho2']:+.4f}  "
                f"rho4 {dm['rho4']:+.4f}  m {dm['m']:+.4f}")
            res[f"{name}|{w}|dmft"] = dm
            for N, Kd, G in sizes:
                # K = 64 at the same g: gains scale as sqrt(K / 16) so that g^2 = (a+^2 + a-^2) / 2K is unchanged, and
                # Jbar is kept by construction only when a+ = a-; for the tilted cell K = 64 is skipped
                if Kd != 16 and P.a_plus != P.a_minus:
                    continue
                Pk = al.replace(P, a_plus=P.a_plus * np.sqrt(Kd / 16), a_minus=P.a_minus * np.sqrt(Kd / 16))
                runs = sim_runs(Pk, w, N, T, 4, G, seed=100 + N + Kd, Kdeg=Kd)
                row = {}
                for k, kk in (("q", "q"), ("q_ab", "q_ab"), ("m", "m")):
                    row[k] = agg(runs, kk)[:2]
                rho = agg(runs, "rho")
                row["rho2"], row["rho4"] = (rho[0][1], rho[1][1]), (rho[0][3], rho[1][3])
                out(f"  {w:13s} N={N:5d} K={Kd:2d}    " + "  ".join(
                    f"{k} {row[k][0]:+.4f}±{row[k][1]:.4f}" for k in ("q", "q_ab", "rho2", "rho4", "m")))
                res[f"{name}|{w}|N{N}K{Kd}"] = {k: [float(v[0]), float(v[1])] for k, v in row.items()}
        out("")
    # transient of m(t), q(t): tilted cell, rounds 0..8
    P = cells["tilted a=(4,3), c=0.15"]
    out("Transient, tilted cell (common BUY bias c = 0.15): m(t) and q(t), rounds 0..8; simulation N = 1024, 8 graphs.")
    for w in W3:
        d = al.dmft(P, w, T=10, M=200000, seed=12)
        runs = sim_runs(P, w, 1024, 10, 2, 8, seed=300)
        mt, mse, _ = agg(runs, "m_t")
        qt, qse, _ = agg(runs, "q_t")
        out(f"  {w:13s} m DMFT {fmt(d['m'][:9])}\n  {'':13s} m sim  {fmt(mt[:9])}  s.e. {mse[:9].max():.4f}")
        out(f"  {'':13s} q DMFT {fmt(d['q'][:9])}\n  {'':13s} q sim  {fmt(qt[:9])}  s.e. {qse[:9].max():.4f}")
        res[f"transient|{w}"] = dict(m_dmft=d["m"][:9].tolist(), m_sim=mt[:9].tolist(), q_dmft=d["q"][:9].tolist(),
                                     q_sim=qt[:9].tolist(), m_se=float(mse[:9].max()), q_se=float(qse[:9].max()))
    # Curie-Weiss reference for the same cell: s = tanh(Jbar m + a0 h0 + nu xi + c), no disorder field
    z, wz = np.polynomial.hermite_e.hermegauss(80)
    wz = wz / wz.sum()
    e = al.effective(P, "mutual")
    m, sd = 0.0, np.hypot(P.a0, P.nu)
    for _ in range(500):
        m = wz @ np.tanh(e["Jbar"] * m + P.c + sd * z)
    q_cw = wz @ np.tanh(e["Jbar"] * m + P.c + sd * z) ** 2
    out(f"  Curie-Weiss (no disorder field, any world): m = {m:.4f}, q = {q_cw:.4f}")
    res["transient|curie_weiss"] = dict(m=float(m), q=float(q_cw))
    # field shape: excess kurtosis of the logit x on graphs
    out("\nShape of the field: excess kurtosis of x = atanh s over agents and rounds, N = 1024, against the DMFT's own")
    out("value (the reaction term makes the total field a mixture, so the DMFT value is not 0 in reciprocal worlds).")
    out("For K inputs of +-1 the cavity part has excess kurtosis -(2/K) (g^2 / Var x)^2: -0.07 at a = (4,4).")
    for name, P in cells.items():
        row = []
        for w in W3:
            runs = sim_runs(P, w, 1024, T, 2, 2, seed=500)
            kd = al.dmft(P, w, T=T, M=200000, seed=13)["x_kurt"]
            row.append(f"{w} graph {agg(runs, 'x_kurt')[0]:+.3f}, DMFT {kd:+.3f}")
            res[f"kurt|{name}|{w}"] = [float(agg(runs, "x_kurt")[0]), float(kd)]
        out(f"  {name:24s} " + "   ".join(row))
    OUT["A"] = res
    json.dump(res, open(HERE / "analyst_part_A.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------
def chi_local(P, world, T=40, M=200000, seeds=(0, 1, 2), L=25, jbar=None):
    """Dressed static susceptibility chi = sum_{tau <= L} R(tau) of the mean call (last round's kernel), mean and
    s.e. over seeds, and the mean Lam(omega) and lag-1 autocorrelation of calls. The kernel is negligible beyond
    lag 25 for Lam < 0.7; longer sums only add Monte Carlo noise (0.0025 per lag at M = 2e5)."""
    chi, lam, c1 = [], [], []
    for sd in seeds:
        d = al.dmft(P, world, T=T, M=M, seed=sd, jbar=jbar)
        R = d["R"][T, T - 1::-1][:L]
        chi.append(R.sum())
        lam.append(d["eff"]["g2"] * np.abs([(R * np.exp(-1j * om * np.arange(1, L + 1))).sum() for om in (0, np.pi / 2, np.pi)]) ** 2)
        c1.append(d["C"][T, T - 1] - d["m"][T] * d["m"][T - 1])
    chi = np.array(chi)
    return chi.mean(), chi.std(ddof=1) / np.sqrt(len(chi)), np.mean(lam, 0), float(np.mean(c1)), d


def part_B():
    out = Tee(HERE / "out_analyst_B.txt")
    res = {}
    out("Part B. Stability.")
    out("\nB1. Mutual world, logistic sampled calls, no news, no notes (an equilibrium system): chi = sum_tau R(tau) = 1 + c1,")
    out("    Lam(0) = g^2 (1 + c1)^2, Lam(pi) = g^2 (1 - c1)^2, c1 = lag-1 autocorrelation of calls (zero without memory),")
    out("    and <sigma(t) sigma(t+2)> = <s^2>. DMFT, lags 1..25, M = 2e5, 3 seeds (80 lags with memory).")
    for a, kap in ((2.0, 0.0), (3.0, 0.0), (3.5, 0.0), (1.5, 0.5), (2.0, 0.5)):
        P = Agent(a, a, nu=0.0, kappa=kap)
        # with memory the kernel decays slowly: sum 80 lags there (M = 1.2e5, 2 seeds; noise 0.02)
        chi, se, lam, c1, d = chi_local(P, "mutual") if kap == 0 else chi_local(P, "mutual", T=80, M=120000, seeds=(0, 1), L=80)
        g2 = d["eff"]["g2"]
        out(f"    a = {a}, kappa = {kap}: g^2 = {g2:.4f}  chi = {chi:.4f}±{se:.4f}  1 + c1 = {1 + c1:.4f}  Lam(0), Lam(pi) = {lam[0]:.4f}, {lam[2]:.4f}  "
            f"g^2 (1 ± c1)^2 = {g2 * (1 + c1)**2:.4f}, {g2 * (1 - c1)**2:.4f}  C_sig(t, t-2) = {d['C'][-1, -3]:.4f}  <s^2> = {d['q'][-1]:.4f}")
        res[f"B1|a{a}k{kap}"] = dict(g2=g2, chi=float(chi), chi_se=float(se), c1=c1, Lam0=float(lam[0]), Lampi=float(lam[2]),
                                    C2=float(d["C"][-1, -3]), q=float(d["q"][-1]))
    out("    The identity on actual graphs (any N, K; with memory kappa the couplings stay symmetric):")
    for a, kap, N in ((3.0, 0.0, 256), (4.0, 0.0, 256), (3.0, 0.5, 256), (4.0, 0.0, 1024)):
        runs = sim_runs(Agent(a, a, nu=0.0, a0=0.4, kappa=kap), "mutual", N, 200, 2, 4, seed=22, burn=20)
        i2, i2se, _ = agg(runs, "id2")
        q, qse, _ = agg(runs, "q")
        out(f"    a = {a}, a0 = 0.4, kappa = {kap}, N = {N}: <sigma(t) sigma(t+2)> = {i2:.4f}±{i2se:.4f}   <s^2> = {q:.4f}±{qse:.4f}")
        res[f"B1id|a{a}k{kap}N{N}"] = [float(i2), float(q)]
    out("    With news the law is not logistic and the identity becomes <sigma(t) sigma(t+2)> = <phi(u)^2> < <s^2>:")
    runs = sim_runs(Agent(4, 4, nu=0.5), "mutual", 1024, 200, 2, 2, seed=23, burn=20)
    out(f"    a = 4, nu = 0.5: <sigma(t) sigma(t+2)> = {agg(runs, 'id2')[0]:.4f}   <s^2> = {agg(runs, 'q')[0]:.4f}")
    res["B1id|nu0.5"] = [float(agg(runs, "id2")[0]), float(agg(runs, "q")[0])]

    out("\nB2. Tree form at K = 16. Mutual world, a+ = a- = a, no news: Lam_tree = (K - 1) tanh^2(a / K); summed squared")
    out("    correlations of calls at equal times (even distances) 1 + K t^2 Lam / (1 - Lam^2) and at lag 1 (odd distances)")
    out("    K t^2 / (1 - Lam^2), t = tanh(a / K). Simulation: N = 2048, 3000 rounds, products of two replicas (no bias).")
    rng = np.random.default_rng(5)
    N = 2048
    J = al.trust_graph(N, K, "mutual", rng)
    for a in (2.0, 3.0, 3.6):
        P = Agent(a, a, nu=0.0)
        Tl = 3000
        o = al.simulate(al.couplings(J, P), P, Tl, al.draw_inputs(N, Tl, 2, rng))
        sg = o["sig"][200:]
        c0 = [(sg[:, r].T @ sg[:, r]) / sg.shape[0] for r in range(2)]
        c1 = [(sg[1:, r].T @ sg[:-1, r]) / (sg.shape[0] - 1) for r in range(2)]
        chi0, chi1 = (c0[0] * c0[1]).sum() / N, (c1[0] * c1[1]).sum() / N
        t2 = np.tanh(a / K) ** 2
        lam = (K - 1) * t2
        out(f"    a = {a}: Lam_tree = {lam:.3f} (K = inf: g^2 = {a * a / K:.3f});  equal time: sim {chi0:.3f}, tree "
            f"{1 + K * t2 * lam / (1 - lam**2):.3f};  lag 1: sim {chi1:.3f}, tree {K * t2 / (1 - lam**2):.3f}")
        res[f"B2|a{a}"] = dict(lam_tree=float(lam), chi0_sim=float(chi0), chi0_tree=float(1 + K * t2 * lam / (1 - lam**2)),
                               chi1_sim=float(chi1), chi1_tree=float(K * t2 / (1 - lam**2)))
    out(f"    boundary Lam_tree = 1: a_c = {al.bethe_gain(K):.3f} (a- = a+), {al.bethe_gain(K, 0.75):.3f} (a- = 0.75 a+), "
        f"{al.bethe_gain(K, 0.5):.3f} (a- = 0.5 a+); K = inf: 4.000, 4.525, 5.060")
    res["B2|a_c"] = dict(r1=al.bethe_gain(K), r075=al.bethe_gain(K, 0.75), r05=al.bethe_gain(K, 0.5))

    out("\nB3. Lam(omega) at omega = 0, pi/2, pi per world, from the solver (nested: 3000 sites x 64 copies sharing notes and")
    out("    news, T = 40). Reference design nu = 0.5, a0 = 0.3 unless stated. Monte Carlo error about 0.02.")
    out("    one-way closed form: Lam = g^2 <(E_thermal sech^2 x)^2>.  mutual estimate: (g beta_eff)^2, beta_eff = (1 + pi nu^2/2)^(-1/2).")
    table = []
    grid = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (4, 3), (5.3, 4), (4, 2), (6, 3)]
    for ap, am in grid:
        P = Agent(ap, am, **REF)
        row = dict(a_plus=ap, a_minus=am, g2=al.effective(P, "mutual")["g2"], Jbar=al.effective(P, "mutual")["Jbar"],
                   gamma_antag=al.effective(P, "antagonistic")["gamma_eff"])
        line = f"    a = ({ap}, {am}): g^2 = {row['g2']:.3f}, Jbar = {row['Jbar']:.2f}, gamma_eff(ant) = {row['gamma_antag']:+.2f} |"
        for w in W3:
            d = al.dmft(P, w, T=40, M=3000, copies=64, seed=31)
            row[w] = d["Lam"].tolist()
            row[w + "_q"] = float(d["q"][-8:].mean())
            line += f" {w[:3]} {fmt(d['Lam'], 2)}"
        table.append(row)
        out(line)
    res["B3"] = table
    out("    sensitivity at a = (4, 4):")
    for tag, kw in (("nu = 0, a0 = 0", dict(nu=0.0, a0=0.0)), ("nu = 0.5, a0 = 0", dict(nu=0.5, a0=0.0)),
                    ("nu = 1.0, a0 = 0.3", dict(nu=1.0, a0=0.3)), ("nu = 0.5, a0 = 0.6", dict(nu=0.5, a0=0.6)),
                    ("nu = 0.5, a0 = 0.3, kappa = 0.5", dict(nu=0.5, a0=0.3, kappa=0.5)),
                    ("nu = 0.5, a0 = 0.3, kappa = 1.0", dict(nu=0.5, a0=0.3, kappa=1.0))):
        P = Agent(4, 4, **kw)
        line = f"      {tag:34s}|"
        for w in W3:
            # nothing is shared between copies when nu = a0 = 0: all sites are equivalent, use the pooled kernel
            d = al.dmft(P, w, T=40, M=3000, copies=64, seed=32) if (P.nu or P.a0) else al.dmft(P, w, T=40, M=200000, seed=32)
            res[f"B3s|{tag}|{w}"] = d["Lam"].tolist()
            line += f" {w[:3]} {fmt(d['Lam'], 2)}"
        out(line)

    out("\nB4. One-way world: Lam = g^2 d^2 with d = <sech^2(sqrt(g^2 + nu^2) z)> (a0 = 0) stays below 2/pi at any gain.")
    z = np.linspace(-8, 8, 400001)
    wz = np.exp(-z**2 / 2)
    wz = wz / wz.sum()
    for nu in (0.0, 0.5):
        vals = []
        for g in (0.5, 1.0, 2.0, 4.0, 16.0):
            d = wz @ (1 / np.cosh(np.hypot(g, nu) * z) ** 2)
            vals.append(g * g * d * d)
        out(f"    nu = {nu}: Lam at g = 0.5, 1, 2, 4, 16: {fmt(vals)}   (2/pi = {2 / np.pi:.3f})")
        res[f"B4|nu{nu}"] = [float(v) for v in vals]
    out("    Mutual world with news: chi(g) = sum_tau R(tau) and Lam = g^2 chi^2 (a0 = 0; lags 1..25, M = 2e5, 3 seeds);")
    out("    boundary where g chi(g) = 1, taking chi at g = 0.75 (a = 3), where the kernel sum has converged.")
    for nu in (0.0, 0.25, 0.5, 0.75, 1.0):
        chis = [chi_local(Agent(a, a, nu=nu), "mutual")[:2] for a in (2.0, 3.0)]
        be, be2 = al.beta_eff(nu), 1 / np.sqrt(1 + np.pi * nu**2 / 2)
        gc = 1 / chis[1][0]
        out(f"    nu = {nu:4.2f}: chi(g=0.5) = {chis[0][0]:.3f}±{chis[0][1]:.3f}, chi(g=0.75) = {chis[1][0]:.3f}±{chis[1][1]:.3f}; slope of phi at 0 = {be:.3f}, "
            f"(1 + pi nu^2/2)^-1/2 = {be2:.3f};  g_c = {gc:.3f}, a_c(K=inf) = {4 * gc:.2f}, a_c(K=16 tree) = {al.bethe_gain(K) * gc:.2f}")
        res[f"B4|chi|nu{nu}"] = dict(chi05=float(chis[0][0]), chi075=float(chis[1][0]), chi075_se=float(chis[1][1]), beta_slope=float(be),
                                     beta_probit=float(be2), g_c=float(gc), a_c_inf=float(4 * gc), a_c_tree=float(al.bethe_gain(K) * gc))
    out("    Antagonistic world: Lam(pi/2) against the fade ratio r = a-/a+ at fixed g^2, with the mean coupling set to zero")
    out("    in the solver to isolate reciprocity from herding (nu = 0.5, a0 = 0; pooled kernel, M = 2e5, T = 40).")
    for g2 in (0.64, 1.0, 1.44, 4.0):
        line = f"    g^2 = {g2}:"
        for r in (1.0, 0.75, 0.5, 0.25, 0.0):
            ap = np.sqrt(2 * K * g2 / (1 + r * r))
            d = al.dmft(Agent(ap, r * ap, nu=0.5), "antagonistic", T=40, M=200000, seed=42, jbar=0.0)
            line += f"  r = {r}: gamma_eff = {d['eff']['gamma_eff']:+.2f}, Lam = {d['Lam'][1]:.2f};"
            res[f"B4|antag|g2{g2}|r{r}"] = dict(gamma_eff=d["eff"]["gamma_eff"], Lam=float(d["Lam"][1]))
        out(line)
    out("\nB5. Herding margin. Response of the mean call to a small common bias, chi_u = dm/dc = chi / (1 - Jbar chi);")
    out("    the swarm herds (Curie-Weiss instability) when Jbar chi = 1. chi per world from the solver at Jbar = 0,")
    out("    then the tilt Jbar_c = 1 / chi at which herding sets in. nu = 0.5, a0 = 0.3; lags 1..25, M = 2e5, 3 seeds.")
    for a in (2, 3, 4, 5):
        line = f"    g^2 = {a * a / K:.3f}:"
        for w in W3:
            chi = chi_local(Agent(a, a, **REF), w)[0]
            line += f"  {w} chi = {chi:.3f}, Jbar_c = {1 / chi:.2f};"
            res[f"B5|a{a}|{w}"] = float(chi)
        cw = al.beta_eff(np.hypot(0.5, 0.3))
        out(line + f"  Curie-Weiss chi = {cw:.3f}, Jbar_c = {1 / cw:.2f}")
        res[f"B5|a{a}|cw"] = float(cw)
    out("    check on graphs (N = 1024, 4 graphs, T = 40): mean belief m under a common bias c = 0.1 at a = (4.4, 3.6), Jbar = 0.4")
    for w in W3:
        P = Agent(4.4, 3.6, c=0.1, **REF)
        runs = sim_runs(P, w, 1024, 40, 2, 4, seed=52, burn=10)
        d = al.dmft(P, w, T=40, M=200000, seed=53)
        out(f"    {w:13s} m sim = {agg(runs, 'm')[0]:.4f}±{agg(runs, 'm')[1]:.4f}   DMFT = {d['m'][10:].mean():.4f}")
        res[f"B5|m|{w}"] = [float(agg(runs, "m")[0]), float(agg(runs, "m")[1]), float(d["m"][10:].mean())]
    OUT["B"] = res
    json.dump(res, open(HERE / "analyst_part_B.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------
INTENS = {"quiet (2,2)": (2, 2), "mid (3,3)": (3, 3), "near (4,4)": (4, 4), "close (5,5)": (5, 5), "tilted (4,3)": (4, 3)}


def tau_c(c):
    """Correlation time in rounds from the even lags of c: |c_4 / c_2| = exp(-2 / tau)."""
    r = abs(c[3] / c[1]) if c[1] != 0 else np.nan
    return -2 / np.log(r) if 0 < r < 1 else np.nan


def part_C():
    out = Tee(HERE / "out_analyst_C.txt")
    res = {}
    out("Part C. Signatures per world. Reference design nu = 0.5, a0 = 0.3. Simulations on K = 16 graphs, N = 1024,")
    out("6 graphs x 4 replicas, T = 40, rounds 5..40. rho = echo function (lags 1..4); c = lag correlation of replica")
    out("differences of beliefs (lags 1..4); S = 1 + 2 sum_{tau<=4} c_tau cos(omega tau) at omega = 0, pi/2, pi;")
    out("X2 = rho2 - rho1^2; tau = correlation time from |c4/c2|; A = (amplification, its equal-time part, its lag-1 part)")
    out("from the spin-glass susceptibility of the calls (analyst_lib.chi_sg; 1 / (1 - Lam) on a tree).\n")
    variants = {"sync": dict(), "sync, memory kappa = 0.5": dict(kappa=0.5), "async": dict()}
    for vname, kw in variants.items():
        mode = "async" if vname == "async" else "sync"
        out(f"=== {vname}")
        for iname, (ap, am) in INTENS.items():
            if vname != "sync" and iname in ("mid (3,3)", "close (5,5)"):
                continue
            P = Agent(ap, am, **REF, **kw)
            for w in W3:
                G, T = (3, 30) if mode == "async" else (6, 40)
                runs = sim_runs(P, w, 1024, T, 4, G, seed=700, mode=mode, pair=True)
                rho, c, S, A = (agg(runs, k)[0] for k in ("rho", "c", "S", "A"))
                d = None
                if mode == "sync":
                    d = al.dmft(P, w, T=40, M=200000, seed=71)
                rec = dict(rho=rho[:4].tolist(), c=c[:4].tolist(), S=S.tolist(), X2=float(rho[1] - rho[0] ** 2),
                           tau=float(tau_c(c)), A=A.tolist(), q=float(agg(runs, "q")[0]), D=float(agg(runs, "D")[0]),
                           rho_dmft=None if d is None else d["rho"][:4].tolist())
                res[f"{vname}|{iname}|{w}"] = rec
                out(f"  {iname:15s} {w:13s} rho {fmt(rho[:4])} c {fmt(c[:4], 2)} S {fmt(S, 2)} X2 {rec['X2']:+.3f} "
                    f"tau {rec['tau']:.1f} A {fmt(A, 3)} q {rec['q']:.3f}"
                    + ("" if d is None else f" | DMFT rho {fmt(d['rho'][:4])}"))
        out("")
    out("N dependence near the boundary (sync, no memory): c2, c4, tau, amplification A, per N (K = 16).")
    for iname in ("near (4,4)", "close (5,5)"):
        ap, am = INTENS[iname]
        P = Agent(ap, am, **REF)
        for w in ("mutual", "antagonistic", "oneway"):
            line = f"  {iname:15s} {w:13s}"
            for N, G in ((256, 16), (1024, 6), (4096, 2)):
                runs = sim_runs(P, w, N, 40, 4, G, seed=800 + N, pair=True)
                c, cse, _ = agg(runs, "c")
                A, Ase, _ = agg(runs, "A")
                As = np.array([r["A"][0] for r in runs])
                rec = dict(c2=float(c[1]), c2_se=float(cse[1]), c4=float(c[3]), tau=float(tau_c(c)), Asum=float(As.mean()),
                           Asum_se=float(As.std(ddof=1) / np.sqrt(len(As))), rho2=float(agg(runs, "rho")[0][1]),
                           q=float(agg(runs, "q")[0]))
                res[f"Ndep|{iname}|{w}|N{N}"] = rec
                line += f"  N={N}: c2 {rec['c2']:+.3f}±{rec['c2_se']:.3f} c4 {rec['c4']:+.3f} tau {rec['tau']:.1f} A {rec['Asum']:.3f}±{rec['Asum_se']:.3f} rho2 {rec['rho2']:+.3f} q {rec['q']:.3f};"
            out(line)
    out("\nSensitivity of the echo to what the note transmits (leak = share of the sender's belief carried by the note),")
    out("near (4,4), sync, N = 1024, 3 graphs:")
    for leak in (0.0, 0.25, 0.5):
        line = f"  leak = {leak}:"
        for w in W3:
            runs = sim_runs(Agent(4, 4, leak=leak, **REF), w, 1024, 40, 4, 3, seed=900)
            rho, c = agg(runs, "rho")[0], agg(runs, "c")[0]
            line += f"  {w} rho2 {rho[1]:+.3f} c2 {c[1]:+.3f} q {agg(runs, 'q')[0]:.3f};"
            res[f"leak{leak}|{w}"] = dict(rho2=float(rho[1]), c2=float(c[1]), q=float(agg(runs, "q")[0]))
        out(line)
    OUT["C"] = res
    json.dump(res, open(HERE / "analyst_part_C.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------
def part_D():
    out = Tee(HERE / "out_analyst_D.txt")
    res = {}
    out("Part D. Kick twins. Flip one analyst's call as its readers see it at t0 (D = -2 sigma_i(t0)), replay t0+1, t0+2.")
    out("r_A, r_D: belief change of an ally-tie / faded-tie reader at t0+1 per unit kick; E: belief change of the kicked")
    out("analyst at t0+2 per unit kick; E_x the same in logit; lin = linear response, r = (+a+/K, -a-/K) <d>, E = Gamma <d_i d_j>.")
    out("N = 256, 4 graphs x 2 replicas x t0 in {8, 12}, every analyst kicked once per (replica, t0): 4096 twins per cell.")
    out("sd = s.d. of one twin's echo per unit kick; n3 = twins for |E| = 3 s.e.; crn = calls redrawn with the base run's")
    out("uniforms, ind = calls of every changed analyst redrawn independently.\n")
    N = 256
    for iname, (ap, am) in INTENS.items():
        P = Agent(ap, am, **REF)
        for w in W3:
            rng = np.random.default_rng(1000)
            acc = {k: [] for k in ("rA", "rD", "E_crn", "E_ind", "Ex_crn", "lin_rA", "lin_rD", "lin_E")}
            e = al.effective(P, w)
            for g in range(4):
                J = al.trust_graph(N, K, w, rng)
                W = al.couplings(J, P)
                inp = al.draw_inputs(N, 16, 2, rng)
                base = al.simulate(W, P, 16, inp)
                Wd = W.toarray()
                for rep in range(2):
                    for t0 in (8, 12):
                        ag = np.arange(N)
                        tw = al.kick_twins(W, J, P, base, inp, rep, t0, ag, crn=True)
                        tw2 = al.kick_twins(W, J, P, base, inp, rep, t0, ag, crn=False, rng=rng)
                        D = tw["D"]
                        # readers j of kicked i: J[j, i] != 0
                        ally, faded = (J[:, ag] > 0).T, (J[:, ag] < 0).T          # (B, N): reader j of twin b
                        acc["rA"].append((tw["ds1"] * D[:, None])[ally] / 4)
                        acc["rD"].append((tw["ds1"] * D[:, None])[faded] / 4)
                        acc["E_crn"].append(tw["ds2"][ag, ag] * D / 4)
                        acc["E_ind"].append(tw2["ds2"][ag, ag] * D / 4)
                        s2b = base["s"][t0 + 2, rep]
                        x2 = np.arctanh(np.clip(s2b[ag] + tw["ds2"][ag, ag], -1 + 1e-12, 1 - 1e-12)) - np.arctanh(s2b[ag])
                        acc["Ex_crn"].append(x2 * D / 4)
                        d1, d2 = 1 - base["s"][t0 + 1, rep] ** 2, 1 - s2b**2
                        acc["lin_rA"].append(np.full(1, P.a_plus / K * d1.mean()))
                        acc["lin_rD"].append(np.full(1, -P.a_minus / K * d1.mean()))
                        # (K2) on the graph: d_i sum_j W_ij d_j W_ji
                        acc["lin_E"].append(d2 * ((Wd * Wd.T) @ d1))
            a = {k: np.concatenate(v) for k, v in acc.items()}
            E, sd_c, sd_i = a["E_crn"].mean(), a["E_crn"].std(), a["E_ind"].std()
            n3 = lambda sd, E_: 9 * sd**2 / E_**2 if abs(E_) > 1e-9 else np.inf
            rec = dict(rA=float(a["rA"].mean()), rD=float(a["rD"].mean()), rA_lin=float(a["lin_rA"].mean()),
                       rD_lin=float(a["lin_rD"].mean()), E=float(E), E_ind=float(a["E_ind"].mean()),
                       E_se=float(sd_c / np.sqrt(len(a["E_crn"]))), E_lin=float(a["lin_E"].mean()),
                       E_x=float(a["Ex_crn"].mean()), E_x_lin=float(e["Gamma"] * (1 - base["s"][8:] ** 2).mean()),
                       sd_crn=float(sd_c), sd_ind=float(sd_i), n3_crn=float(n3(sd_c, E)), n3_ind=float(n3(sd_i, a["E_ind"].mean())),
                       max_abs_echo=float(np.abs(a["E_crn"]).max()),
                       rel=float(E / (K / 2 * (a["rA"].mean() ** 2 + a["rD"].mean() ** 2))) if w == "mutual" else
                       (float(E / (K * a["rA"].mean() * a["rD"].mean())) if w == "antagonistic" else None))
            res[f"{iname}|{w}"] = rec
            out(f"  {iname:15s} {w:13s} r_A {rec['rA']:+.4f} (lin {rec['rA_lin']:+.4f})  r_D {rec['rD']:+.4f} (lin {rec['rD_lin']:+.4f})  "
                f"E {rec['E']:+.4f}±{rec['E_se']:.4f} (lin {rec['E_lin']:+.4f}; ind {rec['E_ind']:+.4f})  E_x {rec['E_x']:+.4f} (lin {rec['E_x_lin']:+.4f})  "
                f"sd crn {rec['sd_crn']:.3f} ind {rec['sd_ind']:.3f}  n3 crn {rec['n3_crn']:.0f} ind {rec['n3_ind']:.0f}"
                + (f"  max|echo| {rec['max_abs_echo']:.1e}" if w == "oneway" else f"  E/pred(r) {rec['rel']:.3f}"))
        out("")
    OUT["D"] = res
    json.dump(res, open(HERE / "analyst_part_D.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------------
def part_E():
    out = Tee(HERE / "out_analyst_E.txt")
    res = {}
    out("Part E. Sizing. One run = one graph, R = 2 replicas (shared notes and news), T = 20, rounds 5..20.")
    out("mean and run-to-run s.d. of each statistic; 24 runs at N = 256, 12 at N = 1024. Reference design, sync.")
    out("The amplification A needs R = 4 (two replica pairs) and is given for R = 4.\n")
    keys = ("q", "q_ab", "rho2", "X2", "c2", "S0", "Spi2", "Spi", "m", "id2")
    for iname, (ap, am) in INTENS.items():
        P = Agent(ap, am, **REF)
        for N, G in ((256, 24), (1024, 12)):
            for w in W3:
                runs = sim_runs(P, w, N, 20, 2, G, seed=2000 + N)
                for r in runs:
                    r.update(rho2=r["rho"][1], X2=r["rho"][1] - r["rho"][0] ** 2, c2=r["c"][1], S0=r["S"][0], Spi2=r["S"][1], Spi=r["S"][2])
                runs4 = sim_runs(P, w, N, 20, 4, max(G // 2, 4), seed=3000 + N, pair=True)
                As = np.array([r["A"][0] for r in runs4])
                rec = {k: [float(agg(runs, k)[0]), float(agg(runs, k)[2])] for k in keys}
                rec["A"] = [float(As.mean()), float(As.std(ddof=1))]
                res[f"{iname}|N{N}|{w}"] = rec
                out(f"  {iname:15s} N={N:4d} {w:13s} " + "  ".join(f"{k} {rec[k][0]:+.3f} ({rec[k][1]:.3f})" for k in keys + ("A",)))
        out("")
    OUT["E"] = res
    json.dump(res, open(HERE / "analyst_part_E.json", "w"), indent=1)


def part_G():
    out = Tee(HERE / "out_analyst_G.txt")
    res = {}
    out("Part G. Where the boundary is on K = 16 graphs, from long simulations. N = 2048, one graph per world, 4 replicas,")
    out("1500 rounds (the first 300 dropped). A = amplification (analyst_lib.chi_sg); on a tree 1/A = 1 - Lam, so 1/A falls")
    out("linearly to zero at the boundary, rounded at finite N (at the boundary A is of order N^(1/3) = 13). c20, c40 = lag")
    out("correlations of replica differences of beliefs at lags 20 and 40. a_c = zero of the least-squares line through the")
    out("points with 1/A > 0.15 (the curve is slightly convex, so this overestimates a_c by a few percent).\n")
    rng = np.random.default_rng(78)
    N = 2048
    Js = {w: al.trust_graph(N, K, w, rng) for w in W3}
    cases = [("mutual", 0.0, 0.0, (2.6, 3.0, 3.3, 3.6, 4.0, 4.4)), ("mutual", 0.5, 0.0, (3.0, 3.5, 4.0, 4.5, 5.0, 5.5)),
             ("mutual", 0.5, 0.3, (3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0)), ("antagonistic", 0.5, 0.3, (3.0, 4.0, 4.5, 5.0, 6.0, 7.0)),
             ("oneway", 0.5, 0.3, (4.0, 6.0, 8.0, 12.0)), ("mutual", 0.25, 0.15, (3.0, 3.5, 4.0, 4.5, 5.0))]
    for w, nu, a0, gains in cases:
        pts = []
        for a in gains:
            P = Agent(a, a, nu=nu, a0=a0)
            T = 1500
            o = al.simulate(al.couplings(Js[w], P), P, T, al.draw_inputs(N, T, 4, rng))
            A = al.chi_sg(o["sig"], burn=300)[0]
            c, _ = al.diff_autocorr(o["s"], burn=300, L=40)
            pts.append((a, 1 / A))
            res[f"{w}|nu{nu}|a0{a0}|a{a}"] = dict(A=float(A), c2=float(c[1]), c20=float(c[19]), c40=float(c[39]))
            out(f"  {w:13s} nu = {nu}, a0 = {a0}, a = {a:4.1f}: A = {A:6.2f}  1/A = {1 / A:.3f}  c2 {c[1]:+.3f}  c20 {c[19]:+.3f}  c40 {c[39]:+.3f}")
        good = np.array([p_ for p_ in pts if p_[1] > 0.15])
        if len(good) >= 3 and w != "oneway":
            slope, icpt = np.polyfit(good[:, 0], good[:, 1], 1)
            ac = -icpt / slope
            pred = al.bethe_gain(K) * np.sqrt(1 + np.pi * nu**2 / 2)
            out(f"  -> a_c = {ac:.2f} from {len(good)} points   (tree form x news factor, no notes: {pred:.2f})")
            res[f"{w}|nu{nu}|a0{a0}|a_c"] = dict(a_c=float(ac), tree_news=float(pred), n_points=len(good))
        out("")
    OUT["G"] = res
    json.dump(res, open(HERE / "analyst_part_G.json", "w"), indent=1)


def part_F():
    """Collect the part JSONs and the frozen mapping into analyst_predictions.json."""
    import subprocess
    parts = {p: json.load(open(HERE / f"analyst_part_{p}.json")) for p in "ABCDEG" if (HERE / f"analyst_part_{p}.json").exists()}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    frozen = dict(
        note="theory/analyst_scenario.md", frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        parent_commit=commit, code=["theory/checks/analyst_lib.py", "theory/checks/check_analyst.py"],
        status="frozen before any analyst-scenario data exist; calibration plugs numbers into `mapping`, nothing else changes",
        law="x_i(t) = sum_j W_ij sigma_j(t) + a0 h0_i + nu xi_i(t+1) + c + kappa sigma_i(t); s_i(t+1) = tanh x_i(t); "
            "sigma_i(t+1) = +-1 with P(+1) = (1 + s_i(t+1))/2; W_ij = +a_plus/K (ally), -a_minus/K (faded)",
        mapping=dict(
            inputs=["a_plus", "a_minus", "nu = a_n sd(news)", "a0 = a_0 sd(note)", "c", "kappa = R0(2) / (R0(1) <d>)", "K"],
            Jbar="(a_plus - a_minus) / 2", g2="(a_plus^2 + a_minus^2) / (2 K)",
            Gamma=dict(mutual="g2", oneway="0", antagonistic="-a_plus a_minus / K"),
            gamma_eff_antagonistic="-2 a_plus a_minus / (a_plus^2 + a_minus^2)",
            Lam_oneway="g2 <(E_thermal sech^2 x)^2>, at most 2/pi",
            Lam_mutual="(K-1) [tanh^2(b a_plus/K) + tanh^2(b a_minus/K)] / 2 with b = (1 + pi nu^2 / 2)^(-1/2) (a0 = 0, kappa = 0); "
                       "standing notes lower it (reference design: boundary at rms gain 5.9 instead of 5.0, part G); memory "
                       "multiplies Lam(0) by (1 + c1)^2 and Lam(pi) by (1 - c1)^2; otherwise analyst_lib.dmft(P, 'mutual', "
                       "copies=64)['Lam'] below the boundary and the long simulation of part G near it",
            Lam_antagonistic="equal to Lam_mutual, at omega = pi/2, when a_minus = a_plus and a0 = 0; otherwise analyst_lib.dmft",
            response_kernel="R(tau + 1) = (1 - q) rho_tau, rho_tau = sum zeta_i(t) s_i(t + tau) / sum zeta_i(t)^2, zeta = sigma - s",
            echo="rho_2 = Gamma <d_i d_j> to leading order; exact value from analyst_lib.dmft(...)['rho'][1] (K = inf) or "
                 "check_analyst.sim_runs (K = 16 graph)",
            every_number="analyst_lib.dmft(Agent(...), world) and check_analyst.sim_runs(Agent(...), world, N, T, R, graphs) "
                         "with the calibrated Agent; the tables below are these functions on a grid"),
        reference_design=dict(K=K, **REF, call="sampled", p_neg=0.5),
        intensity_grid={k: list(v) for k, v in INTENS.items()},
        boundary=dict(rms_gain="sqrt((a_plus^2 + a_minus^2) / 2), half log-odds per unit normalized field",
                      mutual_no_news_no_notes_tree=al.bethe_gain(K), mutual_Kinf=4.0,
                      mutual_by_news_solver={k.split("nu")[1]: v for k, v in parts.get("B", {}).items() if k.startswith("B4|chi|")},
                      located_by_long_simulation={k: v for k, v in parts.get("G", {}).items() if k.endswith("a_c")},
                      oneway="no boundary: Lam <= 2/pi = 0.637, amplification <= 2.75"),
        index=dict(A="closure: solver against graphs at N = 256, 1024, 4096 and K = 64; transient m(t), q(t); kurtosis",
                   B="stability: exact mutual-world results, tree check, Lam(omega) table, news, antagonistic ratio, herding",
                   C="signatures: keys '<schedule>|<intensity>|<world>' with rho, c, S, X2, tau, A, q, D; N dependence; leak",
                   D="kick twins: keys '<intensity>|<world>' with rA, rD, E, linear values, noise per twin, twins for 3 s.e.",
                   E="sizing: keys '<intensity>|N<N>|<world>', each statistic as [mean, run-to-run s.d.] at T = 20",
                   G="long simulations: amplification and long-lag correlations against gain; boundary location"),
        parts=parts)
    json.dump(frozen, open(HERE / "analyst_predictions.json", "w"), indent=1)
    print("wrote", HERE / "analyst_predictions.json")


if __name__ == "__main__":
    todo = sys.argv[1:] or list("ABCDEGF")
    for p in todo:
        t = time.time()
        {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E, "G": part_G, "F": part_F}[p]()
        print(f"[part {p}: {time.time() - t:.0f} s]", flush=True)
