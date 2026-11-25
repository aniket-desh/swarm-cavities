"""Asynchronous control of the de-biased sign-rule test (theory/analyst_signrule.md, amendment): schedule, checks
and frozen numbers. Law, cell and notation: analyst_signrule_check.py ((S1)-(S3), cell P0, chat = -0.637).

Block-random schedule with B blocks. Round 0 is synchronous (no inbox). In every round t >= 1 the analysts are split
by a fresh random permutation into B blocks of N / B, shared by the replicas of a graph. The blocks update one after
another; all analysts of a block update together. An analyst updating in round t reads, for each colleague, that
colleague's LATEST call and note: the round-t ones if the colleague's block came earlier in this round, otherwise
the round-(t-1) ones. Its own previous call and note are its round-(t-1) ones, its news item is xi_i(t), its call is
drawn from tanh(x - chat) with the stored uniform U(replica, t, i). Every analyst updates exactly once per round.
B = 1 is the synchronous design; B = N is one analyst at a time (analyst_lib.simulate, mode "async").

Where the echo goes (amendment (S6)). Let p = (N/B - 1) / (N - 1) be the probability that two given analysts share
a block. The call sigma_i(t) reaches a reciprocated partner j and comes back to i
    at lag 1 with probability  w1 = (1 - p)(3 + p) / 4      (j later than i in round t and i not later than j in
                                                             round t+1; or j earlier than i in round t+1)
    at lag 2 with probability  w2 = ((1 + p) / 2)^2         (j not later than i in round t+1, i not later than j
                                                             in round t+2: i reads sigma_j(t+1) a second time)
and w1 + w2 = 1. To leading order rho_1 = kappa d + w1 G d and the direct lag-2 term is w2 G d, with G of (S4).

Parts (run: uv run python theory/checks/analyst_signrule_async_check.py [1 2 3 4 5]):
  1  code checks: B = 1 equals the synchronous simulation; B = N equals analyst_lib.simulate(mode="async")
  2  w1, w2 against the simulation for B = 1 ... N (law without memory)
  3  predictions for the schedule of the amendment (B = 16) and for B = N: echo function per world and law,
     contrasts with the one-way world, run-to-run s.d.
  4  calibration s.e. of the contrasts (bootstrap draws, common random numbers), asynchronous and synchronous
  5  sizing of falsifier 7 and the frozen JSON
"""
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyst_lib as al  # noqa: E402
import analyst_signrule_check as sc  # noqa: E402

N, K, T, R, BURN, W3, SEED = sc.N, sc.K, sc.T, sc.R, sc.BURN, sc.W3, sc.SEED
B_DESIGN = 16                      # the amendment's schedule: 16 blocks of 16
PART = HERE / "analyst_signrule_async_part.json"
_lines = []


def out(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    _lines.append(line)


def weights_w(B, n=N):
    """(w1, w2) of the module docstring for B blocks of n / B."""
    p = (n / B - 1) / (n - 1)
    return (1 - p) * (3 + p) / 4, ((1 + p) / 2) ** 2


def simulate_blocks(J, L: sc.Law, T_, inp, B, rng):
    """(S1)-(S3) under the block-random schedule. Returns X, s, sr, sig with shape (T+1, R, N), as sc.simulate."""
    CA, CB = sc.weights(J, L)
    R_, n = inp["U"].shape[1:]
    X = np.empty((T_ + 1, R_, n))
    sig = np.empty_like(X)
    X[0] = sc._priv(L, inp, 0)
    sig[0] = np.where(inp["U"][0] < (1 + np.tanh(X[0] - L.chat)) / 2, 1.0, -1.0)
    for t in range(T_):
        cur_sig, cur_sr = sig[t].copy(), np.tanh(X[t])          # latest call and raw belief of every analyst
        priv = np.broadcast_to(sc._priv(L, inp, t + 1), (R_, n))
        for blk in np.array_split(rng.permutation(n), B):       # one permutation per round, shared by the replicas
            Xb = cur_sig @ CA[blk].T + cur_sr @ CB[blk].T + priv[:, blk] + L.kappa * sig[t][:, blk]
            X[t + 1][:, blk] = Xb
            cur_sig[:, blk] = np.where(inp["U"][t + 1][:, blk] < (1 + np.tanh(Xb - L.chat)) / 2, 1.0, -1.0)
            cur_sr[:, blk] = np.tanh(Xb)
        sig[t + 1] = cur_sig
    return dict(X=X, s=np.tanh(X - L.chat), sr=np.tanh(X), sig=sig)


def runs(L, world, G, B, tag=0, R_=R):
    """G runs on the graphs and inputs of analyst_signrule_check.runs (same notes, news, uniforms as the
    synchronous predictions); the block order has its own seed. Dict of arrays (G,)."""
    acc = []
    for g in range(G):
        inp = sc.draw_inputs(N, T, R_, np.random.default_rng([SEED, 7, W3.index(world), g, N, tag]))
        o = simulate_blocks(sc.graph(world, g), L, T, inp, B, np.random.default_rng([SEED, 71, W3.index(world), g, B]))
        st = sc.run_stats(o)
        st["r12"] = st["rho1"] - st["rho2"]                      # lag-1 minus lag-2 echo of the run
        acc.append(st)
    return {k: np.array([a[k] for a in acc]) for k in acc[0]}


def load_part():
    return json.load(open(PART)) if PART.exists() else {}


def save_part(key, val):
    d = load_part()
    d[key] = val
    json.dump(d, open(PART, "w"), indent=1, default=float)


# ----------------------------------------------------------------------------------------------------------------
def part_1():
    out("Part 1. Code checks (cell P0).")
    L1 = sc.cell_laws()["M1"]
    res = {}
    out("  B = 1 against analyst_signrule_check.simulate (laws M1 and M2, residual on): largest |X - X_ref|, calls that differ.")
    for name, L in sc.cell_laws().items():
        for w in W3:
            inp = sc.draw_inputs(N, T, R, np.random.default_rng(3))
            a = simulate_blocks(sc.graph(w, 0), L, T, inp, 1, np.random.default_rng(0))
            b = sc.simulate(sc.graph(w, 0), L, T, inp)
            dx, ds = float(np.abs(a["X"] - b["X"]).max()), int((a["sig"] != b["sig"]).sum())
            out(f"    {name} {w:13s} {dx:.1e}, {ds}")
            res[f"B1|{name}|{w}"] = [dx, ds]
            assert dx < 1e-12 and ds == 0
    out("  B = N against analyst_lib.simulate(mode='async') (calls-only law, no residual, the same permutations):")
    La = replace(L1, eps=0.0)
    P = al.Agent(La.uA, -La.uD, nu=La.nu, a0=La.a0, c=La.c - La.chat, kappa=La.kappa)
    for w in W3:
        J = sc.graph(w, 1)
        inp = sc.draw_inputs(N, T, R, np.random.default_rng(4))
        a = simulate_blocks(J, La, T, inp, N, np.random.default_rng(9))
        b = al.simulate(al.couplings(J, P), P, T, inp, mode="async", rng=np.random.default_rng(9))
        ds, dg = float(np.abs(a["s"] - b["s"]).max()), int((a["sig"] != b["sig"]).sum())
        out(f"    {w:13s} largest |s - s_ref| {ds:.1e}, calls that differ {dg}")
        res[f"BN|{w}"] = [ds, dg]
        assert ds < 1e-12 and dg == 0
    save_part("checks", res)


def part_2():
    out("\nPart 2. Where the echo goes. Calls-only law of cell P0 with kappa = 0 (no memory), 150 runs per entry, design")
    out("size. D1 = rho_1(mutual) - rho_1(one-way), D2 the same at lag 2, D3 at lag 3; G d = the synchronous D2 (B = 1).")
    out("Predicted: D1 = w1 G d, D2 = w2 G d to leading order. Entries: mean (s.e.).")
    L = replace(sc.cell_laws()["M1"], kappa=0.0)
    res, Gd = {}, None
    for B in (1, 2, 4, 8, 16, 32, N):
        m, o = runs(L, "mutual", 150, B), runs(L, "oneway", 150, B)
        D = {k: (m[k].mean() - o[k].mean(), float(np.hypot(sc.ms(m[k])[1], sc.ms(o[k])[1]))) for k in ("rho1", "rho2", "rho3")}
        Gd = D["rho2"][0] if B == 1 else Gd
        w1, w2 = weights_w(B)
        res[str(B)] = dict(D1=D["rho1"], D2=D["rho2"], D3=D["rho3"], w1=w1, w2=w2, Gd=Gd)
        out(f"  B = {B:3d}: D1 {D['rho1'][0]:+.4f} ({D['rho1'][1]:.4f})  w1 G d {w1 * Gd:+.4f}   D2 {D['rho2'][0]:+.4f} ({D['rho2'][1]:.4f})  "
            f"w2 G d {w2 * Gd:+.4f}   D3 {D['rho3'][0]:+.4f} ({D['rho3'][1]:.4f})   w1 {w1:.3f} w2 {w2:.3f}")
    save_part("weights", res)


STATS = ("rho1", "rho2", "rho3", "X2", "r12", "m", "q")


def part_3():
    G = 300
    out(f"\nPart 3. Predictions, cell P0, design size (N = {N}, K = {K}, T = {T}, R = {R}, rounds {BURN + 1}..{T}), {G} runs per entry,")
    out("on the graphs, notes, news and uniforms of the synchronous predictions. Entries: mean (s.e. of the mean)")
    out("[run-to-run s.d.]. B = 16 is the amendment's schedule; B = 256 is one analyst at a time; B = 1 synchronous.")
    res = {}
    for B in (B_DESIGN, N, 1):
        for name, L in sc.cell_laws().items():
            for w in W3:
                r = runs(L, w, G, B)
                rec = {k: sc.ms(r[k]) for k in STATS}
                res[f"B{B}|{name}|{w}"] = rec
                out(f"  B = {B:3d} {name} {w:13s} " + "  ".join(f"{k} {rec[k][0]:+.4f} ({rec[k][1]:.4f}) [{rec[k][2]:.4f}]" for k in STATS))
    out("\n  Contrasts with the one-way world under the same schedule: D1 = rho_1 - rho_1(one-way), D2 = rho_2 - rho_2(one-way),")
    out("  DX2 the same for X2 = rho_2 - rho_1^2. Mean (s.e.); [s.d. of the difference of one run from each world].")
    for B in (B_DESIGN, N, 1):
        for name in ("M1", "M2"):
            for w in ("mutual", "antagonistic"):
                row = []
                for lab, k in (("D1", "rho1"), ("D2", "rho2"), ("DX2", "X2"), ("D1-D2", "r12")):
                    a, b = res[f"B{B}|{name}|{w}"][k], res[f"B{B}|{name}|oneway"][k]
                    v = (a[0] - b[0], float(np.hypot(a[1], b[1])), float(np.hypot(a[2], b[2])))
                    res[f"B{B}|{name}|{w}|{lab}"] = v
                    row.append(f"{lab} {v[0]:+.4f} ({v[1]:.4f}) [{v[2]:.4f}]")
                out(f"  B = {B:3d} {name} {w:13s} " + "  ".join(row))
    out(f"\n  Replicas against graphs, B = {B_DESIGN}, law M1: run-to-run s.d. with 4 replicas per graph (150 runs).")
    for w in ("mutual", "oneway"):
        r = runs(L := sc.cell_laws()["M1"], w, 150, B_DESIGN, tag=4, R_=4)
        res[f"R4|B{B_DESIGN}|M1|{w}"] = {k: sc.ms(r[k]) for k in ("rho1", "rho2", "r12")}
        out(f"  {w:13s} rho1 [{sc.ms(r['rho1'])[2]:.4f}]  rho2 [{sc.ms(r['rho2'])[2]:.4f}]  rho1 - rho2 [{sc.ms(r['r12'])[2]:.4f}]"
            f"   (2 replicas: [{res[f'B{B_DESIGN}|M1|{w}']['rho1'][2]:.4f}], [{res[f'B{B_DESIGN}|M1|{w}']['rho2'][2]:.4f}], [{res[f'B{B_DESIGN}|M1|{w}']['r12'][2]:.4f}])")
    save_part("predictions", res)


def part_4():
    cal = sc.calib()["cells"][sc.CELL]
    nd, G = 200, 32
    out(f"\nPart 4. Calibration s.e. of the contrasts: {nd} bootstrap draws of the cell-P0 fit, each on the same {G} graphs with the")
    out(f"same random numbers and block orders, B = {B_DESIGN} and B = 1. ratio = D2(B = {B_DESIGN}) / D2(B = 1) within a draw.")
    out("Entries: value at the point estimate on these graphs, s.d. over draws, 2.5% and 97.5% of draws.")
    res = {}
    for name, mk in (("M1", sc.law_M1), ("M2", sc.law_M2)):
        def contrasts(p):
            L = mk(p, cal["chat"])
            v = {}
            for B in (B_DESIGN, 1):
                o = runs(L, "oneway", G, B)
                for w in ("mutual", "antagonistic"):
                    r = runs(L, w, G, B)
                    v[f"B{B}|{w}|D1"] = r["rho1"].mean() - o["rho1"].mean()
                    v[f"B{B}|{w}|D2"] = r["rho2"].mean() - o["rho2"].mean()
                    v[f"B{B}|{w}|DX2"] = r["X2"].mean() - o["X2"].mean()
            for w in ("mutual", "antagonistic"):
                v[f"{w}|ratio"] = v[f"B{B_DESIGN}|{w}|D2"] / v[f"B1|{w}|D2"]
            return v
        est = contrasts(cal[name]["est"])
        dr = [contrasts(d) for d in cal[name]["draws"][:nd]]
        for k in est:
            x = np.array([d[k] for d in dr])
            res[f"{name}|{k}"] = dict(est=float(est[k]), sd=float(x.std(ddof=1)), lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)))
            if not k.startswith("B1|") and "antagonistic|ratio" not in k:
                out(f"  {name} {k:26s} {est[k]:+.4f}  sd {res[f'{name}|{k}']['sd']:.4f}  [{res[f'{name}|{k}']['lo']:+.4f}, {res[f'{name}|{k}']['hi']:+.4f}]")
    save_part("calibration", res)


def part_5():
    d = load_part()
    P = d["predictions"]
    Ssync = json.load(open(HERE / "analyst_signrule_part_S.json"))
    out("\nPart 5. Sizing of falsifier 7. Statistic: T7 = D2_sync / 2 - D2_async, D2 = rho_2(mutual) - rho_2(one-way), the")
    out("synchronous one from the experiment's own synchronous runs (2 mutual, 5 one-way; X2 and rho_2 contrasts coincide")
    out(f"there). The falsifier fires if T7 < 0; the prediction is confirmed at 3 s.e. if T7 > 3 s.e.(T7). B = {B_DESIGN}.")
    out("s.e.(T7)^2 = [sd_m,sync^2 / 2 + sd_o,sync^2 / 5] / 4 + [sd_m,async^2 + sd_o,async^2] / n, n asynchronous runs per world.")
    res = {}
    for name in ("M1", "M2"):
        sm, so = Ssync[f"{name}|mutual|R2"], Ssync[f"{name}|oneway|R2"]
        D2s = sm["rho2"][0] - so["rho2"][0]
        var_s = (sm["rho2"][1] ** 2 / 2 + so["rho2"][1] ** 2 / 5) / 4
        am, ao = P[f"B{B_DESIGN}|{name}|mutual"], P[f"B{B_DESIGN}|{name}|oneway"]
        D2a, D1a = am["rho2"][0] - ao["rho2"][0], am["rho1"][0] - ao["rho1"][0]
        T7 = D2s / 2 - D2a
        va2, va1 = am["rho2"][2] ** 2 + ao["rho2"][2] ** 2, am["rho1"][2] ** 2 + ao["rho1"][2] ** 2
        need = va2 / ((T7 / 3) ** 2 - var_s) if (T7 / 3) ** 2 > var_s else float("inf")
        rec = dict(D2_sync=D2s, D2_async=D2a, D1_async=D1a, T7=T7, sd_async_rho2=[am["rho2"][2], ao["rho2"][2]],
                   sd_async_rho1=[am["rho1"][2], ao["rho1"][2]], n_for_3se=need, by_n={})
        out(f"  {name}: D2_sync {D2s:+.4f}, D2_async {D2a:+.4f} (ratio {D2a / D2s:.2f}), T7 {T7:+.4f}; run s.d. of rho_2: async mutual {am['rho2'][2]:.4f}, "
            f"async one-way {ao['rho2'][2]:.4f}; of rho_1: {am['rho1'][2]:.4f}, {ao['rho1'][2]:.4f}.  n for 3 s.e.: {need:.1f}")
        for n in (1, 2, 3, 4):
            z7 = T7 / np.sqrt(var_s + va2 / n)
            z1 = D1a / np.sqrt(va1 / n)
            zfire = (D2s / 2) / np.sqrt(var_s + va2 / n)       # how far D2_async = D2_sync (echo did not move at all) is from the threshold
            rec["by_n"][n] = dict(T7_in_se=float(z7), D1_in_se=float(z1), se_T7=float(np.sqrt(var_s + va2 / n)), se_D1=float(np.sqrt(va1 / n)))
            out(f"     n = {n} per world: T7 = {z7:.1f} s.e. (s.e. {np.sqrt(var_s + va2 / n):.4f});  lag-1 contrast D1 = {D1a:+.4f} = {z1:.1f} s.e.;"
                f"  an echo that did not move would sit {zfire:.1f} s.e. on the other side")
        out("     runs per world for T7 = 3 s.e. if the measured synchronous contrast is D2_sync and the asynchronous one is the")
        out(f"     predicted fraction {D2a / D2s:.2f} of it (same noise):  " + "   ".join(
            f"D2_sync {x:.3f}: " + (f"{va2 / ((x * (0.5 - D2a / D2s) / 3) ** 2 - var_s):.1f}" if (x * (0.5 - D2a / D2s) / 3) ** 2 > var_s else "not reachable")
            for x in (0.030, 0.040, 0.050, 0.0575, 0.070)))
        rec["n_for_3se_by_D2_sync"] = {f"{x:.4f}": (float(va2 / ((x * (0.5 - D2a / D2s) / 3) ** 2 - var_s)) if (x * (0.5 - D2a / D2s) / 3) ** 2 > var_s else None)
                                       for x in (0.030, 0.040, 0.050, 0.0575, 0.070)}
        d12 = P[f"B{B_DESIGN}|{name}|mutual|D1-D2"]
        rec["D1_minus_D2"] = dict(value=d12[0], sd_one_run_pair=d12[2], in_se={n: float(d12[0] / (d12[2] / np.sqrt(n))) for n in (1, 2, 3, 4)})
        out(f"     within the asynchronous runs alone: D1 - D2 = {d12[0]:+.4f} (synchronous: {P[f'B1|{name}|mutual|D1-D2'][0]:+.4f}), s.d. of one run pair {d12[2]:.4f}: "
            + ", ".join(f"{d12[0] / (d12[2] / np.sqrt(n)):.1f} s.e. at n = {n}" for n in (1, 2, 3, 4)))
        for w in ("antagonistic",):
            aa = P[f"B{B_DESIGN}|{name}|{w}"]
            D1 = aa["rho1"][0] - ao["rho1"][0]
            sd = np.hypot(aa["rho1"][2], ao["rho1"][2])
            rec["antagonistic_D1"] = D1
            rec["antagonistic_D1_in_se_n2"] = float(D1 / (sd / np.sqrt(2)))
            out(f"     asynchronous antagonistic: D1 = {D1:+.4f}, {D1 / (sd / np.sqrt(2)):.1f} s.e. with 2 runs per world, {D1 / (sd / np.sqrt(5)):.1f} with 5")
        res[name] = rec
    save_part("sizing", res)
    d = load_part()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    w1, w2 = weights_w(B_DESIGN)
    C = d["calibration"]
    pred = {}
    for name in ("M1", "M2"):
        pred[name] = {}
        for w in W3:
            pred[name][w] = {k: dict(value=P[f"B{B_DESIGN}|{name}|{w}"][k][0], mc_se=P[f"B{B_DESIGN}|{name}|{w}"][k][1],
                                     run_sd=P[f"B{B_DESIGN}|{name}|{w}"][k][2]) for k in STATS}
        for w in ("mutual", "antagonistic"):
            for lab in ("D1", "D2", "DX2"):
                v = P[f"B{B_DESIGN}|{name}|{w}|{lab}"]  # noqa: E501
                pred[name][w][lab] = dict(value=v[0], mc_se=v[1], sd_one_run_pair=v[2], calib_sd=C[f"{name}|B{B_DESIGN}|{w}|{lab}"]["sd"])
        pred[name]["mutual"]["D2_ratio_async_over_sync"] = C[f"{name}|mutual|ratio"]
    frozen = dict(
        note="theory/analyst_signrule.md, amendment (asynchronous control)",
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), parent_commit=commit,
        code="theory/checks/analyst_signrule_async_check.py",
        status="frozen before any asynchronous analyst swarm data exist; synchronous runs of experiments/09-analyst-signrule "
               "were in progress and were not read",
        schedule=dict(B=B_DESIGN, block_size=N // B_DESIGN, p_same_block=(N / B_DESIGN - 1) / (N - 1), w1=w1, w2=w2,
                      rule="round 0 synchronous; each round t >= 1: fresh random partition into B blocks shared by the replicas "
                           "of a graph; blocks update in sequence; an analyst reads each colleague's latest call and note "
                           "(round t if the colleague's block came earlier, else round t-1), its own round-(t-1) call and note, "
                           "news xi_i(t), and its call is drawn from tanh(x - chat) with the stored uniform; one update per "
                           "analyst per round"),
        design=dict(N=N, K=K, T=T, R=R, burn=BURN, cell=sc.CELL, chat=sc.calib()["cells"][sc.CELL]["chat"],
                    statistics="echo function on the sampling belief s = tanh(x - chat), rounds 5..20, lags in rounds"),
        predictions=pred, sizing=d["sizing"],
        falsifier_7="fires if D2_async > D2_sync / 2 with T7 = D2_sync / 2 - D2_async below 0 by more than 3 s.e.; "
                    "D2 = rho_2(mutual) - rho_2(one-way); positive check D1_async = rho_1(mutual) - rho_1(one-way) > 0 at 3 s.e.",
        parts=d)
    json.dump(frozen, open(HERE / "analyst_signrule_async_predictions.json", "w"), indent=1, default=float)
    print("wrote", HERE / "analyst_signrule_async_predictions.json")


if __name__ == "__main__":
    fns = {"1": part_1, "2": part_2, "3": part_3, "4": part_4, "5": part_5}
    todo = sys.argv[1:] or list(fns)
    for p in todo:
        t = time.time()
        fns[p]()
        print(f"[part {p}: {time.time() - t:.0f} s]", flush=True)
    name = "analyst_signrule_async_out.txt" if todo == list(fns) else f"analyst_signrule_async_out_{''.join(todo)}.txt"
    with open(HERE / name, "w") as f:
        f.write("\n".join(_lines) + "\n")
