"""Analyst calibration 2 (EXPERIMENT.md in this folder). GPU side: single-agent decisions only (no swarm, no notes
generated), all at K = 16. Raw records go to results/analyst-calib2-20261125/; fits in analyze.py (also used here
for the pre-registered selection of the combined cell).

Base probes are those of experiments/08-analyst-calibration/run.py (functions imported, file unchanged), built from
the same pilot note pool (exported to note_pool.json here, same order), so phase-A cell P0 repeats 08's max/real
probes exactly and checks reproducibility. Wording "max" (08's strongest ally gain) everywhere.

  tok      every variant: decision words single tokens after its prefix; one rendered prompt per new variant
  phaseA   K = 16, 1,000 base probes per cell:
             P0 default (question "close up", BUY listed first, BUY/SELL tokens, ordinary fading)
             P1 SELL listed first            P2 question "close down"        P3 "close down", SELL first
             P4 YES/NO to "close up" (YES = BUY)                             P5 YES/NO to "close down" (YES = SELL)
             F1 faded calls shown reversed by the desk                        F2 reversed + "as informative as an ally's"
             F3 = F2 without the faded colleagues' notes
             T0 = P0 with template notes for colleagues and for the analyst's own previous note (no pilot text)
  select   pre-registered (M1 point estimates; cells with >= 5% missing or with abar < 0.8 abar(P0) excluded):
           polarity* = argmin |c| over P0-P5; fade* = argmax a_-/a_+ over P0, F1, F2, F3;
           best = polarity* with fade*; written to selection.json
  best     the best variant at K = 16, unless it is already a phase-A cell

Run on the worker:  uv run --no-sync python experiments/08b-analyst-calibration-2/run.py tok phaseA select best
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402

import envs.analyst as A  # noqa: E402
from engine.ledger import append_jsonl, read_jsonl  # noqa: E402


def load(name: str, path: Path):
    """Import a module by path (08's and this folder's scripts share file names)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R08 = load("run08", ROOT / "experiments" / "08-analyst-calibration" / "run.py")

TAG = "analyst-calib2-20261125"
OUT = ROOT / "results" / TAG
WORDING = "max"
N_BASE = 1000
BATCH = 512
V = A.PromptVariant
CELLS_A = {"P0": V(), "P1": V(order="sell_first"), "P2": V(question="down"),
           "P3": V(question="down", order="sell_first"), "P4": V(tokens="yesno"),
           "P5": V(tokens="yesno", question="down"), "F1": V(fade="reversed"), "F2": V(fade="reversed_equal"),
           "F3": V(fade="reversed_equal_nonote"), "T0": V()}
TEMPLATE_CELLS = {"T0"}
POLARITY, FADES = ("P0", "P1", "P2", "P3", "P4", "P5"), ("P0", "F1", "F2", "F3")
PROGRESS = os.environ.get("ORCH_PROGRESS")


def progress(msg: str):
    print(msg, flush=True)
    if PROGRESS:
        Path(PROGRESS).write_text(msg + "\n")


def note_pool() -> dict:
    """08's run.note_pool() rebuilt from note_pool.json (same rows, same order, same terciles)."""
    rows = [tuple(r) for r in json.loads((HERE / "note_pool.json").read_text())["rows"]]
    pool = {}
    for c in (1, -1):
        rc = sorted([r for r in rows if r[0] == c], key=lambda r: c * r[2])
        n3 = max(len(rc) // 3, 1)
        pool[c] = {"hes": rc[:n3], "conf": rc[-n3:], "all": rc}
    return pool


def probes_at(pool, K: int, n: int):
    """08's base probes and design rows with K colleague slots (08's module constant K patched for the call)."""
    k0 = R08.K
    R08.K = K
    try:
        bps = R08.base_probes(pool, n)
        rows = [R08.design_row(bp) for bp in bps]
    finally:
        R08.K = k0
    return bps, rows


def context(bp, name: str):
    """08's probe context; for the template cells every note, the analyst's own included, is a template."""
    if name not in TEMPLATE_CELLS:
        return R08.probe_context(bp, "real")
    from dataclasses import replace
    ctx = R08.probe_context(bp, "template")
    own = A.CALLS[bp["own"]]
    return replace(ctx, own=(own, A.template_note(own, np.random.default_rng([R08.PROBE_SEED, bp["b"], 9]))))


def run_cell(llm, name: str, v: A.PromptVariant, K: int, bps, design):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"design_K{K}.json").write_text(json.dumps(design))
    path = OUT / f"{name}_K{K}.jsonl"
    done = {r["b"] for r in read_jsonl(path)}
    todo = [bp for bp in bps if bp["b"] not in done]
    be = A.AnalystVLLM(llm, WORDING, f"{TAG}-{name}-K{K}", K=K, logprobs=R08.LOGPROBS, variant=v)
    t0 = time.time()
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        for bp, b in zip(chunk, be.beliefs([context(bp, name) for bp in chunk])):
            append_jsonl(path, {"b": bp["b"], "cell": name, "K": K, "variant": asdict(v),
                                "status": "ok" if b["ok"] else "missing",
                                **{k: b[k] for k in ("p_buy", "x", "valid_mass", "censored", "lp_buy", "lp_sell",
                                                     "lp_min", "top_other", "n_prompt_tokens")}})
    return len(todo), time.time() - t0


def step_tok(llm):
    pool = note_pool()
    bps, _ = probes_at(pool, 16, 2)
    for name, v in CELLS_A.items():
        be = A.AnalystVLLM(llm, WORDING, "tok", logprobs=R08.LOGPROBS, variant=v)
        ids = be.decision_ids(context(bps[0], name))
        print(name, v, "up/down ids", be.buy_id, be.sell_id, "prompt tokens", len(ids), flush=True)
        if name in ("P3", "P5", "F3", "T0"):
            print(be.tok.decode(ids), flush=True)


def step_phaseA(llm):
    bps, design = probes_at(note_pool(), 16, N_BASE)
    for k, (name, v) in enumerate(CELLS_A.items()):
        n, dt = run_cell(llm, name, v, 16, bps, design)
        progress(f"phaseA {k + 1}/{len(CELLS_A)} {name}: {n} calls in {dt:.0f}s")


def step_select(llm=None):
    A2 = load("analyze08b", HERE / "analyze.py")
    fits = {name: A2.fit_cell(name, 16) for name in CELLS_A}
    abar = {nm: float(np.sqrt((f["a_plus"] ** 2 + f["a_minus"] ** 2) / 2)) for nm, f in fits.items()}
    miss = {name: sum(r["status"] != "ok" for r in A2.load_cell(name, 16)[1].values()) / N_BASE for name in CELLS_A}
    ok = [nm for nm in CELLS_A if miss[nm] < 0.05 and abar[nm] >= 0.8 * abar["P0"]]   # gains guard (review)
    pol = min([nm for nm in POLARITY if nm in ok], key=lambda nm: abs(fits[nm]["c"]))
    fade = max([nm for nm in FADES if nm in ok], key=lambda nm: fits[nm]["a_minus"] / fits[nm]["a_plus"])
    vp, vf = CELLS_A[pol], CELLS_A[fade]
    best = V(question=vp.question, order=vp.order, tokens=vp.tokens, fade=vf.fade)
    same = [nm for nm, v in CELLS_A.items() if v == best and nm not in TEMPLATE_CELLS]
    sel = {"polarity": pol, "fade": fade, "best": asdict(best), "best_is_cell": same[0] if same else None,
           "missing": miss, "abar": abar, "eligible": ok, "c": {nm: fits[nm]["c"] for nm in POLARITY},
           "r": {nm: fits[nm]["a_minus"] / fits[nm]["a_plus"] for nm in FADES}}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1), flush=True)


def step_best(llm):
    sel = json.loads((OUT / "selection.json").read_text())
    if sel["best_is_cell"]:
        progress(f"best variant is phase-A cell {sel['best_is_cell']}; not rerun")
        return
    bps, design = probes_at(note_pool(), 16, N_BASE)
    n, dt = run_cell(llm, "best", V(**sel["best"]), 16, bps, design)
    progress(f"best K=16: {n} calls in {dt:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["tok", "phaseA", "select", "best"])
    args = ap.parse_args()
    llm = R08.make_llm() if set(args.steps) - {"select"} else None
    for st in args.steps:
        t0 = time.time()
        {"tok": step_tok, "phaseA": step_phaseA, "select": step_select, "best": step_best}[st](llm)
        print(f"== step {st} done in {time.time() - t0:.0f}s", flush=True)
