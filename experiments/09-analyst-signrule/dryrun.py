"""CPU dry run of the 09 pipeline: run.py base + kicks for all 12 runs, then analyze.py, with a mock backend obeying the
frozen M1 law including its residual (a deterministic function of the prompt or of (analyst, session, round)).
Usage: uv run python experiments/09-analyst-signrule/dryrun.py OUTDIR prompt|white"""
import sys, zlib, importlib.util, time
from pathlib import Path
sys.path.insert(0, "/workspace/swarm-cavity")
import numpy as np
import envs.analyst as A
tmp = Path(sys.argv[1]); tmp.mkdir(exist_ok=True); MODE = sys.argv[2]   # MODE: 'prompt' (residual = f(whole prompt)) or 'white' (f(analyst, session, round))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
R = load("run09", "/workspace/swarm-cavity/experiments/09-analyst-signrule/run.py")
L = R.PRED["laws"]["M1"]
class LawMock(A.MockAnalyst):
    def __init__(self):
        super().__init__(L["uA"], -L["uD"], L["nu"], L["a0"], L["c"], L["kappa"], K=16)
        self.eps, self.wording, self.variant = L["eps"], "max", A.PromptVariant()
    def beliefs(self, ctxs, key=None):
        out = super().beliefs(ctxs, key)
        for c, b in zip(ctxs, out):
            key = (c.agent_id, c.session, c.round_t, c.news_pct, c.own, c.allies, c.faded) if MODE == "prompt" else (c.agent_id, c.session, c.round_t)
            h = zlib.crc32(repr(key).encode())
            x = b["x"] + self.eps * np.random.default_rng(h).standard_normal()
            b.update(x=x, p_buy=(1 + np.tanh(x)) / 2)
        return out
R.RESULTS = tmp / "results"
R.backend = lambda llm, name: LawMock()
t = time.time()
R.step_base(None, R.RUNS)
print("base", time.time() - t)
R.step_kicks(None, R.RUNS)
R.step_abase(None, R.ASYNC_RUNS)                                  # asynchronous control (Amendment 1)
print("kicks", time.time() - t)
AN = load("an09", "/workspace/swarm-cavity/experiments/09-analyst-signrule/analyze.py")
AN.RES = tmp / "results"; AN.OUT = tmp / "an"
AN.main()
