"""Binary coin-toss task with known ground truth and an exactly known Bayesian evidence field per agent.

Story shown to the agents (envs/prompts.py): a fair coin was tossed once and nobody saw it land. Each agent
holds k sensor reports; sensor j says HEADS or TAILS and is right with a stated probability rho_j,
independently of the other sensors. The claim under discussion is "the coin landed HEADS".

Field. With a 50/50 prior, a Bayesian who believes the stated reliabilities has log-odds of HEADS

    h_priv = sum_j eps_j * logit(rho_j),     eps_j = +1 for a HEADS report, -1 for TAILS,      (1)

which is the quenched private field of theory/claim1.md eq. (1), in the sign convention s = +1 <-> HEADS
(s = 2p - 1 with p the stated probability of HEADS). The game draws a target h ~ N(truth * mu, sigma^2)
per agent and writes k reports whose displayed reliabilities (whole percent) realize it; h_priv is then
recomputed from the displayed numbers, so (1) holds exactly for the text the agent reads. Rounding moves
h_priv away from the target by at most k * 0.02 or so; both are stored.

Calibration. Over the population, P(HEADS | h) = sigmoid(2 mu h / sigma^2) for h ~ N(+-mu, sigma^2), which
equals the Bayesian sigmoid(h) exactly when sigma^2 = 2 mu (the Nishimori condition). At that point the
stated reliabilities are honest at the ensemble level; elsewhere they are nominal. Individual reports are
constructed to hit the target, not sampled from the coin, so per-sensor honesty is not claimed.

Everything is a deterministic function of the seed: make_game(..., seed) twice gives identical text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_MIN, RHO_MAX = 0.51, 0.95  # displayed reliability range; 50% would be an uninformative sensor


def logit(p):
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class Clue:
    side: str          # "HEADS" or "TAILS"
    reliability: int   # displayed percent, e.g. 68

    @property
    def log_odds(self) -> float:
        """This report's contribution to eq. (1)."""
        return (1.0 if self.side == "HEADS" else -1.0) * float(logit(self.reliability / 100))


@dataclass(frozen=True)
class Evidence:
    clues: tuple[Clue, ...]
    h_target: float

    @property
    def h_priv(self) -> float:
        """Exact Bayesian log-odds of HEADS of the displayed reports, eq. (1)."""
        return float(sum(c.log_odds for c in self.clues))

    def text(self) -> str:
        lines = [f"- Sensor {j + 1} (right {c.reliability}% of the time) reports {c.side}."
                 for j, c in enumerate(self.clues)]
        return "Your sensor reports (they stay the same for the whole game):\n" + "\n".join(lines)


def evidence_for_field(h: float, k: int, rng: np.random.Generator, spread: float = 0.5) -> Evidence:
    """k reports whose log-odds sum to h, up to rounding of the displayed percentages.

    Split h into k parts l_j = h/k + d_j with zero-sum perturbations d_j ~ N(0, spread^2) (so weak evidence
    can contain reports on both sides), then show side = sign(l_j) and reliability = sigmoid(|l_j|) in whole
    percent, clipped to [51, 95]. Clipping or rounding changes the realized h_priv, which is what the
    agent's evidence actually implies and what every analysis uses.
    """
    d = spread * rng.standard_normal(k)
    l = h / k + d - d.mean()
    clues = []
    for x in l:
        rho = 1.0 / (1.0 + np.exp(-abs(x)))
        pct = int(np.clip(round(100 * rho), 100 * RHO_MIN, 100 * RHO_MAX))
        clues.append(Clue("HEADS" if x >= 0 else "TAILS", pct))
    return Evidence(tuple(clues), float(h))


@dataclass(frozen=True)
class Game:
    truth: int                 # +1: the coin landed HEADS (claim true), -1: TAILS
    mu: float                  # evidence quality: E[h_priv] = truth * mu
    sigma: float
    k: int
    seed: int
    evidence: tuple[Evidence, ...]

    @property
    def N(self) -> int:
        return len(self.evidence)

    @property
    def h_priv(self) -> np.ndarray:
        """(N,) realized fields, eq. (1)."""
        return np.array([e.h_priv for e in self.evidence])

    @property
    def h_target(self) -> np.ndarray:
        return np.array([e.h_target for e in self.evidence])


def make_game(N: int, mu: float = 0.5, sigma: float = 1.0, k: int = 3, seed: int = 0,
              truth: int | None = None) -> Game:
    """N agents' evidence for one coin toss. truth=None draws it from the seed.

    Targets h_i ~ N(truth * mu, sigma^2), i.i.d. across agents, then rendered by ``evidence_for_field``.
    sigma = sqrt(2 mu) makes the stated reliabilities calibrated over the population (module docstring).
    """
    rng = np.random.default_rng([seed, 0x5EED])
    if truth is None:
        truth = int(rng.choice([-1, 1]))
    if truth not in (-1, 1):
        raise ValueError("truth must be +1 or -1")
    targets = truth * mu + sigma * rng.standard_normal(N)
    ev = tuple(evidence_for_field(h, k, rng) for h in targets)
    return Game(truth, mu, sigma, k, seed, ev)
