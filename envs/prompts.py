"""The one prompt module shared by the swarm (envs/swarm.py) and the single-agent probes (cavity/probes.py),
so the probes measure exactly the agent the swarm runs.

Layout (engine/agent.py renders system[0], system[1] as the cached prefix, then the user turn):

    system[0]  rules(H)              identical for every agent and round of an experiment
    system[1]  this agent's evidence fixed for the game (envs/evidence_game.Evidence.text)
    user       dynamic(ctx, H)       session/round tag, the agent's own last H answers, this round's inbox

Option B of theory/claim1.md section 5: the inbox has two labeled groups, "sources you tend to agree with"
(J_ij = +1) and "sources you tend to disagree with" (J_ij = -1), wording adapted from
third_party/physics-of-agents/lib/datagen/samplers.py (Apache-2.0), which also shuffles each inbox. Each
message shows the sender's stated probability of HEADS and its short public message. Peers are not
named, so an agent cannot track individual senders across rounds.

Output fields are engine/agent.py's (rationale, public_message, calibrated_belief_prob). CLAUDE.md
guardrail 4: no prompt text calls any field or text private, hidden or secret (that wording drew
reasoning_extraction refusals); tests/test_envs.py checks the rendered prompts for those words.

The session tag ("Discussion 3, round 2.") makes requests of different replicas or probe samples
distinct. engine.Runner replays identical params for free, so without it every replica of a swarm would
receive the same cached reply and the replicas would never diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.agent import AgentRequest

FORBIDDEN_WORDS = ("private", "hidden", "secret")  # guardrail 4; checked in tests


def rules(H: int) -> str:
    """Shared rules. H = number of the agent's own past answers shown (0 = memoryless)."""
    memory = ("- You also see your own answers from recent rounds.\n" if H > 0 else "")
    return (
        "You are taking part in a group estimation game with several other participants.\n\n"
        "The setup\n"
        "- A fair coin was tossed once. Nobody in the group saw how it landed. The question for everyone is "
        "whether the coin landed HEADS.\n"
        "- Every participant has their own set of sensor reports about the toss. Each sensor says HEADS or "
        "TAILS and is right with the probability shown next to it. Different sensors, including other "
        "participants' sensors, are independent of each other.\n"
        "- Your sensor reports are listed below. They stay the same for the whole game.\n\n"
        "How a round works\n"
        "- The game runs over several rounds. In each round you give your current probability that the coin "
        "landed HEADS, a short message to the group, and a short rationale.\n"
        "- From the second round on, you see the messages some other participants sent in the previous "
        "round. Each message shows the probability of HEADS that participant stated and the short message "
        "they wrote.\n"
        "- Messages arrive in two groups: messages from sources you tend to agree with, and messages from "
        "sources you tend to disagree with. Which sources fall in which group stays the same for the whole "
        "game. A group can be empty.\n"
        f"{memory}\n"
        "How to answer\n"
        "- calibrated_belief_prob: your probability, between 0 and 1, that the coin landed HEADS. Use 0.5 if "
        "you think both sides are equally likely. Avoid exactly 0 and 1, since nothing in this game makes "
        "either side certain.\n"
        "- public_message: one or two sentences, at most 30 words, that other participants will read next "
        "round. Say what you currently think and why. Do not invent sensor reports you were not given.\n"
        "- rationale: one or two sentences on why you hold your current probability.\n"
        "- Combine your own sensor reports with the messages you receive in whatever way you judge best. "
        "You may change your probability from round to round.\n"
        "- Reply only with the requested fields."
    )


@dataclass(frozen=True)
class InboxMsg:
    prob: float              # sender's stated probability of HEADS
    text: str                # sender's public message
    sender: int | None = None  # for bookkeeping only; never shown


@dataclass(frozen=True)
class AgentContext:
    """Everything one agent update depends on. The LLM backend renders it with this module; the mock
    backend (envs/swarm.MockBackend) reads the same numbers: h_priv, own history, and the inbox."""
    agent_id: int | str
    session: int | str       # replica / probe-sample tag shown to the model (see module docstring)
    round_t: int             # 0 = first round, no inbox
    h_priv: float            # evidence log-odds (envs/evidence_game eq. (1)); not shown as a number
    evidence: str            # evidence text, system[1]
    history: tuple[tuple[float, str], ...] = ()   # own past (prob, public_message), oldest first
    agree: tuple[InboxMsg, ...] = ()
    disagree: tuple[InboxMsg, ...] = ()
    theta: float = 0.0       # extra numeric field; only the mock backend can receive it (white probes)
    meta: dict = field(default_factory=dict, compare=False, hash=False)


def _fmt_msgs(msgs) -> str:
    if not msgs:
        return "(none this round)"
    return "\n".join(f'- Stated probability of HEADS {m.prob:.2f}: "{m.text.strip()}"' for m in msgs)


def dynamic(ctx: AgentContext, H: int) -> str:
    """The uncached user turn for one agent and round."""
    parts = [f"Discussion {ctx.session}, round {ctx.round_t + 1}."]
    if H > 0:
        hist = ctx.history[-H:]
        if hist:
            first = ctx.round_t - len(hist) + 1
            lines = [f'- Round {first + k}: probability of HEADS {p:.2f}. Your message: "{m.strip()}"'
                     for k, (p, m) in enumerate(hist)]
            parts.append("Your answers in recent rounds (oldest first):\n" + "\n".join(lines))
        else:
            parts.append("You have no earlier answers yet.")
    if ctx.round_t == 0 and not ctx.agree and not ctx.disagree:
        parts.append("This is the first round. There are no messages yet.")
    else:
        parts.append("Messages from sources you tend to agree with:\n" + _fmt_msgs(ctx.agree))
        parts.append("Messages from sources you tend to disagree with:\n" + _fmt_msgs(ctx.disagree))
    parts.append(f"Give your answer for round {ctx.round_t + 1}.")
    return "\n\n".join(parts)


def make_request(ctx: AgentContext, H: int) -> AgentRequest:
    return AgentRequest(agent_id=ctx.agent_id, system_rules=rules(H), evidence=ctx.evidence,
                        dynamic=dynamic(ctx, H))
