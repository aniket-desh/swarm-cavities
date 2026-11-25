# Refusal A/B on claude-haiku-5-5 (2026-11-25)

The engine's live check saw `stop_reason: "refusal"` with `stop_details.category: "reasoning_extraction"` on 4 of 8
Haiku 5.5 calls. `refusal_ab.py` (needs `live_check.py` beside it for the shared prompt) ran 12 realtime calls per arm
on the same game prompt, with commit c6e1f04's engine:

| arm | output fields and wording | refusals | beliefs (non-refused) |
|---|---|---|---|
| A | `private_reasoning` ("never shown to peers") + rule "Your private reasoning is never shown to anyone" | 6/12 | 0.60–0.65 |
| B | `rationale` ("why you hold your belief"), no private/hidden wording | 0/12 | 0.60–0.67 |
| C | no reasoning field | 0/12 | 0.60–0.67 |

Total cost $0.0080 (ledger experiment_id `refusal-ab-20261125-042342`). Beliefs are the same across arms, so the
rename doesn't change the measured belief. The engine now uses `rationale`. Describing any output as private or
hidden reasoning reads as a chain-of-thought extraction attempt to the safety classifier.
