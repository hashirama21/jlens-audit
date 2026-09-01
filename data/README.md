# data/

## pairs.jsonl — schema
```json
{"id": "inj_03", "family": "injection|bug|false_premise|conflict",
 "anomalous": "<text A>", "clean": "<text B, twin>",
 "anomaly_char_span": [412, 468], "anomaly_token_span": [95, 108],
 "diff_tokens": 9, "human_checked": true, "notes": "..."}
```
- `anomaly_token_span`: positions (in the Qwen tokenization) where the anomaly lives — used by checks.evidence and upper_bound. To be computed with `python -m src.gen_pairs span`.
- `human_checked`: set to true ONLY after human reading (side by side, highlighted diff). Reject if the anomaly is lexically blatant or undetectable for a human.

## multihop.jsonl (validation, step 3)
```json
{"prompt": "The capital of the country where sushi originated is", "pivot": "sushi", "intermediate": "Japan", "answer": "Tokyo"}
```
~40 two-hop questions, filtered: the model must answer correctly (`answer`).

## Generation constraints (reminder, see docs/runbook §4)
- Twin within < 10% token difference, same length ±5%, same register.
- Anomaly position varies from one pair to another.
- Injection: natural instruction, WITHOUT keywords (ignore / instructions / system / all-caps).
- Bug: a single line differs, silent semantic error.
- False premise: a single fact differs, plausible.
- Conflict: discreet incompatibility between system instruction and request.
