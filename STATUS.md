# STATUS — what is in this zip, what cannot be, and why

| Item | State | Explanation |
|---|---|---|
| src/ (13 modules), prompts, docs, env | ✅ complete | syntax verified |
| data/pairs_pilot.jsonl | ✅ 8 pairs (2/family), hand-written | `human_checked=false`: to review (`python -m src.gen_pairs review`) — a Neel requirement, not an option |
| data/pairs.jsonl (40) | ⚙️ produced by `python -m src.gen_pairs generate` on the pod | the remaining 32 are generated via OpenRouter with the templates, pilots first; then full human review, `span`, `stats` |
| data/multihop.jsonl | ✅ 40 items | filter on the pod: keep those where the model answers `answer` correctly |
| notebooks/main.ipynb | ✅ created | SETUP / lenses / smoke / validation / capability / timing cells |
| experiments.md | ✅ session 0 filled in | the following sessions are to be written by you, at the end of each session |
| lenses/, Qwen3.6-27B model | ❌ impossible in a zip | 54 GB + ~10 GB, downloaded on the pod by `env/download.sh` (~10-20 min) |
| scans/, results/, judge/outputs/ | ❌ empty by nature | these are the OUTPUTS of the 20 hours; fabricating them would be exactly what Neel rejects. They fill up with `scan`, `conditions`, `metrics`, `checks` |
| src/lens.py `# ADAPTER` | ✅ resolved | format read in the HF README (one `lens.pt` per lens, `J`+`source_layers`); go/no-go `identity_check` + `orientation_check` |
| config.py JUDGE_A/JUDGE_B, GENERATOR | ✅ set | Anthropic/OpenAI judges (IDs verified on OpenRouter); 3rd-family generator `gemini-2.5-pro`; env-overridable |
| Input framing (F1) | ✅ chat template | `USE_CHAT_TEMPLATE=True`; scan/capability/span via `to_input_ids`; re-run `gen_pairs span` if the flag is flipped |

Order of operations on the pod: `env/download.sh` → read `lenses/README.md` → fix `src/lens.py` → `python -m src.validate --smoke` → notebook cells 3-6 → go/no-go → `gen_pairs generate/review/span/stats` → the clock starts.
