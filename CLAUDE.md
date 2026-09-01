# CLAUDE.md — rules for the agent

Project: blind audit of J-Lens / R-lens / logit lens on Qwen3.6-27B.
Question: does a lens's signal in a blind scan survive when the text-reconstruction channel is neutralized?
Hypotheses to discriminate: H1 internal readout / H2 text inversion / H3 judge apophenia.

## Non-negotiable rules
- The model and the lenses are loaded ONCE in the "SETUP" cell of the persistent kernel (notebooks/main.ipynb).
  Never reload, never restart the kernel without asking me.
- Every experiment writes its results to results/<name>.json and its figures to figs/<name>.png. Never a result that is only displayed.
- Before concluding that an experiment "works", show me 3 raw examples.
- The design of the conditions is fixed in src/conditions.py; do not add a condition without asking me.
- Never touch data/pairs.jsonl after human validation (human_checked=true).
- Lens format: follow lenses/README.md (the README of the HF repo camilablank/workspace-lenses) to the letter.
  The points marked `# ADAPTER` in src/lens.py must be resolved by reading that README, not by guessing. If ambiguous: ask.
- At the end of every session: add an entry to experiments.md (done / verified / doubt / next step).
- The judge prompt must NEVER name an anomaly family or an instrument.
- Before any scan-only condition run: execute `python -m src.checks leak` and show me the result.

## What I do myself (do not delegate)
- Reading the 40 pairs; reading the 30 scans; independent recomputation of the AUCs; pivot decisions; writing the doc and the exec summary.
