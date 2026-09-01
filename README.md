# jlens-audit

Blind audit of activation-readout lenses (J-Lens / R-lens / logit lens) on Qwen3.6-27B:
does a scan (all positions × layers) handed to an LLM judge detect anomalies in the text the model reads,
and does the signal survive the controls for text inversion (H2) and apophenia (H3)?

## Reproduction (10 lines)
```bash
pip install -r env/requirements.txt
bash env/download.sh                      # model + lenses (Qwen3.6-27B, camilablank/workspace-lenses)
export OPENROUTER_API_KEY=...
python -m src.validate --smoke            # step 2: 'sushi -> Japan' conformity (after resolving the # ADAPTER points)
python -m src.validate                    # step 3: multihop pass@10 (data/multihop.jsonl), J vs R -> results/, figs/
python -m src.gen_pairs generate          # step 4a: corpus (generator model != judges)
python -m src.gen_pairs review            # step 4b: human validation (human_checked) — NOT delegable
python -m src.gen_pairs span              # step 4c: anomaly_token_span (Qwen tokenizer)
python -m src.capability                  # step 4d: does the model see the anomalies in the clear?
python -m src.scan                        # step 5: scans of the 80 texts x 3 instruments -> scans/
python -m src.checks leak                 # step 6: leak audit (mandatory before the conditions)
python -m src.conditions                  # step 7: the 7 conditions x judges -> judge/outputs/ (resumable)
python -m src.metrics                     # step 9: AUC/CI, triplet, figures -> results/, figs/
python -m src.checks sample --n 30        # step 8: draw the 30 scans to read by hand
```
Local tests (no GPU): `python -m pytest` — pure functions + dry-run of steps 6-9 with a mocked judge.
See docs/ for the full plan and runbook.

## Status of the `# ADAPTER` points
See src/lens.py — the exact lens format (one matrix per layer? bias? norm?) must be read in lenses/README.md.
The conformity test `python -m src.validate --smoke` ("sushi -> Japan") must pass before anything else.
