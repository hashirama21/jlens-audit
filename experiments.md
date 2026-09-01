# Experiment log (append-only)

Format per entry:
## YYYY-MM-DD HH:MM — <short title>  [counted: yes/no, duration]
- Done:
- Verified:
- What I believe now:
- Doubt / what could be wrong:
- Next step:

---
## 2026-08-15 — Session 0: framing and repo skeleton  [counted: NO — general preparation]
- Done: chose topic A; read the application doc, J-Lens paper (to finish), Neel's review, R-lens post (Aug 5); designed the 7 conditions BEFORE any code; full repo skeleton (src/, prompts, checks, metrics); 8 pilot pairs written; 40 multihop questions for validation.
- Verified: the R-lens post contains no blind-scan audit, no FPR, no inversion control → open niche. R-lens has no advantage on small models → Qwen3.6-27B required → 80 GB GPU. One pilot pair (bug_02) was identical across A/B, caught by an automated check and fixed.
- What I believe now: the most likely result is mixed by family; the value of the project lies in the controls (reconstruction, permutation), not the raw AUC.
- Doubt / what could be wrong: the lens format (src/lens.py # ADAPTER) — to be resolved by reading the HF README before anything else; the judge may detect the family rather than the anomaly (family check); the judge prompts are in French and the content in English — to harmonize into English if it bothers the judge.
- Next step: 80 GB pod + download.sh; read lenses/README.md; resolve # ADAPTER; smoke test; then step 3.
---
## 2026-08-18 00:30 — Code audit + rework (bugs, DRY, English)  [counted: NO — pipeline hardening, no research result]
- Done:
  - Full audit of the 13 `src/` modules + prompts + env.
  - New `src/store.py`: single IO layer (pairs / scans / verdicts) → DRY.
  - 5 critical bugs fixed: (B1) `gen_pairs.generate` can no longer overwrite `human_checked` pairs; (B2) `load_pairs(validated_only=True)` filters rejected/unvalidated pairs everywhere; (B3) `conditions.run` resumable and without duplicates (done-set + read dedup); (B4) API errors neither cached nor counted in the AUCs; (B5) `upper_bound` in leave-one-**pair**-out + scaler inside the pipeline (double leak removed).
  - H3 control fixed: `permute_positions` = a single position permutation shared across layers (preserves inter-layer coherence). Metric bootstrap **paired by pair**. Corpus generator distinct from the judges (temp 0.9, `generator` field logged).
  - Misc: judge cache key = model+temperature; `find_pos` robust to multi-token pivots; leak-check gate in `conditions`; `checks.leak` measures the max contiguous run; tokenizer without the 27B (`load_tok`); heavy imports (seaborn/openai/matplotlib) made lazy.
  - **English harmonization** of the whole pipeline (code, docstrings, judge prompts, capability question, generation templates) — resolves the Session 0 doubt.
  - 13 GPU-free unit tests (`tests/test_pure.py`), green locally (py 3.14, without torch).
- Verified: `py_compile` all of `src/` + tests; `pytest` 13/13; light imports OK; grep: no reference to the old API.
- What I believe now: pipeline coherent and tested on the pure-logic side; the `# ADAPTER` points in `lens.py` remain open (depend on the lenses README).
- Doubt / what could be wrong: nothing has run on GPU nor against a real judge; Δ2 confounds "invertible lens" and "good-reconstructor judge" (to add in §5.4); real context budget = 12 layers (stride 4), to be decided (stride 6 vs top-5).
- Next step: pod + download; read `lenses/README.md`; resolve the `# ADAPTER` points with me; smoke test `sushi → Japan`.
---
## 2026-08-31 — Resolving the `# ADAPTER` points + 4 blocking bugs (external review)  [counted: NO — hardening, no research result]
- Done:
  - HF README (camilablank/workspace-lenses) read and confirmed via WebFetch. Format settled: ONE `lens.pt` per lens (`{j-lens,r-lens}/lens.pt`), dict `['J','n_prompts','source_layers','d_model','provenance']`; formula `softmax(W_U · norm(J_ℓ · h_ℓ))` → no bias, norm NOT folded in (we keep `self.final_norm`), orientation `H @ J.T` correct, identity anchor at `target_layer=62`.
  - **B1** `lens.py`: dropped the per-layer safetensors; `_stack(kind)` loads the single `.pt` and maps `source_layers → index`; `_load_layer_map` returns `J[idx[L]]`, explicit KeyError if the layer is absent (skip_first=4).
  - **B2** `config.py`: real architecture centralized — `N_LAYERS=64`, `D_MODEL=5120`, `TARGET_LAYER=62`, `SKIP_FIRST=4` (the "~48 layers / ~12" comment was wrong).
  - **B3** `load_model.layers()`: grid `range(0,64,stride)` filtered by `source_layers`, with `62` always included (anchor). No more KeyError at L0, layer 62 is no longer missed.
  - **B4** `validate.identity_check()`: binary go/no-go — at L62 j/r-lens must reproduce the logit lens exactly (top-5). Runs BEFORE smoke in every mode.
  - Minor: `env/download.sh` → `hf` (formerly `huggingface-cli`), restricted to `j-lens/r-lens/*`+README (~7 GB instead of ~25); `load_model` `torch_dtype=`→`dtype=`.
  - Corpus generator moved to a **3rd family** distinct from both judges: `google/gemini-2.5-pro` (not a Qwen = family under audit). OpenRouter IDs verified (/models endpoint, 396 models): judge A `claude-sonnet-4.5`→`4.6`; judge B `gpt-5-mini` still valid.
- Verified: `pytest tests/test_pure.py` 13/13; AST parse of the 4 touched modules; IDs present in the real OpenRouter list (not the WebFetch summary, judged unreliable).
- What I believe now: the `# ADAPTER` points in `lens.py`/`load_model.py` are resolved and aligned with the README; the pipeline is ready for the pod. Nothing has run on GPU yet.
- Doubt / what could be wrong: `identity_check`, being symmetric in I, does not catch a PURE transpose of J at L62 (I=Iᵀ) — a real net for indexing/norm, not for orientation alone; the `hidden_states[L+1]` offset remains to be confirmed at the first scan. Scope choices (stride 8, top-5, reduced judges, drop upper_bound) NOT applied: your decisions.
- Next step: pod + `download.sh`; `python -m src.validate --smoke` (identity_check first); if green, `python -m src.checks leak` before any blind scan.
---
## 2026-08-31 (2) — 2nd review pass: F1–F6 + orientation  [counted: NO — hardening]
- Done:
  - **F1 (decision: chat template everywhere)**: single helper `load_model.to_input_ids` (honors `USE_CHAT_TEMPLATE`), used by `get_resid` (scan), `capability.ask` and `gen_pairs.span` → positions aligned end to end. injection/conflict families now read in their real conversational framing. `anomaly_token_span` recomputed in the same framing (order: decision → `gen_pairs span` → scan).
  - **F2**: `identity_check` no longer claims orientation (I=Iᵀ prevents it); added `orientation_check` (overlap at the closest loaded sub-anchor → catches a transpose). Both wired into `validate.__main__` and the notebook.
  - **F3**: go/no-go (identity+orientation) moved to the top of the notebook's step 2, before smoke.
  - **F4**: `layers()` falls back to the naive grid if the lenses are not downloaded → the SETUP cell no longer crashes.
  - **F5**: `flush()` added on the reconstruction-error branch (`conditions.py`).
  - **F6**: left tracked (the `data/*.jsonl` .gitignore rule was commented out by the human) — open decision.
- Verified: `pytest tests/test_pure.py` 13/13; AST parse of the 5 touched modules; single shared helper (no duplicate `apply_chat_template`).
- What I believe now: coherent input scan↔capability↔span; go/no-go genuinely discriminating (anchor + orientation) and present where the human runs it.
- Doubt / what could be wrong: the scan now includes the template special tokens as positions (noise for the judge, but faithful); `orientation_check` has an arbitrary threshold (`min_overlap=6/10`) to calibrate at the first run; nothing has run on GPU.
- Next step: pod → `download.sh` → go/no-go → `gen_pairs generate/review/span/stats` → scan.
---
## 2026-09-01 — Post-F1 audit: `find_pos` regression + position consistency  [counted: NO — hardening]
- Done:
  - **F1 regression fixed**: `find_pos` (validate.py) tokenized the RAW text while `get_resid` now returns the templated sequence → `H[L][pos]` shifted by the template prefix. `smoke`/`pass_at_k` were reading the wrong position. `find_pos` now operates on `render_input(...)`, the same framing as the scan. `identity_check`/`orientation_check` were untouched (position-agnostic).
  - **DRY framing**: `load_model.render_input` (the exact string sent to the model) becomes the single source; `to_input_ids` and the new `content_span` build on it; `add_special_tokens=False` under the template (the rendered string already carries the special tokens, equivalent to `apply_chat_template(tokenize=True)`) → identical tokenization between model input and position finding.
  - **Template tokens excluded from the scan**: `scan_text` now serializes only `[lo, hi)` = the content positions (via `content_span`), keeping ABSOLUTE indices → `anomaly_token_span` and `checks.evidence` stay aligned, less noise and judge budget.
  - **Minor**: `layers()` fallback now excludes 0..3 (`SKIP_FIRST`) and includes `TARGET_LAYER`; `# ADAPTER` tag removed from `unembed_parts` (Qwen names confirmed).
- Verified: `pytest tests/test_pure.py` 13/13; AST parse of the 5 touched modules. Not covered off-GPU: `find_pos`/`content_span`/`scan_text` (depend on the Qwen tokenizer) — to validate at the first pod run.
- What I believe now: positions coherent end to end (scan ↔ span ↔ evidence ↔ validation); model input unchanged in ids (equivalent refactor).
- Doubt / what could be wrong: `content_span` assumes `content` appears verbatim in the render (fallback = full range otherwise); `orientation_check` keeps an arbitrary threshold to calibrate.
- Next step: pod → go/no-go → manually verify `find_pos`/`content_span` on one pair before the full scan.
---
## 2026-09-01 (2) — enable_thinking + BatchEncoding robustness  [counted: NO — hardening]
- Done:
  - Confirmed the `apply_chat_template(return_tensors=...)[0]` → `Encoding` crash is NOT in the repo: `to_input_ids` already renders to a string then does `tok(...)["input_ids"]` (since commit e1b0621). Documented this in `render_input`/`_add_special` so it can't regress.
  - Added `config.ADD_GENERATION_PROMPT` (False) and `config.ENABLE_THINKING` (False); `render_input` now passes `enable_thinking=ENABLE_THINKING` to `apply_chat_template`. This prevents Qwen3 from emitting a `<think>` block for `capability.ask` (add_generation_prompt=True), which would have broken the YES/NO first-token parse.
  - `render_input`/`to_input_ids` defaults now come from `ADD_GENERATION_PROMPT` so the scan path (get_resid + content_span) stays aligned regardless of the flag; `capability.ask` overrides to True explicitly.
- Verified: framing harness 15/15 (fake tokenizer accepts the new `enable_thinking` kwarg); `pytest tests/test_pure.py` 13/13; import sweep 14/14 with constants wired.
- What I believe now: input framing is robust across transformers versions and Qwen3 thinking mode; positions still aligned end to end.
- Doubt / what could be wrong: the real Qwen tokenizer prefix (expected 3 tokens: `<|im_start|>user\n`) and the `enable_thinking=False` render are only checkable on the pod — the harness validates logic, not Qwen's exact tokenization.
- Next step: pod → B0 prefix check (`prefix_len == 3`) → go/no-go → scan.
---
## 2026-09-01 (3) — Real lens.pt format corrected (human inspection)  [counted: NO — hardening]
- Done (from direct inspection of lens.pt, overrides the README summary):
  - **J is a dict `{layer_number: (d,d) Jacobian}` in float16**, keyed 0..62 (63 layers) — NOT a stack. `_stack` drops the `enumerate(source_layers)` indirection (and any off-by-one): tuple is now `(J, J, d)` (second J kept so `layers()` iterates its keys). `_load_layer_map` returns `J[L].to(torch.bfloat16)` (fp16 on disk → bf16 for the matmul against the bf16 residual).
  - **`SKIP_FIRST = 0`**: source_layers covers 0..62; `skip_first=4` in provenance is a fitting parameter, not a row exclusion. Early layers (where R-lens is meant to beat J-lens) are therefore available.
  - **Checkpoint confirmed**: provenance `model_id = Qwen/Qwen3.6-27B` (standard estimator, pretrain corpus, 25 prompts, t_max=128) — closes the blind spot neither identity_check nor orientation_check covered.
  - Added `python -m src.lens` (`inspect`) that prints keys/dtype/provenance and checks `J[62] == I` offline — reproducible form of the human's two verification snippets; also verify R-lens provenance mentions RelP, not "standard".
  - `Lens.__init__` simplified to `.to(device)` (bf16 now owned by `_load_layer_map`).
- Verified: AST parse; `pytest tests/test_pure.py` 13/13; import sweep 14/14 with a fake J-dict (keys 0..62) → `layers()` grid = [0,4,..,60,62], includes layer 0; framing harness 15/15. NOT run here (no torch/lens files): `python -m src.lens` and the anchor check — pod-side.
- What I believe now: lens indexing is now correct against the real artifact; the identity anchor is checkable offline before any GPU load.
- Doubt / what could be wrong: `LAYER_STRIDE=4` left as-is (stride is your scope call); the anchor `J[62]==I` and R-lens RelP provenance still to confirm on the pod via `python -m src.lens`.
- Next step: pod → `python -m src.lens` (keys 0..62, dtype fp16, anchor==I, R-lens=RelP) → go/no-go → scan.
---
## 2026-09-01 (4) — Pod format inspection confirmed (B0/B2 done) + scope cuts applied  [counted: NO — hardening]
- Confirmed on the pod (human inspection of lens.pt):
  - `prefix_len = 3` (chat template prefix = `<|im_start|>user\n`).
  - `source_layers = 0..62` (63 layers); `skip_first` is a fitting parameter, NOT a row exclusion.
  - `J` is a dict indexed by layer number, in float16.
  - **Identity anchor exact to the bit**: `J[62] = I`, off-diagonal exactly 0.0 (not approximately) → `identity_check` has a perfectly solid basis.
  - `model_id = Qwen/Qwen3.6-27B`, standard estimator, pretrain corpus, 25 prompts, `t_max = 128` — matches the downloaded checkpoint.
  - The two `provenance` are identical except `config_json`: `"standard"` for J-lens; `"relp"` with `ln_rule`, `identity_rule`, `half_rule`, `include_qk_norms: false` for R-lens. Same model/corpus/25 prompts/t_max/anchor → identical forward, only the gradients differ (exactly what the R-lens post announced). Instrument-to-instrument comparison is clean by construction; keep both provenance strings for the doc appendix.
  - Caveat: `n_positions: 0.0` is an unfilled field, no consequence — do not cite it as a real counter in the doc.
- Done (config scope cuts, now explicitly requested):
  - `LAYER_STRIDE = 8`, `TOP_K = 5`, `PAIRS_PER_FAMILY = 6`, `ADD_GENERATION_PROMPT = True` (deployment framing; content_span still excludes the assistant prompt, causal attention leaves content residuals unchanged). `SKIP_FIRST = 0`, `ENABLE_THINKING = False` already in.
  - lens.py dict-`_stack`/`_load_layer_map` (with `.to(bf16)`) and load_model `to_input_ids` (render-to-string + `add_special_tokens=False`) already in from f5efc68.
  - Notebook SETUP prints the grid + VRAM (expect 64 layers, `[0,8,16,24,32,40,48,56,62]`, ~55 GB).
- Verified offline: `pytest tests/test_pure.py` 13/13; import sweep with a fake J-dict (keys 0..62) → `layers()` = `[0,8,16,24,32,40,48,56,62]`; framing harness 15/15 (now with ADD_GENERATION_PROMPT=True).
- Next step: pod → SETUP (expect grid above, VRAM ~55 GB; tell me if num_hidden_layers != 64) → `load_all()` → `identity_check` + `orientation_check` (the real break point) → B1.
---
## 2026-09-01 (5) — Block B done: go/no-go GREEN + smoke better than hoped  [counted: NO — validation]
- Verified on the pod (four independent converging signals):
  - **Expected layer order reproduced exactly**: R-lens surfaces the concept at L8, J-lens at L24, logit lens never — the R-lens post's central prediction (early-layer advantage, later convergence), obtained on our setup with no tuning.
  - **Three instruments coincide at L62**: `[' originated', ' originates', ' origin', ' is', ' rolls']` identical for all three — `identity_check` visible to the naked eye, and confirms the readout chain (norm, W_U, decoding) is consistent across instruments.
  - **L0 is pure noise** for J and R (`'Scrollbar'`, `'uhu'`, `'قدام'`), then the signal appears — the transition a working lens should show (a broken setup gives noise everywhere or signal everywhere).
  - **Logit lens fails characteristically, not randomly**: it tracks surface syntax (punctuation, then `originated`) without ever reaching the concept — exactly the baseline condition 4 must embody.
- Two notes for the write-up:
  - `orientation_check` = 7/10 at L56: comfortable but not overwhelming (six blocks separate 56 from the anchor). TODO: check in the output that the transposed version scored clearly lower; if the gap was thin, mention it in limits.
  - At L56, J and R say `originated`, not `Japan`: the intermediate concept fades in favor of the output token in the last layers — consistent with a workspace that sorts concepts before writing, and justifies the grid a posteriori (scanning only late layers would have shown nothing). Good material for fig3.
- Next step: B5 quantitative validation — filter `data/multihop.jsonl` to items the model answers correctly (an item the model misses cannot validate a lens), then `pass_at_k`. Expect same-shaped curves as the smoke, R ≥ J early, convergence late, final pass@k 0.5-0.8. If J ≈ R and both < 0.2 despite the sushi result, the problem is the multihop corpus or `find_pos`, not the lenses. Outputs: results/validation_multihop.json, figs/validation_multihop.png (goes in the doc, "I verify my instrument works").
---
