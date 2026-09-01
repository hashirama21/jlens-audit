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
## 2026-09-01 (6) — gen_pairs: injection JSON failure fixed  [counted: NO — pipeline hardening]
- Symptom on the pod: `gen_pairs generate` printed `[skip] injection #k invalid JSON` (family-specific).
- Cause: `max_tokens=1500` truncated `injection` — the only family with two 150-400 token versions in one JSON object (300-800 tokens of text + JSON escaping + anomaly_text) → missing closing brace → parse fail.
- Fix: `complete(..., want_json=True, max_tokens=4000)`. `want_json=True` already returns a parsed dict (handling fences/preamble) or an error dict with `_error`+`_raw`, so gen_pairs' hand-rolled `re.search`/`json.loads` is dropped (DRY; `import re` removed). On failure the raw output is written to `results/raw_fail_{fam}_{k}.txt` — no more blind debugging.
- Note (future, not the current failure): `_parse_json`'s greedy `\{.*\}` fallback in judge.py could over-capture if a response has trailing braces after the JSON; json.loads(raw) is tried first so well-formed JSON (incl. Python code with inner braces in `bug`) is unaffected. Left as-is.
- Resumability unchanged: `have[fam]` counts existing and resumes at have+1; re-run seeds from the pilots — pass `seed_path="pairs.jsonl"` on re-run if a partial pairs.jsonl already exists, to keep the successful pairs.
- Verified offline: AST, import sweep, pytest 13/13. Not run here (needs OpenRouter + generator): the generation itself.
---
## 2026-09-01 (7) — OpenRouter budget wall (~$20): immediate cuts applied  [counted: NO — budget]
- Context: OpenRouter balance nearly empty (~2787 tokens on a single request → 402s). Block E (judges) is the big spend (~4M input tokens on the expensive judge at the full plan → well over $20).
- Applied now (safe, directed):
  - `TOP_K = 5 → 3`: smoke shows the concept in the top-3 at every layer where it appears, so ~40% less judge context at little cost. LIMIT to declare in the doc.
  - gen_pairs `max_tokens = 4000 → 2500` + `TEMPLATES["injection"]` target 150-400 → 150-250 tokens (together, so injection still fits without truncation). Pairs need to be matched, not long. Changed BEFORE generating the 6 injection pairs so the family stays length-consistent (mixing 400- and 200-token pairs would be a parasitic within-family variable in the intra-family AUC).
- Pending YOUR decision (not applied — design change to conditions.py + budget stakes; measure-first):
  - Judge routing: gpt-5-mini (judgeB) primary on ALL conditions; Sonnet (judgeA) second opinion only on conditions 1 (prompt_only), 2 (scan+jlens), 7 (reconstruction). Keeps Cohen κ where Δ1/Δ2/Δ3 live; low tariff everywhere else. Requires updating the e2e expected-count invariant.
  - Cut prompt v2 (`PROMPT_VERSIONS=["v1"]`) ONLY if the 3-item pilot projects > $15. Cheapest scientific sacrifice (lose a robustness test, declared in limits) — never drop a judge (that costs κ).
  - Untouchable in all cases: conditions 1, 2, 5, 7 (else no Δ1/Δ2/apophenia control → nothing for the exec summary).
- Yours on the pod: recharge before re-running generate; verify exact prices on openrouter.ai/models for both ids; run block E on 3 items first, read real cost on the usage page, extrapolate (cache keyed by condition,item,instrument,judge,prompt_v → nothing recomputed).
- Verified offline: pytest 15/15 with TOP_K=3.
---
## 2026-09-01 (8) — gen_pairs: max_tokens back to 4000 + self-label guard  [counted: NO — corpus quality]
- Credits restored; generator produces correct content, but two outputs were truncated mid-field (docstring / loop) at max_tokens=2500 → those cut-offs were the `invalid JSON` (no closing brace). Set `max_tokens = 4000`.
- Serious corpus defect caught: the generator self-labelled its own bug (`# Mistake: limit should be inclusive`). Such a pair is useless — the prompt-only judge (condition 1) sees "Mistake" and detects for free, inflating the baseline and crushing Δ1; and it violates the family rule (error must be semantic and silent).
  - Hardened `TEMPLATES["bug"]`: no comment/docstring/name/string may signal the error; and the anomalous version must NOT raise (the `n = n/2` → float → math.isqrt crash was noise, not the intended silent error — an anomalous version that throws doesn't test what we think).
  - Added an automatic `_TELLS` guard in generate (next to the identical-versions check): skip when a tell (mistake/bug/error/wrong/incorrect/intentional/deliberate/fixme/todo/flaw/off-by-one) is in the anomalous but NOT the clean twin. Does not replace human review; saves reading 30 pairs to discard half. `note:` deliberately excluded — injection instructions are legitimately note-shaped (verified: no false positive on "Note:" / "Note for the assistant").
  - NB: `import re` restored (I had removed it when dropping the hand-rolled JSON parse).
- Verified offline: AST, import, _TELLS behavior (self-label→skip, injection Note→keep, tell-in-both→keep), pytest 13/13.
- Next: pod → regenerate with max_tokens=4000 + hardened template; read the first two `bug` pairs by hand before letting the other families run.
---
## 2026-09-01 (9) — gen_pairs: patch format for substitution families  [counted: NO — corpus quality]
- Template hardening worked (no more self-annotation; first bug pair was a clean inverted-condition). But outputs still truncated in the `clean` field: the generator wrote `anomalous` in full then repeated near-identical code in `clean` — paying twice for a one-line diff, doubling the output past any reasonable limit.
- Fix: change what we ask, not the ceiling. For substitution families (`_PATCH_FAMILIES = {bug, false_premise}`) the model returns the correct text ONCE + a one-line/one-span patch (`clean`, `anomalous_line_old`, `anomalous_line_new`); `generate` rebuilds `anomalous = clean.replace(old, new)` after checking `clean.count(old) == 1` (skip otherwise). Benefits: ~half the output, "exactly one line differs" true by construction, and exact `anomaly_text = old -> new` for free. injection/conflict keep both full versions (diff is an addition/rephrasing, not a substitution).
- bug template also now steers away from "expert" anomalies (inclusive-vs-exclusive quartiles etc.) toward obvious-once-seen logic errors — those subtle ones would drop the bug family under 50% on the capability test.
- `_dump_raw` helper factored out (raw output on any failure). `import re` present.
- Watch at review (noted): the IQR example used inclusive vs exclusive (docstring said exclusive) — too subtle; the new template forbids that class.
- Verified offline: AST; end-to-end generate() with a stubbed generator (patch rebuild exact for bug + false_premise, full versions kept for injection/conflict, uniqueness skip works); pytest 13/13.
- Next: pod → regenerate; read first two bug pairs + first false_premise pair by hand.
---
## 2026-09-01 (10) — judge.complete: surface API errors + stop caching empties  [counted: NO — infra]
- Two bugs in judge.complete:
  - **Silent API errors**: exceptions were caught into `last`, retried, and never printed during the loop. Added a per-attempt trace `[api] {model} attempt i/N: {type}: {msg}`. (The final return already surfaces `last` in `description`.)
  - **Cache poisoning (the real trap)**: with `want_json=False`, an empty response `raw=""` was cached as a success (`json.dump("")`), so every later call with the same prompt re-read `""` from disk without touching the API — re-running generate stayed empty forever. Fix: `if raw.strip():` before caching; empty is returned but never cached. (want_json=True was already safe: `_parse_json("")` → None → error dict, uncached.)
- Pod action required (polluted entries already on disk): purge sub-10-byte cache files —
  `find "$(python -c 'from src.judge import CACHE; print(CACHE)')" -name "*.json" -size -10c -print -delete`
  then the 30-second direct probe: `complete(GENERATOR_MODEL, 'Reply only: {"ok": true}', temperature=0, want_json=False, max_tokens=100)` — with the trace, any API error is now visible.
- Verified offline: stubbed client — empty (want_json F and T) caches 0 files, non-empty/valid-json caches 1; pytest 13/13.
- Note: reconstruction (condition 7) uses want_json=False, so it was exposed to the same cache trap — now fixed there too.
- **Root cause revealed by the probe**: the direct call returned `API error: 'OPENROUTER_API_KEY'` → the key simply isn't exported on the pod (`client()` did `os.environ["OPENROUTER_API_KEY"]` → KeyError, swallowed as a retryable API error). Fix on the pod: `export OPENROUTER_API_KEY=...`.
- Hardened `client()`: explicit key check → clear `RuntimeError("OPENROUTER_API_KEY is not set — export it ...")`. Hoisted `client()` out of the retry loop (after the cache check) so a missing key fails fast instead of being retried 3× and buried; cache hits still work without a key. Verified offline: missing key → RuntimeError, cache hit → returns without touching client, empties still uncached.
---
## 2026-09-01 (11) — gen_pairs crash on null field + full raw dumps  [counted: NO — robustness]
- Key was set (client() constructs). But `generate` crashed at `_TELLS.search(d["anomalous"])` with a TypeError: an item passed the key-presence check but had `anomalous`/`clean` as null (model returned `"anomalous": null`), and `re.search(None)` throws.
- Fix: every path now guarantees non-empty STRING anomalous/clean before the checks — patch families require the three fields be `str` (isinstance), non-patch requires anomalous+clean be `str`, then a shared non-empty guard. Null/non-string/empty → `_dump_raw` + skip, no crash. Verified offline: null bug.clean and null conflict.anomalous both skip with a dump, valid injection/false_premise kept, no crash.
- Also: `complete` now stores the FULL raw (not raw[:2000]) in the parse-failure `_raw`, so `results/raw_fail_*.txt` shows exactly where the JSON was cut — real truncation vs malformation is now visible.
- Open hypothesis for the recurring `bug` parse failures: gemini-2.5-pro is a reasoning model; reasoning tokens may eat the 4000-token output budget before the JSON closes. Check the full raw dump: if it ends mid-JSON with no trailing text, raise max_tokens (8000) or use a non-reasoning generator. Do NOT confuse with unescaped newlines (also visible in the raw).
- Added a `[completion]` trace in complete() (`finish_reason`, chars, max_tokens) — the decisive signal for the truncation: `finish_reason=length` → raise max_tokens (patch format needs ~1000 for bug, not 4000); `finish_reason=stop` with a cut-off `clean` → the model stops mid-code, a template/model problem.
- `gen_pairs`: `clean.replace(old, new, 1)` (explicit single substitution; count==1 checked just above) — defensive against later edits.
- Verified offline: AST, crash-proof test, pytest 13/13.
- Next (pod): single test call, report the `[completion] ... finish_reason=...` line — that decides the truncation fix. Pipeline otherwise frozen.
---
## 2026-09-01 (12) — truncation root cause: gemini reasoning eats the token budget  [counted: NO — infra]
- Probe: `finish_reason=length chars=69 max_tokens=600` → gemini-2.5-pro is a reasoning model; reasoning tokens consume max_tokens, leaving ~69 chars of visible JSON before the cap. Confirmed truncation, not malformation.
- Fixed a real latent bug the review flagged: `_key(model, temperature, want_json, text)` omitted `max_tokens`, so a cached result at one budget would be served for another. Now `_key(model, temperature, want_json, max_tokens, text)` (verified: different max_tokens → different key). NB: this invalidates old cache entries (orphaned, harmless).
- Parse-failure description now carries `finish_reason` — `JSON parse failed (finish_reason=length|stop)` distinguishes truncation from a model stopping mid-code.
- generate() max_tokens LEFT at 4000 (pipeline frozen pending the 1200 direct test). CAUTION for the tuning step: the proposed `1200 if bug else 600` would re-truncate injection/conflict — those keep TWO full versions (~500 tokens) + gemini reasoning, so they need MORE than 600, not less; only bug/false_premise are compact patch formats. Generation tokens are cheap, so err high per family.
- Verified offline: AST, cache-key test, pytest 13/13.
---
## 2026-09-01 (13) — generator switched to gemini-2.5-flash (definitive truncation fix)  [counted: NO — infra]
- Confirmed the reasoning-budget cause: 600/1200/3000 all → `finish_reason=length`, only ~376 chars at 3000. Not a max_tokens problem — gemini-2.5-pro's thinking tokens consume the output budget before the JSON closes.
- Fix: `GENERATOR_MODEL` default `google/gemini-2.5-pro` → `google/gemini-2.5-flash`. Still Google (third family, distinct from Anthropic judgeA + OpenAI judgeB); code + one-line mutation doesn't need a reasoning model. max_tokens stays 4000 (a cheap ceiling on Flash; no per-family tuning needed). Env-overridable.
- Escalation if Flash still shows finish_reason=length (it's a hybrid, thinking can still trigger): `google/gemini-2.5-flash-lite`, or disable reasoning via OpenRouter (`extra_body={"reasoning": {"enabled": False}}`) — offered, not applied.
- Verified offline: GENERATOR_MODEL loads as gemini-2.5-flash, pytest 13/13.
- Next (pod): pull, purge stale cache if needed, run generate (or the direct Flash test) and check `[completion] finish_reason=stop` with valid JSON.
---
## 2026-09-01 (14) — FIRST real result: blocks E/F/G run on 11 pairs  [counted: YES — ~2.5 h wall, incl. debugging]
- Done (first end-to-end judge run against the real scans; all numbers below are MINE, pending your independent AUC recompute):
  - **Block E** (jlens, 2 judges, v1+v2) then **Block F** (rlens+logit, 2 judges, v1 only) over the 11 human-validated pairs (inj_01/02, bug_01/02/03/05, fp_01/02/03/04/06). Both judges kept in F so condition 7 (reconstruction) still pairs cross-judge. Then **Block G** = `python -m src.metrics` → results/metrics.csv, triplet.csv, inter_judge.json, figs/fig1_triplet.png, fig2_auc_family.png. fig3_example NOT yet made.
  - **Two judge bugs fixed mid-run (your approval via prompts):** (a) judgeB `gpt-5-mini` returned `finish_reason=length`, empty JSON on ~every call — a reasoning model burning the 600-token budget before emitting JSON (same class as entries 12/13 on the generator). Switched `JUDGE_B` → `openai/gpt-4.1-mini` via .env (non-reasoning, still OpenAI = 3rd-family design intact, no code change). (b) `judge.py`: OpenAI client had NO timeout → one stalled call hung the whole run (froze fp_01 at key 12/64); and `call()` default `max_tokens=600` was too low for judgeA (claude rambles a preamble past the cap on verbose logit/degenerate inputs → finish=length). Fixed: `OpenAI(..., timeout=90)` and `call(max_tokens=1200)`.
  - Windows-local run: forced `PYTHONUTF8=1` (scans are UTF-8, Windows default cp1252 crashed every `read_text`); no code change.
- Verified:
  - **Leak gate GREEN** (CLAUDE.md, shown before any scan run): `results/leak_check.csv`, 66 scans, `shared_ngrams=0` everywhere, `frac=0.0`; `max_contig=2` except bug_05/rlens=3 (below the n=4 threshold → signal, not harness leak).
  - **579/592 valid verdicts (2.2 % loss).** Scan conditions (2/3/4/5) essentially complete: only 1 scan key lost (bug_03 permuted jlens judgeB). Coverage 64/64 on the 4 full-matrix items, 46-48 on the 7 E/F items.
  - **Budget:** finished at **$14.97 / $19** (E=$2.83, F=$3.44). Real per-item ≈ $1.07 full-matrix, measured on fp_01 with a clean before/after usage snapshot from openrouter /credits.
- What I believe now (FACTUAL, not a conclusion — you recompute + decide): triplet (Δ pooled judges/prompts, 95 % paired-bootstrap CI):
  - **Δ3 permuted−chance > 0, CI excludes 0 on all three instruments** (jlens +0.28 [0.02,0.46], rlens +0.20 [0.01,0.39], logit +0.29 [0.11,0.45]) → apophenia (H3) present: shuffled scans still score above chance.
  - **Δ1 scan−prompt < 0 everywhere** (jlens −0.21, rlens −0.16 [CI<0], logit −0.12) → the blind scan never beats reading the text (prompt-only AUC 0.95/0.96).
  - **Δ2 scan−reconstruction > 0 for rlens (+0.49 [0.23,0.79]) and logit (+0.32 [0.05,0.57])**, jlens inconclusive (+0.11 [−0.30,0.47]).
  - Inter-judge (scan v1): κ=0.49, Spearman=0.50 (moderate).
- Doubt / what could be wrong:
  - **Δ2 is fragile/biased**: the reconstruction condition lost ~10 keys to `finish_reason=content_filter` (claude refusing to read reconstructed text), **concentrated on the false_premise family** (fp_02/03/04/06 clean AND anomalous); reconstruction also has a huge false-alarm-on-clean rate (0.5-0.9). Δ2 uses reconstruction → computed on fewer, non-random items. Not recoverable without a design change (deterministic safety refusals). The 2 non-filter losses are gpt-4.1-mini reconstruction degenerating into repetition (bug_01/rlens).
  - n=11 pairs → per-family AUCs (fig2, n_boot=300 on ~4 items) are **exploratory only**, as planned; report the pooled AUC as the headline, not the family cells.
  - **Judge-identity change** (judgeB gpt-5-mini → gpt-4.1-mini) alters results vs the frozen plan (entry 7) and MUST be declared in limits; κ is preserved (still 2 judges) but the OpenAI judge is now non-reasoning.
  - Mixed max_tokens across items (4 cached items scored at 600, 7 new at 1200) — the resumability key ignores max_tokens so valid 600 verdicts were kept; anomaly/confidence are unaffected by the cap, only truncation is, so pooling is fine — but note it.
  - All AUCs here are MY computation; per CLAUDE.md you recompute independently before anything is reported.
- Next step: you read 3 raw examples per condition + independently recompute the pooled AUCs; make fig3_example (annotated anomalous scan); decide the pivot (H1/H2/H3) and whether the content_filter loss forces a reconstruction-channel rethink; then the exec summary.
---
