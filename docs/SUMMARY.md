# Experimentation Summary — J-Lens / R-Lens / Logit-Lens Blind Audit on Qwen3.6-27B

> Agent-produced, factual summary of the whole project to date. The final write-up, the exec
> summary of record, the independent AUC recompute and the pivot decision remain the human's
> (CLAUDE.md). Numbers here are the agent's computation; interpretation is flagged as such.
> Companion documents: `experiments.md` (append-only lab log), `results/Result.md` (full session
> record + analysis §1-18), `results/executive_summary.pdf` (exec summary draft).

---

## 1. The question

Does a lens's signal in a **blind scan** (numeric per-position/per-layer top-k token stacks, no
prose) survive when the **text-reconstruction channel is neutralized**? Three hypotheses are
discriminated:
- **H1 — internal readout**: the lens surfaces a genuine internal signal, beyond surface text.
- **H2 — text inversion**: the apparent signal is just the scan letting a reader reconstruct the text.
- **H3 — judge apophenia**: the judge manufactures "anomalies" from anomaly-shaped noise.

Model under audit: **Qwen3.6-27B** (64 layers, d_model 5120). Instruments: **J-lens**, **R-lens**
(camilablank/workspace-lenses), and the **logit lens** as the no-internal-machinery baseline.

## 2. Design (fixed before any code)

- **Corpus**: anomalous/clean **text pairs**, 3 families — `bug` (silent logic error), `false_premise`
  (a wrong fact), `injection` (an embedded instruction). Human-validated only.
- **Instruments**: each pair scanned by the 3 lenses -> `{id}_{version}_{instrument}.json`, top-k=3
  per (position, layer), layer stride 8 -> grid [0,8,16,24,32,40,48,56,62].
- **Two judges** (different families, to avoid self-grading): judgeA = Anthropic claude-sonnet-4.6,
  judgeB = OpenAI (originally gpt-5-mini). Generator = Google gemini (3rd family).
- **7 conditions**, the controls carrying the argument: 1 prompt-only (text baseline), 2/3/4 scan
  (per instrument), 5 scan_permuted (positions shuffled = H3 control), 7 reconstruction (judge A
  rebuilds text from the scan, judge B reads only that = H2 control).
- **Guards**: leak audit before any scan run; judge prompt never names a family or instrument;
  bootstrap resampled by pair.

## 3. Phase A — pipeline hardening (2026-08-15 -> 09-01, before the run; experiments.md 0-13)

- Topic and the 7 conditions designed before code; repo skeleton; 8 pilot pairs; 40 multihop
  questions for instrument validation.
- Single IO layer `store.py`; ~5 critical bugs fixed (no overwrite of validated pairs; validated-only
  gate; resumable/dedup runs; API errors never cached nor counted; leave-one-pair-out for the upper
  bound); English harmonization; 13 GPU-free unit tests.
- **Lens format** resolved from the HF README then confirmed by direct inspection: one `lens.pt`,
  `J` a dict keyed by layer number (fp16), **identity anchor `J[62] = I` exact to the bit**, R-lens =
  RelP (same model/corpus, only the gradients differ). Config centralized (N_LAYERS=64, D_MODEL=5120,
  TARGET_LAYER=62). Scope cuts applied under budget: LAYER_STRIDE=8, TOP_K=3, 6 pairs/family.
- **Corpus generation** hardened: injection JSON truncation fixed; a self-labelling guard (the
  generator once wrote `# Mistake:` into its own bug — would inflate the baseline); patch format for
  substitution families; **generator switched gemini-2.5-pro -> flash** because the reasoning model
  spent the token budget on thinking and never closed the JSON (this exact failure recurred later on
  the judge — see §5).
- **Go/no-go GREEN on the pod** (experiments.md 5): R-lens surfaces the concept at **L8**, J-lens at
  **L24**, logit lens never; all three coincide at L62 (identity anchor visible); L0 is pure noise ->
  the expected early-noise-then-signal transition. The R-lens post's central early-layer prediction
  reproduced with no tuning. NB: the J-space readable window peaks ~L24 and collapses by L56-62 — a
  fact that matters for §6's propagation caveat.

## 4. Phase B — the run (this session, 2026-09-01, local Windows, judges via OpenRouter)

Scans (66 files) and validated pairs (**11**: inj_01/02, bug_01/02/03/05, fp_01/02/03/04/06) were
already present; the GPU/model was not reloaded. Executed:
- **Leak gate** (mandatory): 0 shared 4-grams across 66 scans -> GREEN (but see §6 caveat).
- **Block E** (jlens, 2 judges, v1+v2) + **Block F** (rlens+logit, 2 judges, v1) -> **579/592 valid
  verdicts** (2.2% loss; 1 scan key, 12 reconstruction keys mostly content_filter on false_premise).
- **Block G** metrics + figures. Independent from-scratch AUC recompute matches the pipeline to <1e-6.
- **Block H** sanity checks: evidence fidelity, false-alarm rates, 30-row review worksheet (objective
  columns pre-filled, judgment left to the human).
- **Content controls** (post-review, no GPU/API): does the anomaly leave an objective fingerprint in
  the scan (§6)?

## 5. Bugs & anomalies encountered this session (all fixed)

1. **judgeB (gpt-5-mini) reasoning-truncation** — every JSON verdict came back empty
   (finish_reason=length): a reasoning model burns the 600-token budget before emitting JSON. ~50%
   error rate on the first probe. Fixed: `JUDGE_B -> gpt-4.1-mini` (non-reasoning, still OpenAI).
   Same class of failure as the Phase-A generator.
2. **No client timeout** — one stalled call hung the whole run (froze at 12/64). Fixed: `timeout=90`.
3. **max_tokens=600 too low for claude** on verbose scans -> finish_reason=length. Fixed: 1200.
4. **content_filter** — claude refuses to read some reconstructed texts, biased onto false_premise
   (~10 lost keys) -> makes the D2/reconstruction channel fragile.
5. **Windows cp1252 vs UTF-8** crash on every scan read -> fixed at the root (`encoding="utf-8"`).

## 6. Results (agent computation; interpret independently)

- **Text baseline near-ceiling**: prompt-only AUC 0.95 (judgeA) / 0.96 (judgeB).
- **Pooled blind-scan AUC**: logit **0.84** >= rlens 0.80 >= jlens 0.75 (n=11/11). The "dumb" logit
  lens leads. On v1 per-family (fig2 CORRECTED — the submitted v1+v2 heatmap inverted this via n=2
  cells): logit's best family is **bug 0.906**, ahead of jlens 0.61 / rlens 0.66 — the dominance
  survives decomposition.
- **Triplet** (dAUC, 95% paired-bootstrap CI, pooled): D1 scan-prompt < 0 everywhere (scan never
  beats text); D2 scan-reconstruction > 0 for rlens/logit but the reconstruction channel is broken
  (fragile); D3 permuted-chance > 0 on all three (order-invariant judge behaviour).
- **Inter-judge**: kappa 0.49, Spearman 0.50 (moderate).
- **Judge fidelity (H3-relevant)**: ~52% of cited evidence tokens are absent from the scan; high
  false-alarm on clean twins (reconstruction 0.5-1.0). Evidence and confidence do not predict
  correctness.
- **Content control** (5 aligned single-token pairs): a one-token change alters the scan downstream
  (identical input there) in 25% early / 60% mid cells; upstream = 0 (causal sanity passed). **BUT
  no matched neutral-substitution null** (needs GPU, DECLARED) — so this is a fact about transformers
  under causal attention, not a measurement about anomalies. The monotone rise with depth fits
  residual diffusion, not the L24-peaked concept window.
- **Anomaly-specificity (free test)**: downstream of the anomaly, the added vocabulary shifts toward
  **incorrect / wrong / error / correct** for the semantic-error families (fp_02, fp_03, bug_05),
  across all three lenses. Suggestive that the model represents the error — but noisy (~12 tokens/cell),
  n=5, null-less, and **not lens-specific**.

## 7. Post-review corrections (2026-09-02)

An external recompute from the CSVs flagged, and we fixed: (a) fig2 corrupted by n=2 v2 cells ->
regenerated on v1, logit ranking corrected (bug > fp); (b) the leak gate tests 4-grams while the real
risk is the ~100% mono-token echo -> reconciled; (c) the propagation control overclaimed -> withdrawn
to "a one-token perturbation propagates", with the neutral-null explicitly declared as not run.

## 8. Honest conclusion (agent opinion — to verify; not the pivot)

No evidence of a **lens-specific** internal readout: the logit lens leads, the signal is
order-invariant, and everything anomaly-related the lenses surface, the logit lens surfaces too. The
judge detects noisily and partly apophenically (fabricated evidence, clean false-alarms, moderate
kappa). There is a **plausible but unproven** generic "the model represents that something is wrong"
signal (downstream error-vocabulary shift), which would only become a result with the
neutral-substitution null (GPU). The audit's value is in the controls — and in naming its own
decisive open control rather than overclaiming.

## 9. Budget & artifacts

Spend **$14.97 / $19**. Key files: `results/{metrics,triplet,inter_judge,leak_check,evidence_check,
family_false_alarm,blockH_worksheet,anomaly_in_scan,anomaly_propagation,anomaly_vocab_shift,fig2_v1_auc}.csv/json`,
`figs/{fig1_triplet,fig2_auc_family_v1}.png`, `judge/outputs/verdicts.jsonl`, `results/Result.md`,
`results/executive_summary.pdf`, `experiments.md`.

## 10. Open items

- **Human (not delegated, CLAUDE.md)**: read the 30-row worksheet + classify false positives a/b/c;
  independent AUC recompute; the pivot decision; the final exec summary.
- **Needs GPU**: the matched neutral-substitution null (the one control that would turn §6 into a
  result); optionally scale n beyond 11.
- **Security (pending)**: revoke/rotate the exposed OpenRouter API key.
