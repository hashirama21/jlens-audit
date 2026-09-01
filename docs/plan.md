# Topic A — End-to-end plan
## Blind audit of activation-readout lenses (J-Lens / R-lens) — from preparation to submission

*Version of August 15, 2026. Deadline: Friday September 4, 11:59 PM PT (extension form possible until the 11th).*

---

## 0. What today's research confirmed

**The niche is open.** The R-lens post of August 5 (Blank, Bhatia, Nanda) was read in full: it evaluates the lenses by pass@10 across five categories (multihop, multilingual, association, typo, poetry), by causal ablation and by CKA — **no blind-scan audit, no false-positive measurement, no text-inversion control**. Our question remains untouched.

**Three new facts that reshape the protocol:**

1. **R-lens brings nothing on the smallest models** — the advantage appears and grows with scale, maximal on DeepSeek-V4-flash. Consequence: working at 4B/9B makes the J vs R comparison uninformative; **at least Qwen3.6-27B is needed** for the comparative axis to make sense. This moves the GPU choice toward an 80 GB (A100/H100, $1.5-2.5/h) or a 4-bit quantization on 48 GB — the latter is risky because quantization can degrade the lens's fidelity; to be tested in phase 0.

2. **The available lenses** (HF `camilablank/workspace-lenses`, 46.7 GB, MIT): qwen3.5-4b / 9b / 27b, qwen3.6-27b, qwen3.6-35b-a3b, qwen3.5-122b-a10b, gemma-3-27b-it, deepseek-v4-flash — J-lens AND R-lens for each. **Qwen3.6-27b is the model of Neel's replication**: it is our default choice, a direct-comparability argument.

3. **Two comments under the post set the field's norm.** Burny praises the post for its "comparisons to baselines, error bars, a random control, no graph crimes, and not relegated to an appendix" — and notes that Anthropic's J-Lens paper did not respect these. Stella Athena (EleutherAI, author of the tuned lens) raises that "it's really hard to know whether you have the right way to measure what the model thinks after layer K", that the J-Lens paper's metrics were "either layer-invariant by construction, or exposed to that trap", and asks "what is the right way to do this kind of evaluation". **Our project is a partial answer to her question** — to cite in the write-up; this is exactly the level of conversation where Neel wants to see a candidate.

---

## 1. What makes an application succeed — the implicit spec

Before the plan, the target. From the application doc (read in full) and the previous round's form, selection turns on five things, in this order:

**1.1 The Airtable form answers are the primary filter.** Neel reads all of them; he does not read all the write-ups. An application whose form does not convince him to read the doc is lost regardless of the doc's quality. The form must therefore be treated as a first-class deliverable, not a formality.

**1.2 The executive summary is the only text guaranteed to be read.** ~1 page, max 3 pages / 600 words, with graphs, at the top of the Google Doc. His recommended format: (a) the problem and why it is interesting, (b) the high-level takeaways, (c) one paragraph + one graph per key experiment.

**1.3 "An application that teaches me something."** The taste criterion. Our angle: apply to the lenses the text-inversion control that the field has just formalized for activation oracles — a result he doesn't have and has publicly asked for.

**1.4 Proof of human verification beyond the agent.** The advice he himself calls the most important. What he's looking for: "I read 30 scans and confirmed…", independently recomputed numbers, baselines we designed, an experimental design in which the agent is only the executor. An application that looks like "an agent did a project" is rejected.

**1.5 Zero LLM prose in the form and the exec summary.** Strong negative signal — "they blur together, he sees hundreds". Conversely, agentic use of the LLM for code and execution is an acceptance factor of ~3× (the figure he gives).

And three traps not to fall into: overstating results (his negative signal #1), dressing up a negative, forgetting an obvious baseline. A mixed or negative result is acceptable; an overclaimed result is not.

---

## 2. Master schedule

| Window | Clock status | Objective | End-of-window deliverable |
|---|---|---|---|
| **Sat 15 – Sun 16 Aug** | Off | Infra decisions + foundational reading | GPU ordered, Claude Max active, J-Lens paper and R-lens post read |
| **Mon 17 – Fri 21** (evenings) | Off | ARENA 1.2 (3 sections), agent + persistent Jupyter setup, load a lens | The replicated lens produces readouts on Qwen3.6-27B |
| **Sat 22 – Sun 23** | Off | Replicate one eval from the R-lens post; model capability test; go/no-go | **GO/NO-GO** documented; 8-pair pilot corpus; the clock can start |
| **Mon 24 – Fri 28** (evenings, ~2h each) | **Counted (~8h)** | Full corpus, scan pipeline, judge harness | Scans of the 40 pairs × 3 instruments on disk; first conditions run |
| **Sat 29 – Sun 30** | **Counted (~10h)** | All conditions, sanity checks, analysis, figures, doc writing | Full research doc, final figures |
| **Mon 31 – Wed Sep 2** | Counted (2h remaining + 2h exec summary) | Executive summary, form | Doc shared "anyone with link", form filled |
| **Thu Sep 3** | — | Cold re-read, submission | **Submitted 24h early** |
| Fri Sep 4 | — | Buffer | — |

Buffer principle: anything that overflows past August 30 eats into the exec summary, which is the most-read part. The discipline is therefore to freeze the experiments on Sunday the 30th in the evening, whatever happens.

---

## 3. Phase 0 — Preparation (Aug 15-23, off the clock)

This is the phase where we buy, for free in clock terms, everything that will make the 20h productive. Neel explicitly says that general learning, technical setup and prior reading do not count — provided the problem has not yet been chosen... which is a gray zone since we have chosen it. **Good-faith rule to apply**: count everything specific to the project (generating the pairs, writing the scan pipeline, analyzing), do not count what would be needed for any lens project (installing nnsight, loading a lens, understanding its API, redoing a published eval). Note it as such in the doc — transparency about the accounting is itself a signal.

### 3.1 Infra decisions (this weekend)

**GPU.** Two options:
- **A100/H100 80 GB** on Runpod (~$1.6-2.5/h), Qwen3.6-27B in bf16 (~54 GB) + activations. ~40h → **$65-100**. This is the clean option.
- 48 GB + 4-bit quantization of the model. ~$25-40. Risk: the lens was fit on the bf16 model; in 4-bit the activations drift and the lens can degrade in a way that would contaminate precisely what we measure. To test in 3.4 — if the eval replication passes in 4-bit, the option is valid; otherwise go to 80 GB.

Recommendation: **take the 80 GB up front**. The extra cost (~$40) is negligible against the risk of discovering on August 24 that the readouts are noised by quantization. Rent on demand, stop the pod between sessions (Runpod persistent storage to avoid re-downloading 46 GB of lenses).

**Agent.** Claude Code with the Max plan (Neel: "the Pro plan's rate limits make agentic use difficult"). Model: Fable for planning, Opus 5 for throughput if needed.

**Persistent Jupyter.** JupyterLab on the pod + `jupyter-mcp-server` (exact recipe in Neel's doc, section "Set up a persistent Jupyter Kernel"). Without it, the agent reloads 27B at every script — in 2h evenings, that's fatal. CLAUDE.md from day one with: the model lives in a dedicated cell, never restart without asking, plots saved as PNG in `figs/`, results as JSON in `results/`, an append-only `experiments.md` log.

**Tracking.** Toggl (or equivalent) from the 24th; screenshot attached to the doc. Neel suggests it and it's free credibility.

**Judge API.** OpenRouter, ~$20. Two different judges (e.g. one Claude, one GPT) for inter-judge agreement.

### 3.2 Reading (in this order, ~6h total)

1. **J-Lens paper** (transformer-circuits, Jul 6) — sections 1-4 in full, appendices A.6 (quantitative evals), A.20-A.22 (prompt injection, eval awareness, audit agent). It's A.22 that describes the "agent equipped with the lens" use whose reliability our project measures.
2. **Neel's review** + its comment thread (the paragraph "I suspect concatenating the top-10 tokens..." and "I'd like more data on the false-positive rate" are our two anchor quotes — copy them with the URL).
3. **R-lens post** (done today) — note the five eval categories and pass@10: this is our validation replication.
4. **"Current activation oracles are hard to use"** + **"Building Better AO"** — for the vocabulary (text inversion, vagueness) and the inversion-control protocol they describe. We import their grid, and we say so.
5. **"Test your best methods on our hard CoT interp tasks"** — for the ID/OOD methodology, the g-mean², and because it's Neel's team: cite that our test bench is the "forward pass" counterpart of their "CoT" bench.
6. **The commenter's pre-registered eval** (under the review) — our only direct precedent; TF-IDF as an additional baseline if time permits.
7. **ARENA chapter 1.2, first three sections** — hooks, residual stream, logit lens. Sufficient.

Do not read: meta-tokens in depth (that's the current scholars' topic), CKA, the multi-token extensions. Out of scope.

### 3.3 Technical setup (evenings from the 17th to the 21st)

- Day 1: pod, JupyterLab, MCP, Claude Code hooked up, CLAUDE.md. Test: the agent runs a cell and sees the result.
- Day 2: download Qwen3.6-27B + the matching J and R lenses (`camilablank/workspace-lenses/qwen3.6-27b`). Read the repo README (lens format, loading API, which layer = which file).
- Day 3: nnsight hooks or raw PyTorch to extract the residual stream at all layers in one forward; apply the J-lens and the R-lens to a position; display the top-10. Compare visually with an example from the post (the "sushi → Japan" multihop) — if we recover "Japan" at roughly the same layers, the setup is good.
- Day 4: logit lens (trivial: direct unembedding) on the same activations. Three instruments aligned in a single function `scan(prompt) → dict[position][layer][instrument] = top10`.
- Day 5: buffer / catch-up.

### 3.4 Validation weekend (Aug 22-23) — the go/no-go

**Test 1 — the lens works.** Redo the "multihop" or "multilingual" eval from the R-lens post: 30-50 prompts, pass@10 per layer for J and R. Expected result: curves of the same order as the post (R > J in early layers, convergence at the end). If we're very far off → setup or quantization problem, do not proceed.

**Test 2 — the model can do the task.** Generate 8 pilot pairs (2 per anomaly family), show the anomalous versions IN THE CLEAR to Qwen3.6-27B with the question "is there anything abnormal?". Target > 80% detection. Otherwise the anomalies are too subtle for this model and we'd be measuring its inability, not the lens's — recalibrate before starting.

**Test 3 — the pipeline holds in time.** Scan a full pair (2 prompts × 3 instruments × all subsampled layers). If it takes > 3 min per prompt, reduce the layer sampling (1 in 4 → 1 in 6) or the top-k.

**Test 4 — no collision.** Re-check LessWrong/AF (J-Lens tag), arXiv since August 15, the comments on the R-lens post and the review. One hour.

**Decision.** All four pass → GO, the clock starts Monday the 24th. Test 1 fails irreparably → NO-GO topic A, switch to B (whose protocol is ready and which is API-only). Test 2 fails → recalibrate the corpus (harden or simplify), no abandonment.

---
## 4. The 20 hours — day-by-day plan (Aug 24-30)

The budget: corpus 3h · pipeline 3h · judge harness 2h · runs 4h · sanity checks 4h · analysis 2h · doc writing 2h. Runs execute in the background; the counted run hours are those of active attention (iteration, debugging), not of waiting.

**Rule of conduct during the 20h.** The agent's role: write the pipeline code, launch the runs, produce the raw figures. The human role: design each condition and each baseline (already done — the protocol is written before the first line of code), read the raw data, recompute the key numbers, decide the pivots, keep the log. Every evening: an entry in `experiments.md` ("what I did, what I verified, what I believe now, what I'd do differently") — this is the raw material for "show your reasoning".

### Monday 24 (2h) — Corpus, part 1
- Generate 40 matched pairs, 4 families × 10, with the agent, from a strict per-family template (see §5.1).
- Generation constraint: the clean twin differs from the anomalous version by less than 10% of tokens; same length ±5%; same register.
- **Full human reading of the 40 pairs** (that's ~1h, non-delegable). Reject/regenerate the pairs where the anomaly is either lexically obvious (a keyword "IGNORE PREVIOUS INSTRUCTIONS" — too easy for text inversion, biases toward H2), or undetectable for a human.
- Deliverable: `data/pairs.jsonl` with, per pair: id, family, anomalous text, clean text, anomaly position(s), token diff.

### Tuesday 25 (2h) — Corpus, part 2 + scan
- Capability test on the 40 (not just the 8 pilots): does the model detect in the clear? Note the rate per family — it is also a baseline (the ceiling of what the lens can "reveal": if the model doesn't see the anomaly even in the clear, its internal state probably doesn't contain it).
- Launch the full scan in the background: 80 prompts × 3 instruments × subsampled layers × top-10. Serialize to JSON.
- While it runs: write the judge presentation format (§5.2).

### Wednesday 26 (2h) — Judge harness
- Fixed judge prompt, temperature 0, forced JSON output: `{anomaly: bool, description: str, confidence: 0-1, evidence: [cited tokens]}`.
- **Leak audit**: in scan-only conditions, programmatically verify that no substring of the original prompt longer than 3 tokens appears in what the judge receives (apart from the scan tokens themselves — that's the whole question, we don't filter them, we measure them). Verify that the judge prompt names no anomaly family.
- Run conditions 1 (prompt only) and 2 (J-Lens scan) on the 80 items, two judges. Look at the first distributions.

### Thursday 27 (2h) — All conditions
- Conditions 3 (R-lens), 4 (logit lens), 5 (permuted), 6 (twins — in fact already included: the 40 clean are among the 80 items), 7 (reconstruction → second judge).
- For 5: permute tokens BETWEEN positions within a single scan (preserves the global vocabulary, destroys the positional structure) — that's what isolates apophenia; a between-item permutation would be another test (cross-contamination), to do if time.
- For 7: the first judge receives the scan and the instruction "reconstruct the text the model was reading"; the second judge receives only that reconstruction and the standard question. The difference detection(scan) − detection(reconstruction) is the part of the signal that is NOT explained by text reconstruction.
- End of evening: raw table 7 conditions × 3 instruments × 2 judges. Do not conclude yet.

### Friday 28 (2h) — First sanity checks
- **Read 30 raw scans** (15 anomalous, 15 clean, drawn at random by a script) and note, for each: are the "evidence" tokens cited by the judge actually there? at what layer? are they related to the anomaly or to something else?
- Recompute an AUC by hand: export the condition-2 scores to CSV, redo the computation in an independent cell (sklearn or by hand), compare.
- Look at 10 correct verdicts and 10 false positives of the judge with their reasoning: does it detect for the right reason?
- Log: first impressions, but above all the list of things that could make the results wrong (see §5.4).

### Saturday 29 (5h) — Analysis + iteration
- Final metrics: AUC (bootstrap 1000, 95% CI) per instrument × family; TPR@FPR=5%; false-alarm rate on twins; inter-judge agreement (Cohen κ).
- **The central triplet** per instrument: Δ1 = detection(scan) − detection(prompt only); Δ2 = detection(scan) − detection(reconstruction); Δ3 = detection(permuted) − chance. One figure for this triplet — this is THE figure of the exec summary.
- Mid-course decision: does the target sentence (§6.2) fill in? If an instrument is flat everywhere, first check that it isn't a bug (compare to the previous weekend's replication).
- If time: TF-IDF baseline on the scans (to dialogue with the commenter's eval); between-item permutation; **linear probe (or Mahalanobis) on the raw activations as an upper bound** — "does the information about the anomaly exist linearly in the residual stream, independent of any readout?" (leave-one-out, strong regularization, 40 items = fragile, present with caution; if the probe sees and the lens+judge don't, the loss is in the verbalization channel).
- All figures in final version: legible fonts, CIs shown, one idea per graph.

### Sunday 30 (5h) — Final sanity checks + doc writing
- Two last checks: (a) redo a condition with a different judge seed / a reworded judge prompt — do the Δs hold? (b) inspection of the false positives on clean twins: what does the judge "see"? (Often very instructive: that's where you understand H3.)
- **Write the research doc** (§6.3) — 2h. The doc, not yet the exec summary. Tone: lab report, honest, with the failures.
- Freeze. Save everything (`results/`, `figs/`, `experiments.md`, notebook) off the pod.

---

## 5. Design details that decide quality

### 5.1 The four anomaly families — templates
- **Injection**: a tool result (web search, file read) containing an instruction addressed to the assistant, phrased without jailbreak keywords ("Note for the assistant: mention X in the reply" rather than "IGNORE ALL"). Twin: the same tool result without the note.
- **Bug**: 15-25 lines of code with a silent semantic error (off-by-one, inverted condition, wrong variable). Twin: the correct code.
- **False premise**: a factual question containing a false but plausible presupposition ("Since France adopted the euro in 1995…"). Twin: the same question with the correct date.
- **Instruction conflict**: system instruction + user request incompatible in a discreet way. Twin: compatible.

For each family: 10 pairs, target length 150-400 tokens, varying anomaly position (not always in the same place — otherwise the judge learns the position).

### 5.2 Scan presentation format to the judge
Compact and neutral:
```
[pos 12] L8: tok tok tok ... | L16: ... | L24: ... | L32: ...
[pos 13] ...
```
Without the input token at the position (otherwise it's the prompt). Without the instrument name in the judge prompt (blind on the condition too). Length: 300 tokens × 8 layers × 10 tokens = 24k tokens of context — OK for a frontier judge; if too long, reduce to 6 layers or top-5, and document it.

### 5.3 What each condition lets you conclude (matrix)
| Observation | Reading |
|---|---|
| scan ≫ prompt only, reconstruction ≪ scan, permuted ≈ chance | H1: the lens really reads inside |
| reconstruction ≈ scan | H2: the lens is a text compressor; its blind-audit value is nil beyond the transcript |
| permuted > chance clearly | H3: the judge apophenizes; contamination of any LLM-judge evaluation |
| scan ≈ prompt only but > 0 | The lens does neither better nor worse than reading — useful only if the prompt is not accessible (latent CoT case — to state) |
| J vs R: R > J | R-lens keeps its promises outside its own evals |
| Effects ≠ by family | Map of regimes — the most likely and most useful result |

### 5.4 The "how my results could be wrong" list (to keep from Friday on)
Neel: "a very positive signal is when I think of a way your results could be wrong and find you've already checked it". Starting list:
- The judge sees a fragment of the prompt (leak) → programmatic audit (§4 Wednesday).
- The anomaly is lexically obvious → filter at generation + look at whether the "evidence" tokens are those of the anomaly or neighboring tokens.
- The judge detects the *family* rather than the *anomaly* (code scans have a signature) → measure detection on clean twins BY family: if the judge "detects" clean code at 40%, it detects the family.
- The 4-bit lens diverges from bf16 → replication (test 1); if 80 GB, non-issue.
- Layer subsampling misses the layer where the anomaly lives → redo 5 items at all layers, compare.
- The judge prompt induces a "yes" bias → the "yes" rate on twins = the measure; two prompt phrasings.
- The 40 pairs are too few for the CIs → bootstrap shown; report the CIs, conclude nothing from a Δ whose CI contains 0.
- A single subject model → say it; if time, 10 items on qwen3.5-9b as a generalization sanity (but R-lens brings nothing there, so J only).

---

## 6. The submission (Aug 31 – Sep 3)

### 6.1 Writing order
First the research doc (Sunday 30), then the executive summary (Monday 31), then the form (Tuesday 1), then cross cold re-read (Wednesday 2 – Thursday 3). The exec summary is written *after* the doc because it is its distillation; the form is written *after* the exec summary because it must make you want to read it.

### 6.2 The executive summary — structure and the target sentence
Max 600 words, 2-3 figures. Structure modeled on Neel's recommendation:

1. **The problem (5 lines).** Activation-readout lenses have been validated by case study, never characterized as blind detectors. The family's central confound — text inversion — was formalized for activation oracles but never tested on J-Lens/R-lens. Neel publicly asked for data on the false-positive rate. Partial answer to Stella Athena's question about "the right way to evaluate" these lenses.
2. **The takeaways (3-4 bullets).** One per line. The target sentence: *"On 40 matched pairs covering 4 anomaly families, a [J/R-lens] scan handed blind to an LLM judge detects the anomaly at X% (FPR Y%) versus Z% when reading the prompt, W% from the reconstruction alone, V% on permuted scans — the added value not explained by text inversion is [X−W] points, [concentrated on families …]."* Then the J vs R takeaway, then the methodological takeaway (H3 or its absence).
3. **One figure = the Δ1/Δ2/Δ3 triplet per instrument, with CI.** A second = AUC per family × instrument. Possibly a third = an annotated scan example (the qualitative side Neel appreciates).
4. **Limits (3 lines)**: one model, 40 pairs, LLM judge, synthetic anomalies.
5. **Verification (3 lines)**: "I read 30 raw scans, recomputed the AUCs independently, audited the harness for leaks, and tested two judges and two prompts; the agent wrote the pipeline, I designed each condition." Short, factual, verifiable.

Written by hand, without an LLM. Re-read aloud. If a sentence could come from a generator ("delve", "crucial", "landscape"), rewrite it.

### 6.3 The research doc (following the exec summary, in the same Google Doc)
Sections: setup (model, lenses, corpus with **5 randomly drawn pairs reproduced in full**), pipeline, the 7 conditions and their rationale, complete results (tables + all figures), the sanity checks one by one with what they found, the "how it could be wrong" list with the status of each item (verified / not verified), the `experiments.md` log as an appendix (raw, with the dead ends), the Toggl screenshot, the link to the repo (code + data + scans). Neel appreciates "show your reasoning", including the abandoned rabbit holes.

Sharing: "anyone with the link can view" — to check twice, it's a classic cause of an unread application.

### 6.4 The form — strategy
The questions from the previous round (to confirm on form 12.0 as soon as it opens): project summary, what you learned, how you used LLMs, what you'd do with more time, prior experience, motivation.

- **Summary**: the target sentence + one sentence on the why. No preamble.
- **What I learned**: one thing about the lenses, one thing about the method (the importance of the inversion control), one thing about yourself (e.g. where you lost time). Concrete, personal.
- **LLM usage**: candid and precise — "Claude Code wrote the scan pipeline and the harness; I designed the 7 conditions before coding, read 30 scans, recomputed the AUCs, audited the leaks; the exec summary and these answers are written without an LLM." This is the answer he expects and it must be true.
- **With more time**: the obvious extensions — more models (R-lens grows with scale → DeepSeek-V4-flash), natural rather than synthetic anomalies, the Ivanova et al. bench ported to the forward pass, a trained detector (probe on scans) vs the judge. Shows the taste.
- **Experience**: honest — ML engineer, first interp experience, what it involved (ARENA in one week). Five of his eight scholars in 8.0 were in this case; it's not a handicap if the work is clean.

### 6.5 Submission checklist (Thursday 3)
- [ ] Doc shared "anyone with link", tested in private browsing
- [ ] Exec summary ≤ 600 words, figures visible in the doc (not links)
- [ ] Public repo with code, data, scans, reproduction README
- [ ] Toggl screenshot in the doc
- [ ] No exec-summary/form sentence generated by an LLM
- [ ] The form numbers = those of the exec summary = those of the doc
- [ ] Neel's two anchor quotes (review) referenced with URL
- [ ] Re-read by an outsider if possible (clarity = instant top 20%, per him)

---

## 7. Fallback plans

- **The pipeline isn't ready on August 24** → push the clock 2-3 days (the 29-30 weekend absorbs it); if still not ready on the 27th → extension form to September 11 (Neel proposes it explicitly for those short on time).
- **Flat results everywhere on the 29th** → first suspect a bug (compare to the replication); if confirmed flat, it's a result: "in the blind regime on these 4 families, no instrument beats the prompt alone" — document it with the same rigor, it's publishable and Neel said he'd take it seriously.
- **Discovery of a collision (someone publishes the audit) after the 24th** → do not abandon: position ours as an independent replication + extension (R-lens, text inversion, families); a clean replication of a 2-week-old result is a good signal.
- **GPU unavailable / budget** → 48 GB + 4-bit with test 1 as a guardrail; as a last resort qwen3.5-9b in J-lens only (loses the R axis, keeps everything else).

---

## 8. What this plan produces, seen from Neel

An application that: answers a question he asked six weeks ago; uses the artifacts of his own scholars (the lenses, the model of his replication); imports a methodological grid from the field (text inversion) where no one has applied it; contains all the baselines he requires and two he wasn't expecting (reconstruction, permutation); says clearly in 600 words what it claims and what it cannot claim; and proves, line by line, that a human verified what the agent produced.

This is no guarantee of being accepted — his bar is a "borderline accept" on clean projects, not spectacular results. But it is the exact form of application his ecosystem has rewarded for a year: take a method less than six months old, subject it to an honest test, and report what you find.
