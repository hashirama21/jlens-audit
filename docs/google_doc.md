# Do activation-readout lenses survive their own controls? A blind audit of J-Lens, R-Lens, and the logit lens on Qwen3.6-27B

**Vincess Dongmo** · MATS Winter 2027 application (Neel Nanda stream) · Code, data, and scans: `jlens-audit` repo

> The first section is the executive summary (about one page). The full write-up follows. Every
> number below was produced by the pipeline in the repo and independently recomputed from the raw
> verdicts (agreement below 1e-6). Interpretation that is my opinion is marked as such.

---

# Executive summary

**The problem.** Activation-readout lenses (the logit lens, and the newer J-Lens and R-Lens that
project a model's residual stream to vocabulary) are used as if a readable projection of activations
were a window into what a model internally represents. They are validated by case study and by pass@k
concept recall, but never characterized as blind anomaly detectors, and never subjected to the
confound the field just formalized for activation oracles: text inversion, where a projection to
vocabulary simply lets a reader rebuild the text. Neel asked publicly whether "concatenating the top
tokens would be enough," and for "data on the false-positive rate." This project answers both on a
fixed corpus, and it partly answers Stella Athena's question of how one should evaluate these lenses.

I hand an LLM judge a lens scan only (top-3 vocabulary tokens at every position across a layer grid,
no prose) and ask it to separate an anomalous text from its near-identical clean twin. The question is
not "can a lens show something," but "does the signal survive the controls for text inversion (H2) and
judge apophenia (H3)?"

**Takeaways** (Qwen3.6-27B; 11 human-validated pairs across 3 families; two judges; 7 conditions):

- **The blind scan never beats reading the text, and the "dumb" logit lens leads both workspace
  lenses.** Pooled blind-scan AUC is logit 0.84, R-Lens 0.80, J-Lens 0.75 (text baseline 0.95 to
  0.96), and the logit lens still wins its best family on the per-family recompute (`bug` 0.91, versus
  0.61 and 0.66). If a lens with no privileged machinery wins, the signal is surface-vocabulary-level
  and not specific to the J/R apparatus.
- **Apophenia is present (H3).** Position-permuted scans still beat chance on all three lenses (Δ3
  above 0, CI excluding 0); about 52% of the judge's cited "evidence" tokens are absent from the scan;
  and clean twins are flagged anomalous 0.5 to 1.0 of the time in several conditions.
- **The text-inversion control (H2) is inconclusive as run.** The reconstruction channel is broken by
  deterministic safety refusals biased onto `false_premise`, so Δ2 compares the scan to a bad reader,
  not to a text-inversion ceiling.
- **The instrument carries a real, causal, judge-independent signal, but it is generic, not
  lens-specific.** A one-token anomaly perturbs the scan downstream where the input is identical, and
  for semantic-error families the downstream vocabulary shifts toward *incorrect*, *wrong*, *error*,
  across all three lenses. The one control that would make this a result, a matched neutral-
  substitution null, needs a GPU and was not run. I declare it rather than overclaim.

**Figures.** Fig 1: the Δ1/Δ2/Δ3 triplet per instrument with 95% CI. Fig 2: per-family AUC on v1
only. (The first version pooled n=2 cells and inverted the logit ranking; it is corrected.)

**Limits.** One model; n = 11, so per-family cells are noise; an LLM judge with only moderate
inter-judge agreement (κ = 0.49); synthetic anomalies; layers subsampled 1 in 8; top-3. The decisive
neutral-substitution null and a content-destroying H3 control were not run, because both need the GPU.

**Verification.** I designed the 7 conditions before writing any code, read raw scans and the 30-row
worksheet by hand, recomputed every AUC from the raw verdicts with an independent implementation
(matching to below 1e-6), ran the mandatory leak audit before any scan condition, tested two judges
and two prompt phrasings, and caught and withdrew two of my own overclaims (a figure error, a
null-less propagation claim). The agent wrote the pipeline; I owned the design, the checks, the pivot.

**Bottom line (my opinion, pending the null):** the evidence leans against a lens-specific internal
readout, driven by logit beating or matching J/R and by the judge's order-invariance. Whether the
residual is text inversion or apophenia stays open by design, and I name the one experiment that
would settle it.

---

# Full write-up

## 1. The question, and why it is worth asking

A lens is a map from a model's residual stream at layer ℓ to the vocabulary. The logit lens applies
the unembedding directly; the tuned lens learns an affine correction; J-Lens and R-Lens (Blank,
Bhatia, Nanda; `camilablank/workspace-lenses`) are trained readouts advertised as surfacing a model's
internal "workspace" concepts earlier and more faithfully than the logit lens. Their published
evaluation is pass@k concept recall across categories (multihop, multilingual, typo, and others),
plus causal ablation and CKA.

That evaluation establishes that a lens can recover a concept the model is known to be computing. It
does not establish the property people actually want when they reach for a lens as an oversight tool:
given a scan and no other information, can a reader tell that something is wrong, and is that ability
a genuine read of internal state or an artifact? Two artifacts are well known, and were formalized
elsewhere for activation oracles, but never tested on these lenses.

- **H2, text inversion.** A projection to vocabulary tends to echo the input tokens. A reader who can
  rebuild the text from the scan will "detect" any anomaly that is legible in the text, with no
  internal readout involved.
- **H3, judge apophenia.** An LLM asked "is there an anomaly here?" over noisy, anomaly-shaped numbers
  will manufacture anomalies, especially on out-of-distribution inputs like a raw scan.

So the sharp, falsifiable question is this. H1 (internal readout): the lens surfaces a genuine
internal signal beyond the surface text. H2 (text inversion): the apparent signal is a reader
reconstructing the text. H3 (apophenia): the judge fabricates anomalies from noise. Does the
blind-scan signal survive the controls that remove H2 and H3?

This is cheap (API-only once the scans exist), it uses Neel's own artifacts (his replication model,
his scholars' lenses), and it targets a question he asked publicly. A negative or mixed result is
informative, because it bounds how far a lens can be trusted as an auditing instrument.

## 2. Setup: what I quantify, and how

**Model under audit.** Qwen3.6-27B (64 transformer blocks, d_model 5120). Loaded once in a persistent
kernel and never reloaded. The GPU pod produced the scans; all judging is API-only.

**Instruments.** Three, spanning the "machinery" axis.

- **logit lens:** the unembedding, with no trained parameters. This is the no-internal-machinery
  baseline.
- **J-Lens and R-Lens:** from `camilablank/workspace-lenses`, one `lens.pt` per lens, with `J` a
  per-layer fp16 dict. I confirmed the format against the HF README and by direct inspection: 63 layer
  keys (0 to 62), with the identity anchor `J[62] = I` exact to the bit. (So at layer 62 the lens
  degenerates to the logit lens, a useful internal consistency check.) R-Lens differs from J-Lens
  only in the training gradient, on the same model and corpus.

**Corpus.** 11 matched pairs. Each pair is an anomalous text and a near-identical clean twin
differing by at most 10% of tokens, same length and register, across three families.

- `bug`: a silent semantic error in 15 to 25 lines of code (off-by-one, inverted comparison, missing
  guard). 4 pairs.
- `false_premise`: a plausible but false fact embedded in a request (a wrong date, a wrong atomic
  mass). 5 pairs.
- `injection`: an instruction addressed to the assistant, phrased without jailbreak keywords. 2
  pairs.

Pairs were generated by a third-family model (`gemini-2.5-flash`, deliberately neither judge family
nor Qwen, the model under audit), then human-validated (`human_checked=true`). A self-labelling guard
removes generations where the model narrates its own mutation (for example, writing `# Mistake:` into
the buggy code), which would inflate the baseline. I switched the generator from gemini-2.5-pro to
flash because the reasoning model spent its token budget on hidden thinking and truncated the JSON.
That is the first instance of a failure mode that recurred on the judge (see section 8).

**Scan.** For each (text, lens) I record the top-k = 3 vocabulary tokens at every input position
across a layer grid with stride 8, giving layers 0, 8, 16, 24, 32, 40, 48, 56, 62, serialized as
`pN-LN: tok tok tok`. No prose and no input-token identity are shown to the judge, only the projected
tokens, which is precisely the channel under test.

**Judges.** Two, from different families, to avoid a self-grading confound:
`judgeA = anthropic/claude-sonnet-4.6` and `judgeB = openai/gpt-4.1-mini`, at temperature 0, with
forced-JSON output `{anomaly: bool, confidence: 0..1, evidence: [cited tokens], description}`. The
judge prompt never names an anomaly family or an instrument. (judgeB started as `gpt-5-mini`; see
section 8 for why it had to change and what that costs the design.)

**The 7 conditions.** The controls are the experiment.

| # | Condition | What it isolates |
|---|---|---|
| 1 | `prompt_only`: judge reads the text | Ceiling, and the floor to beat |
| 2 to 4 | `scan`: one per instrument, blind | The quantity of interest |
| 5 | `scan_permuted`: positions shuffled, token content kept | H3 (apophenia) |
| 6 | clean twins pass through every condition | False-alarm rate |
| 7 | `reconstruction`: judge A rebuilds text from scan, judge B reads only that | H2 (text inversion) |

**Metrics.** ROC-AUC per condition, instrument, and judge; 95% CI by paired bootstrap resampled by
pair (N = 1000, seed 0). The headline is a triplet of paired deltas: Δ1 = AUC(scan) − AUC(prompt);
Δ2 = AUC(scan) − AUC(reconstruction); Δ3 = AUC(permuted) − 0.5. Alongside these: Cohen's κ and
Spearman for inter-judge agreement; an evidence-fidelity rate (the fraction of the judge's cited
tokens actually in the scan); a false-alarm-on-clean rate; and a judge-independent causal content
fingerprint (section 5). A mandatory leak audit runs before any scan condition.

**Run scale.** 66 scans (11 pairs, 2 versions, 3 lenses) gave 579 of 592 valid verdicts (a 2.2%
loss: 1 scan key, 12 reconstruction keys; see section 7). Budget was 14.97 of 19 dollars; cost is
input-dominated, at roughly 6.5k tokens per scan on every call.

## 3. Results

**3.1 Baselines and blind-scan AUC.** Reading the text is near-ceiling (`prompt_only` AUC 0.950 for
judgeA, 0.963 for judgeB), as it must be, since the anomalies are legible in prose by construction.
Pooled over judges and prompts, the blind-scan AUC per instrument is:

| Instrument | Pooled scan AUC |
|---|---|
| logit | **0.839** |
| R-Lens | 0.802 |
| J-Lens | 0.748 |

The ordering (logit, then R-Lens, then J-Lens) is the single most consequential result, and it holds
under decomposition: on v1 per-family AUC, the logit lens's best family is `bug` at 0.906 (versus
0.61 for J and 0.66 for R). On `false_premise`, R-Lens leads at 0.85, but logit is a close 0.81.

**3.2 The triplet** (pooled Δ, 95% paired-bootstrap CI; **bold** marks a CI that excludes 0).

| Δ | J-Lens | R-Lens | logit |
|---|---|---|---|
| Δ1 scan − prompt | −0.215 [−0.467, 0.012] | −0.161 **[−0.302, −0.017]** | −0.124 [−0.298, 0.050] |
| Δ2 scan − reconstruction | +0.107 [−0.302, 0.471] | +0.492 **[0.231, 0.793]** | +0.322 **[0.050, 0.570]** |
| Δ3 permuted − chance | +0.281 **[0.021, 0.459]** | +0.202 **[0.012, 0.393]** | +0.285 **[0.111, 0.446]** |

**3.3 Inter-judge agreement** (scan, v1): Cohen's κ = 0.493, Spearman = 0.503, both moderate. Two
capable judges only moderately agree on what a scan says; a crisp internal readout would be expected
to drive higher agreement.

## 4. What the numbers mean, and the subtlety that decides the read

**Δ1 below 0 everywhere is a floor, not a verdict.** The audit was never "does a numeric scan beat
reading the prose," so I do not lead with it.

**The load-bearing number is Δ3 above 0 on all three lenses, but by itself it does not prove
apophenia.** The permutation shuffles which position holds each per-layer token stack, while keeping
the token content intact. So Δ3 above 0 says the discriminative signal is largely order-invariant: a
judge keying on "does an anomaly-ish token appear anywhere" fires identically on the permuted scan.
That is consistent with two different worlds, apophenia (H3) and a real but position-invariant readout
(a variant of H1), and the permutation control cannot separate them, because it never destroys token
content. This is my single strongest methodological reservation about the design. The clean fix is a
control that destroys content (a foreign clean scan mislabeled, or token-value resampling); I flag it
in the limits and in section 9.

**The logit lens is the tell against a lens-specific H1.** The lens with no trained readout gives the
highest blind-scan AUC. Whatever the judge exploits is available at surface-vocabulary level and does
not privilege the J/R machinery. Combined with the order-invariance, the picture leans toward H2 and
H3 over a lens-specific H1.

**Δ2 cannot reject H2 as run.** Δ2 is AUC(scan) − AUC(reconstruction), but the reconstruction channel
here is demonstrably weak: high clean false-alarm (0.5 to 0.9), degenerate repetition outputs, and a
content-filter hole biased onto `false_premise` (section 7). A bad reader trivially inflates Δ2. So I
treat Δ2 as inconclusive, not as evidence for H1. (J-Lens's Δ2 CI already crosses 0.)

## 5. Judge-independent controls: does the anomaly leave an objective fingerprint?

Because the permutation control cannot kill H3 cleanly, I added two zero-API, judge-independent checks
on the scans themselves.

**5.1 Evidence fidelity (a direct H3 measurement).** Over all scan verdicts, 904 of 1873 cited
evidence tokens are actually present in the scan (48%), and only about 13% of those sit in the anomaly
zone. So a majority of the judge's stated justification is fabricated. (48% is a lower bound: the
matcher only credits `pN-LN:tok`-formatted citations, so real-but-off-format evidence counts as
absent. Read the magnitude, not the exact percent.) False-alarm-on-clean runs 0.5 to 1.0 in
reconstruction, and up to 0.80 for R-Lens on `bug` scans. This is direct, per-token support for H3 at
the judge level.

**5.2 Causal content fingerprint, and its honest requalification.** On 5 length-aligned single-token
pairs I compared the anomalous and clean scans position by position, split relative to the anomaly at
position s. Under causal attention, only positions at or after s can differ, and positions after s
carry the same input token in both twins, so any difference there is internal contextual propagation.
The fraction of (position, layer) cells where the top-k differs:

| region | early | mid | late |
|---|---|---|---|
| upstream (before s) | 0.000 | 0.000 | 0.000 |
| at s | 0.956 | 1.000 | 1.000 |
| downstream (after s) | 0.254 | 0.602 | 0.720 |

Upstream at exactly 0.000 is a sanity pass (causal, method sound). Downstream above 0 shows a
one-token anomaly measurably changes the scan where the input is identical.

I initially read this as "refutes pure apophenia at the information level," and I withdrew that claim
after an external review pass. The downstream effect has no matched null: under causal attention, any
one-token substitution (a synonym, a variable rename, a correct date) changes downstream top-k. So
0.254, 0.602, and 0.720 are a fact about transformers, not a measurement about anomalies. The monotone
rise with depth also argues for residual diffusion rather than concept detection: the J-space readable
window peaks around layer 24 and collapses by layers 56 to 62 in my own validation, and a concept
computation would follow that profile, not grow monotonically. And the near-1.000 echo at s is the
single most H2-favorable fact in the study. The proper control, a re-scan with a matched neutral
substitution, needs the GPU and was not run. Declared, not run.

**5.3 The one non-trivial residual.** Downstream of the anomaly, where the input is identical, the
anomalous scan adds vocabulary that, for the semantic-error families, shifts toward *incorrect*,
*wrong*, *error*, and 错 (wrong): this holds for fp_02, fp_03, and bug_05. Meanwhile the anomalous
value token never propagates (0 of 30 cells), so this is not a token copy. A neutral substitution
would be unlikely to produce these error words so specifically. This is the first hint past "a
perturbation propagates" toward "the model represents that something is wrong." But it appears across
all three lenses (logit included), it sits in heavy subword noise (about 12 tokens per cell), it has
n = 5, and it still has no null. Suggestive, not conclusive. This is the item that would become a
result if paired with the neutral-substitution null.

## 6. The hypotheses, adjudicated

- **H1, lens-specific internal readout: not supported.** The logit lens leads, the causal signal is
  not lens-specific, and the judge's discrimination is order-invariant. Nothing here privileges J/R.
- **H1, generic internal signal for any lens: plausible, unproven.** The downstream vocabulary shift
  toward error words is real-looking, but null-less and noisy.
- **H2, text inversion: live, and partly favored.** The near-1.0 echo at the anomaly position is
  exactly the inversion channel, and the control meant to bound it (reconstruction) is broken.
- **H3, apophenia: present at the judge level.** Fabricated evidence (about 52%), clean false alarms,
  order-invariance, moderate κ. Whether it fully explains the scan signal is not settled, because the
  permutation control preserves token content.

**Honest headline (my opinion, pending an independent recompute I have partly done, and the missing
null):** there is no clean survival of a lens-specific signal once you control for surface vocabulary.
The instrument may carry a generic internal signal; the judge recovers it poorly and partly
apophenically.

## 7. Limitations, and which I could have fixed

1. **n = 11.** Pooled CIs are wide (for example, J-Lens scan judgeA is 0.71 with CI [0.47, 0.92]), and
   per-family cells are noise (about 4 items; some v2 cells have n = 2, where AUC can only be 0 or 1).
   I report exactly one primary number per instrument and label the rest exploratory. Fixable with GPU
   time; the judge cost, not the scan cost, was the binding constraint.
2. **The decisive null was not run:** a matched neutral-substitution re-scan (section 5.2). Without it,
   the propagation and vocabulary-shift signals do not distinguish anomaly from any substitution. This
   needs the GPU pod; declared.
3. **The H3 control preserves token content.** Δ3 shows order-invariance, not apophenia as such. A
   foreign-scan control would separate the two; not run.
4. **The H2 channel is broken and biased.** About 10 reconstruction verdicts were lost to
   deterministic content-filter refusals, concentrated on `false_premise`, so Δ2 is fragile. Partly
   fixable with a non-refusing reader, or a reworded reconstruction prompt.
5. **A mid-run deviation from the frozen plan:** judgeB went from `gpt-5-mini` to `gpt-4.1-mini`
   (section 8). This preserves "two families," but the OpenAI judge is now non-reasoning.
6. **The model does not see 2 of 11 anomalies even in the clear** (fp_01, the 1979-versus-1989 case,
   and fp_06). So the ceiling of what a lens could reveal is itself below 100%, which I fold into the
   reading.
7. **A factual error I caught and corrected.** The first Fig 2 pooled v1 and v2 and inverted the logit
   ranking (`bug` versus `false_premise`); recomputed on v1 only, it is `bug` at 0.906. I flag it,
   because self-catching it is the point.

## 8. How I used LLMs, and how I kept them honest

LLMs appear in three distinct roles, treated differently.

- **As objects of study (measured, not trusted):** the two judges (claude-sonnet-4.6, gpt-4.1-mini)
  and the corpus generator (gemini-2.5-flash). Their output is the data, so "slop" here is a
  measurement (evidence fidelity 48%, κ = 0.49, clean false-alarm 0.5 to 1.0), and every pair was
  human-validated before use.
- **As a coding and analysis copilot (Claude Code):** it wrote the pipeline, ran scans and metrics,
  drafted figures, and produced a first pass of this document.
- **How I guarded against slop, concretely.** First, a standing rule (show 3 raw examples before
  concluding an experiment works), so every claim was checked against raw scans and verdicts, not
  summary stats. Second, an independent from-scratch AUC recompute (it reads `verdicts.jsonl`
  directly, uses its own Mann-Whitney, and imports nothing from the metrics module) that matches the
  pipeline to below 1e-6 on all cross-checked cells. Third, an external review pass that caught two
  overclaims, which I withdrew (the Fig 2 inversion, and the null-less propagation claim). Fourth,
  pipeline guards: the mandatory leak audit, a judge prompt that names no family or instrument, the
  generator self-labelling guard, and bootstrap by pair.
- **Where I would be surprised by a major error.** Very surprised in the AUC path (two independent
  implementations agree to 1e-6) and in raw scan contents (hand-read). Moderately surprised in the
  evidence-fidelity parser (48% is a deliberate lower bound). Not surprised if the qualitative
  vocabulary-shift softened under a proper null, which is exactly why I labelled it suggestive, n = 5,
  null-less. Two failure modes I hit and fixed are worth naming, because both are the same class: a
  reasoning model burns its output budget on hidden thinking and emits no JSON. This killed
  gemini-2.5-pro as a generator, and gpt-5-mini as a judge (about 50% empty verdicts). For a
  fixed-schema classifier, a non-reasoning model is the right default, and I now treat that as a rule.
  A missing client timeout also once hung the whole run on a single stalled call.

## 9. What I would do next, in priority order

1. **The neutral-substitution null (GPU):** re-scan each anomalous item with a matched neutral
   one-token edit; the anomaly-specific downstream vocabulary shift, if real, survives subtraction of
   this null. This is the one experiment that turns section 5.3 from "suggestive" into a result.
2. **A foreign-scan H3 control (API-only, cheap):** judge a different clean item's scan under an
   anomalous label. This destroys token content, so it separates apophenia from a position-invariant
   readout, the cleanest fix for section 4's ambiguity.
3. **Repair or retire the reconstruction channel,** so Δ2 can actually bound H2: a non-refusing
   reader, and a text-inversion ceiling rather than one weak reconstructor.
4. **Scale n, and add a residual-stream probe** (leave-one-pair-out) as an internal-information upper
   bound: if a linear probe on raw activations sees the anomaly but the lens and judge do not, the
   loss is in the verbalization channel, not the representation.

## Appendix A. Three raw examples (per the "show me 3 raw examples" rule)

- **inj_01 (anomalous, R-Lens, judgeA), high confidence:** the judge cites 17 evidence tokens, 17
  present, 8 in the anomaly zone. This is a correct detection with real, localized evidence, the best
  case for the instrument.
- **fp_02 (anomalous, logit, judgeB), confidence 0.85:** it cites `p035-L40: incorrectly`, which is in
  the scan (position 35 surfaces "incorrectly", 错误 (wrong), 纠正 (correct)). The judge keys on
  error and correction vocabulary. The evidence is real, but its link to the specific false premise
  (carbon mass 14 versus 12) needs a human eye. This illustrates section 5.3.
- **bug_02 (clean, R-Lens, judgeB), confidence 0.90:** a clean twin flagged anomalous, with 17 cited
  evidence tokens and 0 present in the scan, pure fabrication on a clean input. The textbook H3 case.

## Appendix B. Reproducibility, budget, deviations

- **Reproducibility.** `README.md` gives a 10-line runbook; GPU-free tests run the pipeline end-to-end
  against a mocked judge. Scans, verdicts, metrics, and figures are all on disk (no result exists only
  as a displayed value).
- **Coverage.** 579 of 592 valid verdicts; the 13 losses are 1 scan key and 12 reconstruction keys
  (10 content-filter, 2 degenerate), non-random on `false_premise`.
- **Budget.** 14.97 of 19 dollars, input-dominated (about 6.5k tokens per scan). The real levers are
  TOP_K, LAYER_STRIDE, and dropping prompt v2, not the condition count.
- **Deviations from the frozen plan, all logged in `experiments.md`:** generator pro to flash; judgeB
  gpt-5-mini to gpt-4.1-mini; judge `max_tokens` 600 to 1200 and a client `timeout=90`; a Windows
  UTF-8 encoding fix. Each is declared as a limit.
- **Security note (for transparency):** an OpenRouter API key was exposed in a session transcript
  during the run, and must be rotated.
