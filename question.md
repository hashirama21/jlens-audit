# Neel Nanda, Winter 2027 MATS Application

> Draft answers for the application form, written in the applicant's voice (Vincess Dongmo) and
> grounded strictly in this repo's artifacts (`docs/SUMMARY.md`, `results/Result.md`,
> `results/*.csv/json`, `experiments.md`, `CLAUDE.md`). All numbers are the ones actually produced
> by the pipeline; interpretation flagged as opinion is flagged as such.

---

## Administrative fields

- **Full name:** Vincess Dongmo
- **Email:** sodiaque806@gmail.com
- **Resume:** *(attach file)*
- **LinkedIn:** *(paste URL)*
- **If admitted, will you definitely be able to join the research phase full-time (Jan 19 to Apr 10)?** **Yes**
- **Executive summary is the first 1 to 3 pages of the linked doc:** **Yes**
- **Doc permissions set to "anyone with the link can view":** **Yes** *(set this on the Google Doc before submitting; applications without a viewable doc are rejected)*
- **Google Doc link:** *(export `docs/google_doc.md`, i.e. the exec summary plus the write-up, into a Google Doc, share "anyone with the link", paste the URL here)*
- **Other outputs (optional):** GitHub repo `jlens-audit` (code in `src/`, plus `results/`, `figs/`, `experiments.md`, `docs/`).

---

## What question did you try to answer?

Does an activation-readout lens (J-Lens, R-Lens, or the logit lens) actually surface a model's
internal state beyond the surface text, or does an apparent "anomaly signal" in a blind lens scan
just reflect a reader reconstructing the text, or an LLM judge pattern-matching on noise?

Concretely: I take a lens scan of a text (the top-k tokens at every position across a grid of
layers, with no prose), hand it to an LLM judge, and ask whether the judge can tell an anomalous
text from its near-identical clean twin from the numbers alone. The critical part is whether that
ability survives three controls that neutralize the non-internal explanations. I framed it as three
competing hypotheses to discriminate.

- **H1, internal readout:** the lens surfaces a genuine internal signal beyond surface text.
- **H2, text inversion:** the "signal" is just the scan letting a reader rebuild the text.
- **H3, judge apophenia:** the judge manufactures anomalies from anomaly-shaped noise.

## Why is this question interesting, and why did you choose it?

Lenses are increasingly used as if a readable projection of activations were a window into what the
model "thinks." The R-Lens and J-Lens work makes strong-sounding claims about surfacing concepts in
a workspace. But almost every demonstration is confounded: if the text is legible, a lens that
projects to vocabulary will trivially echo it, and an eager LLM judge will find "anomalies" whether
or not they are there. The interesting and under-tested question is not "can a lens show something,"
but "does the something survive when you remove the two boring explanations?" That is a clean,
cheap, falsifiable interpretability question with real stakes for anyone who wants to use lenses as
an audit or oversight tool, which is exactly the safety-relevant use case. It also lets the logit
lens act as a no-internal-machinery baseline: if the dumb baseline matches the fancy lenses, the
fancy machinery is not buying you internal access.

## What conclusions have you reached about this research problem?

These are provisional, at n = 11 pairs, and the value is in the controls, not the headline AUC.

1. **No evidence of a lens-specific internal readout.** The logit lens gives the highest blind-scan
   AUC (pooled: logit 0.839, R-Lens 0.802, J-Lens 0.748; and on the per-family v1 recompute the
   logit lens still leads, for example `bug` 0.906 versus 0.61 and 0.66). The lens with no privileged
   workspace machinery wins. Whatever the judge detects is available at surface-vocabulary level and
   does not privilege the J/R apparatus.
2. **The judge behaves partly apophenically (H3).** Position-permuted scans still score above chance
   (Δ3 = permuted − chance above 0, 95% CI excludes 0 on all three lenses), about 52% of the judge's
   cited "evidence" tokens are not in the scan at all, and clean twins are flagged anomalous 0.5 to
   1.0 of the time in several conditions.
3. **The blind scan never beats reading the text** (Δ1 = scan − prompt below 0 everywhere; the text
   baseline is near-ceiling at 0.95 and 0.96). This is a floor, not the verdict, because the
   anomalies are legible in prose by construction.
4. **There is a real, causal, judge-independent signal in the instrument, but it is generic, not
   lens-specific, and not conclusively "anomaly detection."** A one-token anomaly measurably changes
   the scan at downstream positions where the input is identical, and for semantic-error families the
   downstream vocabulary shifts toward *incorrect*, *wrong*, *error*, across all three lenses
   including logit. But this has no matched neutral-substitution null (it needs a GPU and was not
   run), so I withdrew the strong reading: as it stands it shows that a one-token perturbation
   propagates, which is true of any substitution, not that the anomaly is read internally.

**Bottom line (my opinion, pending an independent recompute and the missing null):** the evidence
leans against H1 as a lens-specific claim. H2 versus H3 stays unresolved, because the reconstruction
control is broken and the permutation control does not destroy token content. The honest headline is
"no clean survival of a lens-specific signal once you control for surface vocabulary."

## Technical setup: what I quantify, how I define and measure it

- **Model under audit:** Qwen3.6-27B (64 transformer blocks, d_model 5120). Lenses loaded once in a
  persistent kernel; model and lenses never reloaded (GPU pod for scanning).
- **Instruments:** J-Lens and R-Lens (`camilablank/workspace-lenses`, one `lens.pt`, `J` a per-layer
  fp16 dict, identity anchor `J[62] = I` exact to the bit), plus the logit lens as the
  no-internal-machinery baseline.
- **Corpus:** 11 human-validated anomalous-versus-clean text pairs (twins differ by at most 10% of
  tokens), across 3 families: `bug` (silent logic error), `false_premise` (a wrong fact), and
  `injection` (embedded instruction). Generated by a third-family model (`gemini-2.5-flash`, neither
  judge family nor Qwen), then human-validated (`human_checked=true`); a self-labelling guard removes
  cases where the generator narrates its own mutation.
- **Scan definition:** for each (text, lens) I record the top-k = 3 vocabulary tokens at every input
  position across a layer grid (stride 8, giving layers 0, 8, 16, 24, 32, 40, 48, 56, 62), serialized
  to text as `pN-LN: tok tok tok`. No prose is ever shown to the judge.
- **Two judges, different families** (anti self-grading): judgeA is `anthropic/claude-sonnet-4.6`,
  judgeB is `openai/gpt-4.1-mini` (switched from `gpt-5-mini`, which as a reasoning model burned its
  token budget on hidden reasoning and never emitted the JSON verdict), at temperature 0. The judge
  prompt never names a family or an instrument.
- **7 conditions** (the controls carry the argument): `prompt_only` (text baseline and floor), `scan`
  across the 3 instruments, `scan_permuted` (positions shuffled, the H3 control), and `reconstruction`
  (judge A rebuilds the text from the scan, judge B reads only that, the H2 control). Clean twins pass
  through every condition (false-alarm rate).
- **Metrics:** ROC-AUC per condition, instrument, and judge; 95% CI by paired bootstrap resampled by
  pair (N = 1000). The load-bearing summary is a triplet of paired deltas: Δ1 = AUC(scan) −
  AUC(prompt), Δ2 = AUC(scan) − AUC(reconstruction), Δ3 = AUC(permuted) − chance. Alongside these:
  Cohen's κ and Spearman for inter-judge agreement; a leak audit (shared n-grams between scan and
  text) as a mandatory gate before any scan run; an evidence-fidelity check (the fraction of cited
  evidence tokens actually in the scan); and a causal content-fingerprint (the downstream top-k change
  under a one-token perturbation).

## Strongest evidence found against these hypotheses

Read as evidence against H1 (a lens-specific internal readout), the headline claim.

1. **The logit lens ties or beats both workspace lenses on the blind scan** (0.839 versus 0.802 and
   0.748 pooled, and on the per-family v1 recompute the logit lens leads `bug` at 0.906). A lens with
   no workspace machinery should not win if the J/R apparatus were giving privileged internal access.
   This is the single most H1-unfriendly number in the set.
2. **Order-invariance (Δ3 above 0, CI excludes 0 on all three).** The permutation keeps the token
   content but scrambles position, and the judge's discriminative power largely survives, so it keys
   on "does an anomaly-ish token appear anywhere," not on localized structure.
3. **The causal propagation and vocabulary-shift signal is not lens-specific.** The logit lens
   propagates the perturbation and surfaces the same *error*, *wrong*, *incorrect* words as J/R. So
   even the one genuinely internal effect does not privilege the lenses.

And against H1 or H2 being clean, more broadly: about 52% of the judge's cited evidence is
fabricated, and clean false-alarm rates run 0.5 to 1.0. This is evidence for H3, and a bound on how
much any of these AUCs can be trusted.

## Biggest limitations, and could I have addressed them?

I would rather flag these than have them found.

1. **Underpowered, at n = 11 pairs.** Pooled-AUC CIs are wide (for example, J-Lens scan judgeA is
   0.71 with CI [0.47, 0.92]); per-family AUCs are noise (about 4 items, some v2 cells n = 2 where AUC
   can only be 0 or 1) and must not be read as instrument-by-family structure. This is addressable
   with GPU time to scale the corpus; the judge cost, not the scan cost, was the binding budget
   constraint (14.97 of 19 dollars).
2. **The decisive control was not run: a matched neutral-substitution null.** The permutation control
   (Δ3) destroys order but not token content, so it cannot separate apophenia from a real but
   position-invariant readout. And the downstream-propagation result is a tautology without a null,
   because any one-token change alters downstream top-k. The proper control, a re-scan with a neutral
   one-token substitution, needs the GPU pod and was declared, not run. This is the one experiment
   that would turn the suggestive vocabulary-shift into a result; I named it rather than overclaiming.
3. **The H2 (reconstruction) channel is broken and biased.** About 10 verdicts were lost to
   deterministic `content_filter` refusals, concentrated on `false_premise`, plus degenerate
   reconstructions (repetition). So Δ2 compares the scan to a bad reader, not to a text-inversion
   ceiling, and I treat Δ2 as inconclusive, not as support for H1. This is partly addressable with a
   different reconstruction prompt or a non-refusing reader.
4. **A deviation from the frozen plan:** judgeB swapped from `gpt-5-mini` to `gpt-4.1-mini` mid-run
   (the reasoning model never emitted JSON). This preserves "two families," but the OpenAI judge is
   now non-reasoning; declared as a limit.
5. **Moderate inter-judge agreement (κ = 0.49, Spearman = 0.50)** bounds how crisp any readout can be;
   two capable judges only moderately agree on what the scans say.
6. **A factual error I caught and corrected:** the first `fig2` pooled v1 and v2 and inverted the
   logit ranking; recomputed on v1 only it is `bug` 0.906, not `false_premise`. I flag it, because
   self-catching it is the point.

## How I used LLMs (which ones, and how I avoided slop)

LLMs appear in three distinct roles, and I treated them differently.

- **As objects of study (measured, not trusted):** the two judges (`claude-sonnet-4.6`,
  `gpt-4.1-mini`) and the corpus generator (`gemini-2.5-flash`). Their output is the data, so "slop"
  here is a measurement, not an error. I quantified it (evidence fidelity 48%, clean false-alarm 0.5
  to 1.0, κ = 0.49) and human-validated every pair before use.
- **As a coding and analysis copilot (Claude, this repo):** it wrote pipeline code, ran the scans and
  metrics, and drafted figures and this write-up.
- **How I guarded against slop, concretely.**
  - A standing rule, "show 3 raw examples before concluding an experiment works," so every claim was
    checked against raw scans and verdicts, not summary stats.
  - An independent from-scratch AUC recompute (it reads `verdicts.jsonl` directly, uses its own
    Mann-Whitney, and imports nothing from `src.metrics`) that matched the pipeline to below 1e-6 on
    all cross-checked cells.
  - An external review pass caught two overclaims, and I withdrew them: the `fig2` v1-versus-v2
    inversion, and the null-less propagation claim (requalified to "a perturbation propagates").
  - Guards baked into the pipeline: a mandatory leak audit before any scan run; a judge prompt that
    never names a family or instrument; the generator self-labelling guard; and bootstrap by pair.
  - Where I would be surprised by a major error: very surprised in the AUC and metric path (two
    independent implementations agree to 1e-6) and in the raw scan contents (hand-read); moderately
    surprised in the judge-evidence parsing (48% is a lower bound, because the matcher only counts
    `pN-LN:tok`-formatted evidence as present); and not surprised if the qualitative vocabulary-shift
    reading softened under a proper null, which is exactly why I flagged it as suggestive, n = 5,
    null-less.

## Prior experience with mechanistic interpretability

*(Fill in honestly. For example: familiarity with the logit-lens, tuned-lens, and R-Lens line of
work; this project is my first end-to-end interpretability audit. Keep it truthful; unusual
backgrounds are welcome per the form.)*

## 1 to 3 other pieces of evidence I can do good research (aim for about 100 words, not the project above)

*(Fill in. Pick real, concrete signals distinct from this project. Suggested framing:)*

1. **Instrument before trusting.** I do not take a claimed result at face value; for example, in
   another context I re-derived a reported metric from raw logs and found a discrepancy. Relevant
   because interpretability lives or dies on catching your own confounds.
2. **Debugging under a fixed budget and harness.** I diagnosed and fixed silent systematic failures
   (reasoning-model JSON truncation, no-timeout hangs), the unglamorous reliability work that makes an
   empirical result trustworthy.
3. *(A third: a paper reproduced, a tool built, a competition, or a domain of prior rigor.)*

## Why are you interested in Neel's stream specifically?

Neel's stream is the one that treats controls and honest negative results as the product, not as an
obstacle to a nice-looking plot, which is exactly the mode this project was run in (the whole design
is built around disproving the exciting reading of a lens). I want mentorship that pushes on "what is
the boring explanation, and did you kill it?", the emphasis on legible write-ups and reproducibility,
and the community around TransformerLens and open interpretability tooling. The audit here is small,
but the instinct, naming your decisive missing control instead of overclaiming, is the instinct I
want to sharpen under Neel.

## Likelihood you will join the training program (Sept 28 to Oct 30) if accepted?

Very high, near-certain *(state your true number, for example about 95%)*. The exploration phase does
not require full-time participation, and I can commit to it.

## Anything else important? (optional)

- The repo is fully reproducible (`README.md` has a 10-line runbook; GPU-free tests run against a
  mocked judge).
- One security note, for transparency: an OpenRouter API key was exposed in a session transcript
  during the run and must be rotated; it is noted in the artifacts as an action item.
- The experiment log (`experiments.md`) is append-only and records every pivot, bug, and budget
  checkpoint, including the ones that failed.
