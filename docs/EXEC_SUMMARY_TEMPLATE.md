# [Title — one line, factual]  (≤ 600 words, 2-3 figures — TO BE WRITTEN BY HAND, NO LLM)

## The problem (≈ 5 lines)
- Lenses validated by case study, never characterized as blind detectors.
- Text inversion: formalized for activation oracles, never tested on J/R-lens.
- Neel (J-Lens review, Jul 6): [quote 1: "concatenating the top-10 tokens... would be enough"]; [quote 2: "I'd like data on the false-positive rate"]. URLs.
- Partial answer to S. Athena's question under the R-lens post ("what is the right way to evaluate these lenses").

## Takeaways (3-4 bullets, one number each)
- On 40 pairs / 4 families, blind [J/R] scan → LLM judge: AUC X [CI] (false alarm Y%), vs Z (prompt only), W (reconstruction only), V (permuted). Added value not explained by text inversion: X−W.
- J vs R: ...
- By family: ...
- Methodological (H3): ...

## Figures
- fig1_triplet.png — Δ1/Δ2/Δ3 per instrument (CI).   - fig2_auc_family.png.   - (fig3 annotated example)

## Limits (3 lines)
One model (Qwen3.6-27B); n=40; LLM judge (2 judges, κ=…); synthetic anomalies; layers subsampled 1/4; top-10.

## Verification (3 lines, factual)
I read 30 raw scans (table in appendix), recomputed the AUCs independently, audited the harness for leaks (results/leak_check.csv), tested 2 judges × 2 prompts. The agent wrote the pipeline; I designed each condition before coding.
