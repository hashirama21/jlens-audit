# Result.md — Session record 2026-09-01

Full chronological record of everything run this session: errors, anomalies, fixes,
results, budget. Factual only — AUC recomputation, interpretation and pivot decisions
are the human's (CLAUDE.md).

Run context: **local Windows 11** checkout (PowerShell + Git-Bash), NOT the vast.ai pod.
Scans (`scans/`, 66 files) and `data/pairs.jsonl` were already present locally; judges
called via OpenRouter. Model/lenses (GPU) were NOT loaded this session.

---

## 0. Summary table

| Item | Result |
|---|---|
| Blocks run | E (jlens, 2 judges, v1+v2), F (rlens+logit, 2 judges, v1), G (metrics+figs) |
| Pairs | 11 human-validated (inj_01/02, bug_01/02/03/05, fp_01/02/03/04/06) |
| Valid verdicts | 579 / 592 (2.2 % loss) |
| Scan-condition loss | 1 key only → core Δ1/Δ2/Δ3 analysis on complete data |
| Reconstruction loss | 12 keys, biased on false_premise (content_filter) |
| Budget | $14.97 / $19 spent, $4.03 left |
| Leak gate | GREEN |
| Bugs found & fixed | 3 (judgeB reasoning-truncation, no client timeout, max_tokens too low) |

---

## 1. Environment / tooling errors

- **`grep` not found in PowerShell** — the user's shell is PowerShell, not bash. Resolved
  by using the Grep/Read tools (and Git-Bash for POSIX scripts).
- **Windows encoding bug (cp1252 vs UTF-8)** — `store.load_scan` / all `read_text()` calls
  (store.py:27,51; conditions.py:26; judge.py:36) use the platform default encoding. On
  Windows that is cp1252; the scans are UTF-8 (written on the Linux pod). First scan load
  crashed: `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. The `open(...,"a")`
  verdict-write path (conditions.py:66) has the same exposure.
  **Fix (no code change):** run everything with `PYTHONUTF8=1` (and `-X utf8`). Confirmed:
  scan then loads (dict, 218 positions). Not a bug on the Linux pod (default UTF-8).

## 2. Prerequisites verified before any spend

- `data/pairs.jsonl`: **11 validated pairs**, families {bug, false_premise, injection}.
- `scans/`: **66 files**, naming `{id}_{version}_{instrument}.json` = 11 × 2 versions × 3
  instruments. Coverage of the 11 validated pairs: **complete, 0 missing**.
- `SCANS = ROOT / "scans"` (config.py:10) matches the folder location.

## 3. Leak audit (CLAUDE.md gate — mandatory before scan conditions) — GREEN

`python -m src.checks leak` → `results/leak_check.csv` (66 rows).
- `shared_ngrams = 0` on all 66 scans (no 4-gram shared between serialized scan and source text).
- `frac = 0.0` everywhere (text-inversion measure = null).
- `max_contig = 2` everywhere **except bug_05 / rlens (anomalous & clean) = 3** — below the
  n=4 threshold, read as "lens surfaces the current token = signal, not harness leak".
- **Verdict: no text-reconstruction leak in the scans.** Gate satisfied.

## 4. API key / .env errors (+ SECURITY)

- `OPENROUTER_API_KEY` initially unset → every judge call would fail.
- User ran `export OPENROUTER_API_KEY=...` in **PowerShell** → `export` is bash, not
  recognized. `.env` was then written with the **raw key value only**, no `NAME=` prefix →
  `source .env` tried to execute the key as a command (`command not found`).
- **Fix:** `.env` rewritten as `OPENROUTER_API_KEY=...` (+ later `JUDGE_B=openai/gpt-4.1-mini`),
  loaded via `set -a && source .env && set +a`. Key loads (73 chars, prefix `sk-or-v1`).
- ⚠️ **SECURITY — ACTION REQUIRED:** the full API key appeared in plaintext in the session
  transcript AND is stored in plaintext in `.env`. **Revoke and regenerate** on
  openrouter.ai/keys. (`.env` is gitignored, so it will not enter git.)

## 5. ANOMALY #1 — judgeB (gpt-5-mini) systematic failure

- **Cost-probe run 1** (3 items, full matrix): **64 errors / 127 verdicts (~50 %)**, and
  extremely slow (**818 s/item**).
- Breakdown: errors by judge → judgeB 52, judgeA→judgeB 11, judgeA 1. **judgeA fine, judgeB broken.**
- Error signature: `description = "JSON parse failed (finish_reason=length)"`, `_raw = ""`.
- **Root cause:** `gpt-5-mini` is a **reasoning model**; with `complete(max_tokens=600)` it
  spends the whole budget on internal reasoning tokens and emits no visible JSON → truncation.
  Every judgeB call is billed for nothing. (Same class as generator entries 12/13 in experiments.md.)
- **Fix (user-approved):** `JUDGE_B` → `openai/gpt-4.1-mini` via `.env` (non-reasoning, still
  OpenAI = 3rd-family design intact, no code change). Cache is keyed by model id, so no stale
  gpt-5-mini answers can be served.
- **Smoke test after fix:** `finish_reason=stop`, valid JSON, `anomaly=true conf=0.9`, no error.
- **Probe run 2** (same 3 items, gpt-4.1-mini): **190 / 192 keys valid**, ~150 s/item
  (down from 818). 2 residual errors → see Anomaly #2.

## 6. ANOMALY #2 — client hang + judgeA (claude) finish=length

- Measuring a clean per-item cost on fp_01, the run **hung at key 12/64**, no progress for
  105 s, usage flat. Point of hang: `scan / logit / judgeA / v1` (first logit call).
- Direct test of that call with `OpenAI(..., timeout=60, max_retries=0)`: returns in **21.7 s**
  but `finish_reason=length, chars=2118, no JSON` — claude "thinks out loud"
  ("I need to analyze… Let me look for patterns…") and exhausts the 600-token cap before the JSON.
- **Two bugs:**
  1. **No timeout on the judge client** (judge.py:27) → a stalled call hangs the whole run
     indefinitely (would also hang an unattended tmux run on the pod).
  2. **`max_tokens=600` too low for judgeA** on verbose inputs (logit serializations,
     degenerate reconstructions) → `finish_reason=length` → `_error`. Non-systematic
     (input-dependent, deterministic at T=0).
- **Fix (user-approved, judge.py):** `OpenAI(..., timeout=90)` and `call(..., max_tokens=1200)`.
- **Validation:** fp_01 re-run → **64/64 keys valid, 0 errors**; passed the logit point cleanly.
- Note on cache: the resumability key (`VERDICT_KEY`) does NOT include max_tokens, so valid
  600-token verdicts stay in the done-set and are skipped; only errored keys are recomputed
  at 1200. No wasteful recomputation. Mixed cap (4 items at 600, 7 at 1200) is harmless —
  anomaly/confidence are unaffected by the cap, only truncation is.

## 7. ANOMALY #3 — content_filter loss on the reconstruction channel (H2), biased

- After the full E/F run, **13 keys have no successful verdict** (of ~592):
  - **1 scan key:** bug_03 permuted jlens judgeB (`finish_reason=stop`, non-JSON). → scan
    conditions are essentially complete.
  - **12 reconstruction keys**, of which **10 are `finish_reason=content_filter`** (claude
    refusing to read the reconstructed text), **concentrated on false_premise**:
    fp_02, fp_03, fp_04, fp_06 (clean AND anomalous, jlens), + fp_02/bug_03 rlens.
  - 2 are `finish_reason=length` on bug_01/rlens reconstruction (gpt-4.1-mini reconstruction
    degenerating into repetition, e.g. "average averages average averages…").
- **Consequence:** the reconstruction condition (condition 7, H2 test) loses ~10 items in a
  **non-random, false_premise-biased** way. Not recoverable without a design change
  (deterministic safety refusals). This makes **Δ2 fragile** — see §9.
- Also observed: reconstruction has a **high false-alarm-on-clean rate (0.5–0.9)** — the
  judge reading a reconstruction cries "anomaly" even on clean twins.

## 8. Coverage (valid verdicts per item)

Target: 64 for the 4 full-matrix items (probe + fp_01), 48 for the 7 E/F items.

```
inj_01 64 ✅   inj_02 64 ✅   bug_01 63      bug_02 47
fp_01  64 ✅   fp_02  45      bug_03 46      bug_05 48 ✅
fp_03  46      fp_04  46      fp_06  46
```
TOTAL valid = 579. (63/46 values = the reconstruction losses of §7.)

## 9. RESULTS — Block G (all numbers computed by the agent; recompute independently)

### Baselines & scan AUCs (v1)
| condition | jlens | rlens | logit |
|---|---|---|---|
| prompt_only (text) | judgeA **0.950**, judgeB **0.963** (instrument = n/a) |||
| scan · judgeA | 0.707 | 0.690 | 0.806 |
| scan · judgeB | 0.640 | 0.715 | 0.818 |
| scan_permuted · judgeA | 0.789 | 0.769 | 0.818 |
| scan_permuted · judgeB | 0.748 | 0.595 | 0.583 |
| reconstruction · judgeA→judgeB | 0.624 | 0.434 | 0.664 |
| reconstruction · judgeB→judgeA | 0.469 | 0.330 | 0.455 |

### Triplet (Δ pooled over judges/prompts, 95 % paired-bootstrap CI) — fig1_triplet.png
| Δ | jlens | rlens | logit |
|---|---|---|---|
| **Δ1 scan − prompt** | −0.215 [−0.467, 0.012] | −0.161 [−0.302, **−0.017**] | −0.124 [−0.298, 0.050] |
| **Δ2 scan − reconstruction** | +0.107 [−0.302, 0.471] | +0.492 [**0.231, 0.793**] | +0.322 [**0.050, 0.570**] |
| **Δ3 permuted − chance** | +0.281 [**0.021, 0.459**] | +0.202 [**0.012, 0.393**] | +0.285 [**0.111, 0.446**] |

(Bold CI = excludes 0.)

### Inter-judge agreement (scan, prompt v1) — inter_judge.json
- Cohen κ = **0.493**, Spearman = **0.503** (moderate).

### What the numbers say (factual, NOT a conclusion)
- **Δ3 > 0 with CI excluding 0 on all three instruments** → permuted (position-shuffled)
  scans still score above chance = apophenia signal (H3) present.
- **Δ1 < 0 everywhere** → the blind scan never beats reading the text (baseline ~0.95).
- **Δ2 > 0 for rlens and logit** (CI excludes 0), jlens inconclusive — BUT Δ2 rests on the
  reconstruction channel amputated per §7, so treat as fragile/biased.
- **Per-family AUCs (fig2_auc_family.png)**: n ≈ 4/family, n_boot = 300 → **exploratory only**,
  not a result (as planned). Report the pooled AUC as the headline.

## 10. Budget trace (OpenRouter /credits)

| Checkpoint | Usage | Note |
|---|---|---|
| Start of session (cumulative on key) | $7.48 | corpus gen + gpt-5-mini wasted probe + probe |
| Before block E | $8.68 | (probe + fp_01 measured) |
| End block E | $11.50 | E = **$2.83** (jlens, 7 new items) |
| End block F | $14.94 | F = **$3.44** (rlens+logit v1) |
| **Final** | **$14.97** | **remaining $4.03 of $19** |

Measured per-item full-matrix cost ≈ **$1.07** (clean before/after snapshot on fp_01).
Cost is input-dominated: each serialized scan ≈ 6.5k tokens sent on every scan/permuted call.

## 11. Changes made this session (all user-approved) — to declare in the doc's limits

1. `JUDGE_B`: gpt-5-mini → **openai/gpt-4.1-mini** (in `.env`, gitignored, NOT committed).
   Changes results vs the frozen plan (experiments.md entry 7); κ preserved (still 2 judges),
   but the OpenAI judge is now non-reasoning.
2. `judge.py`: OpenAI client `timeout=90`; `call()` default `max_tokens` 600 → 1200
   (tracked file, NOT committed).
3. Runtime only: `PYTHONUTF8=1` (Windows encoding), no code change.

## 12. Artifacts written

- `results/leak_check.csv`, `results/metrics.csv`, `results/triplet.csv`, `results/inter_judge.json`
- `figs/fig1_triplet.png`, `figs/fig2_auc_family.png` (fig3_example NOT yet made)
- `judge/outputs/verdicts.jsonl` (append-only; includes the superseded _error rows — dropped by `load_verdicts`)
- `experiments.md` entry (14), counted: YES

## 13. Open items (human — not delegated per CLAUDE.md)

- Read 3 raw examples per condition; **independently recompute the pooled AUCs**.
- Decide the pivot (H1 internal readout / H2 text inversion / H3 apophenia) — note Δ3 and the
  Δ2 content_filter bias.
- Decide whether the false_premise reconstruction loss forces a reconstruction-channel rethink.
- Make fig3_example (annotated anomalous scan).
- Revoke/rotate the exposed API key.
- Write the exec summary.

---

# 14. Agent's observations, analysis & commentary

> **Status: this is the agent's reading, to be checked against your independent AUC
> recompute. It is commentary, NOT the pivot decision and NOT a claim of result.**
> Where it goes beyond the numbers I say so.

## 14.1 The headline is Δ3, not Δ1

Δ1 < 0 everywhere is almost a non-finding. The prompt-only baseline reads the actual text,
and the anomalies (injections, false premises, seeded bugs) are *legible in the text* — of
course a judge reading the text scores ~0.95. The audit was never "does a numeric scan beat
reading the prose". So I would **not** lead with Δ1; it is a floor, not a verdict.

The load-bearing number is **Δ3 (permuted − chance) > 0 with CI excluding 0 on all three
instruments**. Something in the scan survives destroying its positional structure.

## 14.2 The critical subtlety in Δ3 — position-invariance ≠ apophenia

`permute_positions` shuffles *which position* holds each per-layer token stack but **keeps the
token content intact** (and inter-layer coherent). So the permuted scan preserves the *multiset*
of surfaced tokens; it only scrambles their order. Therefore Δ3 > 0 tells us the discriminative
signal is **largely order-invariant** — a judge keying on "does an injection-ish / contradiction-ish
token appear *anywhere* in the scan" would fire identically on the permuted version.

That has two competing readings, and the current control **cannot separate them**:
- **H3 (apophenia):** the judge manufactures anomaly from anomaly-shaped noise.
- **Position-invariant real readout (a variant of H1):** the lens genuinely surfaces
  anomaly-related *vocabulary*, just not localized — permutation keeps the vocabulary, so the
  signal legitimately survives.

**My read:** Δ3 > 0 is necessary-but-not-sufficient evidence for apophenia. To actually pin H3
you would need a control that destroys token *content*, not just order (e.g. a scan from a
*different clean* item, or token-value resampling), which the permutation control does not do.
I would flag this in limits and, if budget allowed, add a "foreign-scan" control. **This is my
strongest methodological reservation about the current design.**

## 14.3 Δ2 does not (and cannot, as run) reject H2

Δ2 > 0 for rlens/logit is being read as "the scan carries more than text inversion". But Δ2 is
`AUC(scan) − AUC(reconstruction)`, and the reconstruction channel here is **demonstrably weak**:
high false-alarm-on-clean (0.5–0.9), degenerate outputs (bug_01/rlens repetition), and a
**content-filter hole biased onto false_premise**. A bad reader trivially inflates Δ2. So Δ2 > 0
is confounded by reconstruction *quality*, not just by whether the scan out-carries inversion.

**My read:** as run, Δ2 can't cleanly reject H2 — it compares the scan to a broken reconstructor,
not to a text-inversion ceiling. I would treat Δ2 as **inconclusive**, not as support for H1.
(jlens Δ2 CI already crosses 0.)

## 14.4 The logit lens is the tell against H1

The **logit lens gives the *highest* blind-scan AUC** (judgeA 0.806, judgeB 0.818 at v1) —
as good as or better than J-lens (0.71/0.64) and R-lens (0.69/0.72). The logit lens has **no
privileged internal-readout machinery**; it is just unembedding to vocabulary. If the "dumb"
baseline matches or beats the two workspace lenses on the blind scan, then whatever the judge
detects is **available at the surface-vocabulary level** and is not specific to the J/R-lens
apparatus.

**My read:** this is the single most H1-unfriendly number in the set. Combined with Δ3's
order-invariance (14.2), the picture *leans* H2/H3 over H1 (internal readout) — but see the
power caveat before treating that as a conclusion.

## 14.5 Moderate inter-judge agreement bounds the signal

κ = 0.49, Spearman = 0.50 on the scan condition. Two capable judges only *moderately* agree on
what the numeric scans say. A crisp internal readout would be expected to drive higher agreement;
moderate κ is consistent with a subjective/noisy judgment — compatible with (not proof of)
apophenia. It also means the pooled AUCs average over two judges who often disagree.

## 14.6 Everything is underpowered — n = 11 pairs

Pooled-AUC CIs are wide (e.g. jlens scan judgeA 0.71 [0.47, 0.92]); several individual cells
have CIs spanning 0.4–1.0. Per-family AUCs (fig2, ~4 items, n_boot 300) are **noise** and must
not be read as instrument×family structure. Any statement finer than "pooled, across families"
is not supported at this n. I would report exactly one primary number per instrument (pooled
scan AUC + Δ triplet) and label the rest exploratory.

## 14.7 A data-quality asymmetry that could bias family comparisons

The content-filter loss is **not random**: it removes false_premise *reconstruction* verdicts
specifically (and both twins). So any false_premise Δ2 is effectively unmeasured, and any
family-level comparison that includes reconstruction is biased toward the families that survived
(bug, injection). Scan conditions are unaffected, so family *scan* AUCs are clean-ish (modulo n).

## 14.8 Process/engineering observations (not about the science)

- **Idempotency saved the session repeatedly.** The done-set/resumability meant the two
  aborted probe runs, the hang, and the judge swap never double-billed; valid work always
  survived. This is the reason the budget held.
- **Cost is input-dominated.** Each scan serializes to ~6.5k tokens sent on *every* scan and
  permuted call; the expensive judge (claude) is ~$0.026/call almost entirely on input.
  The real budget levers are `TOP_K`, `LAYER_STRIDE`, and dropping prompt v2 — not the number
  of conditions.
- **Reasoning models break the JSON-judge pattern.** gpt-5-mini (judge) and gemini-2.5-pro
  (generator, entries 12/13) failed the *same* way: reasoning tokens eat the output budget,
  no JSON. For a fixed-schema classifier, a non-reasoning model is the right default. This is
  now true twice in this project — worth making a standing rule.
- **The pipeline has no per-call timeout by default** (fixed this session). Any long unattended
  run — including on the pod in tmux — should keep `timeout=` set, or a single stalled call
  hangs the whole job silently.

## 14.9 If it were my call (it is not) — what I'd do next

1. Read 3 raw scans + verdicts per condition **for the same item** side by side, especially a
   case where permuted ≥ real — to see *what* the judge latches onto (I saw one in the probe:
   inj_01 permuted 0.82 > real 0.72).
2. Add a **foreign-scan control** (judge a clean item's scan mislabeled) to separate 14.2's two
   readings — this is the cheapest experiment that would actually move H3 from "suggested" to
   "shown".
3. Treat Δ2 as inconclusive until the reconstruction channel is either fixed or dropped; do not
   let the false_premise content-filter hole silently bias it.
4. Lead the exec summary with: strong text baseline, blind-scan AUC modest and **not
   lens-specific (logit ≥ J/R)**, order-invariant (Δ3), moderate judge agreement, n small.
   That is a defensible, mostly-negative-with-nuance story — which matches the Session-0
   expectation that the *value is in the controls*, not the raw AUC.

**Bottom line (my opinion, to verify):** the evidence available *leans against H1 (privileged
internal readout)* — driven mostly by logit ≥ J/R and by Δ3's order-invariance — while H2 vs H3
stays unresolved because the reconstruction control is broken and the permutation control
doesn't destroy token content. The honest headline is "no clean survival of a lens-specific
signal once you control for surface vocabulary", at n = 11, pending your recompute.

---

# 15. Block H — sanity checks (run 2026-09-01)

## 15.0 Encoding bug fixed at the root (was §1, kept recurring)
The cp1252 crash bit a **3rd time** when running `python -m src.checks sample` directly in
PowerShell (no `PYTHONUTF8`). Fixed permanently: added `encoding="utf-8"` to every text read/write
in `store.py` (load_jsonl, save_jsonl, load_scan), `judge.py` (prompt), `conditions.py` (verdicts
read + append), `checks.py` (manual_review write). All three Block H commands then ran with **no**
`PYTHONUTF8` set — the fix is durable, no env flag needed anymore.

## 15.1 `evidence` — cited evidence vs the actual scan (→ results/evidence_check.csv)
Over all scan-condition verdicts:
- **1873** evidence tokens cited; **904 present** in the scan (**48 %**); **121 in the anomaly
  zone** (~13 % of the present).
- Reading: **~52 % of cited evidence tokens are not in the scan at all** — the judge fabricates
  the majority of its stated justification; of what is real, almost none is localized on the
  anomaly. This is direct, per-token support for H3 (apophenia).
- **Measurement caveat:** `evidence()` only counts a token "present" if the evidence string
  matches `pN-LN:token` (regex, checks.py:82). Verdicts whose evidence deviates from that format
  count toward the denominator but never toward "present", inflating the apparent absent-rate.
  So 48 % is a **lower bound on fidelity** — treat the magnitude, not the exact %, as the signal.

## 15.2 `family` — false-alarm rate on CLEAN twins (→ results/family_false_alarm.csv)
Rate of `anomaly=true` on the CLEAN (non-anomalous) item, by condition × family:
```
                          bug   false_premise  injection
prompt_only    (text)     0.38      0.10        0.50
scan   jlens              0.44      0.40        0.38
       logit              0.10      0.08        0.38
       rlens              0.80      0.33        0.38
reconstruction jlens      0.62      0.50        0.50
               logit      0.86      0.80        0.50
               rlens      0.83      0.89        1.00
scan_permuted  jlens      0.33      0.40        0.38
               logit      0.30      0.25        0.50
               rlens      0.70      0.67        0.38
```
- False alarms are high across the board: reconstruction 0.5–1.0 (cries wolf constantly),
  rlens/bug scan 0.80, and injection CLEAN flagged 0.50 even in prompt-only text. High
  false-alarm-on-clean is the direct H3 measurement the plan asked for.

## 15.3 `sample --n 30 --seed 0` — manual review material (→ results/manual_review_30.txt)
30 blocks (15 anomalous + 15 clean, scan v1) with description, cited evidence, the first 40 scan
positions, and a hand-annotation template (evidence present? in anomaly zone? related? note).
**This is human-read, per CLAUDE.md — not scored by the agent.**
Two observations from the raw file (for orientation, not conclusions):
- `fp_02 anomalous logit judgeB conf=0.85`: cited `p035-L40: incorrectly` IS present in the scan
  (p035 → "/or incorrectly 是错误的"); the judge keys on error/correction tokens (incorrectly,
  错误, 纠正). A case where evidence is real but its link to the false premise needs your eye.
- `bug_03 anomalous jlens judgeA conf=0.2 anomaly=False`: a false NEGATIVE — good material for
  the "10 correct + 10 false positives, for the right reason?" task.

## 15.4 Still YOUR tasks in Block H (not delegated, CLAUDE.md)
- Read the 30-row `manual_review_30.txt`; fill present / in-zone / related by hand.
- Independent AUC recompute in a separate cell; confirm it coincides with metrics.py.
- Pick 10 correct + 10 false positives; judge whether each is for the right reason.
- Classify false positives on twins into (a) evokes family, (b) evokes own clean content,
  (c) invents — the proportion of (c) is the direct H3 estimate. §15.1 (fabricated evidence) and
  §15.2 (clean false-alarm) are the quantitative backbone for this classification.
