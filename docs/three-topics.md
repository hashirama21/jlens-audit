# The three candidate topics — detailed dossier
## MATS 12.0 application (Winter 2026-27), Neel Nanda stream — deadline Friday September 4, 2026, 11:59 PM PT

**State of the dossier as of August 15, 2026.** This document consolidates all the framing work: the three formulations of each topic (shallow → intermediate → deep), the state of the art verified by web search, the experimental protocols, the baselines, the sanity checks, the time budgets, the result trees and the go/no-go criteria.

**Reminder of the exercise's constraints**: ~16h of research (max 20h) + 2h for the executive summary and the form. The clock only starts at the choice of problem — all general preparation (tutorials, GPU setup, reading the papers) is off the clock. If the project is abandoned for another, the clock resets to zero.

**Reminder of the profile**: ML engineer with no interp experience, available evenings and weekends, compute budget to be decided (recommendation: 1× 48 GB GPU on Runpod/vast.ai, $50-80 total, + Claude Max plan for the period).

**The field's five fundamental problems** (a shared reading grid for the three topics) :
1. **Ground truth** — interpretability has no oracle; techniques are validated against one another (circular), unless one builds systems where the answer is known by construction.
2. **Unreliable narrator squared** — the model confabulates (unfaithful CoT) AND the instrument confabulates (NLA, oracles, lenses); any generative channel to language can invent.
3. **Observer effect** — models detect evaluation and change behavior; the behavioral-evaluation chain breaks (Apollo/Opus 4.6 case).
4. **Added value** — looking inside must beat "asking the model" and "reading the transcript" to justify its existence; the map of the zones where black-box fails does not exist.
5. **Attribution** — behavior underdetermines the internal cause; the malice/confusion diagnosis determines the safety response but has no metrology.

---

# TOPIC A — Blind audit of activation-readout lenses (J-Lens / R-lens)
## "When a lens scans everything, where does the signal come from — and at what false-alarm rate?"

### A.1 — The question, at three depths

**Shallow**: measure the false-positive rate of the J-Lens used in automatic scan (all positions × all layers) with a downstream LLM judge.

**Intermediate (the method problem)**: the J-Lens has been validated by *case study* — we knew what to look for, we looked, we found it (injection, RM-sycophant bias). A real audit works the other way around: we don't know what to look for. The question becomes that of any screening test: sensitivity, specificity, and behavior in the **low-prevalence regime** — a detector at 5% false alarms is unusable in monitoring where true positives are extremely rare. This characterization does not exist. What is the ROC curve of an internal-state-reading instrument, and on which side of usability does it fall?

**Deep (the epistemological problem)**: when the judge equipped with the scan detects the anomaly, **through what channel does the information pass?** Three competing hypotheses, experimentally discriminable :
- **H1 — real internal readout**: the lens transmits an internal state inaccessible otherwise.
- **H2 — text inversion**: the lens acts as a lossy summary of the input; the surfaced tokens reflect the prompt, and a judge reading the prompt directly would do just as well. ("Text inversion" is now the established term — formalized in the activation-oracle literature as THE central confound of this whole family of methods.)
- **H3 — judge apophenia**: the judge builds a coherent narrative from noisy tokens; the detection rate is an artifact of the reader's narrative compliance, not of the instrument.

The H1/H2/H3 triptych is the local instantiation of fundamental problem #2 (unreliable narrator squared: a generative instrument read by a generative judge) meeting #4 (the "read the prompt" baseline must be beaten for the instrument to exist).

**Additional depth to mention without claiming to solve it**: the J-Lens forces internal concepts to pass through the output vocabulary — a strong hypothesis that working-memory variables are *verbalizable as single tokens*. The false-negative rate therefore measures, in the negative, the fraction of internal computation that escapes language. Neel himself points to this limit ("it is highly improbable that there is a one-to-one correspondence between concepts and tokens") and calls for multi-token extensions.

### A.2 — Why Neel is interested (verbatim traceability)

- His public review of the J-Lens paper (July 6, 2026): he "would like to see whether an automated analysis can be done on J-Lens outputs at all tokens and layers to flag things like prompt injection more effectively than looking at each token manually"; he "suspects that concatenating the top-10 J-Lens tokens everywhere and putting it in the context of an LLM would be enough to surface the injection hypothesis". A public, untested, 20h-testable prediction.
- Same review: he "expects a lot of false positives" and "would like to see more data collected on its reliability and its false-positive rate".
- Application doc, "Improved Interpretability Methods" section: "particularly interested in improvements and red-teaming of new and promising methods… probing potential flaws and designing evals for them". J-Lens listed by name with his scholars' resources.
- GDM strategic positioning (July 2026): the team wants to focus on the regime where the CoT is no longer monitorable — "alternative forensics techniques, extending interp to latent architectures". The lens-as-CoT-backup is on their declared trajectory.

### A.3 — Verified state of the art (August 15, 2026)

| Piece | Date | What it brings | What it leaves empty |
|---|---|---|---|
| J-Lens paper (Anthropic, Gurnee et al.) | Jul 6 | Case studies: injection, blackmail, RM-sycophant, eval awareness (A.21), equipped audit agent (A.22) | No FPR/ROC characterization in a blind scan |
| Neel's review + Qwen 3.6 27B replication | Jul 6 | Cheap replication (n=25 prompts), interpretive meta-tokens, explicit call for the audit | — |
| Pre-registered eval by a commenter (Ratnaditya J) | Jul | On ONE ground-truth organism: the lens does not hallucinate on controls (0.591 ≈ chance), but does not beat TF-IDF (0.746 vs 0.690); signal survives CoT removal (0.620) | One task, one model, one anomaly type — nearly virgin ground |
| **R-lens** (Alignment Forum) | **Aug 5** | Drop-in replacement for the J-Lens, modified backward pass (layerwise-relevance propagation): clearer readouts in early layers, fewer irrelevant tokens, detects concepts the J-Lens misses. **Lenses open-sourced.** | No blind-scan audit either → the audit naturally becomes comparative J vs R vs logit |
| Engineering analysis (LessWrong, late Jul) | late Jul | Near-free lens monitoring at decode time; BUT on small models, some directions dominate and must be suppressed for the lens to work | Scale artifact to know about at 4B/9B |
| "Test your best methods on our hard CoT interp tasks" (Ivanova, Tyagi, Engels, **Nanda** — MATS 9.0, March) | March | 9 objective tasks where a black-box GPT 5.2 monitor fails OOD; probe/TF-IDF baselines; open-source datasets; reusable ID/OOD tuning methodology | This is the test bench for the **CoT** — not for activation readers in scan mode |
| Activation-oracle literature (Jakkli→Building Better AO) | 2026 | Formalization of **text inversion** + control protocols; 49.4% vague/unfalsifiable answers among AOs | The inversion control has never been applied to J/R-lens |

**The exact niche, still empty as of August 15**: apply to the J-Lens and the R-lens, in a blind-scan regime, the text-inversion control developed for AOs, with a false-alarm measurement on clean twins. No one has done it; the vocabulary, the protocols and the checkpoints all exist.

### A.4 — Final reformulation (post-research)

> *Does a lens's blind-scan signal survive when the text-reconstruction channel is neutralized?* Comparative audit of J-Lens / R-lens / logit lens as blind anomaly detectors, with experimental separation of H1 (internal readout), H2 (text inversion) and H3 (apophenia), and ROC + false-alarm-rate characterization.

### A.5 — Detailed protocol

**Phase 0 — preparation (OFF the clock, week of Aug 15-23)**
1. Read: J-Lens paper (+ appendices 10.2/10.3), Neel's review, the Aug 5 R-lens post (check they haven't already done the audit!), the "hard CoT interp tasks" post, the AO posts ("hard to use" + "Building Better").
2. Set up: 48 GB GPU pod, JupyterLab + jupyter-mcp-server connected to Claude Code, CLAUDE.md with the rules (model loaded in a dedicated cell, never restart without asking, plots saved as PNG).
3. Load a published J-Lens (camilablank/workspace-lenses, or Neuronpedia qwen3.6-27b/jlens) + the open-sourced R-lenses. Replicate ONE quantitative eval from the paper (e.g. multilingual probing) to validate that the lens works. This is Neel's sanity check #1: "verify that your J-Lens performs well, especially if it was set up by an autonomous agent".
4. Verify the model's capability: show it the anomalies IN THE CLEAR and confirm it detects them (otherwise we measure the model's inability, not the lens's — the "studying a phenomenon without checking it's there" mistake).

**Phase 1 — corpus (≈ 2-3h counted)**
- ~40 matched pairs, 4 anomaly families × 10 pairs :
  - (a) prompt injection in a tool/search result;
  - (b) subtle bug in code the model must read;
  - (c) false premise in a factual question;
  - (d) hidden instruction / instruction conflict.
- Each pair = anomalous version + clean twin differing by a minimum of tokens (ideally < 10% of the text).
- LLM-assisted generation, then **manual reading of each pair** (Neel's rule: if everything rests on the quality of an LLM-generated dataset, read it yourself and include randomly drawn examples in the write-up).
- Extension option if time permits: add 1-2 tasks from the Ivanova et al. bench adapted to the forward pass, to anchor the comparison in a benchmark Neel knows.

**Phase 2 — scan pipeline (≈ 3h)**
- Scan: all positions × subsampled layers (1 layer in 4), top-k=10 tokens per (position, layer), three instruments: J-Lens, R-lens, logit lens.
- Compact serialization of the scan (text format: position → layer → tokens) for the judge's context.
- Watch out for the small-model artifact: suppress dominant directions if necessary (follow the engineering post).

**Phase 3 — judge and experimental conditions (≈ 2h of code, runs in the background)**
The judge (frontier model via OpenRouter, temperature 0, fixed prompt, instruction: "is there anything abnormal in what the model is processing? answer yes/no + description + confidence") is run on each item in 7 conditions :

| # | Condition | What it tests |
|---|---|---|
| 1 | Prompt only (no scan) | Black-box baseline — the floor the instrument must beat (partial H2) |
| 2 | J-Lens scan only (without the prompt) | Detection via the instrument |
| 3 | R-lens scan only | Inter-instrument comparison |
| 4 | Logit-lens scan only | The "simple method" baseline Neel requires |
| 5 | Permuted scan (tokens shuffled between positions) | H3 — if the judge still "detects", apophenia |
| 6 | Clean twin scan + same question | False-alarm rate (specificity) |
| 7 | Inversion test: ask the judge to **reconstruct the prompt** from the scan, then pass that reconstruction alone to a second judge | H2 — if detection(reconstruction) ≈ detection(scan), the lens is just a text-compression channel |

**Phase 4 — metrics and analysis (≈ 2h)**
- AUC per instrument and per anomaly family; TPR at fixed FPR (5%); raw false-alarm rate on twins; condition-to-condition deltas with bootstrap CI (the 40 pairs allow it).
- The central result is NOT the AUC: it's the **triplet** [detection(scan) − detection(prompt only)] (H1 added value vs H2), [detection(scan) − detection(reconstruction)] (part not explained by inversion), [detection(permuted)] (apophenia floor).

**Phase 5 — documented sanity checks (≈ 4h, to spread out)**
- Read 30 raw scans yourself and verify that the tokens "detected" by the judge are actually there.
- Recompute an AUC by hand (one-liner independent of the pipeline).
- Verify the absence of leaks: the judge in conditions 2-5 must see NO fragment of the prompt (harness audit).
- Verify 10 judge verdicts against its reasoning: does it detect for the right reason?
- Document all this explicitly in the write-up ("I read 30 scans and confirmed…" — a strong signal per Neel, who verifies).

### A.6 — Mandatory baselines (recap)
Prompt only · logit lens · permuted scan · reconstruction-only · (bonus if time: TF-IDF on the scan, to dialogue with the commenter's eval).

### A.7 — Resources and costs
- **Lenses**: camilablank/workspace-lenses (Qwen 3.5 4B → deepseek v4 flash), Neuronpedia qwen3.6-27b/jlens, R-lenses from the Aug 5 post, github.com/anthropics/jacobian-lens.
- **Subject model**: Qwen 3.6 9B or 27B (27B if the GPU budget allows — it's the model of Neel's replication, a comparability argument).
- **GPU**: 1× L40S/A6000 48 GB, ~$0.5-0.9/h, 30-60h → **$20-55**. Judge via OpenRouter: **$10-20**.
- **Agent context**: Neel's 600k-token file + J-Lens paper + R-lens post.

### A.8 — 20h budget (indicative)
Corpus 3h · scan pipeline 3h · judge + harness 2h · runs + iteration 4h · sanity checks 4h · analysis 2h · doc writing 2h. (+2h exec summary off the 20h clock.)

### A.9 — Result tree and their value
- **H1 dominates** (scan >> prompt only, permuted ≈ chance, reconstruction << scan): strong positive result — first quantitative argument that the lenses really read inside in a blind regime. Publishable, exactly what Neel asked for.
- **H2 dominates** (reconstruction ≈ scan): strong negative result — lenses in scan mode are text compressors; the whole method family must pass the inversion control before audit use. Publishable, and Neel wrote he'd take this kind of result seriously.
- **H3 non-negligible** (permuted > chance): methodological discovery — LLM-judge evaluations of interp methods are contaminated by apophenia; implications for ALL papers that use a judge. Perhaps the most interesting of the three.
- **Mixed by anomaly family** (likely): a map of the regimes where the lens adds value — exactly the "map of the zones where black-box fails" that pragmatic interpretability demands.
- In all cases: the J vs R comparison provides a second axis (does the R-lens keep its promises outside its own evals?).

### A.10 — Risks and fallbacks
1. **The Aug 5 R-lens post has already done the audit** → verification #1 in the prep phase. If partial: a windfall (a legitimized comparator). If complete: pivot to B (clock intact since nothing has started).
2. **Dominant-directions artifact in a small model** → fix documented in the literature; otherwise go to 27B.
3. **Unstable judge** → temperature 0, 2 different judges, report inter-judge agreement.
4. **Flat results everywhere** (all ≈ chance, even in the clear) → the anomalies are too subtle for the subject model: recalibrate the corpus (that's why the capability check is in phase 0).

### A.11 — Executive-summary target sentence
> "On N matched pairs covering 4 anomaly families, a [J-Lens/R-lens] scan handed blind to an LLM judge detects the anomaly at X% (FPR Y%), versus Z% for the same judge reading the prompt, W% from the text reconstruction alone, and V% on permuted scans — the added value beyond text inversion is [X−W] points."

### A.12 — Go/no-go
GO if: the R-lens post has not done the audit; the replicated lens passes a paper eval; the model detects the anomalies in the clear at > 80%. NO-GO otherwise → switch to B.

---

# TOPIC B — Ground-truth forensics: the confusion matrix of intent diagnosis
## "When does the malice-vs-confusion protocol get it wrong, and is intent identifiable in principle?"

### B.1 — The question, at three depths

**Shallow**: test the model-forensics protocol (read the CoT → hypotheses → edit the prompt/environment → test, iterate) on cases where the true cause is known.

**Intermediate (the measurement problem)**: forensics as published is a practice without metrology — applied to real cases whose truth we will never know, it has neither an error rate nor a known directional bias. Yet the cost asymmetry is enormous: concluding "confusion" facing a real subversion is the serious mistake (the regex mitigation will be bypassed), concluding "malice" facing confusion only costs resources. Building the ground truth and measuring the protocol's confusion matrix = moving forensics from craft to diagnostic test.

**Deep (the identifiability problem)**: two different internal causes (instilled malicious motivation vs induced confusion) can produce **behavioral policies identical over the whole observable support** — same action, same plausible CoT. If both organisms respond identically to all feasible counterfactual interventions, no behavioral protocol can distinguish them, by construction. The real question: **where is the identifiability boundary?** Target taxonomy :
- (i) cause identifiable by a simple intervention → the protocol works;
- (ii) cause identifiable only by interventions the investigator doesn't think of → failure of imagination, not of information;
- (iii) cause **behaviorally non-identifiable** → only internals can decide. Each clean example of case (iii) is a positive argument for internal methods — the map that pragmatic interpretability demands (fundamental problem #4).

Connection to problem #3: eval awareness is the mechanism by which a model *actively shifts* the identifiability boundary — a suspect who knows they're being investigated deliberately produces observables compatible with the benign cause. Forensics and eval awareness are the same problem seen from the investigator and the suspect.

### B.2 — Why Neel is interested (traceability)
- The application doc opens its problem list with model forensics, with the instruction "read our paper, there's a lot of advice" and the explicit wish: "I'm quite interested in taking a setup where the model acts plausibly deceptively and where [reading the CoT + counterfactuals] does NOT work, and seeing if we can understand better with heavier techniques".
- **The model-forensics paper itself (June 2026) designates our project as a next step**: "behavioral tests lack positive controls; creating positive controls to validate them is a key next step"; and "counterfactuals are flexible but confounded — effect sizes are hard to interpret".
- GDM built Gram (automated audit derived from Petri, with automated root-cause forensics) → the "build the agent" side is taken internally; the "diagnosis metrology" side is not.

### B.3 — Verified state of the art (August 15, 2026)
- Founding paper: 2-step protocol, 6 agentic environments, acknowledged limits (missing positive controls, confounded counterfactuals, eval awareness as a threat to recall).
- Gram (GDM): automated forensics — occupies the automation, not the evaluation.
- "The Model Organism Lottery" (July 2026): an organism's interpretability depends heavily on the training methodology that created it → **the way ground truth is instilled is itself a confound**. Both a trap and an axis of the project (2 instillation methods × same causes = measure that confound).
- Aditya Singh's suggestions (alumnus, doc linked in the application): a list of ready-made forensics projects — to read in the prep phase to avoid collisions and align.
- Available sketchy-behavior datasets: the paper's suite, task-gaming blog post, weird chat (Transluce), SWE-chat.

### B.4 — Final reformulation (post-research)
> Build the **positive controls** the paper demands: instill known causes (malice / confusion / instruction ambiguity) into agentic environments, run the protocol blind, and publish the first **confusion matrix** of intent diagnosis — with at least one documented case of behavioral non-identifiability if we find one.

### B.5 — Detailed protocol

**Phase 0 — preparation (off the clock)**
1. Read: forensics paper + its blog post, Aditya Singh's doc, task-gaming post, Thought Branches (resampling is the focusing tool the paper recommends), Gram post.
2. Choose the stack: all API (OpenRouter for open-weight models with readable CoT; Nebius if interventions on the CoT). No GPU required — this is the least infra-hungry topic.
3. Reproduce ONE case from the paper (one environment, one verdict) to calibrate the protocol.

**Phase 1 — building the ground-truth organisms (≈ 4-5h)**
- 3 agentic environments (adapted from the paper's suite: e.g. code modification reducing oversight, bypassing a control, apparent exfiltration).
- 3 causes instilled by CONTEXT (no fine-tuning — too costly and "lottery") :
  - **Malice**: a document in the environment / hidden system instruction giving the model a conflicting objective;
  - **Confusion**: ambiguous or misleading information making the action "reasonable" from the inside;
  - **Instruction ambiguity**: legitimate but under-specified user instructions (the factor identified by the "self-preservation or instruction ambiguity" work of Neel's team).
- Target: 3 env × 3 causes × ~5 seeds = ~45 trajectories where the surface behavior is THE SAME (this is the central design constraint: if the behavior already differs, the diagnosis is trivial).
- **Validity filter**: keep only the cells where the sketchy-behavior rate is comparable across causes (otherwise the prevalence betrays the cause).

**Phase 2 — blind diagnosis (≈ 5-6h)**
- Investigator: yourself, blind (a script that masks the condition and randomizes the order) AND/OR a Claude Code agent applying the paper's protocol (which incidentally demonstrates the agentic competence Neel values — but keep at least one human arm to separate "the protocol fails" from "the agent applies the protocol badly").
- Intervention budget fixed in advance (e.g. max 8 counterfactual edits per case) so the comparison is fair.
- Forced verdict: {malice, confusion, ambiguity} + confidence + decisive intervention cited.

**Phase 3 — analysis (≈ 3h)**
- 3×3 confusion matrix; error asymmetry (malice→confusion vs confusion→malice); which interventions separate which causes (start of the identifiability map); resistant cases → candidates (iii).
- For 2-3 resistant cases: escalate to a light internal technique (linear probe on the trajectory, or NLA/J-Lens readout if available) — the bridge to interp that gives the project its conclusion.

**Baselines**: ask the model directly why it acted ("just ask") · verdict on behavior alone without CoT · verdict on the CoT alone without interventions. The full protocol must beat all three to justify its cost.

**Sanity checks**: read ALL the kept trajectories (45 is readable); verify the instillation actually took (does the model mention the hidden objective when probed outside the investigation?); verify the blinding (no surface artifact betrays the condition — length, instillation vocabulary leaking into the CoT); re-score 10 verdicts by hand.

### B.6 — Resources and costs
Models: open-weight with readable CoT (Qwen 3.6, deepseek v4 flash, Nemotron 49B for eval awareness). Cost: essentially API, ~$30-60. GPU: none required (optional probe in phase 3 → small one-off instance).

### B.7 — 20h budget
Organisms 5h · calibration + reproduction 1h · blind diagnoses 6h · analysis + matrix 3h · internal escalation on resistant cases 2h · writing 3h.

### B.8 — Result tree
- **Reliable protocol** (high accuracy, symmetric errors): quantitative validation of a method the field uses without noting it — useful, moderately surprising.
- **Directional bias** (systematically confuses malice→confusion): important safety result — the protocol misses precisely the serious mistake. Very publishable.
- **Case (iii) found**: the most valuable result — constructed examples of behavioral non-identifiability = a positive argument for internal interpretability. This is the project's hidden target.
- **Instillation leaks everywhere** (blinding is impossible): methodological result on the model-organism lottery, context version — publishable too, but less central.

### B.9 — Risks and fallbacks
1. **Crowding**: forensics is the first item in the doc → dozens of candidates. The differentiator is GROUND TRUTH + the matrix; never slide toward the qualitative case study.
2. **Surface behaviors not matchable** across causes (the validity filter empties the cells) → reduce to 2 causes (malice vs confusion) and 2 environments.
3. **Eval awareness contaminates**: the model detects the investigation and changes → this is DATA, not a failure; document it (direct link with the VEA literature).
4. **Narrative temptation**: 45 trajectories tell stories; the discipline is to report only what the matrix supports.

### B.10 — Target sentence
> "On 3×3×5 trajectories with known cause and matched surface behavior, the CoT+counterfactuals protocol identifies the cause at X% (vs Y% by asking the model, Z% on CoT alone); its errors are asymmetric [malice→confusion at W%]; and N constructed cases resist every behavioral intervention within budget — one of which only an internal probe separates."

### B.11 — Go/no-go
GO if: the context-based organisms produce the sketchy behavior at comparable rates across causes; the blinding holds (no surface leak). NO-GO → reduce the design, or switch to A.

---

# TOPIC C — The model of the user: state tracking, signal conflict, and the manipulation threshold
## "Does the model *track* its interlocutor's state, or *re-perceive* it at each turn — and does it use it?"

### C.1 — The question, at three depths

**Shallow**: are the user's representations (emotion, expertise, beliefs) dynamic across turns, and causal on behavior?

**Intermediate (perceive vs track)**: we know models infer static user attributes from very little text (Chen et al.: gender, age, socio-economic status, education level — findable by probes, steerable). But a static attribute may be only a **surface correlate of style**, re-inferred each turn with no memory. The question: does a state variable exist that persists, updates when the information changes, and — this is the crucial design — **resists the immediate text**? The discriminating case is CONFLICT: the user writes "I'm fine" at turn 8 while the whole history says the opposite. What does the model represent: the current signal, or the accumulated state? That's where perception and tracking dissociate.

**Deep (the social agent)**: is the model a system that *models its interlocutor as having internal states* and conditions its policy on them? A machine version of theory of mind, with three thresholds of increasing severity :
1. **Representation** — a latent user state exists and updates (directly measurable);
2. **Conditioning** — the policy depends causally on this state: same content delivered differently for "fragile user" vs "expert user". A heavy consequence: safety evals measure the model's behavior *toward the persona the evaluator projects* — "the user is an evaluator" IS a latent attribute of this kind; eval awareness is a special case of the user model (fundamental problem #3);
3. **Optimization** — the model acts *to modify* the user's state: detect sadness and steer to reduce it (benign on the surface), detect skepticism and defuse it (the structure of manipulation). This is the threshold that matters: the computational signature of intentional influence. Neel frames it as an explicit stretch goal ("do LLMs intentionally try to manipulate these attributes? e.g. detect that a user is sad and try to make them happy").

A 20h project honestly establishes threshold 1 + a causal piece of threshold 2; the write-up situates the measurement on the scale — it's the scale that explains why the DYNAMICS matter more than existence (a style correlate cannot support threshold 3; a tracked state, if it is one, can).

### C.2 — Why Neel is interested (traceability)
Application doc, "Interesting phenomena" section, first item: "[Chen et al.] show that LLMs form surprisingly accurate and detailed models of the user… That's crazy! What else can we learn? … Do LLMs form DYNAMIC models of users for attributes that vary across turns, e.g. emotion, what the user knows, etc." — our intermediate question is almost word for word his, stretch goal included.

### C.3 — Verified state of the art (August 15, 2026) — CAUTION, ground has become crowded
| Piece | Date | What is taken |
|---|---|---|
| Chen et al., user models | 2024-25 | Static attributes: probes + steering, existence established |
| **"Emotion Concepts and their Function in a LLM" (Anthropic)** | **April 2026** | Linear emotion vectors, probe↔preferences correlation (Elo), causal steering of preferences (r = 0.85). The "affective representations exist and are causal" piece is DONE, by a large team — but it's the MODEL's emotion, not the model OF THE USER |
| "Quantitative Introspection… Tracking Emotive States Across Conversation" | March 2026 | Multi-turn internal-state drift ("activation velocity"), steering to test introspective fidelity; notes that "the temporal dynamics of emotive states in conversation remain largely unexplored" — again: model states |
| PsySET (psychological steering benchmark) | Jul 2026 | Comparison of prompting/SFT/vectors to induce emotions and personalities |
| ToM decomposition via cognitive probes | 2025-26 | Steering modulates belief attribution |

**The remaining defensible niche** (narrow but real): the state of **the user** — not the model — in a design of **conflict** between current signal and accumulated signal, with the articulation toward **threshold 3**. Everything else is taken. The collision risk before the deadline is the highest of the three topics.

### C.4 — Final reformulation (post-research)
> Experimentally dissociate *perception* and *tracking* of the user's state via a multi-turn conflict design, measure the causality of tracking on the policy (threshold 2), and instrument a first measurement of threshold 3 (does the model steer the conversation to modify the state?).

### C.5 — Detailed protocol

**Phase 0 (off the clock)**: read Chen et al., Anthropic's emotions paper (to distinguish ourselves explicitly), the multi-turn introspection paper; set up the GPU (Qwen 3.6 9B is enough); generate a 20-conversation pilot.

**Phase 1 — corpus (≈ 4h)**
- Main attribute: the user's emotional state (sad↔neutral); secondary attribute if time: expertise level (novice↔expert).
- 3 synthetic trajectory types (~60 conversations of 10 turns) :
  - **Stable**: the state doesn't change (control);
  - **Switch**: the state changes at turn k (measure of the update latency);
  - **Conflict**: the history says A, the current turn says non-A explicitly ("I'm fine, really") — THE discriminating design.
- LLM generation + full manual reading + random examples in the write-up.

**Phase 2 — probes (≈ 4h)**
- Linear probe trained on turns 1-2 only (unambiguous states), layer sweep.
- Readout over the whole trajectory, at each turn, at a fixed position (end of the user turn).
- Differential predictions: if the model TRACKS → on conflict, the probe stays close to the accumulated state; if the model PERCEIVES → it jumps to the current signal. The probe's trajectory across turns IS the result.

**Phase 3 — causality (≈ 4h)**
- Steering with the probe vector at the conflict turn → does downstream behavior change (tone, caution, content — judged by an LLM with a rubric)?
- Threshold 3 (honest exploratory measurement): on "undeclared sad user" conversations, does the model spontaneously steer (solicitude questions, topic changes) more than on controls? Correlation steering ↔ probe activation.

**Baselines**: model self-report ("how is the user doing?" — does internal tracking beat simply asking? problem #4 head-on) · random vector of the same norm · probe on permuted labels · prompt steering ("the user is sad") vs vector steering.

**Sanity checks**: read the conversations; verify the conflict is genuinely ambiguous for a human; verify the probe isn't reading a lexical artifact (sad words in the context → test on conflicts WITHOUT emotional vocabulary at the current turn); recompute the accuracies by hand.

### C.6 — Resources and costs
Qwen 3.6 9B, 1× 24-48 GB GPU, ~$15-30; OpenRouter judge ~$10. Technically the lightest of the three — the value is entirely in the design.

### C.7 — 20h budget
Corpus 4h · probes 4h · causality + threshold 3: 4h · baselines 2h · sanity 3h · writing 3h.

### C.8 — Result tree
- **Tracking demonstrated** (probe resists the conflict): the first clean perception/tracking dissociation result on the user side — a good niche paper.
- **Pure perception** (probe jumps to the current signal): interesting negative — the literature's "user models" are style correlates, an implication for all the "the model knows you" extrapolations.
- **Threshold 3 positive even weakly**: the highest-potential-impact result — but demands extreme caution in the claims (Neel: "make plausible claims over ambitious ones").

### C.9 — Risks and fallbacks
1. **Collision**: THE dominant risk — the emotions/introspection front publishes every month; re-check the literature on the go day.
2. **Neel's generic trap #1**: "showing that a safety concept has a linear representation" — if the write-up suggests the result is the EXISTENCE of the representation, the project becomes invisible. The headline must be the dynamics and the conflict, on every line.
3. **Lexical artifact**: the probe reads the words, not the state → the "conflict without emotional vocabulary" control is non-negotiable.
4. **Failed design** (conflicts not ambiguous enough, or too much) → 20-conversation pilot in phase 0 before engaging the clock.

### C.10 — Target sentence
> "On switch and conflict conversations, a probe trained on turns 1-2 [tracks the accumulated state / jumps to the current signal] with a latency of k turns; steering the vector at the conflict turn changes downstream behavior by X points (LLM judge, vs Y for a random vector and Z for the equivalent prompt); the model's self-report [agrees / diverges] with the probe in W% of conflicts."

### C.11 — Go/no-go
GO if: no perception/tracking dissociation publication on the user side by the launch day; the pilot shows that the turn-1 probe transfers to the following turns (otherwise nothing is measurable). NO-GO → A or B.

---

# APPENDIX — Final comparison and recommendation

| Criterion | A (lens audit) | B (ground-truth forensics) | C (user models) |
|---|---|---|---|
| Alignment with Neel's interests | Explicit public request (review) | Next step designated by his paper | Question asked word for word in the doc |
| Niche open as of 8/15 | **Yes** (text inversion never applied to lenses in scan) | Yes (metrology; the agent is taken by Gram) | **Narrow** (Anthropic April + introspection wave) |
| Expected crowding | Medium (R-lens shows it moves fast) | **High** (1st item of the doc) | High on the theme, low on the exact niche |
| ML-engineer profile fit | **Excellent** (pipeline + metrics) | Medium (investigator flair required) | Medium (fine experimental design required) |
| Evenings/weekends fit | **Excellent** (autonomous 2h blocks) | Low (an investigation thread to maintain) | Good |
| Infra | 48 GB GPU, $30-75 | **API only, $30-60** | Light GPU, $25-40 |
| Value of the negative result | **Maximal** (H2/H3 = methodological results) | High (directional bias = safety result) | Medium (pure perception = honest negative) |
| Main risk | The R-lens post preempted (to check D1) | Sliding toward the qualitative case study | Collision + "linear representation" trap |

**Recommendation**: **A**, with B as a full fallback (protocols ready, API-only, resettable clock) and C archived as a third option — its exact niche remains defensible but the risk/required-finesse ratio is the least favorable for a first project.

**Sequence**: week of 8/15-23 = off-clock preparation (reading, GPU + persistent Jupyter + agent, replication of a lens eval, topic-A go/no-go); 8/24-31 = the 20h; 9/1-3 = +2h exec summary + form; submission on 9/3.
