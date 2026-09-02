"""Render the (DRAFT) executive summary to results/executive_summary.pdf using matplotlib
PdfPages (no external deps). Text is flowed across A4 pages; fig1/fig2 embedded as image pages.
Content = agent synthesis, provisional (pivot + final exec summary are the human's, CLAUDE.md).
"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from src.config import RESULTS, FIGS

PAGE = (8.27, 11.69)  # A4 portrait inches
TOP, BOT, LEFT = 0.945, 0.06, 0.08
LH = 0.0160           # line height (figure fraction)
WRAP = 100            # body wrap width (chars)

# ---- content: list of (kind, text) ----
# kinds: h1, h2, body, bullet, space, hr
C = [
 ("h1", "Blind Audit of J-Lens / R-Lens / Logit-Lens on Qwen3.6-27B"),
 ("sub", "Executive Summary (DRAFT - agent synthesis, provisional). Final exec summary & pivot are the human's."),
 ("hr", ""),
 ("h2", "1. Question and hypotheses"),
 ("body", "Does a lens's signal in a blind scan survive when the text-reconstruction channel is "
          "neutralized? Three competing hypotheses are discriminated:"),
 ("bullet", "H1 - internal readout: the lens surfaces a genuine internal signal, beyond surface text."),
 ("bullet", "H2 - text inversion: the apparent signal is just the scan letting a reader reconstruct the text."),
 ("bullet", "H3 - judge apophenia: the judge manufactures 'anomalies' from anomaly-shaped noise."),
 ("space", ""),
 ("h2", "2. Method"),
 ("body", "11 human-validated anomalous/clean text pairs (families: bug, false_premise, injection). "
          "Each pair is scanned by 3 instruments (J-lens, R-lens, logit lens) into per-position, per-layer "
          "top-k token stacks. Two LLM judges (judgeA = claude-sonnet-4.6; judgeB = gpt-4.1-mini) score "
          "each scan for anomaly + confidence, blind to any anomaly family or instrument name. Seven fixed "
          "conditions include the controls that carry the argument: prompt-only (text baseline), blind scan, "
          "scan_permuted (positions shuffled = H3 control), and reconstruction (judge A rebuilds text from "
          "the scan, judge B reads only that = H2 control). A leak audit gates every scan run."),
 ("space", ""),
 ("h2", "3. Headline results (all figures/numbers computed this session; see Result.md)"),
 ("body", "Leak gate GREEN: 0 shared 4-grams across 66 scans; no text-reconstruction leak. 579/592 valid "
          "verdicts (2.2% loss). Independent from-scratch AUC recompute coincides with the pipeline to <1e-6."),
 ("bullet", "Text baseline is near-ceiling: prompt-only AUC 0.95 (judgeA) / 0.96 (judgeB)."),
 ("bullet", "Pooled blind-scan AUC per instrument: logit 0.84  >=  rlens 0.80  >=  jlens 0.75 (n=11/11)."),
 ("bullet", "Triplet (dAUC, 95% paired-bootstrap CI), pooled over judges/prompts:"),
 ("bullet", "   D1 scan - prompt:  jlens -0.21 [-0.47,0.01] | rlens -0.16 [-0.30,-0.02] | logit -0.12 [-0.30,0.05]"),
 ("bullet", "   D2 scan - reconstruction:  jlens +0.11 [-0.30,0.47] | rlens +0.49 [0.23,0.79] | logit +0.32 [0.05,0.57]"),
 ("bullet", "   D3 permuted - chance:  jlens +0.28 [0.02,0.46] | rlens +0.20 [0.01,0.39] | logit +0.29 [0.11,0.45]"),
 ("bullet", "Inter-judge agreement (scan, v1): Cohen kappa 0.49, Spearman 0.50 (moderate)."),
 ("bullet", "Evidence fidelity: ~52% of tokens the judge CITES are absent from the scan; ~13% of the present "
            "ones fall in the anomaly zone. High false-alarm on CLEAN twins (reconstruction 0.5-1.0)."),
 ("space", ""),
 ("h2", "4. What the data asserts (factual)"),
 ("bullet", "D3 > 0 with CI excluding 0 on all three instruments: shuffling positions does NOT destroy the "
            "discriminative signal -> a large part of it is order-invariant."),
 ("bullet", "D1 < 0 everywhere: the blind scan never beats reading the text."),
 ("bullet", "The logit lens - which has no privileged internal-readout machinery - gives the HIGHEST blind-scan "
            "AUC, matching/beating the two workspace lenses."),
 ("bullet", "The judge fabricates the majority of its cited evidence and flags clean twins often."),
 ("space", ""),
 ("h2", "4b. Content control (judge-independent, causal) - the decisive addition"),
 ("body", "Does the anomaly leave an objective fingerprint in the scan (not just via the judge)? On the 5 "
          "length-aligned single-token pairs, compare the anomalous vs clean scan top-k per position, split "
          "by region relative to the anomaly at position s. Under causal attention only positions >= s can be "
          "affected; positions > s carry the SAME input token in both twins, so any difference there is the "
          "anomaly's internal contextual propagation. Fraction of (position, layer) cells that differ:"),
 ("bullet", "upstream (<s):   0.000 / 0.000 / 0.000  (early/mid/late) - causal sanity check PASSED (exactly 0)."),
 ("bullet", "at (=s):         0.956 / 1.000 / 1.000  - trivial: the input token itself differs (echo). Ignore."),
 ("bullet", "downstream (>s): 0.254 / 0.602 / 0.720  - SAME input token -> internal propagation of the anomaly."),
 ("body", "So a one-token anomaly measurably changes the lens scan at downstream positions (identical input "
          "there) in 25% of early-layer and 60% of mid-layer cells - an objective, causal, judge-independent "
          "fingerprint, past the late-layer echo. It is NOT lens-specific: the logit lens propagates as much "
          "(0.20/0.64 early/mid) as J/R. Caveat: 'differ' means the top-k changed, not that it changed in an "
          "anomaly-INTERPRETABLE way - it proves the information exists and propagates, not that the judge reads it."),
 ("space", ""),
 ("h2", "5. Bugs and anomalies encountered (and fixed)"),
 ("bullet", "judgeB (gpt-5-mini) reasoning-truncated EVERY JSON verdict (finish_reason=length, empty) -> "
            "switched to gpt-4.1-mini (non-reasoning, still OpenAI family)."),
 ("bullet", "Judge client had no timeout -> a stalled call hung the whole run; max_tokens=600 too low for "
            "claude on verbose scans -> added timeout=90 and raised to 1200."),
 ("bullet", "content_filter: claude refuses to read some reconstructed texts, biased onto false_premise "
            "(~10 lost keys) -> makes D2 fragile."),
 ("bullet", "Windows cp1252 vs UTF-8 crash on every scan read -> fixed at the root (encoding='utf-8')."),
 ("space", ""),
 ("h2", "6. Limitations / caveats"),
 ("bullet", "n = 11 pairs: pooled CIs are wide; per-family AUCs are exploratory noise."),
 ("bullet", "The permutation control preserves token CONTENT (only order is shuffled), so D3>0 cannot separate "
            "apophenia (H3) from a real-but-position-invariant readout. This is the key design gap."),
 ("bullet", "D2 compares the scan to a BROKEN reconstructor (high false-alarm, degenerate outputs, "
            "content_filter hole) -> it cannot cleanly reject H2."),
 ("bullet", "Judge B was changed mid-project (gpt-5-mini -> gpt-4.1-mini); results differ from the frozen plan."),
 ("space", ""),
 ("h2", "7. Provisional conclusion (AGENT OPINION - to be verified, not the pivot)"),
 ("body", "Two-level picture. The INSTRUMENT (scan) carries a real, internal, causal signal: the content "
          "control shows a one-token anomaly propagates to downstream early/mid layers, judge-independently "
          "(causal sanity passed). This refutes 'pure apophenia' at the INFORMATION level - anomalous and "
          "clean scans are objectively distinguishable - and shows the signal is not merely surface text (H2). "
          "BUT it is NOT lens-specific (logit propagates as much as J/R), and the JUDGE exploits it poorly: "
          "blind-scan AUC is modest and never beats reading the text, is highest for the 'dumb' logit lens, "
          "shows order-invariant behaviour (D3), fabricates most cited evidence, and false-alarms on clean "
          "twins. Honest headline: a real internal signal exists and propagates, but it is not privileged to "
          "the J/R-lens machinery and is not cleanly recovered by the judge, at n=11. Apophenia is a valid "
          "critique of the judge, not of the scan's information content."),
 ("space", ""),
 ("h2", "8. Potential leads / next steps"),
 ("bullet", "Content control DONE and POSITIVE (see 4b): the anomaly propagates internally. Next, test whether "
            "the downstream change is anomaly-INTERPRETABLE (not just any reshuffle) - e.g. does an anomaly-"
            "trained probe read it, or does the surfaced vocabulary shift toward the anomaly semantics."),
 ("bullet", "Foreign-scan control (judge a different clean item's scan) to bound the JUDGE-side apophenia "
            "directly, now that the scan is known to contain real signal."),
 ("bullet", "Fix or drop the reconstruction channel before trusting D2 (content_filter + degeneracy)."),
 ("bullet", "Classify ALL false positives (not just the 5 in the seed-0 sample) into (a) evokes family / "
            "(b) evokes own content / (c) invents; proportion (c) is the direct H3 estimate."),
 ("bullet", "Scale n beyond 11 if budget allows; report only pooled AUC + triplet as primary."),
 ("space", ""),
 ("h2", "9. Budget and artifacts"),
 ("body", "Spend $14.97 / $19. Artifacts: results/{metrics,triplet,inter_judge,leak_check,evidence_check,"
          "family_false_alarm,blockH_worksheet}, figs/fig1_triplet.png, fig2_auc_family.png, "
          "judge/outputs/verdicts.jsonl, Result.md (full record), experiments.md entry (14)."),
]

STYLE = {
 "h1":   dict(size=14.5, weight="bold"),
 "sub":  dict(size=8.3, style="italic", color="#555555"),
 "h2":   dict(size=11, weight="bold", color="#1a3a6b"),
 "body": dict(size=8.6),
 "bullet": dict(size=8.6),
}

pdf = PdfPages(str(RESULTS / "executive_summary.pdf"))
def savepage(f):
    pdf.savefig(f)

def new_page():
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    return fig, TOP

fig, y = new_page()
def emit(x, y, s, **kw):
    fig.text(x, y, s, ha="left", va="top", family="DejaVu Sans", wrap=False, **kw)

for kind, text in C:
    if kind == "space":
        y -= LH * 0.6; continue
    if kind == "hr":
        fig.add_artist(plt.Line2D([LEFT, 0.92], [y, y], color="#cccccc", lw=0.8, transform=fig.transFigure))
        y -= LH; continue
    if kind in ("h1", "sub", "h2"):
        if y < BOT + 3*LH:
            savepage(fig); plt.close(fig); fig, y = new_page()
        if kind == "h2": y -= LH*0.5
        for ln in textwrap.wrap(text, 92) or [""]:
            emit(LEFT, y, ln, **STYLE[kind]); y -= LH*(1.35 if kind=="h1" else 1.05)
        y -= LH*0.25
        continue
    # body / bullet
    st = STYLE[kind]
    indent = LEFT + (0.02 if kind == "bullet" else 0)
    prefix = "- " if kind == "bullet" else ""
    wrapped = textwrap.wrap(text, WRAP - (2 if kind=="bullet" else 0)) or [""]
    for i, ln in enumerate(wrapped):
        if y < BOT:
            savepage(fig); plt.close(fig); fig, y = new_page()
        emit(indent, y, (prefix if i == 0 else "  ") + ln, **st)
        y -= LH
savepage(fig); plt.close(fig)

# ---- image pages ----
for img, cap in [("fig1_triplet.png", "Figure 1 - H1/H2/H3 triplet per instrument (95% paired-bootstrap CI)"),
                 ("fig2_auc_family.png", "Figure 2 - AUC per family x instrument (exploratory, n~4/family)")]:
    p = FIGS / img
    if not p.exists():
        continue
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.93, cap, ha="center", va="top", size=10, weight="bold")
    ax = fig.add_axes([0.08, 0.30, 0.84, 0.55]); ax.axis("off")
    ax.imshow(mpimg.imread(str(p)))
    savepage(fig); plt.close(fig)

d = pdf.infodict(); d["Title"] = "J/R/Logit-Lens Blind Audit - Executive Summary (DRAFT)"
pdf.close()
print("-> results/executive_summary.pdf")
