"""Block H #1 — OBJECTIVE scaffold only. No judgment.
Reproduces checks.sample's exact 30-row selection (seed 0), computes per-row
present/in_zone with the SAME logic as checks.evidence, and joins the anomaly
context so the human can fill 'related?' and the FP class (a/b/c) by hand.
"""
import re
import pandas as pd
from src import store
from src.config import RESULTS

N, SEED = 30, 0
EVID = re.compile(r"p(\d+)-L(\d+):(.*)")

df = store.load_verdicts()
df = df[(df.condition == "scan") & (df.prompt_v == "v1")]
an = df[df.version == "anomalous"].sample(N // 2, random_state=SEED)
cl = df[df.version == "clean"].sample(N - N // 2, random_state=SEED)
sample = pd.concat([an, cl])

pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl")}

rows = []
for _, r in sample.iterrows():
    sc = store.load_scan(r.id, r.version, r.instrument)["scan"]
    span = pairs.get(r.id, {}).get("anomaly_token_span")
    n_ev = len(r.evidence or [])
    present = in_zone = 0
    for e in (r.evidence or []):
        m = EVID.match(str(e))
        if not m:
            continue
        pos = m.group(1).lstrip("0") or "0"
        L = str(int(m.group(2)))
        tk = m.group(3).strip()
        ok = tk in [t.strip() for t in sc.get(pos, {}).get(L, [])]
        present += ok
        if ok and span and span[0] <= int(pos) <= span[1]:
            in_zone += 1
    # objective classification of the verdict outcome (not a judgment about WHY)
    y = int(r.version == "anomalous")
    pred = bool(r.anomaly)
    outcome = ("TP" if (y and pred) else "FN" if (y and not pred)
               else "FP" if (not y and pred) else "TN")
    anomaly_text = pairs.get(r.id, {}).get("anomaly_text", "")
    rows.append(dict(
        id=r.id, ver=r.version, inst=r.instrument, judge=r.judge,
        pred_anom=pred, conf=round(float(r.confidence), 2), outcome=outcome,
        n_ev=n_ev, present=present, in_zone=(in_zone if span else "n/a"),
        span=(f"{span[0]}-{span[1]}" if span else "n/a"),
        family=pairs.get(r.id, {}).get("family", ""),
        anomaly_text=(anomaly_text[:80] if anomaly_text else ""),
    ))

out = pd.DataFrame(rows)
# stable order: FP first (the H3 material), then FN, then TP/TN
out["_o"] = out.outcome.map({"FP": 0, "FN": 1, "TP": 2, "TN": 3})
out = out.sort_values(["_o", "id"]).drop(columns="_o").reset_index(drop=True)
out.to_csv(RESULTS / "blockH_worksheet.csv", index=False)

# empty judgment columns for the human
JUDGE_COLS = "| related?(y/n) | FP_class(a/b/c) | note |"
JUDGE_SEP = "|---|---|---|"
lines = ["# Block H #1 — worksheet (objective pre-filled; YOU fill the last 3 columns)",
         "",
         "Objective columns computed exactly like `src.checks.evidence`. FP rows first (they are",
         "the H3 material). `related?` and `FP_class` are YOUR judgment — see the scan blocks in",
         "`results/manual_review_30.txt` for the same items.",
         "",
         "FP_class legend (false positives on CLEAN twins only): (a) evokes the family, "
         "(b) evokes the item's own clean content, (c) invents. Proportion of (c) = direct H3.",
         "",
         "| # | id | ver | inst | judge | pred | conf | outcome | n_ev | present | in_zone | span | family | anomaly_text " + JUDGE_COLS,
         "|--|----|-----|------|-------|------|------|---------|------|---------|---------|------|--------|--------------" + JUDGE_SEP]
for i, r in out.iterrows():
    lines.append(f"| {i+1} | {r.id} | {r.ver} | {r.inst} | {r.judge} | {r.pred_anom} | {r.conf} "
                 f"| {r.outcome} | {r.n_ev} | {r.present} | {r.in_zone} | {r.span} | {r.family} "
                 f"| {r.anomaly_text} |   |   |   |")
(RESULTS / "blockH_worksheet.md").write_text("\n".join(lines), encoding="utf-8")

# console summary (objective)
print("outcomes:", dict(out.outcome.value_counts()))
print("FP rows (clean flagged anomalous) =", int((out.outcome == "FP").sum()),
      "| FN rows =", int((out.outcome == "FN").sum()))
print("evidence fidelity on these 30: present", int(out.present.sum()), "/ cited",
      int(out.n_ev.sum()))
print("-> results/blockH_worksheet.md  (+ .csv)")
