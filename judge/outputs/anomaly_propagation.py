"""Decisive refinement (aligned single-token pairs only): does the anomaly propagate to
DOWNSTREAM positions in the lens scan?

Causal attention: a one-token swap at position s can only affect the residual stream at
positions >= s. So compare the anomalous vs clean scan top-k:
  - pos < s  : must be IDENTICAL (sanity: causal, upstream untouched).
  - pos == s : differs trivially (the input token itself differs -> current-token echo).
  - pos > s  : SAME input token in both versions -> any difference is the anomaly's internal
               contextual propagation. Difference here (esp. early/mid layers) = genuine
               internal signal (H1). ~0 = the anomaly is purely local/surface (H2/echo only).
Objective only.
"""
import pandas as pd
from src import store
from src.config import RESULTS, INSTRUMENTS

ALIGNED = ["fp_01", "fp_02", "fp_03", "fp_04", "bug_05"]  # n_pos anom == clean, span len 1


def band(L):
    return "early" if L <= 16 else "mid" if L <= 40 else "late"


def norm(t):
    return str(t).strip()


pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl", validated_only=True)}
rows = []
for pid in ALIGNED:
    s, e = pairs[pid]["anomaly_token_span"]
    for inst in INSTRUMENTS:
        a = store.load_scan(pid, "anomalous", inst)["scan"]
        c = store.load_scan(pid, "clean", inst)["scan"]
        for pos in a:
            if pos not in c:
                continue
            region = "upstream(<s)" if int(pos) < s else "at(=s)" if int(pos) == s else "downstream(>s)"
            for L in a[pos]:
                at = [norm(t) for t in a[pos].get(L, [])]
                ct = [norm(t) for t in c[pos].get(L, [])]
                rows.append(dict(id=pid, instrument=inst, region=region, band=band(int(L)),
                                 differ=int(at != ct)))

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "anomaly_propagation.csv", index=False)

print("=== fraction of (pos,layer) cells where anomalous top-k != clean top-k ===")
print("    upstream should be ~0 (causal sanity); downstream>0 in early/mid = internal signal\n")
piv = df.pivot_table(index="region", columns="band", values="differ", aggfunc="mean").round(3)
piv = piv.reindex(index=["upstream(<s)", "at(=s)", "downstream(>s)"], columns=["early", "mid", "late"])
print(piv.to_string())

print("\n=== downstream-only, per instrument (early/mid = the H1-relevant bands) ===")
d = df[df.region == "downstream(>s)"]
print(d.pivot_table(index="instrument", columns="band", values="differ", aggfunc="mean").round(3)
      .reindex(columns=["early", "mid", "late"]).to_string())

print("\n=== downstream-only, per pair (mean over layers/instruments) ===")
print(d.groupby("id")["differ"].mean().round(3).to_string())

n_up = df[df.region == "upstream(<s)"]["differ"].mean()
print(f"\nsanity: upstream differ rate = {n_up:.4f} (expected ~0 under causal attention)")
print("-> results/anomaly_propagation.csv")
