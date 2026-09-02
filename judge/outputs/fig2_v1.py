"""Regenerate fig2 on v1 ONLY (the submitted fig2 pooled v1+v2; on bug/false_premise the v2
logit/rlens cells have n=2 = one pair -> AUC is 0 or 1 coin-flips that corrupt the mean and
invert the logit ranking). On v1 all instruments share the same n. AUC per family x instrument,
judges pooled via mean signed score per (id,version). No API.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import store
from src.config import RESULTS, FIGS, INSTRUMENTS

df = store.load_verdicts()
df = df[(df.condition == "scan") & (df.prompt_v == "v1")].copy()
df["s"] = df["confidence"].astype(float) * np.where(df["anomaly"].astype(bool), 1, -1)
pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl", validated_only=True)}
df["family"] = df["id"].map(lambda i: pairs[i]["family"])


def auc(pos, neg):
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + 1 + j) / 2.0
        i = j
    R = sum(ranks[k] for k in range(len(xs)) if xs[k][1] == 1)
    nP, nN = len(pos), len(neg)
    return (R - nP * (nP + 1) / 2.0) / (nP * nN) if nP and nN else np.nan


fams = ["bug", "false_premise", "injection"]
M = pd.DataFrame(index=fams, columns=INSTRUMENTS, dtype=float)   # metrics.py method: mean of per-judge AUCs
N = pd.DataFrame(index=fams, columns=INSTRUMENTS, dtype=int)
for fam in fams:
    for inst in INSTRUMENTS:
        per_judge = []
        for judge in sorted(df.judge.dropna().unique()):
            g = df[(df.family == fam) & (df.instrument == inst) & (df.judge == judge)]
            pos = list(g[g.version == "anomalous"].s)
            neg = list(g[g.version == "clean"].s)
            if pos and neg:
                per_judge.append(auc(pos, neg))
        M.loc[fam, inst] = np.mean(per_judge) if per_judge else np.nan
        N.loc[fam, inst] = len(df[(df.family == fam) & (df.instrument == inst) &
                                  (df.version == "anomalous")].id.unique())

print("=== fig2 on v1 only: AUC per family x instrument (judges pooled) ===")
print(M.round(3).to_string())
print("\nn pairs per cell (same across instruments on v1):")
print(N.to_string())
print("\nlogit best family on v1:", M["logit"].idxmax(), "=", round(M["logit"].max(), 3))

plt.figure(figsize=(6, 3.8))
import matplotlib.cm as cm
ax = plt.gca()
im = ax.imshow(M.values.astype(float), vmin=0.4, vmax=1.0, cmap="viridis")
ax.set_xticks(range(len(INSTRUMENTS))); ax.set_xticklabels(INSTRUMENTS)
ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
for i in range(len(fams)):
    for j in range(len(INSTRUMENTS)):
        v = M.values[i, j]
        ax.text(j, i, f"{v:.2f}\n(n={N.values[i,j]})", ha="center", va="center",
                color="white" if v < 0.7 else "black", fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title("AUC per family x instrument - v1 only (blind scan)")
plt.tight_layout()
plt.savefig(FIGS / "fig2_auc_family_v1.png", dpi=160)
M.round(4).to_csv(RESULTS / "fig2_v1_auc.csv")
print("-> figs/fig2_auc_family_v1.png , results/fig2_v1_auc.csv")
