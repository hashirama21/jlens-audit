"""Step 9 — AUC [bootstrap CI], TPR@FPR, false-alarm rate, the Δ1/Δ2/Δ3 triplet,
inter-judge agreement, figures.

The bootstrap resamples by PAIR, not by row: the anomalous item and its clean twin are
dependent (near-identical text), so resampling rows would give anti-conservative CIs.
Pass `groups` (the pair id) to resample whole pairs together."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
from scipy.stats import spearmanr

from .config import RESULTS, FIGS, N_BOOT, FPR_TARGET, SEED, INSTRUMENTS
from . import store


def load():
    df = store.load_verdicts(dedup=True, drop_errors=True)
    df["s"] = df["confidence"].astype(float) * np.where(df["anomaly"].astype(bool), 1, -1)  # signed score
    return df


def _group_indices(groups):
    idx = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)
    return {g: np.asarray(v) for g, v in idx.items()}


def _boot_aucs(y, s, groups, n_boot, seed):
    """Bootstrap AUCs resampling whole groups (pairs) with replacement."""
    y, s = np.asarray(y), np.asarray(s)
    g2i = _group_indices(np.asarray(groups))
    keys = list(g2i)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([g2i[keys[c]] for c in chosen])
        if len(np.unique(y[idx])) == 2:
            out.append(roc_auc_score(y[idx], s[idx]))
    return np.asarray(out)


def auc_ci(y, s, groups=None, n_boot=N_BOOT, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    if groups is None:
        groups = np.arange(len(y))  # degenerate: one item per group == row bootstrap
    boots = _boot_aucs(y, s, groups, n_boot, seed)
    ci = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    return roc_auc_score(y, s), ci


def tpr_at_fpr(y, s, t=FPR_TARGET):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(fpr, t, side="right") - 1
    return float(tpr[max(i, 0)])


def paired_delta(dfA, dfB, n_boot=N_BOOT, seed=SEED):
    """Δ = AUC(A) - AUC(B) on the same pairs, paired bootstrap resampled by pair id."""
    m = dfA.merge(dfB, on=["id", "version"], suffixes=("_a", "_b"))
    y, sa, sb = m["y_a"].values, m["s_a"].values, m["s_b"].values
    g2i = _group_indices(m["id"].values)
    keys = list(g2i)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        chosen = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([g2i[keys[c]] for c in chosen])
        if len(np.unique(y[idx])) == 2:
            d.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    point = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    return point, (np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan))


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load()
    rows = []
    for (cond, inst, judge, pv), g in df.groupby(["condition", "instrument", "judge", "prompt_v"], dropna=False):
        auc, ci = auc_ci(g["y"], g["s"], groups=g["id"])
        far = g[g["version"] == "clean"]["anomaly"].astype(bool).mean()
        rows.append(dict(condition=cond, instrument=inst, judge=judge, prompt_v=pv, auc=auc, ci_lo=ci[0], ci_hi=ci[1],
                         tpr_at_5fpr=tpr_at_fpr(g["y"], g["s"]), false_alarm_clean=far, n=len(g)))
        for fam, gf in g.groupby("family"):
            if gf["y"].nunique() == 2:
                a, c = auc_ci(gf["y"], gf["s"], groups=gf["id"], n_boot=300)
                rows.append(dict(condition=cond, instrument=inst, judge=judge, prompt_v=pv, family=fam,
                                 auc=a, ci_lo=c[0], ci_hi=c[1], n=len(gf)))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "metrics.csv", index=False)

    # Triplet per instrument (judges/prompts pooled: mean signed score per item)
    pool = df.groupby(["id", "version", "y", "condition", "instrument"], dropna=False)["s"].mean().reset_index()
    p1 = pool[pool.condition == "prompt_only"]
    trip = []
    for inst in INSTRUMENTS:
        sc = pool[(pool.condition == "scan") & (pool.instrument == inst)]
        rc = pool[(pool.condition == "reconstruction") & (pool.instrument == inst)]
        pm = pool[(pool.condition == "scan_permuted") & (pool.instrument == inst)]
        d1, c1 = paired_delta(sc, p1)
        d2, c2 = paired_delta(sc, rc)
        a3, ci3 = auc_ci(pm["y"], pm["s"], groups=pm["id"])
        d3, c3 = a3 - .5, (ci3[0] - .5, ci3[1] - .5)
        trip += [dict(instrument=inst, delta="Δ1 scan−prompt", val=d1, lo=c1[0], hi=c1[1]),
                 dict(instrument=inst, delta="Δ2 scan−reconstruction", val=d2, lo=c2[0], hi=c2[1]),
                 dict(instrument=inst, delta="Δ3 permuted−chance", val=d3, lo=c3[0], hi=c3[1])]
    trip = pd.DataFrame(trip)
    trip.to_csv(RESULTS / "triplet.csv", index=False)

    # Inter-judge agreement (scan condition, prompt v1)
    ag = {}
    sc = df[(df.condition == "scan") & (df.prompt_v == "v1")]
    js = sorted(sc.judge.dropna().unique())
    if len(js) >= 2:
        w = sc.pivot_table(index=["id", "version", "instrument"], columns="judge", values=["anomaly", "s"])
        k = cohen_kappa_score(w["anomaly"][js[0]].astype(bool), w["anomaly"][js[1]].astype(bool))
        r = spearmanr(w["s"][js[0]], w["s"][js[1]]).correlation
        ag = dict(kappa=k, spearman=r, judges=js)
        json.dump(ag, open(RESULTS / "inter_judge.json", "w"), indent=2)

    # Fig 1 — triplet
    plt.figure(figsize=(8, 4.2))
    x = np.arange(len(INSTRUMENTS))
    w = .25
    for j, d in enumerate(trip.delta.unique()):
        t = trip[trip.delta == d]
        plt.bar(x + (j - 1) * w, t.val, w, yerr=[t.val - t.lo, t.hi - t.val], capsize=3, label=d)
    plt.axhline(0, color="k", lw=.8)
    plt.xticks(x, INSTRUMENTS)
    plt.ylabel("ΔAUC")
    plt.legend()
    plt.title("H1/H2/H3 triplet per instrument (95% paired bootstrap CI)")
    plt.tight_layout()
    plt.savefig(FIGS / "fig1_triplet.png", dpi=160)

    # Fig 2 — AUC family x instrument (scan condition, pooled)
    fam = res[(res.condition == "scan") & res.family.notna()].groupby(["family", "instrument"])["auc"].mean().unstack()
    plt.figure(figsize=(6, 3.8))
    sns.heatmap(fam, annot=True, fmt=".2f", vmin=.4, vmax=1, cmap="viridis")
    plt.title("AUC per family × instrument (blind scan)")
    plt.tight_layout()
    plt.savefig(FIGS / "fig2_auc_family.png", dpi=160)

    print(res[res.family.isna()][["condition", "instrument", "judge", "prompt_v", "auc", "ci_lo", "ci_hi",
                                  "tpr_at_5fpr", "false_alarm_clean"]].to_string())
    print("\n", trip.to_string(), "\n", ag)


if __name__ == "__main__":
    main()