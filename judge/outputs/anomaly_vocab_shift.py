"""Anomaly-specificity test (no GPU/API): downstream of the anomaly (positions > s, where the
input token is IDENTICAL in both twins), WHICH tokens does the anomalous scan add vs the clean
twin, and which does it drop? Does the surfaced vocabulary shift toward the anomaly's semantics
(the correct fact / error-correction words) or is it an arbitrary reshuffle?

Objective extraction only: we list the added (anom-only) and removed (clean-only) tokens per
pair per band, plus whether the anomalous value token and the clean value token themselves
propagate downstream. The semantic judgment (does this trend toward the anomaly?) is the human's.

Caveat carried from the review: there is NO matched neutral-substitution null (needs GPU). So a
downstream shift here still cannot, by itself, separate anomaly propagation from any-substitution
propagation. This test only asks whether the shifted vocabulary LOOKS anomaly-related.
"""
import pandas as pd
from src import store
from src.config import RESULTS, INSTRUMENTS

ALIGNED = ["fp_01", "fp_02", "fp_03", "fp_04", "bug_05"]
BANDS = {"early(0-16)": {"0", "8", "16"}, "mid(24-40)": {"24", "32", "40"},
         "late(48-62)": {"48", "56", "62"}}


def norm(t):
    return str(t).strip()


def surfaced_downstream(scan, s, band_layers):
    """multiset-as-set of tokens surfaced at positions > s within the band."""
    out = set()
    for pos, layers in scan.items():
        if int(pos) <= s:
            continue
        for L, toks in layers.items():
            if L in band_layers:
                out.update(norm(t) for t in toks)
    return out


pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl", validated_only=True)}
rows = []
for pid in ALIGNED:
    p = pairs[pid]
    s, e = p["anomaly_token_span"]
    a_in = store.load_scan(pid, "anomalous", "jlens")["input_tokens"]
    c_in = store.load_scan(pid, "clean", "jlens")["input_tokens"]
    anom_tok, clean_tok = norm(a_in[s]), norm(c_in[s])
    print("=" * 90)
    print(f"{pid}  ({p['family']})   anomaly@pos{s}:  clean='{clean_tok}'  ->  anomalous='{anom_tok}'"
          f"   [anomaly_text={p.get('anomaly_text')!r}]")
    for inst in INSTRUMENTS:
        a = store.load_scan(pid, "anomalous", inst)["scan"]
        c = store.load_scan(pid, "clean", inst)["scan"]
        for band, Ls in BANDS.items():
            if band == "late(48-62)":
                continue  # echo/flat zone; skip for the semantic question
            A = surfaced_downstream(a, s, Ls)
            C = surfaced_downstream(c, s, Ls)
            added = sorted(A - C)
            removed = sorted(C - A)
            rows.append(dict(id=pid, family=p["family"], instrument=inst, band=band,
                             n_added=len(added), n_removed=len(removed),
                             anom_val_in_added=int(anom_tok in (A - C)),
                             clean_val_in_removed=int(clean_tok in (C - A)),
                             added=" ".join(added), removed=" ".join(removed)))
            print(f"  [{inst:5} {band:11}]  +{len(added):2} added: {' '.join(added)[:88]}")
            print(f"  {'':19}  -{len(removed):2} removed: {' '.join(removed)[:86]}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "anomaly_vocab_shift.csv", index=False)
print("\n" + "=" * 90)
print("OBJECTIVE summary (early+mid downstream, all instruments):")
print("  does the anomalous value token propagate downstream (appear in ADDED)?",
      f"{df.anom_val_in_added.sum()} / {len(df)} cells")
print("  does the clean value token disappear downstream (appear in REMOVED)?",
      f"{df.clean_val_in_removed.sum()} / {len(df)} cells")
print("  median added/removed per cell:", int(df.n_added.median()), "/", int(df.n_removed.median()))
print("-> results/anomaly_vocab_shift.csv  (read the added/removed columns and judge relatedness)")
