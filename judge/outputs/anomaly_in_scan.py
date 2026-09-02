"""Content control (no API): do the anomaly-span tokens surface in the ANOMALOUS scan but
NOT in the clean twin's scan? Stratified by layer band to separate a real early/mid readout
(H1) from the trivial late-layer echo of the current token (surface => H2, not H1).

T = input_tokens of the ANOMALOUS version at anomaly_token_span [s..e] (the anomaly content).
For each instrument and layer band we ask whether each t in T appears among the lens top-k
tokens surfaced ANYWHERE in the anomalous scan vs the clean scan. Positional detail is added
for the 5 length-aligned single-token pairs (top-k at the anomaly position itself).
Objective only — no judgment.
"""
import json
import pandas as pd
from src import store
from src.config import RESULTS, INSTRUMENTS

BANDS = {"early(0-16)": {"0", "8", "16"}, "mid(24-40)": {"24", "32", "40"},
         "late(48-62)": {"48", "56", "62"}}


def norm(t):
    return str(t).strip()


def surfaced(scan, band_layers):
    """set of normalized tokens surfaced anywhere in the scan for the given layer band."""
    out = set()
    for pos, layers in scan.items():
        for L, toks in layers.items():
            if L in band_layers:
                out.update(norm(t) for t in toks)
    return out


pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl", validated_only=True)}
rows = []
pos_rows = []

for pid, p in pairs.items():
    s, e = p["anomaly_token_span"]
    for inst in INSTRUMENTS:
        a = store.load_scan(pid, "anomalous", inst)
        c = store.load_scan(pid, "clean", inst)
        T = [norm(t) for t in a["input_tokens"][s:e + 1] if norm(t)]
        Tset = set(T)
        aligned = a["n_pos"] == c["n_pos"]
        for band, Ls in BANDS.items():
            A = surfaced(a["scan"], Ls)
            C = surfaced(c["scan"], Ls)
            in_a = Tset & A
            in_c = Tset & C
            disc = in_a - C          # surfaced in anomalous, absent from clean twin  <-- the control
            rows.append(dict(id=pid, family=p["family"], instrument=inst, band=band,
                             n_T=len(Tset), in_anom=len(in_a), in_clean=len(in_c),
                             discriminating=len(disc),
                             disc_tokens=" ".join(sorted(disc))[:60]))
        # positional detail at the anomaly site (aligned single-token pairs only)
        if aligned and s == e and str(s) in a["scan"]:
            for L in sorted(a["scan"][str(s)], key=int):
                at = [norm(t) for t in a["scan"][str(s)].get(L, [])]
                ct = [norm(t) for t in c["scan"].get(str(s), {}).get(L, [])]
                pos_rows.append(dict(id=pid, instrument=inst, layer=int(L),
                                     anom_input=norm(a["input_tokens"][s]),
                                     clean_input=norm(c["input_tokens"][s]),
                                     anom_topk="|".join(at), clean_topk="|".join(ct),
                                     differ=int(at != ct)))

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "anomaly_in_scan.csv", index=False)
pos = pd.DataFrame(pos_rows)
pos.to_csv(RESULTS / "anomaly_positional.csv", index=False)

# ---- aggregate summary (objective) ----
print("=== Q: do anomaly-span tokens surface in the ANOMALOUS scan but NOT the clean twin? ===")
print("    (aggregated over 11 pairs x 3 instruments = 33 scans, by layer band)\n")
agg = (df.groupby("band")[["n_T", "in_anom", "in_clean", "discriminating"]].sum())
agg["frac_anom"] = (agg.in_anom / agg.n_T).round(3)
agg["frac_clean"] = (agg.in_clean / agg.n_T).round(3)
agg["frac_disc"] = (agg.discriminating / agg.n_T).round(3)
print(agg.to_string())

print("\n=== same, per instrument x band (frac discriminating = surfaced-in-anom-only / |T|) ===")
g2 = df.groupby(["instrument", "band"]).apply(
    lambda x: pd.Series({"frac_disc": round(x.discriminating.sum() / x.n_T.sum(), 3),
                         "frac_anom": round(x.in_anom.sum() / x.n_T.sum(), 3)}),
    include_groups=False)
print(g2.to_string())

print("\n=== positional test at the anomaly site (5 aligned single-token pairs) ===")
print("    does the top-k AT the anomaly position differ between anomalous and clean twin?")
if len(pos):
    pt = pos.groupby(["id", "instrument"]).apply(
        lambda x: pd.Series({"layers_differ": int(x.differ.sum()), "n_layers": len(x)}),
        include_groups=False)
    print(pt.to_string())
    print("\n  by layer band (fraction of (pair,instrument,layer) cells that differ):")
    pos["band"] = pos.layer.map(lambda L: "early" if L <= 16 else "mid" if L <= 40 else "late")
    print(pos.groupby("band")["differ"].mean().round(3).to_string())
print("\n-> results/anomaly_in_scan.csv , results/anomaly_positional.csv")
