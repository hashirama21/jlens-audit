"""Steps 6/8 — automated sanity checks.
   leak    : source n-grams that reappear in what the judge receives (leak; also a crude inversion measure)
   sample  : draw N scans (half anomalous, half clean) to read by hand, with verdicts + evidence
   family  : rate of 'anomaly=true' on CLEAN items per family (is it detecting the family?)
   evidence: for each verdict, are the 'evidence' tokens actually in the scan, and in the anomaly zone?"""
import re
import argparse
import difflib

import pandas as pd

from .config import RESULTS, TOP_K, INSTRUMENTS
from .serialize import serialize
from . import store


def _tok_words(s):
    return re.findall(r"\w+", s.lower())


def _max_contiguous(seq_a, seq_b):
    """Longest run of consecutive tokens shared between two word lists (STOP criterion:
    isolated shared n-grams = the lens surfacing the current token; a long contiguous run
    = the source text leaking through the harness)."""
    return difflib.SequenceMatcher(a=seq_a, b=seq_b, autojunk=False).find_longest_match(
        0, len(seq_a), 0, len(seq_b)).size


def leak(n=4, path="pairs.jsonl"):
    rows = []
    for p in store.load_pairs(path, validated_only=True):
        for version in ("anomalous", "clean"):
            src = _tok_words(p[version])
            grams = {tuple(src[i:i + n]) for i in range(len(src) - n + 1)}
            for inst in INSTRUMENTS:
                ser = _tok_words(serialize(store.load_scan(p["id"], version, inst)["scan"], TOP_K))
                shared = sum(tuple(ser[i:i + n]) in grams for i in range(len(ser) - n + 1))
                rows.append(dict(id=p["id"], version=version, instrument=inst,
                                 shared_ngrams=shared, src_ngrams=len(grams),
                                 frac=shared / max(len(grams), 1),
                                 max_contig=_max_contiguous(ser, src)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "leak_check.csv", index=False)
    print(df.groupby("instrument")[["frac", "max_contig"]].describe())
    print("\nInterpretation: isolated n-grams = the lens surfaces the current token (signal, not leak)."
          "\nLong contiguous runs (high max_contig) = harness leak -> STOP. Mean frac is a crude inversion measure.")


def sample(n=30, seed=0):
    df = store.load_verdicts()
    df = df[(df.condition == "scan") & (df.prompt_v == "v1")]
    an = df[df.version == "anomalous"].sample(n // 2, random_state=seed)
    cl = df[df.version == "clean"].sample(n - n // 2, random_state=seed)
    out = []
    for _, r in pd.concat([an, cl]).iterrows():
        sc = store.load_scan(r.id, r.version, r.instrument)
        out.append(f"\n===== {r.id} {r.version} {r.instrument} judge={r.judge} anomaly={r.anomaly} conf={r.confidence}\n"
                   f"DESCRIPTION: {r.description}\nEVIDENCE: {r.evidence}\n--- SCAN (first 40 positions) ---\n" +
                   "\n".join(serialize(sc["scan"], TOP_K).split("\n")[:40]) +
                   "\n--- FILL IN BY HAND: evidence present? [ ]  in anomaly zone? [ ]  related to anomaly? [ ]  note:\n")
    (RESULTS / f"manual_review_{n}.txt").write_text("".join(out), encoding="utf-8")
    print(f"-> results/manual_review_{n}.txt (annotate by hand)")


def family():
    df = store.load_verdicts()
    t = df[df.version == "clean"].groupby(["condition", "instrument", "family"], dropna=False)["anomaly"].mean().unstack()
    print("Rate of 'anomaly=true' on CLEAN items (family detection if high):\n", t.round(2))
    t.to_csv(RESULTS / "family_false_alarm.csv")


def evidence():
    df = store.load_verdicts()
    df = df[df.condition == "scan"]
    pairs = {p["id"]: p for p in store.load_pairs("pairs.jsonl")}
    rows = []
    for _, r in df.iterrows():
        sc = store.load_scan(r.id, r.version, r.instrument)["scan"]
        present = in_zone = 0
        span = pairs[r.id].get("anomaly_token_span")  # [start,end] positions, filled at generation if possible
        for e in (r.evidence or []):
            m = re.match(r"p(\d+)-L(\d+):(.*)", str(e))
            if not m:
                continue
            pos, L, tk = m.group(1).lstrip("0") or "0", str(int(m.group(2))), m.group(3).strip()
            ok = tk in [t.strip() for t in sc.get(pos, {}).get(L, [])]
            present += ok
            if ok and span and span[0] <= int(pos) <= span[1]:
                in_zone += 1
        rows.append(dict(id=r.id, version=r.version, instrument=r.instrument, judge=r.judge,
                         n_evidence=len(r.evidence or []), present=present, in_zone=in_zone))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "evidence_check.csv", index=False)
    print(out[["n_evidence", "present", "in_zone"]].sum(),
          "\n(evidence cited but absent from the scan = judge hallucination)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["leak", "sample", "family", "evidence"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"leak": leak, "sample": lambda: sample(a.n, a.seed), "family": family, "evidence": evidence}[a.cmd]()