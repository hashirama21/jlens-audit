"""Step 5 — full scan: for each text (anomalous + clean), each instrument,
all positions x subsampled layers x top-k. One JSON per (id, version, instrument)."""
import json
import time

from tqdm import tqdm

from .config import TOP_K
from .load_model import get_resid, layers, load, content_span
from .lens import load_all
from . import store


def scan_text(text, lenses):
    H, ids = get_resid(text)
    tok, _ = load()
    lo, hi = content_span(tok, text)                         # skip template scaffolding, keep absolute positions
    out = {kind: {} for kind in lenses}
    for L in layers():
        for kind, lens in lenses.items():
            tops = lens.readout_all(H[L], L, TOP_K)          # (seq) lists
            for pos in range(lo, hi):
                out[kind].setdefault(str(pos), {})[str(L)] = tops[pos]
    return out, [tok.decode(int(t)) for t in ids]


def main(path="pairs.jsonl"):
    lenses = load_all()
    items = store.load_pairs(path, validated_only=True)
    if not items:
        raise SystemExit(f"No human-validated pairs in data/{path}.")
    t0 = time.time()
    for p in tqdm(items):
        for version in ("anomalous", "clean"):
            out, toks = scan_text(p[version], lenses)
            for kind, sc in out.items():
                json.dump({"id": p["id"], "version": version, "family": p["family"], "instrument": kind,
                           "n_pos": len(toks), "input_tokens": toks, "scan": sc},
                          open(store.scan_path(p["id"], version, kind), "w"), ensure_ascii=False)
    print(f"{len(items) * 2} texts x {len(lenses)} instruments in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "pairs.jsonl")