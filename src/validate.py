"""Step 3 — quantitative validation that the lenses work (test 1 of the go/no-go).
   --smoke : conformity test 'sushi -> Japan' (R-lens post: R from ~L2, J around ~L14 — order of magnitude).
   otherwise: pass@10 per layer on data/multihop.jsonl (generate via the agent, filtered: the model answers correctly)."""
import json
import argparse

import matplotlib.pyplot as plt

from .config import DATA, RESULTS, FIGS, TOP_K
from .load_model import get_resid, layers, load
from .lens import load_all


def find_pos(text, tok, pivot):
    """Position of the last token overlapping the pivot substring. Robust to multi-token
    pivots (e.g. 'sushi' -> 'su' + 'shi') via character offsets, with a single-token fallback
    for slow tokenizers."""
    lo, pv = text.lower(), pivot.lower()
    start = lo.rfind(pv)
    if start < 0:
        raise ValueError(f"pivot {pivot!r} not found in text")
    end = start + len(pv)
    try:
        offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
        for i in range(len(offsets) - 1, -1, -1):
            s, e = offsets[i]
            if s < end and e > start:  # token span overlaps the pivot span
                return i
    except Exception:
        pass
    ids = tok(text)["input_ids"]
    for i in range(len(ids) - 1, -1, -1):
        if pv in tok.decode(int(ids[i])).lower():
            return i
    raise ValueError(f"pivot {pivot!r} not aligned to any token")


def smoke(lenses):
    tok, _ = load()
    prompt = "The capital of the country where sushi originated is"
    H, ids = get_resid(prompt)
    pos = find_pos(prompt, tok, "sushi")
    print(f"pos(sushi)={pos}")
    for kind, lens in lenses.items():
        first = None
        for L in layers():
            top = lens.readout(H[L][pos], L, TOP_K)
            hit = any("japan" in t.lower() for t in top)
            print(f"{kind:6s} L{L:02d} {'JAPAN' if hit else '     '} {top}")
            if hit and first is None:
                first = L
        print(f"==> {kind}: 'Japan' first appears at layer {first}\n")


def pass_at_k(lenses):
    tok, _ = load()
    items = [json.loads(line) for line in open(DATA / "multihop.jsonl")]
    res = {k: {L: 0 for L in layers()} for k in lenses}
    n = 0
    for it in items:
        H, ids = get_resid(it["prompt"])
        pos = find_pos(it["prompt"], tok, it["pivot"])
        n += 1
        for kind, lens in lenses.items():
            for L in layers():
                top = lens.readout(H[L][pos], L, TOP_K)
                if any(it["intermediate"].strip().lower() in t.strip().lower() for t in top):
                    res[kind][L] += 1
    res = {k: {L: v / n for L, v in d.items()} for k, d in res.items()}
    json.dump(res, open(RESULTS / "validation_multihop.json", "w"), indent=2)
    plt.figure(figsize=(7, 4))
    for k, d in res.items():
        plt.plot(list(d.keys()), list(d.values()), marker="o", label=k)
    plt.xlabel("layer")
    plt.ylabel("pass@10")
    plt.title(f"Multihop pass@10 (n={n})")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGS / "validation_multihop.png", dpi=150)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    lenses = load_all()
    smoke(lenses) if a.smoke else pass_at_k(lenses)