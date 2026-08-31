"""Scan -> text for the judge. WITHOUT the input token, WITHOUT the instrument name.
Permutations for condition 5 (apophenia / H3)."""
import random


def serialize(scan: dict, top_k: int = 10) -> str:
    lines = []
    for pos in sorted(scan, key=int):
        parts = [f"L{int(L):02d}: " + " ".join(t.strip().replace("\n", "\\n") or "∅" for t in toks[:top_k])
                 for L, toks in sorted(scan[pos].items(), key=lambda kv: int(kv[0]))]
        lines.append(f"[p{int(pos):03d}] " + " | ".join(parts))
    return "\n".join(lines)


def permute_positions(scan: dict, seed) -> dict:
    """Shuffle whole positions: a single permutation of the position axis, applied identically
    to every layer. This destroys position<->content alignment (the point of the H3 control)
    while PRESERVING the inter-layer coherence at each position — i.e. the judge still sees a
    self-consistent column per position, only relocated. A per-layer shuffle would be strictly
    more destructive and would make the H3 control artificially easy to pass."""
    rng = random.Random(str(seed))
    positions = sorted(scan, key=int)
    sources = positions[:]
    rng.shuffle(sources)
    return {dst: scan[src] for dst, src in zip(positions, sources)}


def permute_across_items(scan: dict, other_scan: dict) -> dict:
    """Bonus variant: tokens taken from ANOTHER item's scan (cross-item contamination),
    truncated/extended to the same length."""
    positions = sorted(scan, key=int)
    others = sorted(other_scan, key=int)
    return {p: other_scan[others[i % len(others)]] for i, p in enumerate(positions)}