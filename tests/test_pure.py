"""Unit tests for the GPU-free pure functions. These lock in the audit fixes and run on a
laptop (no model, no lenses). Run: `.venv/bin/python -m pytest`."""
import json

import numpy as np
import pytest

from src import store
from src.serialize import serialize, permute_positions, permute_across_items
from src.judge import _parse_json
from src.checks import _max_contiguous
from src import metrics


# --- helpers ---------------------------------------------------------------

def make_scan(n_pos=4, layers=(0, 4)):
    """Cell token encodes its origin as 'pos_layer' so permutations are traceable."""
    return {str(p): {str(L): [f"{p}_{L}"] for L in layers} for p in range(n_pos)}


# --- serialize / permutations ----------------------------------------------

def test_serialize_shape():
    s = serialize(make_scan(2, (0, 4)), top_k=10)
    assert s.splitlines()[0].startswith("[p000]")
    assert "L00:" in s and "L04:" in s


def test_permute_positions_preserves_inter_layer_coherence():
    scan = make_scan(4, (0, 4))
    out = permute_positions(scan, seed=0)
    # Each destination position must draw ALL its layers from a single source position (S2 fix).
    for p in out:
        sources = {out[p][L][0].split("_")[0] for L in out[p]}
        assert len(sources) == 1, "a position must not mix layers from different sources"


def test_permute_positions_is_a_bijection():
    scan = make_scan(5, (0, 4))
    out = permute_positions(scan, seed=1)
    mapped = {next(iter(out[p].values()))[0].split("_")[0] for p in out}
    assert mapped == set(scan.keys())  # every source used exactly once -> vocabulary preserved


def test_permute_positions_deterministic():
    scan = make_scan(6)
    assert permute_positions(scan, "abc") == permute_positions(scan, "abc")


def test_permute_across_items_matches_length():
    a, b = make_scan(4), make_scan(2)
    out = permute_across_items(a, b)
    assert set(out) == set(a)


# --- judge JSON parsing -----------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"anomaly": true, "confidence": 0.8}', True),
    ('here is json {"anomaly": false, "confidence": 0.1} trailing', False),
])
def test_parse_json_ok(raw, expected):
    assert _parse_json(raw)["anomaly"] is expected


def test_parse_json_failure_returns_none():
    assert _parse_json("no json here") is None


# --- leak contiguous run ----------------------------------------------------

def test_max_contiguous():
    src = "the quick brown fox jumps".split()
    scan = "zzz quick brown fox zzz".split()
    assert _max_contiguous(scan, src) == 3  # "quick brown fox"


# --- store: validation gate + verdict dedup ---------------------------------

def test_load_pairs_validated_only(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    rows = [
        {"id": "a", "human_checked": True},
        {"id": "b", "human_checked": False},
        {"id": "c", "human_checked": True, "rejected": True},
    ]
    store.save_jsonl(rows, tmp_path / "pairs.jsonl")
    kept = [p["id"] for p in store.load_pairs(validated_only=True)]
    assert kept == ["a"]
    assert len(store.load_pairs(validated_only=False)) == 3


def test_load_verdicts_dedup_and_drop_errors(tmp_path, monkeypatch):
    vf = tmp_path / "verdicts.jsonl"
    monkeypatch.setattr(store, "VERDICTS", vf)
    key = dict(id="x", version="anomalous", condition="scan", instrument="jlens", judge="judgeA", prompt_v="v1")
    rows = [
        {**key, "confidence": 0.2, "anomaly": True},            # old
        {**key, "confidence": 0.9, "anomaly": True},            # newer duplicate -> keep this one
        {**key, "instrument": "rlens", "_error": True},         # error -> dropped
    ]
    vf.write_text("\n".join(json.dumps(r) for r in rows))
    df = store.load_verdicts()
    assert len(df) == 1
    assert df.iloc[0]["confidence"] == 0.9


# --- metrics: bootstrap resamples by pair -----------------------------------

def test_auc_ci_perfect_separation():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, -0.9, 0.8, -0.8])
    groups = np.array(["p1", "p1", "p2", "p2"])
    auc, (lo, hi) = metrics.auc_ci(y, s, groups=groups, n_boot=200)
    assert auc == 1.0 and lo == 1.0 and hi == 1.0


def test_paired_delta_zero_when_identical():
    import pandas as pd
    base = pd.DataFrame({"id": ["p1", "p1", "p2", "p2"],
                         "version": ["anomalous", "clean", "anomalous", "clean"],
                         "y": [1, 0, 1, 0], "s": [0.7, -0.7, 0.6, -0.6]})
    d, (lo, hi) = metrics.paired_delta(base, base, n_boot=200)
    assert d == 0.0 and lo == 0.0 and hi == 0.0