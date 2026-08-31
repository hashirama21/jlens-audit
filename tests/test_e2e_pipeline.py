"""End-to-end dry run of the API half of the pipeline, GPU-free.

Simulates: validated pairs -> synthetic scans -> leak audit -> the 7 conditions with a
mocked judge -> metrics (figures included) -> sanity checks. Also proves resumability:
re-running `conditions.run` adds zero new rows.

The GPU half (load_model / lens / scan / validate / capability / upper_bound) cannot run
on a laptop; its contract is exercised here through the same file formats scan.py writes.
"""
import json
import random

import pytest

from src import store, checks, conditions, metrics, config
from src.serialize import serialize

FAMILIES = ["injection", "bug", "false_premise", "conflict"]
LAYERS = [0, 4, 8]
VOCAB = "the of and to a in for on with data file result search note code line".split()


# --- synthetic corpus + scans ----------------------------------------------

def make_pairs(n_per_family=2):
    pairs = []
    for f in FAMILIES:
        for k in range(1, n_per_family + 1):
            pairs.append({
                "id": f"{f[:4]}_{k:02d}", "family": f,
                "anomalous": f"some {f} content number {k} with a hidden oddity inside it",
                "clean": f"some {f} content number {k} with a normal phrase inside it",
                "anomaly_token_span": [5, 7], "diff_tokens": 2,
                "human_checked": True, "rejected": False, "notes": "",
            })
    return pairs


def make_scan(pair, version, rng):
    """20 positions x 3 layers x 5 tokens. Anomalous scans carry a detectable token
    ('odd') inside the anomaly zone so the mock judge has a real signal."""
    scan = {}
    for p in range(20):
        scan[str(p)] = {}
        for L in LAYERS:
            toks = rng.sample(VOCAB, 5)
            if version == "anomalous" and 5 <= p <= 7 and L >= 4:
                toks[0] = "odd"
            scan[str(p)][str(L)] = toks
    return {"id": pair["id"], "version": version, "family": pair["family"],
            "instrument": None, "n_pos": 20, "input_tokens": [], "scan": scan}


# --- mock judge -------------------------------------------------------------

def mock_call(judge, text, want_json=True, max_tokens=600):
    if not want_json:  # reconstruction request -> return a text
        return "reconstructed text " + ("odd finding" if "odd" in text else "nothing special")
    hit = "odd" in text
    return {"anomaly": hit, "confidence": 0.9 if hit else 0.2,
            "description": "odd token cluster" if hit else "nothing stands out",
            "evidence": ["p005-L04:odd"] if hit else []}


# --- the dry run ------------------------------------------------------------

@pytest.fixture()
def env(tmp_path, monkeypatch):
    data, scans, results, figs = (tmp_path / d for d in ("data", "scans", "results", "figs"))
    for d in (data, scans, results, figs):
        d.mkdir()
    verdicts = tmp_path / "verdicts.jsonl"
    monkeypatch.setattr(store, "DATA", data)
    monkeypatch.setattr(store, "SCANS", scans)
    monkeypatch.setattr(store, "VERDICTS", verdicts)
    monkeypatch.setattr(checks, "RESULTS", results)
    monkeypatch.setattr(checks, "INSTRUMENTS", ["jlens", "rlens", "logit"])
    monkeypatch.setattr(conditions, "RESULTS", results)
    monkeypatch.setattr(conditions, "call", mock_call)
    monkeypatch.setattr(metrics, "RESULTS", results)
    monkeypatch.setattr(metrics, "FIGS", figs)

    pairs = make_pairs()
    store.save_jsonl(pairs, data / "pairs.jsonl")
    rng = random.Random(0)
    for p in pairs:
        for version in ("anomalous", "clean"):
            for inst in config.INSTRUMENTS:
                sc = make_scan(p, version, rng)
                sc["instrument"] = inst
                store.scan_path(p["id"], version, inst).write_text(json.dumps(sc))
    return dict(results=results, figs=figs, verdicts=verdicts, n_pairs=len(pairs))


def test_full_pipeline(env):
    # Step 6 gate — conditions must refuse to run before the leak audit exists.
    with pytest.raises(SystemExit, match="[Ll]eak"):
        conditions.run()

    checks.leak(n=4)
    assert (env["results"] / "leak_check.csv").exists()

    # Step 7 — all conditions, 2 mock judges.
    conditions.run()
    df = store.load_verdicts()
    n = env["n_pairs"] * 2  # item-versions
    judges = list(config.JUDGES)
    expected = (
        n * len(judges)                                          # prompt_only
        + n * len(judges) * 3 * len(config.PROMPT_VERSIONS) * 2  # scan + permuted
        + n * len(judges) * 3                                    # reconstruction
    )
    assert len(df) == expected
    assert set(df.condition) == {"prompt_only", "scan", "scan_permuted", "reconstruction"}

    # Resumability — a second run must add strictly nothing.
    before = env["verdicts"].read_text()
    conditions.run()
    assert env["verdicts"].read_text() == before

    # The synthetic signal must be recoverable: scan separates, permuted stays imperfect-or-worse.
    m = metrics
    d = m.load()
    sc = d[(d.condition == "scan")]
    auc, _ = m.auc_ci(sc["y"], sc["s"], groups=sc["id"], n_boot=50)
    assert auc > 0.9  # 'odd' token is a clean separator by construction

    # Step 9 — metrics + figures.
    m.main()
    for f in ("metrics.csv", "triplet.csv", "inter_judge.json"):
        assert (env["results"] / f).exists(), f
    for f in ("fig1_triplet.png", "fig2_auc_family.png"):
        assert (env["figs"] / f).exists(), f

    # Step 8 — sanity checks run on the produced artifacts.
    checks.family()
    assert (env["results"] / "family_false_alarm.csv").exists()
    checks.evidence()
    assert (env["results"] / "evidence_check.csv").exists()
    ev = (env["results"] / "evidence_check.csv").read_text()
    assert "jlens" in ev
    checks.sample(n=4, seed=0)
    assert (env["results"] / "manual_review_4.txt").exists()


def test_permuted_breaks_position_signal(env):
    """The permuted control must relocate the 'odd' cluster: serialization differs from
    the original at the anomaly zone for at least one item."""
    from src.serialize import permute_positions
    moved = 0
    for line in (store.DATA / "pairs.jsonl").read_text().splitlines():
        p = json.loads(line)
        sc = store.load_scan(p["id"], "anomalous", "jlens")["scan"]
        pm = permute_positions(sc, p["id"])
        if any(sc[str(q)] != pm[str(q)] for q in range(5, 8)):
            moved += 1
    assert moved > 0
