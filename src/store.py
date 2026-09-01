"""Shared persistence layer (DRY): the single place that knows where pairs, scans and
verdicts live and how they are read. Every other module goes through these helpers so
the on-disk contract exists in one file only.

Design notes:
- `load_pairs(validated_only=True)` is the guard that enforces the CLAUDE.md rule
  "never run on pairs that have not passed human validation": rejected or unchecked
  pairs are dropped here, once, instead of in every downstream script.
- `load_verdicts` deduplicates and drops API-error rows so an interrupted / re-run
  `conditions` job can never inflate the AUC (append-mode is safe again).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA, SCANS, JUDGE_OUT

VERDICTS = JUDGE_OUT / "verdicts.jsonl"

# Key that uniquely identifies one judge verdict; used for dedup and for resumability.
VERDICT_KEY = ("id", "version", "condition", "instrument", "judge", "prompt_v")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_pairs(path: str = "pairs.jsonl", validated_only: bool = False) -> list[dict[str, Any]]:
    """Load the corpus. With `validated_only`, keep only human-accepted pairs
    (human_checked=True and not rejected) — the gate every run downstream of the
    corpus must respect."""
    items = load_jsonl(DATA / path)
    if validated_only:
        items = [p for p in items if p.get("human_checked") and not p.get("rejected")]
    return items


def scan_path(id_: str, version: str, instrument: str) -> Path:
    return SCANS / f"{id_}_{version}_{instrument}.json"


def load_scan(id_: str, version: str, instrument: str) -> dict[str, Any]:
    return json.loads(scan_path(id_, version, instrument).read_text(encoding="utf-8"))


def verdict_key(record: dict[str, Any]) -> tuple:
    return tuple(record.get(k) for k in VERDICT_KEY)


def load_verdicts(dedup: bool = True, drop_errors: bool = True):
    """Return verdicts as a DataFrame, deduplicated on VERDICT_KEY (keep last) and with
    API-error rows removed. pandas is imported lazily so pure-stdlib callers stay light."""
    import pandas as pd

    df = pd.read_json(VERDICTS, lines=True)
    if drop_errors and "_error" in df.columns:
        df = df[df["_error"].fillna(False) != True]  # noqa: E712 (pandas mask)
    if dedup:
        present = [k for k in VERDICT_KEY if k in df.columns]
        df = df.drop_duplicates(subset=present, keep="last")
    return df.reset_index(drop=True)
