"""Step 7 — the seven conditions. FIXED. Do not add any without human validation.

 1 prompt_only     : the judge reads the text (black-box baseline, the floor to beat)
 2 scan            : the instrument's scan, no text            (x3 instruments = conditions 2/3/4)
 5 scan_permuted   : positions shuffled                        (H3 apophenia)
 6 (implicit)      : the 'clean' items go through every condition (false-alarm rate)
 7 reconstruction  : judge A reconstructs the text from the scan; judge B reads the reconstruction only (H2)

The run is idempotent and resumable: each verdict has a unique key (see store.VERDICT_KEY);
already-computed keys are skipped, so a crashed or re-launched job never double-counts.
"""
import json

from tqdm import tqdm

from .config import RESULTS, JUDGES, PROMPT_VERSIONS, INSTRUMENTS, TOP_K
from .serialize import serialize, permute_positions
from .judge import call, prompt
from . import store


def _load_done() -> set:
    """Keys already present (and not errored) in the verdicts file, for resumability."""
    done = set()
    if store.VERDICTS.exists():
        for line in store.VERDICTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("_error"):
                done.add(store.verdict_key(r))
    return done


def run(pairs_path="pairs.jsonl", judges=None, prompt_versions=None, instruments=None):
    judges = judges or list(JUDGES)
    prompt_versions = prompt_versions or PROMPT_VERSIONS
    instruments = instruments or INSTRUMENTS

    # CLAUDE.md gate: no scan-only condition before the leak audit has been run.
    if instruments and not (RESULTS / "leak_check.csv").exists():
        raise SystemExit("Leak audit missing: run `python -m src.checks leak` first (CLAUDE.md rule).")

    items = store.load_pairs(pairs_path, validated_only=True)
    if not items:
        raise SystemExit(f"No human-validated pairs in data/{pairs_path} "
                         f"(set human_checked=true via `python -m src.gen_pairs review`).")

    done = _load_done()

    def emit(fout, record: dict, judge, text):
        """Skip if already done; otherwise call the judge, write the verdict, record the key."""
        key = store.verdict_key(record)
        if key in done:
            return
        v = call(judge, text)
        record = {**record, **(v if isinstance(v, dict) else {})}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        if not record.get("_error"):
            done.add(key)

    with open(store.VERDICTS, "a", encoding="utf-8") as fout:
        for p in tqdm(items):
            for version in ("anomalous", "clean"):
                text = p[version]
                base = {"id": p["id"], "family": p["family"], "version": version,
                        "y": int(version == "anomalous")}
                for judge in judges:
                    # 1 — prompt only
                    emit(fout, {**base, "condition": "prompt_only", "instrument": None,
                                "judge": judge, "prompt_v": "text"},
                         judge=judge, text=prompt("text_only", text))

                    for inst in instruments:
                        sc = store.load_scan(p["id"], version, inst)["scan"]
                        ser = serialize(sc, TOP_K)
                        for pv in prompt_versions:
                            # 2/3/4 — scan
                            emit(fout, {**base, "condition": "scan", "instrument": inst,
                                        "judge": judge, "prompt_v": pv},
                                 judge=judge, text=prompt(f"judge_{pv}", ser))
                            # 5 — permuted
                            emit(fout, {**base, "condition": "scan_permuted", "instrument": inst,
                                        "judge": judge, "prompt_v": pv},
                                 judge=judge,
                                 text=prompt(f"judge_{pv}", serialize(permute_positions(sc, p["id"]), TOP_K)))

                        # 7 — reconstruction (judge A reconstructs, judge B reads)
                        other = next((j for j in judges if j != judge), judge)
                        rkey = store.verdict_key({**base, "condition": "reconstruction",
                                                  "instrument": inst, "judge": f"{judge}->{other}",
                                                  "prompt_v": "text"})
                        if rkey in done:
                            continue
                        recon = call(judge, prompt("reconstruct", ser), want_json=False, max_tokens=1200)
                        if not isinstance(recon, str):  # reconstruction call errored
                            rec = {**base, "condition": "reconstruction", "instrument": inst,
                                   "judge": f"{judge}->{other}", "prompt_v": "text",
                                   "anomaly": False, "confidence": 0.0, "_error": True}
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()
                            continue
                        v = call(other, prompt("text_only", recon))
                        rec = {**base, "condition": "reconstruction", "instrument": inst,
                               "judge": f"{judge}->{other}", "prompt_v": "text",
                               "_recon": recon[:2000], **(v if isinstance(v, dict) else {})}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        if not rec.get("_error"):
                            done.add(rkey)
    print(f"verdicts -> {store.VERDICTS}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "pairs.jsonl")