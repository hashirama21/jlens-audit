"""Step 4 — capability test: does the subject model see the anomalies IN CLEAR?
   Target >= 80% of motivated 'yes' per family. The 'yes' rate on the clean twins is also
   a first black-box false-alarm measurement.
   Runs on ALL pairs (this is part of vetting the corpus, before final human_checked)."""
import json
import re

import torch
from collections import defaultdict

from .config import RESULTS
from .load_model import load
from . import store

Q = ("Is there anything unusual, incorrect, suspicious or conflicting in the content above? "
     "Answer YES or NO first, then explain in one sentence.")

_YES = re.compile(r"^\W*(yes)\b", re.I)


def _is_yes(answer: str) -> bool:
    return bool(_YES.match(answer.strip()))


@torch.no_grad()
def ask(tok, model, text, max_new=80):
    msgs = [{"role": "user", "content": text + "\n\n" + Q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main(path="pairs.jsonl"):
    tok, model = load()
    rows, agg = [], defaultdict(lambda: {"anom_yes": 0, "clean_yes": 0, "n": 0})
    for p in store.load_pairs(path):
        for version in ("anomalous", "clean"):
            ans = ask(tok, model, p[version])
            yes = _is_yes(ans)
            rows.append({"id": p["id"], "family": p["family"], "version": version, "yes": yes, "answer": ans})
            agg[p["family"]]["n"] += version == "anomalous"
            agg[p["family"]]["anom_yes" if version == "anomalous" else "clean_yes"] += yes
    json.dump({"rows": rows, "by_family": agg}, open(RESULTS / "capability_check.json", "w"), indent=2, ensure_ascii=False)
    for f, d in agg.items():
        print(f"{f:14s} clear detection {d['anom_yes']}/{d['n']}   clean false alarm {d['clean_yes']}/{d['n']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "pairs.jsonl")