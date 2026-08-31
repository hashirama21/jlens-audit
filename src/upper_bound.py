"""Bonus (1h) — upper bound: does information about the anomaly exist LINEARLY in the residual
stream, independent of any readout? Strongly regularized logistic probe, leave-one-PAIR-out,
layer by layer, on the activation at the anomaly position vs the matching position of the twin.

Two leakage traps are avoided on purpose:
- leave-one-PAIR-out (GroupKFold by pair id): the anomalous item and its near-identical twin
  must never be split across train/test, otherwise the probe memorizes the pair.
- the StandardScaler lives INSIDE the CV pipeline, so test-fold statistics never leak into
  the fit.
n=40 pairs: FRAGILE. Present with that caveat, never as a headline result."""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .config import RESULTS
from .load_model import get_resid, layers
from . import store


def main():
    pairs = store.load_pairs("pairs.jsonl", validated_only=True)
    X = {L: [] for L in layers()}
    y, groups = [], []
    for gi, p in enumerate(pairs):
        span = p.get("anomaly_token_span")
        if not span:
            continue
        for version, lab in (("anomalous", 1), ("clean", 0)):
            H, ids = get_resid(p[version])
            pos = min(span[1], len(ids) - 1)
            for L in layers():
                X[L].append(H[L][pos].float().cpu().numpy())
            y.append(lab)
            groups.append(gi)  # same group id for both members of a pair
    y, groups = np.array(y), np.array(groups)

    res = {}
    cv = LeaveOneGroupOut()
    for L in layers():
        Xl = np.stack(X[L])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
        s = cross_val_predict(clf, Xl, y, cv=cv, groups=groups, method="decision_function")
        res[L] = float(roc_auc_score(y, s))
    json.dump(res, open(RESULTS / "upper_bound_probe.json", "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()