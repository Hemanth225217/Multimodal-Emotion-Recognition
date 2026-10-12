"""Export the per-utterance MELD test predictions of the third-look model (stacking, frozen v3 list) for the demo.

This refits exactly the stacker of final_scoring_stacking.py (same families, same C, validation probabilities only) and
checks that its test metrics equal the recorded third-look result; it makes no new choice and is not a new test look.
Output: final_review/results/final_scoring/third_look_test_predictions.json (predicted class and probabilities).
Usage (repository root): python final_review/tools/export_stacked_predictions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.ensemble_from_cache import metrics  # noqa: E402
from final_review.analysis.final_scoring import family_probs  # noqa: E402
from final_review.analysis.final_scoring_stacking import feats  # noqa: E402

FS = ROOT / "final_review" / "results" / "final_scoring"


def main():
    cand = json.loads((ROOT / "final_review" / "final_candidates_v3_third_look.json").read_text())
    spec, C = cand["datasets"]["meld"], cand["procedure"]["C"]
    val, test, yv, yt, st = {}, {}, None, None, None
    for f, paths in spec["families"].items():
        val[f], yv, _ = family_probs(paths, "val", yv)
        test[f], yt, st = family_probs(paths, "test", yt)
    clf = LogisticRegression(C=C, max_iter=5000).fit(feats(val, val), yv)
    proba = clf.predict_proba(feats(val, test))
    pred = proba.argmax(1)
    m = metrics(yt, pred)
    rec = json.loads((FS / "TEST_SCORED_meld_THIRD_LOOK.json").read_text())["test"]
    for k, v in rec.items():
        assert abs(m[k] - v) < 1e-9, f"{k}: {m[k]} != recorded {v}"
    out = {"source": "third look (stacking, frozen v3 list); reproduces TEST_SCORED_meld_THIRD_LOOK.json",
           "metrics": m, "sizes": [int(x) for x in st], "pred": pred.tolist(),
           "proba": np.round(proba, 4).tolist()}
    (FS / "third_look_test_predictions.json").write_text(json.dumps(out))
    print("reproduced:", {k: round(v * 100, 2) for k, v in m.items()}, "| utterances:", len(pred))


if __name__ == "__main__":
    main()
