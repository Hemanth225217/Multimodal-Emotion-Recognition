"""Third, disclosed MELD test look (pre-registered 12 Oct 2026): logistic-regression stacking of all candidate families,
the procedure chosen by cross-validation inside the validation set for ACCURACY (accuracy_selection.py: M3).

The stacker (multinomial logistic regression on the concatenated log-probabilities of every family, C = 0.01 as fitted
on the full validation set) is trained on the VALIDATION probabilities only, then applied once to the test
probabilities. Reported: accuracy, weighted F1, macro F1, dialogue-bootstrap 95 % intervals and the paired comparison
with the pre-final ensemble. Refuses to run twice (marker TEST_SCORED_meld_THIRD_LOOK.json).
Usage: python final_review/analysis/final_scoring_stacking.py CANDIDATES_V3.json --score-test
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.ensemble_from_cache import metrics, bootstrap_ci, paired_bootstrap_diff  # noqa: E402
from final_review.analysis.final_scoring import family_probs  # noqa: E402

OUT = ROOT / "final_review" / "results" / "final_scoring" / "TEST_SCORED_meld_THIRD_LOOK.json"


def feats(fams, split_probs):
    return np.concatenate([np.log(np.clip(split_probs[f], 1e-6, 1)) for f in sorted(fams)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--score-test", action="store_true")
    args = ap.parse_args()
    raw = Path(args.candidates).read_bytes()
    cand = json.loads(raw)
    spec = cand["datasets"]["meld"]
    C = cand["procedure"]["C"]
    print("candidates sha256:", hashlib.sha256(raw).hexdigest()[:16], "commit", cand.get("commit"), "| C =", C,
          "|", len(spec["families"]), "families")
    val, yv, sv = {}, None, None
    for f, paths in spec["families"].items():
        val[f], yv, sv = family_probs(paths, "val", yv)
    clf = LogisticRegression(C=C, max_iter=5000).fit(feats(val, val), yv)
    pv = clf.predict(feats(val, val))
    print("VALIDATION (in-sample, optimistic):", {k: round(v * 100, 2) for k, v in metrics(yv, pv).items()})
    if not args.score_test:
        print("test not scored (pass --score-test once the list is frozen and committed)")
        return
    if OUT.exists():
        print("REFUSED: the third look was already scored:", OUT)
        return
    test, yt, st = {}, None, None
    for f, paths in spec["families"].items():
        test[f], yt, st = family_probs(paths, "test", yt)
    pt = clf.predict(feats(val, test))
    m = metrics(yt, pt)
    ca, cw = bootstrap_ci(yt, pt, st)
    ref = np.mean([family_probs(p, "test", yt)[0] for p in spec["reference"]["members"].values()], axis=0).argmax(1)
    d = paired_bootstrap_diff(yt, pt, ref, st)
    res = {"look": "third (disclosed)", "procedure": cand["procedure"], "commit": cand.get("commit"),
           "candidates_sha256": hashlib.sha256(raw).hexdigest()[:16], "scored_at": datetime.now().isoformat(timespec="seconds"),
           "test": m, "ci95": {"accuracy": [float(x) for x in ca], "weighted_f1": [float(x) for x in cw]},
           "reference": {"name": spec["reference"]["name"], "metrics": metrics(yt, ref)},
           "minus_reference": {k: {"mean": float(v[0]), "ci95": [float(x) for x in v[1]], "p_greater": v[2]} for k, v in d.items()}}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("TEST (third look):", {k: round(v * 100, 2) for k, v in m.items()},
          "| 95% CI acc", [round(x * 100, 2) for x in ca], "wF1", [round(x * 100, 2) for x in cw])
    for k, v in res["minus_reference"].items():
        print(f"  minus pre-final {k}: {v['mean']*100:+.2f} [{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}]")


if __name__ == "__main__":
    main()
