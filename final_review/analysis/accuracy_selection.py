"""Choose the MELD ensemble procedure for ACCURACY by cross-validation INSIDE the validation set (pre-registered 12 Oct).

Procedures (all use only validation probabilities; the test set is never read here):
  M1  greedy forward selection by accuracy (start from the best family, add the family that raises accuracy most, stop
      when the gain is below 0.1 points), equal-weight average of the selected families
  M2  M1 followed by a class-prior correction p_k / prior_k ** tau, tau in [-1.0, 1.0] fitted on the training folds
  M3  multinomial logistic regression on the concatenated log-probabilities of all families (C fitted on the training
      folds by an inner split), i.e. stacking
Cross-validation: 5 folds of whole dialogues, repeated 3 times with different shuffles. The procedure with the best mean
held-out accuracy is reported as the choice; final_scoring_accuracy.py then refits it on the full validation set.
Usage: python final_review/analysis/accuracy_selection.py CANDIDATES.json [--extra FAMILY=path1,path2,...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.ensemble_from_cache import TRAIN_CLASS_COUNTS  # noqa: E402

PRIOR = np.array(TRAIN_CLASS_COUNTS, dtype=np.float64) / sum(TRAIN_CLASS_COUNTS)
TAUS = np.round(np.arange(-1.0, 1.01, 0.1), 2)
CS = [0.01, 0.03, 0.1, 0.3, 1.0]


def load_family(paths, split):
    cs = [torch.load(ROOT / p if not Path(p).is_absolute() else p, map_location="cpu", weights_only=False) for p in paths]
    y = np.asarray(cs[0][f"{split}_labels"])
    for c in cs:
        assert np.array_equal(np.asarray(c[f"{split}_labels"]), y)
    return np.mean([np.asarray(c[split], dtype=np.float64) for c in cs], axis=0), y, list(cs[0][f"{split}_sizes"])


def acc(y, p):
    return accuracy_score(y, p.argmax(1))


def greedy(fams, idx, y):
    names = list(fams)
    order = sorted(names, key=lambda f: acc(y[idx], fams[f][idx]), reverse=True)
    chosen, best = [order[0]], acc(y[idx], fams[order[0]][idx])
    while True:
        rest = [f for f in names if f not in chosen]
        if not rest:
            break
        sc = {f: acc(y[idx], np.mean([fams[g][idx] for g in chosen + [f]], 0)) for f in rest}
        f, s = max(sc.items(), key=lambda kv: kv[1])
        if s - best < 0.001:
            break
        chosen.append(f); best = s
    return chosen


def prior_adjust(p, tau):
    q = p / PRIOR ** tau
    return q / q.sum(1, keepdims=True)


def fit_m1(fams, idx, y):
    ch = greedy(fams, idx, y)
    return lambda j: np.mean([fams[f][j] for f in ch], 0), {"families": ch}


def fit_m2(fams, idx, y):
    ch = greedy(fams, idx, y)
    base = np.mean([fams[f] for f in ch], 0)
    tau = max(TAUS, key=lambda t: acc(y[idx], prior_adjust(base[idx], t)))
    return lambda j: prior_adjust(np.mean([fams[f][j] for f in ch], 0), tau), {"families": ch, "tau": float(tau)}


def features(fams, j):
    return np.concatenate([np.log(np.clip(fams[f][j], 1e-6, 1)) for f in sorted(fams)], axis=1)


def fit_m3(fams, idx, y, rng):
    # inner split of the training dialogues' utterances for C
    perm = rng.permutation(idx)
    cut = int(0.8 * len(perm))
    a, b = perm[:cut], perm[cut:]
    best_c = max(CS, key=lambda c: accuracy_score(
        y[b], LogisticRegression(C=c, max_iter=2000).fit(features(fams, a), y[a]).predict(features(fams, b))))
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(features(fams, idx), y[idx])
    return (lambda j: clf.predict_proba(features(fams, j))), {"C": best_c}


def dialogue_folds(sizes, k, rng):
    offs = np.concatenate([[0], np.cumsum(sizes)])
    dias = rng.permutation(len(sizes))
    folds = np.array_split(dias, k)
    return [np.concatenate([np.arange(offs[d], offs[d + 1]) for d in sorted(f)]) for f in folds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--extra", action="append", default=[], help="FAMILY=path1,path2 (added to the MELD families)")
    ap.add_argument("--out", default=str(ROOT / "final_review/results/final_scoring/accuracy_selection_cv.json"))
    args = ap.parse_args()
    spec = json.load(open(args.candidates))["datasets"]["meld"]["families"]
    for e in args.extra:
        k, v = e.split("=", 1)
        spec[k] = v.split(",")
    fams, y, sizes = {}, None, None
    for f, paths in spec.items():
        fams[f], y, sizes = load_family(paths, "val")
    n = len(y)
    print(f"{len(fams)} families, {n} validation utterances, {len(sizes)} dialogues")
    res = {m: {"acc": [], "wf1": []} for m in ("M1", "M2", "M3")}
    choices = {m: [] for m in res}
    for rep in range(3):
        rng = np.random.default_rng(100 + rep)
        folds = dialogue_folds(sizes, 5, rng)
        for k in range(5):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            for m in res:
                if m == "M1":
                    pred, info = fit_m1(fams, train_idx, y)
                elif m == "M2":
                    pred, info = fit_m2(fams, train_idx, y)
                else:
                    pred, info = fit_m3(fams, train_idx, y, rng)
                p = pred(test_idx)
                res[m]["acc"].append(accuracy_score(y[test_idx], p.argmax(1)))
                res[m]["wf1"].append(f1_score(y[test_idx], p.argmax(1), average="weighted"))
                choices[m].append(info)
    summary = {m: {"cv_accuracy": float(np.mean(v["acc"])), "cv_accuracy_sd": float(np.std(v["acc"])),
                   "cv_weighted_f1": float(np.mean(v["wf1"]))} for m, v in res.items()}
    best = max(summary, key=lambda m: summary[m]["cv_accuracy"])
    for m, s in summary.items():
        print(f"{m}: CV accuracy {s['cv_accuracy']*100:.2f} (SD over folds {s['cv_accuracy_sd']*100:.2f}), CV wF1 {s['cv_weighted_f1']*100:.2f}")
    print("chosen procedure:", best)
    # what each procedure picks on the FULL validation set (for information)
    full = np.arange(n)
    rng = np.random.default_rng(7)
    full_info = {"M1": fit_m1(fams, full, y)[1], "M2": fit_m2(fams, full, y)[1], "M3": fit_m3(fams, full, y, rng)[1]}
    print("fitted on full validation:", full_info)
    Path(args.out).write_text(json.dumps({"families": list(fams), "cv": summary, "chosen": best, "full_fit": full_info,
                                          "fold_choices": choices}, indent=1, default=str))


if __name__ == "__main__":
    main()
