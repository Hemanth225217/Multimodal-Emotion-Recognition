"""Ensemble analysis over cached member probabilities (see
evaluation/cache_member_probs.py) with an honest selection protocol.

Protocol -- this is the point of the script:
  * Every candidate ensemble (every subset of the candidate members, equal
    weight average of softmax probabilities) is scored on VALIDATION.
  * The subset with the best validation weighted F1 is the one reported.
    Its test score is looked at once and is the headline number.
  * The best-on-test subset is also printed, clearly labelled optimistic:
    picking the maximum over many ensembles on the very set you report is a
    winner's-curse bias, which is exactly how earlier numbers in this
    project's history were produced.
  * A dialogue-level bootstrap (resampling whole dialogues, since
    utterances inside a dialogue are not independent) gives 95% confidence
    intervals, because on 2,610 test utterances a difference of a few tenths
    of a point is smaller than the sampling noise.

Seed-averaged members: --group NAME=m1,m2,... builds a virtual member whose
probabilities are the mean of the listed members' (e.g. several seeds of the
same recipe).

Example:
  python -m evaluation.ensemble_from_cache --candidates B,D_avg,F_avg,H2_base,H3_large \
      --group D_avg=D_s42,D_s1,D_s2,D_s3 --group F_avg=F_s42,F_s1,F_s2,F_s3
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EMOTION_NAMES, NUM_CLASSES

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "prob_cache"
AMB_DSGDN = {"accuracy": 0.6607, "weighted_f1": 0.6618}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


def load_cache(names):
    cache = {}
    for name in names:
        path = CACHE_DIR / f"{name}.pt"
        if not path.exists():
            raise SystemExit(f"No cached probabilities for '{name}' ({path}) -- run cache_member_probs first.")
        cache[name] = torch.load(path, map_location="cpu", weights_only=False)
    return cache


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def fmt(m):
    return f"{m['accuracy']*100:5.2f}% / {m['weighted_f1']*100:5.2f}% / {m['macro_f1']*100:5.2f}%"


# Training-split class counts (neutral, surprise, fear, sadness, joy, disgust,
# anger), printed by the training scripts as "Class distribution". Stored here
# so the optional prior correction needs no dataset files.
TRAIN_CLASS_COUNTS = [4709, 1205, 268, 684, 1743, 271, 1108]

_DIALOGUE_SIZES = {}


def dialogue_sizes(split):
    """Utterances per dialogue, for the dialogue-level bootstrap. Taken from
    the caches (so the committed caches reproduce everything standalone);
    caches written before sizes were stored fall back to the dataset files."""
    if split in _DIALOGUE_SIZES:
        return _DIALOGUE_SIZES[split]
    from training.dataset import MELDDataset
    dataset = MELDDataset(split=split)
    return [len(dataset[i]["labels"]) for i in range(len(dataset))]


def train_class_priors():
    counts = torch.tensor(TRAIN_CLASS_COUNTS, dtype=torch.float32)
    return counts / counts.sum()


def prior_adjust(probs, priors, tau):
    """Divide each class probability by prior**tau and renormalize.
    tau=0 is a no-op; tau=1 is the fully 'balanced' posterior correction.
    One parameter, fit on validation -- unlike a per-member weight grid it
    has almost no room to overfit."""
    adjusted = probs / priors.pow(tau)
    return adjusted / adjusted.sum(dim=-1, keepdim=True)


def bootstrap_ci(labels, preds, sizes, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    labels, preds = np.asarray(labels), np.asarray(preds)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    per_dialogue = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))]
    rng = np.random.default_rng(seed)
    accs, wf1s = [], []
    for _ in range(samples):
        picked = rng.integers(0, len(sizes), size=len(sizes))
        idx = np.concatenate([per_dialogue[i] for i in picked])
        accs.append(accuracy_score(labels[idx], preds[idx]))
        wf1s.append(f1_score(labels[idx], preds[idx], average="weighted", zero_division=0))
    return (np.percentile(accs, [2.5, 97.5]), np.percentile(wf1s, [2.5, 97.5]))


def paired_bootstrap_diff(labels, preds_a, preds_b, sizes, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Dialogue-level paired bootstrap of (A - B) for accuracy and weighted
    F1, resampling the same dialogues for both so the comparison cancels the
    noise they share. Returns (mean diff, 95% CI, P(diff > 0)) per metric."""
    labels, a, b = np.asarray(labels), np.asarray(preds_a), np.asarray(preds_b)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    per_dialogue = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))]
    rng = np.random.default_rng(seed)
    d_acc, d_wf1 = [], []
    for _ in range(samples):
        picked = rng.integers(0, len(sizes), size=len(sizes))
        idx = np.concatenate([per_dialogue[i] for i in picked])
        d_acc.append(accuracy_score(labels[idx], a[idx]) - accuracy_score(labels[idx], b[idx]))
        d_wf1.append(f1_score(labels[idx], a[idx], average="weighted", zero_division=0)
                     - f1_score(labels[idx], b[idx], average="weighted", zero_division=0))
    out = {}
    for name, d in (("accuracy", d_acc), ("weighted_f1", d_wf1)):
        d = np.asarray(d)
        out[name] = (d.mean(), np.percentile(d, [2.5, 97.5]), float((d > 0).mean()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="comma-separated member/group names to enumerate subsets of")
    parser.add_argument("--group", action="append", default=[], help="NAME=m1,m2,... seed-average virtual member")
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument("--reference", default=None,
                        help="cached member to compare the chosen ensemble against with a paired bootstrap (e.g. B)")
    parser.add_argument("--tau", action="store_true",
                        help="also fit a single minority-class prior-correction exponent on validation")
    args = parser.parse_args()

    groups = {}
    for spec in args.group:
        gname, members = spec.split("=", 1)
        groups[gname] = members.split(",")
    candidates = args.candidates.split(",")

    needed = set()
    for c in candidates:
        needed.update(groups[c] if c in groups else [c])
    cache = load_cache(sorted(needed))

    ref = next(iter(cache.values()))
    for name, entry in cache.items():
        assert entry["val_labels"] == ref["val_labels"], f"val label mismatch for {name}"
        assert entry["test_labels"] == ref["test_labels"], f"test label mismatch for {name}"
    val_labels, test_labels = ref["val_labels"], ref["test_labels"]
    for split in ("val", "test"):
        if all(entry.get(f"{split}_sizes") is not None for entry in cache.values()):
            sizes = ref[f"{split}_sizes"]
            assert sum(sizes) == len(ref[f"{split}_labels"]), f"{split} dialogue sizes do not match the labels"
            assert all(entry[f"{split}_sizes"] == sizes for entry in cache.values()),                 f"{split} dialogue sizes differ between members"
            _DIALOGUE_SIZES[split] = sizes

    probs = {"val": {}, "test": {}}
    for c in candidates:
        parts = groups[c] if c in groups else [c]
        for split in ("val", "test"):
            probs[split][c] = sum(cache[p][split] for p in parts) / len(parts)

    print("=== SOLO CANDIDATES (accuracy / weighted F1 / macro F1) ===")
    print(f"{'member':<14} {'validation':<32} {'test':<32}")
    for c in candidates:
        v = metrics(val_labels, probs["val"][c].argmax(-1).tolist())
        t = metrics(test_labels, probs["test"][c].argmax(-1).tolist())
        print(f"{c:<14} {fmt(v):<32} {fmt(t):<32}")

    rows = []
    for r in range(args.min_size, len(candidates) + 1):
        for combo in combinations(candidates, r):
            avg = {s: sum(probs[s][n] for n in combo) / len(combo) for s in ("val", "test")}
            v = metrics(val_labels, avg["val"].argmax(-1).tolist())
            t = metrics(test_labels, avg["test"].argmax(-1).tolist())
            rows.append((combo, v, t, avg["test"]))

    rows.sort(key=lambda row: (row[1]["weighted_f1"], row[1]["accuracy"]), reverse=True)
    print(f"\n=== ALL {len(rows)} SUBSETS, equal-weight, ranked by VALIDATION weighted F1 (top 12) ===")
    print(f"{'subset':<44} {'validation':<32} {'test':<32}")
    for combo, v, t, _ in rows[:12]:
        print(f"{'+'.join(combo):<44} {fmt(v):<32} {fmt(t):<32}")

    chosen, v, t, chosen_test_probs = rows[0]
    preds = chosen_test_probs.argmax(-1).tolist()
    print(f"\n=== HONEST HEADLINE: subset chosen on validation = {'+'.join(chosen)} ===")
    print(f"validation: {fmt(v)}")
    print(f"test (looked at once): {fmt(t)}")
    acc_ci, wf1_ci = bootstrap_ci(test_labels, preds, dialogue_sizes("test"))
    print(f"95% dialogue-bootstrap CI  accuracy: [{acc_ci[0]*100:.2f}%, {acc_ci[1]*100:.2f}%]   "
          f"weighted F1: [{wf1_ci[0]*100:.2f}%, {wf1_ci[1]*100:.2f}%]")
    print(f"AMB-DSGDN reported {AMB_DSGDN['accuracy']*100:.2f}% / {AMB_DSGDN['weighted_f1']*100:.2f}% -> "
          f"accuracy {'INSIDE' if acc_ci[0] <= AMB_DSGDN['accuracy'] <= acc_ci[1] else 'outside'} our CI, "
          f"weighted F1 {'INSIDE' if wf1_ci[0] <= AMB_DSGDN['weighted_f1'] <= wf1_ci[1] else 'outside'} our CI")
    print(classification_report(test_labels, preds, labels=list(range(NUM_CLASSES)),
                                target_names=EMOTION_NAMES, digits=4, zero_division=0))

    if args.reference:
        ref_name = args.reference
        if ref_name not in probs["test"]:
            ref_cache = load_cache([ref_name])[ref_name]
            ref_preds = ref_cache["test"].argmax(-1).tolist()
        else:
            ref_preds = probs["test"][ref_name].argmax(-1).tolist()
        diff = paired_bootstrap_diff(test_labels, preds, ref_preds, dialogue_sizes("test"))
        print(f"=== PAIRED BOOTSTRAP: chosen ensemble minus {ref_name} (test, dialogue-level) ===")
        for metric, (mean, ci, p_pos) in diff.items():
            print(f"  {metric:<12} mean {mean*100:+.2f} pts   95% CI [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]   "
                  f"P(ensemble better) = {p_pos:.3f}")

    if args.tau:
        priors = train_class_priors()
        avg_val = sum(probs["val"][n] for n in chosen) / len(chosen)
        avg_test = sum(probs["test"][n] for n in chosen) / len(chosen)
        print(f"\n=== MINORITY-CLASS PRIOR CORRECTION on the chosen subset ({'+'.join(chosen)}) ===")
        print("train class priors:", {EMOTION_NAMES[i]: round(priors[i].item(), 4) for i in range(NUM_CLASSES)})
        print(f"{'tau':>5}  {'validation':<32} {'test (shown for transparency)':<32}")
        best = None
        for tau in [round(x * 0.1, 1) for x in range(-5, 11)]:
            tv = metrics(val_labels, prior_adjust(avg_val, priors, tau).argmax(-1).tolist())
            tt = metrics(test_labels, prior_adjust(avg_test, priors, tau).argmax(-1).tolist())
            print(f"{tau:>5}  {fmt(tv):<32} {fmt(tt):<32}")
            key = (tv["weighted_f1"], -abs(tau))
            if best is None or key > best[0]:
                best = (key, tau, tv, tt)
        _, tau_star, tv, tt = best
        print(f"tau chosen on VALIDATION weighted F1: {tau_star}  -> validation {fmt(tv)} | test {fmt(tt)}")
        tau_preds = prior_adjust(avg_test, priors, tau_star).argmax(-1).tolist()
        a_ci, w_ci = bootstrap_ci(test_labels, tau_preds, dialogue_sizes("test"))
        print(f"95% dialogue-bootstrap CI  accuracy: [{a_ci[0]*100:.2f}%, {a_ci[1]*100:.2f}%]   "
              f"weighted F1: [{w_ci[0]*100:.2f}%, {w_ci[1]*100:.2f}%]")
        print(classification_report(test_labels, tau_preds, labels=list(range(NUM_CLASSES)),
                                    target_names=EMOTION_NAMES, digits=4, zero_division=0))

    full = next(row for row in rows if len(row[0]) == len(candidates))
    print(f"=== FULL SET ({'+'.join(full[0])}), equal weight: validation {fmt(full[1])} | test {fmt(full[2])}")

    best_test = max(rows, key=lambda row: (row[2]["weighted_f1"], row[2]["accuracy"]))
    print(f"=== OPTIMISTIC (best of {len(rows)} subsets chosen ON TEST -- winner's-curse biased, not a valid headline): "
          f"{'+'.join(best_test[0])}: {fmt(best_test[2])}")


if __name__ == "__main__":
    main()
