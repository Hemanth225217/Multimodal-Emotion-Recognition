"""Why do Fear and Disgust fail? A data-level look at the ensemble's errors.

Uses cached member probabilities (evaluation/cache_member_probs.py) and the
raw test utterances (same order as the cached labels), so it needs no
checkpoints. For each minority class it reports where the true examples go,
how long they are compared to the rest, how recall varies with utterance
length, and a handful of concrete examples.

Usage:
  python -m evaluation.minority_error_analysis --members D_orig,F_orig,H3_large
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EMOTION_NAMES, NUM_CLASSES
from training.dataset_finetune import FinetuneMELDDataset

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "prob_cache"
FEAR, DISGUST = 2, 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", required=True, help="comma-separated cached members to average")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    caches = [torch.load(CACHE_DIR / f"{m}.pt", map_location="cpu", weights_only=False)
              for m in args.members.split(",")]
    labels = np.array(caches[0]["test_labels"])
    probs = sum(c["test"] for c in caches) / len(caches)
    preds = probs.argmax(-1).numpy()

    dataset = FinetuneMELDDataset(split="test")
    texts = []
    for i in range(len(dataset)):
        texts.extend(dataset[i]["texts"])
    assert len(texts) == len(labels), f"text/label misalignment: {len(texts)} vs {len(labels)}"
    words = np.array([len(t.split()) for t in texts])

    print(f"\nEnsemble of {args.members}: test accuracy {(preds == labels).mean()*100:.2f}%\n")
    for cls in (FEAR, DISGUST):
        name = EMOTION_NAMES[cls]
        mask = labels == cls
        n = int(mask.sum())
        print("=" * 72)
        print(f"{name.upper()}: {n} test utterances; predicted as {name} overall: {int((preds == cls).sum())}")
        row = np.bincount(preds[mask], minlength=NUM_CLASSES)
        print("  where the true examples go: " +
              ", ".join(f"{EMOTION_NAMES[j]} {row[j]} ({row[j]/n*100:.0f}%)" for j in np.argsort(-row) if row[j]))
        print(f"  utterance length (words): {name} mean {words[mask].mean():.1f}, median {np.median(words[mask]):.0f}; "
              f"all others mean {words[~mask].mean():.1f}, median {np.median(words[~mask]):.0f}")
        for lo, hi, label in ((0, 2, "1-2 words"), (3, 5, "3-5 words"), (6, 10, "6-10 words"), (11, 999, "11+ words")):
            bucket = mask & (words >= lo) & (words <= hi)
            if bucket.sum():
                print(f"  {label:<11}: {int(bucket.sum()):>3} true {name} utterances, "
                      f"recall {((preds == cls) & bucket).sum() / bucket.sum() * 100:5.1f}%")
        p_true = probs[torch.from_numpy(mask), cls].numpy()
        print(f"  model's probability for the true class on these: mean {p_true.mean():.3f}, "
              f"max {p_true.max():.3f}; share with p>0.25: {(p_true > 0.25).mean()*100:.0f}%")
        print(f"  examples (text -> predicted):")
        order = np.where(mask)[0]
        for idx in order[:args.examples]:
            print(f"    {texts[idx][:70]!r:<74} -> {EMOTION_NAMES[preds[idx]]}")

    print("=" * 72)
    print("Other classes' utterance length (mean words): " +
          ", ".join(f"{EMOTION_NAMES[c]} {words[labels == c].mean():.1f}" for c in range(NUM_CLASSES)))


if __name__ == "__main__":
    main()
