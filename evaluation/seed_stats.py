"""How much does a single trained instance of one recipe move with its seed,
and what does averaging several seeds buy?

Reads the cached probabilities (evaluation/cache_member_probs.py) of several
seeds of the same recipe, prints each seed's solo scores, the mean and
standard deviation across seeds, and the score of the equal-weight average
of all the seeds -- the quantity that tells you whether "train a few seeds
and average" is worth doing versus trusting one run.

Usage:
  python -m evaluation.seed_stats --family D=D_s42,D_s1,D_s2,D_s3 --family F=F_s42,F_s1,F_s2,F_s3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "prob_cache"


def metrics(labels, probs):
    preds = probs.argmax(-1).tolist()
    return (
        accuracy_score(labels, preds) * 100,
        f1_score(labels, preds, average="weighted", zero_division=0) * 100,
        f1_score(labels, preds, average="macro", zero_division=0) * 100,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", action="append", required=True, help="NAME=member1,member2,...")
    args = parser.parse_args()

    for spec in args.family:
        family, members = spec.split("=", 1)
        names = members.split(",")
        entries = {n: torch.load(CACHE_DIR / f"{n}.pt", map_location="cpu", weights_only=False) for n in names}
        val_labels = entries[names[0]]["val_labels"]
        test_labels = entries[names[0]]["test_labels"]

        print(f"\n=== family {family}: {len(names)} seeds (accuracy / weighted F1 / macro F1, %) ===")
        print(f"{'member':<10} {'validation':<26} {'test':<26}")
        val_rows, test_rows = [], []
        for n in names:
            v = metrics(val_labels, entries[n]["val"])
            t = metrics(test_labels, entries[n]["test"])
            val_rows.append(v)
            test_rows.append(t)
            print(f"{n:<10} {v[0]:6.2f} / {v[1]:6.2f} / {v[2]:6.2f}   {t[0]:6.2f} / {t[1]:6.2f} / {t[2]:6.2f}")

        val_arr, test_arr = np.array(val_rows), np.array(test_rows)
        print(f"{'mean':<10} {val_arr[:,0].mean():6.2f} / {val_arr[:,1].mean():6.2f} / {val_arr[:,2].mean():6.2f}   "
              f"{test_arr[:,0].mean():6.2f} / {test_arr[:,1].mean():6.2f} / {test_arr[:,2].mean():6.2f}")
        print(f"{'std':<10} {val_arr[:,0].std(ddof=1):6.2f} / {val_arr[:,1].std(ddof=1):6.2f} / {val_arr[:,2].std(ddof=1):6.2f}   "
              f"{test_arr[:,0].std(ddof=1):6.2f} / {test_arr[:,1].std(ddof=1):6.2f} / {test_arr[:,2].std(ddof=1):6.2f}")
        print(f"{'range':<10} test accuracy {test_arr[:,0].min():.2f}..{test_arr[:,0].max():.2f}   "
              f"test weighted F1 {test_arr[:,1].min():.2f}..{test_arr[:,1].max():.2f}")

        avg_val = sum(e["val"] for e in entries.values()) / len(entries)
        avg_test = sum(e["test"] for e in entries.values()) / len(entries)
        v, t = metrics(val_labels, avg_val), metrics(test_labels, avg_test)
        print(f"{'seed-avg':<10} {v[0]:6.2f} / {v[1]:6.2f} / {v[2]:6.2f}   {t[0]:6.2f} / {t[1]:6.2f} / {t[2]:6.2f}"
              f"   <- equal-weight average of the {len(names)} seeds")


if __name__ == "__main__":
    main()
