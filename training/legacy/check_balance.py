import sys
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import MELDDataset


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MELD TRAINING DATA BALANCE CHECK")
    print("=" * 70)

    # --------------------------------------------------
    # Load training dataset
    # --------------------------------------------------

    print("\nLoading training dataset...")

    dataset = MELDDataset("train")

    print(f"\nTraining dialogues: {len(dataset)}")

    # --------------------------------------------------
    # Count emotions
    # --------------------------------------------------

    emotion_names = {
        0: "neutral",
        1: "surprise",
        2: "fear",
        3: "sadness",
        4: "joy",
        5: "disgust",
        6: "anger"
    }

    counts = Counter()

    print("\nCounting training utterances...")

    for i in range(len(dataset)):

        sample = dataset[i]

        labels = sample["labels"]

        # Flatten labels
        labels = labels.reshape(-1).tolist()

        for label in labels:
            counts[int(label)] += 1

    # --------------------------------------------------
    # Display distribution
    # --------------------------------------------------

    total = sum(counts.values())

    print("\n" + "=" * 70)
    print("ACTUAL TRAINING DISTRIBUTION")
    print("=" * 70)

    for class_id in range(7):

        emotion = emotion_names[class_id]

        count = counts[class_id]

        percentage = (count / total) * 100 if total > 0 else 0

        print(
            f"{emotion:<10}: {count:>7} "
            f"({percentage:>6.2f}%)"
        )

    print("-" * 70)
    print(f"{'TOTAL':<10}: {total:>7}")
    print("=" * 70)

    # --------------------------------------------------
    # Check balance against target
    # --------------------------------------------------

    target = 5000

    print("\nBALANCING CHECK")
    print("=" * 70)

    print(f"Target minority count: {target}")

    for class_id in range(1, 7):

        emotion = emotion_names[class_id]

        count = counts[class_id]

        if count >= target:
            status = "OK"
            required = 0
        else:
            status = "NEEDS MORE"
            required = target - count

        print(
            f"{emotion:<10}: "
            f"{count:>7} -> "
            f"{target:>7} | "
            f"Additional needed: {required:>7} | "
            f"{status}"
        )

    print("=" * 70)

    print("\nNOTE:")
    print("This script only CHECKS the original training dataset.")
    print("It does NOT create, copy, or modify any files.")
    print("=" * 70)


if __name__ == "__main__":
    main()