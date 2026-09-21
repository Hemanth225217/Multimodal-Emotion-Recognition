"""Evaluate the original text+audio baseline on the untouched MELD test set.

Fixed version: the previous copy of this script (now in evaluation/legacy/)
imported the current tri-modal MultimodalFusionModel and called it with only
(text, audio), which raises a TypeError since that class requires a video
tensor too. This script uses models/baseline_model.py, which matches the
checkpoint's actual saved architecture.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, BASELINE_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from models.baseline_model import BaselineModel


def main():
    print(f"Device: {DEVICE}")
    print(f"Model : {BASELINE_MODEL_PATH}\n")

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = BaselineModel().to(DEVICE)
    model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"].to(DEVICE, dtype=torch.float32)
            audio = batch["audio"].to(DEVICE, dtype=torch.float32)
            labels = batch["labels"].reshape(-1).tolist()

            logits = model(text, audio).reshape(-1, NUM_CLASSES)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels)

    print("=== BASELINE TEST METRICS ===")
    print(f"Accuracy           : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Weighted Precision : {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Weighted Recall    : {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Weighted F1        : {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"Macro F1           : {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")

    print("\n=== PER-CLASS REPORT ===")
    print(classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES, digits=4, zero_division=0,
    ))

    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    print("Actual\\Pred ".ljust(12) + "".join(f"{n[:8]:>10s}" for n in EMOTION_NAMES))
    for name, row in zip(EMOTION_NAMES, cm):
        print(name.ljust(12) + "".join(f"{v:>10d}" for v in row))


if __name__ == "__main__":
    main()
