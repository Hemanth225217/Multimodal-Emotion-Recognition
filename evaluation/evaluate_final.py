"""Comprehensive evaluation of the final model on the untouched MELD test set.

Produces overall accuracy/precision/recall/weighted-F1/macro-F1, a full
per-class precision/recall/F1/support table, and the confusion matrix --
everything needed for the results section of the report.
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

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.inference import load_model, predict_dialogue


def evaluate(model, loader, use_speaker):
    all_preds, all_labels = [], []
    for batch in loader:
        text = batch["text"].to(DEVICE, dtype=torch.float32)
        audio = batch["audio"].to(DEVICE, dtype=torch.float32)
        video = batch["video"].to(DEVICE, dtype=torch.float32)
        labels = batch["labels"].reshape(-1).tolist()

        # Only pass real speaker slots for checkpoints that were actually
        # trained with the speaker embedding -- for an older checkpoint
        # that layer is left at random init (see modules/inference.py's
        # strict=False load), and feeding it real slots would inject
        # untrained noise into a model that never learned to use it.
        speaker_slots = batch["speaker_slots"].to(DEVICE, dtype=torch.long) if use_speaker else None

        result = predict_dialogue(model, text, audio, video, speaker_slots=speaker_slots)
        all_preds.extend(result["predictions"].tolist())
        all_labels.extend(labels)

    return all_labels, all_preds


def main():
    print(f"Device: {DEVICE}")
    print(f"Model : {FINAL_MODEL_PATH}\n")

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model, meta = load_model(FINAL_MODEL_PATH)
    if "val_accuracy" in meta:
        print(
            f"Checkpoint (epoch {meta.get('epoch', '?')}): "
            f"val_accuracy={meta['val_accuracy']:.4f} "
            f"val_macro_f1={meta.get('val_macro_f1', float('nan')):.4f}\n"
        )

    use_speaker = "num_speaker_slots" in meta
    print(f"Speaker embedding: {'trained, using real slots' if use_speaker else 'not present in this checkpoint, skipping'}\n")

    labels, preds = evaluate(model, test_loader, use_speaker)

    print("=== OVERALL TEST METRICS ===")
    print(f"Accuracy           : {accuracy_score(labels, preds):.4f}")
    print(f"Weighted Precision : {precision_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"Weighted Recall    : {recall_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"Weighted F1        : {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"Macro F1           : {f1_score(labels, preds, average='macro', zero_division=0):.4f}")

    print("\n=== PER-CLASS REPORT (precision / recall / F1 / support) ===")
    print(classification_report(
        labels, preds, labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES, digits=4, zero_division=0,
    ))

    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("Actual\\Pred ".ljust(12) + "".join(f"{n[:8]:>10s}" for n in EMOTION_NAMES))
    for name, row in zip(EMOTION_NAMES, cm):
        print(name.ljust(12) + "".join(f"{v:>10d}" for v in row))

    print("\n=== PREDICTION DISTRIBUTION ===")
    total = len(preds)
    for class_id, name in enumerate(EMOTION_NAMES):
        count = preds.count(class_id)
        print(f"{name:10s}: {count:5d} ({100.0 * count / total:5.2f}%)")


if __name__ == "__main__":
    main()
