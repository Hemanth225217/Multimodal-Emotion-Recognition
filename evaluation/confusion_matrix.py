"""Renders the final model's test-set confusion matrix as a PNG for the report."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, PROJECT_ROOT, DEVICE
from training.dataset import MELDDataset
from modules.inference import load_model, predict_dialogue


def main():
    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model, _ = load_model(FINAL_MODEL_PATH)

    all_preds, all_labels = [], []
    for batch in test_loader:
        text = batch["text"].to(DEVICE, dtype=torch.float32)
        audio = batch["audio"].to(DEVICE, dtype=torch.float32)
        video = batch["video"].to(DEVICE, dtype=torch.float32)
        labels = batch["labels"].reshape(-1).tolist()
        result = predict_dialogue(model, text, audio, video)
        all_preds.extend(result["predictions"].tolist())
        all_labels.extend(labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, matrix, title, fmt in [
        (axes[0], cm, "Confusion Matrix (counts)", "d"),
        (axes[1], cm_normalized, "Confusion Matrix (row-normalized)", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(EMOTION_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        threshold = matrix.max() / 2
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                value = matrix[i, j]
                color = "white" if value > threshold else "black"
                ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Final Model — MELD Test Set Confusion Matrix", fontsize=14)
    fig.tight_layout()

    output_path = PROJECT_ROOT / "evaluation" / "confusion_matrix_final.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
