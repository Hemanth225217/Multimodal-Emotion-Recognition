"""Cheap ensemble test: average two already-trained checkpoints' softmax
probabilities, no retraining required.

Both checkpoints were trained on the same frozen RoBERTa-base text features
(only the model/training recipe differs), so they can be fed the same
dataset batch and their probabilities averaged directly:
  - CHECKPOINT_A: RoBERTa + auxiliary losses + adaptive dropout, no speaker
    embedding (git commit c5d115a).
  - CHECKPOINT_B: same + dialogue-relative speaker embeddings (the current
    best, config.FINAL_MODEL_PATH).

This only answers "does averaging these two specific checkpoints help" --
it is not a claim that ensembling in general would or wouldn't help with
differently-trained members (e.g. different seeds, or a DistilBERT member,
which would need their own matching text features and are not tested here).
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.inference import load_model

CHECKPOINT_A_PATH = Path(os.environ.get("ENSEMBLE_CKPT_A", ""))


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    if not CHECKPOINT_A_PATH.exists():
        raise FileNotFoundError(
            f"Set ENSEMBLE_CKPT_A to the first checkpoint's path (got: {CHECKPOINT_A_PATH})"
        )

    print(f"Checkpoint A (no speaker): {CHECKPOINT_A_PATH}")
    print(f"Checkpoint B (current best): {FINAL_MODEL_PATH}\n")

    model_a, meta_a = load_model(CHECKPOINT_A_PATH)
    model_b, meta_b = load_model(FINAL_MODEL_PATH)
    use_speaker_b = "num_speaker_slots" in meta_b

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    labels_all, preds_a, preds_b, preds_ensemble = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            text = batch["text"].to(DEVICE, dtype=torch.float32)
            audio = batch["audio"].to(DEVICE, dtype=torch.float32)
            video = batch["video"].to(DEVICE, dtype=torch.float32)
            labels = batch["labels"].reshape(-1).tolist()
            speaker_slots_b = batch["speaker_slots"].to(DEVICE, dtype=torch.long) if use_speaker_b else None

            logits_a = model_a(text, audio, video, speaker_slots=None).reshape(-1, NUM_CLASSES)
            logits_b = model_b(text, audio, video, speaker_slots=speaker_slots_b).reshape(-1, NUM_CLASSES)

            prob_a = torch.softmax(logits_a, dim=-1)
            prob_b = torch.softmax(logits_b, dim=-1)
            prob_ensemble = (prob_a + prob_b) / 2.0

            labels_all.extend(labels)
            preds_a.extend(torch.argmax(prob_a, dim=-1).tolist())
            preds_b.extend(torch.argmax(prob_b, dim=-1).tolist())
            preds_ensemble.extend(torch.argmax(prob_ensemble, dim=-1).tolist())

    print("=== SOLO A (RoBERTa, no speaker) ===", metrics(labels_all, preds_a))
    print("=== SOLO B (RoBERTa + speaker, current best) ===", metrics(labels_all, preds_b))
    print("=== ENSEMBLE (avg softmax) ===", metrics(labels_all, preds_ensemble))
    print("\n=== ENSEMBLE PER-CLASS REPORT ===")
    print(classification_report(
        labels_all, preds_ensemble, labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES, digits=4, zero_division=0,
    ))


if __name__ == "__main__":
    main()
