"""Cheap ensemble test: average two already-trained checkpoints' softmax
probabilities, no retraining required.

All checkpoints were trained on the same frozen RoBERTa-base text features
(only the model/training recipe or seed differs), so they can be fed the
same dataset batch and their probabilities averaged directly:
  - CHECKPOINT_A: RoBERTa + auxiliary losses + adaptive dropout, no speaker
    embedding (git commit c5d115a).
  - CHECKPOINT_B: same + dialogue-relative speaker embeddings, seed 42 (the
    committed best, config.FINAL_MODEL_PATH / git commit 436cdcb).
  - CHECKPOINT_C (optional, set ENSEMBLE_CKPT_C): same recipe as B, seed 43
    -- a second training run of the identical architecture, used to test
    whether seed diversity adds anything on top of recipe diversity.

This only answers "does averaging these specific checkpoints help" -- it is
not a claim that ensembling in general would or wouldn't help with other
members (e.g. a DistilBERT member, which would need its own matching text
features and is not tested here).
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
CHECKPOINT_C_PATH = Path(os.environ["ENSEMBLE_CKPT_C"]) if os.environ.get("ENSEMBLE_CKPT_C") else None


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
    print(f"Checkpoint B (current best, seed 42): {FINAL_MODEL_PATH}")
    if CHECKPOINT_C_PATH:
        print(f"Checkpoint C (same recipe, seed 43): {CHECKPOINT_C_PATH}")
    print()

    model_a, meta_a = load_model(CHECKPOINT_A_PATH)
    model_b, meta_b = load_model(FINAL_MODEL_PATH)
    use_speaker_b = "num_speaker_slots" in meta_b
    model_c, use_speaker_c = None, False
    if CHECKPOINT_C_PATH:
        model_c, meta_c = load_model(CHECKPOINT_C_PATH)
        use_speaker_c = "num_speaker_slots" in meta_c

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    labels_all, preds_a, preds_b, preds_c = [], [], [], []
    preds_ensemble_ab, preds_ensemble_abc = [], []

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
            prob_ensemble_ab = (prob_a + prob_b) / 2.0

            labels_all.extend(labels)
            preds_a.extend(torch.argmax(prob_a, dim=-1).tolist())
            preds_b.extend(torch.argmax(prob_b, dim=-1).tolist())
            preds_ensemble_ab.extend(torch.argmax(prob_ensemble_ab, dim=-1).tolist())

            if model_c is not None:
                speaker_slots_c = batch["speaker_slots"].to(DEVICE, dtype=torch.long) if use_speaker_c else None
                logits_c = model_c(text, audio, video, speaker_slots=speaker_slots_c).reshape(-1, NUM_CLASSES)
                prob_c = torch.softmax(logits_c, dim=-1)
                preds_c.extend(torch.argmax(prob_c, dim=-1).tolist())
                prob_ensemble_abc = (prob_a + prob_b + prob_c) / 3.0
                preds_ensemble_abc.extend(torch.argmax(prob_ensemble_abc, dim=-1).tolist())

    print("=== SOLO A (RoBERTa, no speaker) ===", metrics(labels_all, preds_a))
    print("=== SOLO B (RoBERTa + speaker, seed 42) ===", metrics(labels_all, preds_b))
    print("=== ENSEMBLE A+B (avg softmax) ===", metrics(labels_all, preds_ensemble_ab))

    if model_c is not None:
        print("=== SOLO C (RoBERTa + speaker, seed 43) ===", metrics(labels_all, preds_c))
        print("=== ENSEMBLE A+B+C (avg softmax) ===", metrics(labels_all, preds_ensemble_abc))
        print("\n=== ENSEMBLE A+B+C PER-CLASS REPORT ===")
        print(classification_report(
            labels_all, preds_ensemble_abc, labels=list(range(NUM_CLASSES)),
            target_names=EMOTION_NAMES, digits=4, zero_division=0,
        ))
    else:
        print("\n=== ENSEMBLE A+B PER-CLASS REPORT ===")
        print(classification_report(
            labels_all, preds_ensemble_ab, labels=list(range(NUM_CLASSES)),
            target_names=EMOTION_NAMES, digits=4, zero_division=0,
        ))


if __name__ == "__main__":
    main()
