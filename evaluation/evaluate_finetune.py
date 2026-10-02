"""Evaluate a fine-tuned checkpoint (training/train_finetune.py) on the real
MELD test set.

Separate from evaluate_final.py because a fine-tuned checkpoint pairs a
trained text encoder with the fusion model -- modules/inference.load_model()
only knows how to load the fusion model against frozen precomputed
features, not a live encoder + raw-text dataset.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, AUDIO_DIM, VIDEO_DIM, NUM_SPEAKER_SLOTS, NUM_SENTIMENT_CLASSES, DEVICE
from training.dataset_finetune import FinetuneMELDDataset
from training.train_finetune import FINETUNE_MODEL_PATH
from models.fusion_model import MultimodalFusionModel
from models.finetune_text_encoder import build_encoder_from_checkpoint


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {FINETUNE_MODEL_PATH}\n")

    checkpoint = torch.load(FINETUNE_MODEL_PATH, map_location=DEVICE)
    print(f"Epoch {checkpoint.get('epoch')}: val_accuracy={checkpoint.get('val_accuracy'):.4f} "
          f"val_weighted_f1={checkpoint.get('val_weighted_f1'):.4f}\n")

    text_encoder = build_encoder_from_checkpoint(checkpoint).to(DEVICE)
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    text_encoder.eval()

    model = MultimodalFusionModel(
        text_dim=checkpoint["text_dim"], audio_dim=checkpoint["audio_dim"], video_dim=checkpoint["video_dim"],
        num_classes=checkpoint["num_classes"], num_speaker_slots=checkpoint["num_speaker_slots"],
        num_sentiment_classes=checkpoint["num_sentiment_classes"], use_graph_fusion=checkpoint["use_graph_fusion"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = FinetuneMELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for sample in test_loader:
            texts = sample["texts"]
            audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            video = sample["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            labels = sample["labels"].tolist()
            speaker_slots = sample["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)

            text = text_encoder(texts, DEVICE, speaker_slots=sample["speaker_slots"])
            logits = model(text, audio, video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES)

            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels)

    print("=== OVERALL TEST METRICS ===")
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
