"""Full modality ablation: does multimodal fusion actually help?

Evaluates the final model on the MELD test set under all 7 modality
combinations (single, pairwise, and tri-modal) by zeroing out the inputs of
excluded modalities. This directly answers whether adding a modality
improves performance, rather than assuming it does.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.inference import load_model

CONFIGS = {
    "text_only": (True, False, False),
    "audio_only": (False, True, False),
    "video_only": (False, False, True),
    "text+audio": (True, True, False),
    "text+video": (True, False, True),
    "audio+video": (False, True, True),
    "text+audio+video": (True, True, True),
}


@torch.no_grad()
def evaluate_config(model, loader, use_text, use_audio, use_video, use_speaker):
    all_preds, all_labels = [], []

    for batch in loader:
        text = batch["text"].to(DEVICE, dtype=torch.float32)
        audio = batch["audio"].to(DEVICE, dtype=torch.float32)
        video = batch["video"].to(DEVICE, dtype=torch.float32)
        labels = batch["labels"].reshape(-1).tolist()
        speaker_slots = batch["speaker_slots"].to(DEVICE, dtype=torch.long) if use_speaker else None

        if not use_text:
            text = torch.zeros_like(text)
        if not use_audio:
            audio = torch.zeros_like(audio)
        if not use_video:
            video = torch.zeros_like(video)

        logits = model(text, audio, video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "per_class_f1": f1_score(
            all_labels, all_preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
        ),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Model : {FINAL_MODEL_PATH}\n")

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model, meta = load_model(FINAL_MODEL_PATH)
    use_speaker = "num_speaker_slots" in meta
    print(f"Speaker embedding: {'trained, using real slots' if use_speaker else 'not present in this checkpoint, skipping'}\n")

    results = {}
    for name, (use_text, use_audio, use_video) in CONFIGS.items():
        results[name] = evaluate_config(model, test_loader, use_text, use_audio, use_video, use_speaker)
        r = results[name]
        print(f"{name:20s} acc={r['accuracy']:.4f}  weightedF1={r['weighted_f1']:.4f}  macroF1={r['macro_f1']:.4f}")

    print("\n=== SUMMARY TABLE ===")
    print(f"{'Configuration':20s}{'Accuracy':>12s}{'Weighted F1':>14s}{'Macro F1':>12s}")
    for name in CONFIGS:
        r = results[name]
        print(f"{name:20s}{r['accuracy']:>12.4f}{r['weighted_f1']:>14.4f}{r['macro_f1']:>12.4f}")

    print("\n=== PER-CLASS F1 BY CONFIGURATION ===")
    print("Emotion".ljust(12) + "".join(f"{n:>18s}" for n in CONFIGS))
    for i, emotion in enumerate(EMOTION_NAMES):
        row = emotion.ljust(12) + "".join(f"{results[n]['per_class_f1'][i]:>18.4f}" for n in CONFIGS)
        print(row)

    tri = results["text+audio+video"]
    print("\n=== MODALITY CONTRIBUTION (tri-modal minus the pair missing it) ===")
    print(
        f"Video : accuracy {tri['accuracy'] - results['text+audio']['accuracy']:+.4f}, "
        f"macroF1 {tri['macro_f1'] - results['text+audio']['macro_f1']:+.4f}"
    )
    print(
        f"Audio : accuracy {tri['accuracy'] - results['text+video']['accuracy']:+.4f}, "
        f"macroF1 {tri['macro_f1'] - results['text+video']['macro_f1']:+.4f}"
    )
    print(
        f"Text  : accuracy {tri['accuracy'] - results['audio+video']['accuracy']:+.4f}, "
        f"macroF1 {tri['macro_f1'] - results['audio+video']['macro_f1']:+.4f}"
    )


if __name__ == "__main__":
    main()
