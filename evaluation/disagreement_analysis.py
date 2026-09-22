"""Emotion disagreement / ambiguity analysis, and adaptive-weight analysis.

For each test utterance, compares what each modality would predict alone
against the fused tri-modal prediction, then studies:
  - how fusion accuracy/confidence changes as modalities agree less
  - whether fusion tends to side with whichever modality it weighted highest
  - average adaptive modality weights, overall and broken down by emotion
"""

import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.inference import load_model, predict_dialogue
from modules.ambiguity import disagreement_level


@torch.no_grad()
def single_modality_predictions(model, text, audio, video, speaker_slots):
    zero_text, zero_audio, zero_video = torch.zeros_like(text), torch.zeros_like(audio), torch.zeros_like(video)

    # Speaker context isn't one of the three modalities under test here, so
    # it stays real (not zeroed) in every branch -- this asks "how much does
    # removing text/audio/video specifically hurt", not "what if we also had
    # no speaker info".
    text_logits = model(text, zero_audio, zero_video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES)
    audio_logits = model(zero_text, audio, zero_video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES)
    video_logits = model(zero_text, zero_audio, video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES)

    return (
        torch.argmax(text_logits, dim=1).cpu().tolist(),
        torch.argmax(audio_logits, dim=1).cpu().tolist(),
        torch.argmax(video_logits, dim=1).cpu().tolist(),
    )


def main():
    print(f"Device: {DEVICE}\nModel : {FINAL_MODEL_PATH}\n")

    test_dataset = MELDDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model, meta = load_model(FINAL_MODEL_PATH)
    use_speaker = "num_speaker_slots" in meta
    print(f"Speaker embedding: {'trained, using real slots' if use_speaker else 'not present in this checkpoint, skipping'}\n")

    records = []
    for batch in test_loader:
        text = batch["text"].to(DEVICE, dtype=torch.float32)
        audio = batch["audio"].to(DEVICE, dtype=torch.float32)
        video = batch["video"].to(DEVICE, dtype=torch.float32)
        labels = batch["labels"].reshape(-1).tolist()
        speaker_slots = batch["speaker_slots"].to(DEVICE, dtype=torch.long) if use_speaker else None

        fusion = predict_dialogue(model, text, audio, video, speaker_slots=speaker_slots)
        text_preds, audio_preds, video_preds = single_modality_predictions(model, text, audio, video, speaker_slots)

        for i, true_label in enumerate(labels):
            fusion_pred = int(fusion["predictions"][i])
            confidence = float(fusion["probabilities"][i, fusion_pred])
            weights = fusion["modality_weights"][i].tolist()
            level, score = disagreement_level(text_preds[i], audio_preds[i], video_preds[i])

            records.append({
                "true": true_label,
                "fusion_pred": fusion_pred,
                "modality_preds": (text_preds[i], audio_preds[i], video_preds[i]),
                "confidence": confidence,
                "weights": weights,
                "level": level,
                "score": score,
            })

    total = len(records)

    print("=== FUSION BEHAVIOUR BY MODALITY DISAGREEMENT ===")
    print(f"{'Level':10s}{'Count':>8s}{'Share':>8s}{'Fusion Acc':>12s}{'Avg Conf':>10s}")
    for level in ("LOW", "MEDIUM", "HIGH"):
        subset = [r for r in records if r["level"] == level]
        if not subset:
            continue
        acc = accuracy_score([r["true"] for r in subset], [r["fusion_pred"] for r in subset])
        avg_conf = sum(r["confidence"] for r in subset) / len(subset)
        print(f"{level:10s}{len(subset):>8d}{100.0 * len(subset) / total:>7.2f}%{acc:>12.4f}{avg_conf:>10.4f}")

    disagreeing = [r for r in records if r["level"] != "LOW"]
    if disagreeing:
        sided_with_top_weight = sum(
            1 for r in disagreeing
            if r["fusion_pred"] == r["modality_preds"][max(range(3), key=lambda i: r["weights"][i])]
        )
        pct = 100.0 * sided_with_top_weight / len(disagreeing)
        print(
            f"\nWhen modalities disagree ({len(disagreeing)} utterances, "
            f"{100.0 * len(disagreeing) / total:.1f}% of the test set), the fused prediction matches "
            f"whichever modality the model adaptively weighted highest {pct:.1f}% of the time."
        )

    print("\n=== AVERAGE ADAPTIVE MODALITY WEIGHTS ===")
    avg = [sum(r["weights"][i] for r in records) / total for i in range(3)]
    print(f"Text={avg[0]:.3f}  Audio={avg[1]:.3f}  Video={avg[2]:.3f}")

    print("\nBy true emotion:")
    print(f"{'Emotion':10s}{'Text':>8s}{'Audio':>8s}{'Video':>8s}{'Count':>8s}")
    for class_id, name in enumerate(EMOTION_NAMES):
        subset = [r for r in records if r["true"] == class_id]
        if not subset:
            continue
        avg_w = [sum(r["weights"][i] for r in subset) / len(subset) for i in range(3)]
        print(f"{name:10s}{avg_w[0]:>8.3f}{avg_w[1]:>8.3f}{avg_w[2]:>8.3f}{len(subset):>8d}")


if __name__ == "__main__":
    main()
