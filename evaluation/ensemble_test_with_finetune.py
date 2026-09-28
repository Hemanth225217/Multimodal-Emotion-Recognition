"""Ensemble test including the fine-tuned checkpoint (H) alongside the
frozen-feature members (B, D, F). Separate from evaluation/ensemble_test.py
because H needs live text encoding via FinetuneMELDDataset, not a frozen
feature file -- verified separately that FinetuneMELDDataset's test-set
ordering exactly matches MELDDataset's (0 mismatches across all 280
dialogues), so H's probabilities can be combined with the others
positionally.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from training.dataset_finetune import FinetuneMELDDataset
from training.train_finetune import FINETUNE_MODEL_PATH
from modules.data_loader import DISTILBERT_TEXT_PATH, ROBERTA_TEXT_PATH, ROBERTA_LARGE_TEXT_PATH
from modules.inference import load_model
from models.fusion_model import MultimodalFusionModel
from models.finetune_text_encoder import FinetuneTextEncoder


def env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


FROZEN_CHECKPOINT_PATHS = {
    "B": FINAL_MODEL_PATH,
    "D": env_path("ENSEMBLE_CKPT_D"),
    "F": env_path("ENSEMBLE_CKPT_F"),
}
FROZEN_FEATURE_CONFIG = {
    "B": (ROBERTA_TEXT_PATH, True),
    "D": (DISTILBERT_TEXT_PATH, True),
    "F": (ROBERTA_LARGE_TEXT_PATH, True),
}


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    active_frozen = {n: p for n, p in FROZEN_CHECKPOINT_PATHS.items() if p and p.exists()}
    print("Frozen members:", list(active_frozen.keys()))

    frozen_models, use_speaker = {}, {}
    for name, path in active_frozen.items():
        model, meta = load_model(path)
        frozen_models[name] = model
        use_speaker[name] = "num_speaker_slots" in meta

    print(f"Fine-tuned member (H): {FINETUNE_MODEL_PATH}")
    checkpoint = torch.load(FINETUNE_MODEL_PATH, map_location=DEVICE)
    text_encoder = FinetuneTextEncoder(checkpoint["text_model_name"]).to(DEVICE)
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    text_encoder.eval()
    finetune_model = MultimodalFusionModel(
        text_dim=checkpoint["text_dim"], audio_dim=checkpoint["audio_dim"], video_dim=checkpoint["video_dim"],
        num_classes=checkpoint["num_classes"], num_speaker_slots=checkpoint["num_speaker_slots"],
        num_sentiment_classes=checkpoint["num_sentiment_classes"], use_graph_fusion=checkpoint["use_graph_fusion"],
    ).to(DEVICE)
    finetune_model.load_state_dict(checkpoint["model_state_dict"])
    finetune_model.eval()

    # One dataset instance per unique (text_path, use_legacy_audio) pair among frozen members.
    needed_configs = {FROZEN_FEATURE_CONFIG[n] for n in active_frozen}
    frozen_datasets = {cfg: MELDDataset(split="test", text_path=cfg[0], use_legacy_audio=cfg[1]) for cfg in needed_configs}
    finetune_dataset = FinetuneMELDDataset(split="test")

    lengths = {len(ds) for ds in frozen_datasets.values()} | {len(finetune_dataset)}
    assert len(lengths) == 1, f"dataset length mismatch: {lengths}"
    num_dialogues = lengths.pop()

    labels_all = []
    probs_by_member = {name: [] for name in list(active_frozen) + ["H"]}

    with torch.no_grad():
        for i in range(num_dialogues):
            frozen_samples = {cfg: ds[i] for cfg, ds in frozen_datasets.items()}
            ft_sample = finetune_dataset[i]

            dialogue_ids = {s["dialogue_id"] for s in frozen_samples.values()} | {ft_sample["dialogue_id"]}
            assert len(dialogue_ids) == 1, f"dataset misalignment at index {i}: {dialogue_ids}"

            reference = next(iter(frozen_samples.values()))
            labels_all.extend(reference["labels"].tolist())
            speaker_slots = reference["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)
            video = reference["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)

            for name, model in frozen_models.items():
                sample = frozen_samples[FROZEN_FEATURE_CONFIG[name]]
                text = sample["text"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
                audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
                slots = speaker_slots if use_speaker[name] else None
                logits = model(text, audio, video, speaker_slots=slots).reshape(-1, NUM_CLASSES)
                probs_by_member[name].append(torch.softmax(logits, dim=-1))

            ft_audio = ft_sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            ft_video = ft_sample["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            ft_speaker_slots = ft_sample["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)
            ft_text = text_encoder(ft_sample["texts"], DEVICE)
            ft_logits = finetune_model(ft_text, ft_audio, ft_video, speaker_slots=ft_speaker_slots).reshape(-1, NUM_CLASSES)
            probs_by_member["H"].append(torch.softmax(ft_logits, dim=-1))

    probs_by_member = {name: torch.cat(t, dim=0) for name, t in probs_by_member.items()}
    active = list(probs_by_member.keys())

    print("\n=== SOLO SCORES ===")
    solo_preds = {}
    for name in sorted(active):
        solo_preds[name] = torch.argmax(probs_by_member[name], dim=-1).tolist()
        print(f"  {name}: {metrics(labels_all, solo_preds[name])}")

    print("\n=== ENSEMBLE SCORES (every combination including B and H) ===")
    ensemble_preds = {}
    for r in range(2, len(active) + 1):
        for combo in combinations(sorted(active), r):
            if "B" not in combo or "H" not in combo:
                continue
            avg_prob = sum(probs_by_member[n] for n in combo) / len(combo)
            preds = torch.argmax(avg_prob, dim=-1).tolist()
            ensemble_preds[combo] = preds
            print(f"  {'+'.join(combo)}: {metrics(labels_all, preds)}")

    if ensemble_preds:
        best_combo = max(ensemble_preds, key=lambda c: metrics(labels_all, ensemble_preds[c])["weighted_f1"])
        print(f"\nBest (including H) by weighted F1: {'+'.join(best_combo)}")
        print(classification_report(
            labels_all, ensemble_preds[best_combo], labels=list(range(NUM_CLASSES)),
            target_names=EMOTION_NAMES, digits=4, zero_division=0,
        ))


if __name__ == "__main__":
    main()
