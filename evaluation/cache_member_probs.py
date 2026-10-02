"""Compute and cache one ensemble member's softmax probabilities on the
validation and test sets, so every later ensemble question (which subset,
what weights, seed-averaging) is a cheap tensor operation instead of
re-running -- for a fine-tuned member, 10-20 minutes of live transformer
encoding on CPU -- every single time.

Usage:
  python -m evaluation.cache_member_probs NAME frozen CKPT --features {roberta,distilbert,roberta_large}
  python -m evaluation.cache_member_probs NAME finetune CKPT

Writes logs/prob_cache/NAME.pt = {"val": [N,7], "test": [N,7],
"val_labels": [...], "test_labels": [...], "meta": {...}}. Validation and
test dialogue order is identical across member types (verified separately
for FinetuneMELDDataset vs MELDDataset), and evaluation/ensemble_from_cache.py
asserts every member's labels match exactly before combining anything.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, DEVICE
from training.dataset import MELDDataset
from modules.data_loader import DISTILBERT_TEXT_PATH, ROBERTA_TEXT_PATH, ROBERTA_LARGE_TEXT_PATH
from modules.inference import load_model

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "prob_cache"
FEATURE_PATHS = {
    "roberta": ROBERTA_TEXT_PATH,
    "distilbert": DISTILBERT_TEXT_PATH,
    "roberta_large": ROBERTA_LARGE_TEXT_PATH,
}


@torch.no_grad()
def frozen_probs(checkpoint_path, features, split):
    model, meta = load_model(Path(checkpoint_path))
    use_speaker = "num_speaker_slots" in meta
    dataset = MELDDataset(split=split, text_path=FEATURE_PATHS[features], use_legacy_audio=True)
    labels, probs = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.extend(sample["labels"].tolist())
        text = sample["text"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
        audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
        video = sample["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
        slots = sample["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long) if use_speaker else None
        logits = model(text, audio, video, speaker_slots=slots).reshape(-1, NUM_CLASSES)
        probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs, dim=0), labels, {"val_epoch_in_ckpt": meta.get("epoch")}


@torch.no_grad()
def finetune_probs(checkpoint_path, split, encoder, model):
    from training.dataset_finetune import FinetuneMELDDataset

    dataset = FinetuneMELDDataset(split=split)
    labels, probs = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        labels.extend(sample["labels"].tolist())
        audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
        video = sample["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
        slots = sample["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)
        text = encoder(sample["texts"], DEVICE, speaker_slots=sample["speaker_slots"])
        logits = model(text, audio, video, speaker_slots=slots).reshape(-1, NUM_CLASSES)
        probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs, dim=0), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("kind", choices=["frozen", "finetune"])
    parser.add_argument("checkpoint")
    parser.add_argument("--features", choices=list(FEATURE_PATHS), default=None)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{args.name}.pt"
    result = {"meta": {"name": args.name, "kind": args.kind, "checkpoint": str(args.checkpoint)}}

    if args.kind == "frozen":
        if not args.features:
            raise SystemExit("--features is required for frozen members (768 is ambiguous: roberta vs distilbert)")
        result["meta"]["features"] = args.features
        for split in ("val", "test"):
            probs, labels, extra = frozen_probs(args.checkpoint, args.features, split)
            result[split], result[f"{split}_labels"] = probs, labels
            result["meta"].update(extra)
    else:
        from models.fusion_model import MultimodalFusionModel
        from models.finetune_text_encoder import build_encoder_from_checkpoint

        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        encoder = build_encoder_from_checkpoint(ckpt).to(DEVICE)
        encoder.load_state_dict(ckpt["text_encoder_state_dict"])
        encoder.eval()
        model = MultimodalFusionModel(
            text_dim=ckpt["text_dim"], audio_dim=ckpt["audio_dim"], video_dim=ckpt["video_dim"],
            num_classes=ckpt["num_classes"], num_speaker_slots=ckpt["num_speaker_slots"],
            num_sentiment_classes=ckpt["num_sentiment_classes"], use_graph_fusion=ckpt["use_graph_fusion"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        result["meta"].update({
            "text_model_name": ckpt["text_model_name"], "seed": ckpt.get("seed"),
            "freeze_layers": ckpt.get("freeze_layers"), "val_weighted_f1_in_ckpt": ckpt.get("val_weighted_f1"),
            "context_past": ckpt.get("context_past", 0), "context_future": ckpt.get("context_future", 0),
        })
        for split in ("val", "test"):
            probs, labels = finetune_probs(args.checkpoint, split, encoder, model)
            result[split], result[f"{split}_labels"] = probs, labels

    torch.save(result, out_path)
    print(f"Saved {out_path}: val {tuple(result['val'].shape)}, test {tuple(result['test'].shape)}")


if __name__ == "__main__":
    main()
