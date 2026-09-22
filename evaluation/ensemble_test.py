"""Ensemble test: average already-trained checkpoints' softmax probabilities,
no retraining required.

Members (each optional except B, set via environment variables so this can
be re-run against whichever checkpoints currently exist without editing
code):
  - CHECKPOINT_A (env ENSEMBLE_CKPT_A): RoBERTa + auxiliary losses +
    adaptive dropout, no speaker embedding (git commit c5d115a). Same
    RoBERTa text features as B, so shares B's dataset instance directly.
  - CHECKPOINT_B: the committed best (config.FINAL_MODEL_PATH, git commit
    436cdcb) -- RoBERTa + speaker embeddings, seed 42. Always included.
  - CHECKPOINT_C (env ENSEMBLE_CKPT_C): same recipe/features as B, a
    different seed. Tested with seeds 43 and 44: both made the ensemble
    *worse* than A+B alone (see README "Results") -- same-recipe seed
    diversity isn't enough here. Kept as an option for the record, not
    because it's expected to help.
  - CHECKPOINT_D (env ENSEMBLE_CKPT_D): DistilBERT + auxiliary losses +
    adaptive dropout, no speaker embedding (git commit c06594c). Trained on
    a *different* text embedding space (DistilBERT, not RoBERTa) -- a
    genuinely different member rather than a reseeded rerun of the same
    recipe, which A/B/C all are relative to each other. Needs its own
    dataset instance built with text_path=DISTILBERT_TEXT_PATH; verified
    separately that this produces identical dialogue ordering, lengths and
    labels to the RoBERTa dataset (0 mismatches across all 280 test
    dialogues), so predictions can be combined positionally.

Prints every solo score plus every ensemble combination that includes B
(since B is the one member always present), so partial runs -- e.g. only D
set alongside B -- are still informative.
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
from modules.data_loader import DISTILBERT_TEXT_PATH
from modules.inference import load_model


def env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


CHECKPOINT_PATHS = {
    "A": env_path("ENSEMBLE_CKPT_A"),
    "B": FINAL_MODEL_PATH,
    "C": env_path("ENSEMBLE_CKPT_C"),
    "D": env_path("ENSEMBLE_CKPT_D"),
}
USES_DISTILBERT = {"D"}  # members in this set are fed the DistilBERT dataset instead of RoBERTa


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    active = {name: path for name, path in CHECKPOINT_PATHS.items() if path and path.exists()}
    print("Active members:")
    for name, path in active.items():
        tag = "  (DistilBERT text)" if name in USES_DISTILBERT else ""
        print(f"  {name}: {path}{tag}")
    print()

    models, use_speaker = {}, {}
    for name, path in active.items():
        model, meta = load_model(path)
        models[name] = model
        use_speaker[name] = "num_speaker_slots" in meta

    roberta_dataset = MELDDataset(split="test")
    distilbert_dataset = None
    if any(name in USES_DISTILBERT for name in active):
        distilbert_dataset = MELDDataset(split="test", text_path=DISTILBERT_TEXT_PATH)
        assert len(distilbert_dataset) == len(roberta_dataset), "dataset length mismatch"

    labels_all = []
    probs_by_member = {name: [] for name in active}  # each entry: list of [utterances, NUM_CLASSES] tensors

    with torch.no_grad():
        for i in range(len(roberta_dataset)):
            sample_r = roberta_dataset[i]
            audio = sample_r["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            video = sample_r["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            speaker_slots = sample_r["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)
            text_roberta = sample_r["text"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            labels_all.extend(sample_r["labels"].tolist())

            text_distilbert = None
            if distilbert_dataset is not None:
                sample_d = distilbert_dataset[i]
                assert sample_d["dialogue_id"] == sample_r["dialogue_id"], "dataset misalignment"
                text_distilbert = sample_d["text"].unsqueeze(0).to(DEVICE, dtype=torch.float32)

            for name, model in models.items():
                text = text_distilbert if name in USES_DISTILBERT else text_roberta
                slots = speaker_slots if use_speaker[name] else None
                logits = model(text, audio, video, speaker_slots=slots).reshape(-1, NUM_CLASSES)
                probs_by_member[name].append(torch.softmax(logits, dim=-1))

    probs_by_member = {name: torch.cat(tensors, dim=0) for name, tensors in probs_by_member.items()}

    print("=== SOLO SCORES ===")
    solo_preds = {}
    for name in sorted(active):
        solo_preds[name] = torch.argmax(probs_by_member[name], dim=-1).tolist()
        print(f"  {name}: {metrics(labels_all, solo_preds[name])}")

    print("\n=== ENSEMBLE SCORES (every combination including B) ===")
    ensemble_preds = {}
    for r in range(2, len(active) + 1):
        for combo in combinations(sorted(active), r):
            if "B" not in combo:
                continue
            avg_prob = sum(probs_by_member[n] for n in combo) / len(combo)
            preds = torch.argmax(avg_prob, dim=-1).tolist()
            ensemble_preds[combo] = preds
            print(f"  {'+'.join(combo)}: {metrics(labels_all, preds)}")

    best_combo = max(ensemble_preds, key=lambda c: metrics(labels_all, ensemble_preds[c])["weighted_f1"])
    print(f"\nBest by weighted F1: {'+'.join(best_combo)}")
    print(classification_report(
        labels_all, ensemble_preds[best_combo], labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES, digits=4, zero_division=0,
    ))


if __name__ == "__main__":
    main()
