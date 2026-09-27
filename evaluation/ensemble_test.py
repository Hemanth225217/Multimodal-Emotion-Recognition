"""Ensemble test: average already-trained checkpoints' softmax probabilities,
no retraining required.

Each member was trained on a specific (text, audio) feature pair; the
dataset instance it needs is looked up from FEATURE_CONFIG below and built
once, then shared by every member with the same pair. All instances are
verified aligned (identical dialogue order/lengths/labels) before combining
predictions positionally.

Members (each optional except B, set via environment variables so this can
be re-run against whichever checkpoints currently exist without editing
code):
  - CHECKPOINT_A (env ENSEMBLE_CKPT_A): RoBERTa text + 300-D legacy audio,
    auxiliary losses + adaptive dropout, no speaker embedding (commit
    c5d115a).
  - CHECKPOINT_B: the committed best (config.FINAL_MODEL_PATH, commit
    436cdcb) -- RoBERTa text + 300-D legacy audio + speaker embeddings,
    seed 42. Always included.
  - CHECKPOINT_C (env ENSEMBLE_CKPT_C): same recipe/features as B, a
    different seed. Tested with seeds 43 and 44: both made the ensemble
    *worse* than A+B alone -- same-recipe seed diversity isn't enough here.
  - CHECKPOINT_D (env ENSEMBLE_CKPT_D): DistilBERT text + 300-D legacy
    audio, auxiliary losses + adaptive dropout, no speaker embedding
    (commit c06594c). A genuinely different text embedding space, unlike
    A/B/C which only differ by seed or a small architectural addition.
  - CHECKPOINT_E (env ENSEMBLE_CKPT_E): RoBERTa text + 768-D Wav2Vec2 audio,
    same recipe as B (commit 14a71df). Solo score was a mixed result
    (weaker weighted/macro F1 than B), tested here anyway since D proved a
    weak solo score doesn't rule out ensemble value.

Prints every solo score plus every ensemble combination that includes B
(since B is the one member always present), so partial runs -- e.g. only
one extra member set -- are still informative.
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
from modules.data_loader import DISTILBERT_TEXT_PATH, ROBERTA_TEXT_PATH, ROBERTA_LARGE_TEXT_PATH
from modules.inference import load_model


def env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


CHECKPOINT_PATHS = {
    "A": env_path("ENSEMBLE_CKPT_A"),
    "B": FINAL_MODEL_PATH,
    "C": env_path("ENSEMBLE_CKPT_C"),
    "D": env_path("ENSEMBLE_CKPT_D"),
    "E": env_path("ENSEMBLE_CKPT_E"),
    "F": env_path("ENSEMBLE_CKPT_F"),
    "G": env_path("ENSEMBLE_CKPT_G"),
}

# (text_path, use_legacy_audio) each member's checkpoint was trained on.
# Explicit for every member -- config.py's default text path has changed
# twice (DistilBERT -> RoBERTa-base -> RoBERTa-large) since A/B/C were
# trained, so relying on "None means default" here would silently point
# older checkpoints at the wrong text embedding space.
FEATURE_CONFIG = {
    "A": (ROBERTA_TEXT_PATH, True),
    "B": (ROBERTA_TEXT_PATH, True),
    "C": (ROBERTA_TEXT_PATH, True),
    "D": (DISTILBERT_TEXT_PATH, True),
    "E": (ROBERTA_TEXT_PATH, False),
    "F": (ROBERTA_LARGE_TEXT_PATH, True),
    "G": (ROBERTA_LARGE_TEXT_PATH, True),
}


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
        text_path, legacy_audio = FEATURE_CONFIG[name]
        tag = f"  (text={'RoBERTa' if text_path is None else text_path.parent.name}, audio={'legacy 300D' if legacy_audio else 'Wav2Vec2 768D'})"
        print(f"  {name}: {path}{tag}")
    print()

    models, use_speaker = {}, {}
    for name, path in active.items():
        model, meta = load_model(path)
        models[name] = model
        use_speaker[name] = "num_speaker_slots" in meta

    # Build one dataset instance per unique (text_path, use_legacy_audio) pair.
    needed_configs = {FEATURE_CONFIG[name] for name in active}
    datasets = {cfg: MELDDataset(split="test", text_path=cfg[0], use_legacy_audio=cfg[1]) for cfg in needed_configs}

    lengths = {len(ds) for ds in datasets.values()}
    assert len(lengths) == 1, f"dataset length mismatch across feature configs: {lengths}"
    num_dialogues = lengths.pop()

    labels_all = []
    probs_by_member = {name: [] for name in active}

    with torch.no_grad():
        for i in range(num_dialogues):
            samples = {cfg: ds[i] for cfg, ds in datasets.items()}
            dialogue_ids = {s["dialogue_id"] for s in samples.values()}
            assert len(dialogue_ids) == 1, f"dataset misalignment at index {i}: {dialogue_ids}"

            reference = next(iter(samples.values()))
            labels_all.extend(reference["labels"].tolist())
            speaker_slots = reference["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)
            video = reference["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)

            for name, model in models.items():
                sample = samples[FEATURE_CONFIG[name]]
                text = sample["text"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
                audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
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
