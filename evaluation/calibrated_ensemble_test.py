"""Does per-member temperature calibration improve the ensemble?

Averaging raw softmax outputs implicitly assumes every member is equally
"confident" in a comparable way. That's not guaranteed -- a model trained
with a different text encoder or a different loss mix can be systematically
over- or under-confident relative to another. Temperature scaling (Guo et
al., 2017) rescales one model's logits by a single learned scalar T before
softmax; for a single model this changes confidence but never accuracy
(dividing every logit by the same T can't change which one is the max), but
inside an ensemble it changes each member's *voting weight* in the average,
which can change the argmax.

T is fit per member on the VALIDATION set (never the test set, to avoid
quietly overfitting a calibration parameter to the numbers being reported)
by a small grid search minimizing NLL, then applied to that member's TEST
set logits before averaging. Reuses the same member definitions as
evaluation/ensemble_test.py (env vars ENSEMBLE_CKPT_A/C/D, B is always the
committed best).
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.data_loader import DISTILBERT_TEXT_PATH
from modules.inference import load_model

TEMPERATURE_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]


def env_path(name):
    value = os.environ.get(name)
    return Path(value) if value else None


CHECKPOINT_PATHS = {
    "A": env_path("ENSEMBLE_CKPT_A"),
    "B": FINAL_MODEL_PATH,
    "C": env_path("ENSEMBLE_CKPT_C"),
    "D": env_path("ENSEMBLE_CKPT_D"),
}
USES_DISTILBERT = {"D"}


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def collect_logits(models, use_speaker, roberta_dataset, distilbert_dataset):
    labels_all = []
    logits_by_member = {name: [] for name in models}

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
            logits_by_member[name].append(logits)

    logits_by_member = {name: torch.cat(t, dim=0) for name, t in logits_by_member.items()}
    return labels_all, logits_by_member


def fit_temperature(logits, labels):
    labels_t = torch.tensor(labels, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in TEMPERATURE_GRID:
        nll = F.cross_entropy(logits / t, labels_t).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def main():
    active = {name: path for name, path in CHECKPOINT_PATHS.items() if path and path.exists()}
    print("Active members:", list(active.keys()), "\n")

    models, use_speaker = {}, {}
    for name, path in active.items():
        model, meta = load_model(path)
        models[name] = model
        use_speaker[name] = "num_speaker_slots" in meta

    val_roberta = MELDDataset(split="val")
    val_distilbert = MELDDataset(split="val", text_path=DISTILBERT_TEXT_PATH) if any(n in USES_DISTILBERT for n in active) else None
    test_roberta = MELDDataset(split="test")
    test_distilbert = MELDDataset(split="test", text_path=DISTILBERT_TEXT_PATH) if any(n in USES_DISTILBERT for n in active) else None

    print("Collecting validation logits to fit temperatures...")
    val_labels, val_logits = collect_logits(models, use_speaker, val_roberta, val_distilbert)

    temperatures = {name: fit_temperature(val_logits[name], val_labels) for name in active}
    print("Fitted temperatures:", temperatures, "\n")

    print("Collecting test logits...")
    test_labels, test_logits = collect_logits(models, use_speaker, test_roberta, test_distilbert)

    uncalibrated_avg = sum(F.softmax(test_logits[n], dim=-1) for n in active) / len(active)
    uncalibrated_preds = torch.argmax(uncalibrated_avg, dim=-1).tolist()

    calibrated_avg = sum(F.softmax(test_logits[n] / temperatures[n], dim=-1) for n in active) / len(active)
    calibrated_preds = torch.argmax(calibrated_avg, dim=-1).tolist()

    print("=== UNCALIBRATED ensemble (equal weight, T=1 for all) ===", metrics(test_labels, uncalibrated_preds))
    print("=== CALIBRATED ensemble (per-member T fit on val) ===", metrics(test_labels, calibrated_preds))


if __name__ == "__main__":
    main()
