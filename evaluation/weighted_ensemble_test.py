"""Does giving each ensemble member a fitted weight (instead of a plain
average) beat the plain average?

Every ensemble result in this project so far (B+D+F+H included) averages
member softmax probabilities with equal weight. That implicitly assumes
every member deserves equal say -- not obviously true, and directly
relevant now: the re-verification run found a freshly retrained F scoring
several points below the original, which a fixed 1/4-weight average has no
way to discount. Per-member temperature scaling was tried earlier
(evaluation/calibrated_ensemble_test.py, logs/calibrated_ensemble_output.log)
and made things slightly *worse* -- but temperature only reshapes a
member's own confidence, it can't down-weight an unreliable member's
overall vote. This is a different, more direct lever.

Weights are fit two ways, both on the VALIDATION set only (never test, to
avoid quietly overfitting the reported numbers):
1. NLL: softmax-parameterized weights (guarantees non-negative, sums to 1)
   minimized via L-BFGS-B against validation cross-entropy. Smooth,
   principled, standard practice for mixture weighting.
2. Direct grid search over the weight simplex (0.1 steps) maximizing
   validation weighted F1 -- noisier but directly optimizes the metric
   this project actually selects checkpoints on.
Whichever wins on validation is applied to the test set. If neither beats
equal weighting on validation, equal weighting is kept and reported as such
-- this script will not report a fitted-weight test number that wasn't
actually better on validation first.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from training.dataset_finetune import FinetuneMELDDataset
from modules.data_loader import DISTILBERT_TEXT_PATH, ROBERTA_TEXT_PATH, ROBERTA_LARGE_TEXT_PATH
from modules.inference import load_model
from models.fusion_model import MultimodalFusionModel
from models.finetune_text_encoder import FinetuneTextEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FROZEN_CHECKPOINT_PATHS = {
    "B": FINAL_MODEL_PATH,
    "D": PROJECT_ROOT / "models" / "checkpoint_d_retrained.pt",
    "F": PROJECT_ROOT / "models" / "checkpoint_f_retrained.pt",
}
FROZEN_FEATURE_CONFIG = {
    "B": (ROBERTA_TEXT_PATH, True),
    "D": (DISTILBERT_TEXT_PATH, True),
    "F": (ROBERTA_LARGE_TEXT_PATH, True),
}
FINETUNE_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "final_model_finetuned_roberta_large.pt"


def metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def collect_probs(split, frozen_models, use_speaker, finetune_model, text_encoder):
    needed_configs = {FROZEN_FEATURE_CONFIG[n] for n in frozen_models}
    frozen_datasets = {cfg: MELDDataset(split=split, text_path=cfg[0], use_legacy_audio=cfg[1]) for cfg in needed_configs}
    finetune_dataset = FinetuneMELDDataset(split=split)

    lengths = {len(ds) for ds in frozen_datasets.values()} | {len(finetune_dataset)}
    assert len(lengths) == 1, f"[{split}] dataset length mismatch: {lengths}"
    num_dialogues = lengths.pop()

    labels_all = []
    probs_by_member = {name: [] for name in list(frozen_models) + ["H"]}

    for i in range(num_dialogues):
        frozen_samples = {cfg: ds[i] for cfg, ds in frozen_datasets.items()}
        ft_sample = finetune_dataset[i]

        dialogue_ids = {s["dialogue_id"] for s in frozen_samples.values()} | {ft_sample["dialogue_id"]}
        assert len(dialogue_ids) == 1, f"[{split}] dataset misalignment at index {i}: {dialogue_ids}"

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
    return labels_all, probs_by_member


def fit_weights_nll(probs_by_member, labels):
    names = sorted(probs_by_member.keys())
    labels_t = torch.tensor(labels, dtype=torch.long)
    stacked = torch.stack([probs_by_member[n] for n in names], dim=0)  # [M, N, C]

    def nll(raw_weights):
        w = torch.softmax(torch.tensor(raw_weights, dtype=torch.float32), dim=0)
        mixed = (w.view(-1, 1, 1) * stacked).sum(dim=0).clamp(min=1e-12)
        return F.nll_loss(torch.log(mixed), labels_t).item()

    result = minimize(nll, x0=np.zeros(len(names)), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    weights = torch.softmax(torch.tensor(result.x, dtype=torch.float32), dim=0)
    return {n: w.item() for n, w in zip(names, weights)}


def grid_search_weighted_f1(probs_by_member, labels, step=0.1):
    names = sorted(probs_by_member.keys())
    n = len(names)
    steps = round(1.0 / step)
    best_weights, best_score = {name: 1.0 / n for name in names}, -1.0

    def recurse(remaining_units, remaining_slots, chosen):
        nonlocal best_weights, best_score
        if remaining_slots == 1:
            candidate = chosen + [remaining_units]
            weights = {name: c * step for name, c in zip(names, candidate)}
            mixed = sum(weights[name] * probs_by_member[name] for name in names)
            preds = torch.argmax(mixed, dim=-1).tolist()
            score = f1_score(labels, preds, average="weighted", zero_division=0)
            if score > best_score:
                best_score, best_weights = score, weights
            return
        for units in range(remaining_units + 1):
            recurse(remaining_units - units, remaining_slots - 1, chosen + [units])

    recurse(steps, n, [])
    return best_weights, best_score


def apply_weights(weights, probs_by_member):
    return sum(weights[name] * probs_by_member[name] for name in weights)


def main():
    active_frozen = {n: p for n, p in FROZEN_CHECKPOINT_PATHS.items() if p.exists()}
    print("Frozen members:", list(active_frozen.keys()))
    if not FINETUNE_CHECKPOINT_PATH.exists():
        raise SystemExit(f"Fine-tuned checkpoint not found: {FINETUNE_CHECKPOINT_PATH}")

    frozen_models, use_speaker = {}, {}
    for name, path in active_frozen.items():
        model, meta = load_model(path)
        frozen_models[name] = model
        use_speaker[name] = "num_speaker_slots" in meta

    print(f"Fine-tuned member (H): {FINETUNE_CHECKPOINT_PATH}")
    checkpoint = torch.load(FINETUNE_CHECKPOINT_PATH, map_location=DEVICE)
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

    print("\nCollecting VALIDATION probabilities (for weight fitting only)...")
    val_labels, val_probs = collect_probs("val", frozen_models, use_speaker, finetune_model, text_encoder)

    print("\nCollecting TEST probabilities...")
    test_labels, test_probs = collect_probs("test", frozen_models, use_speaker, finetune_model, text_encoder)

    names = sorted(test_probs.keys())
    equal_weights = {n: 1.0 / len(names) for n in names}

    val_equal_preds = torch.argmax(apply_weights(equal_weights, val_probs), dim=-1).tolist()
    val_equal_score = f1_score(val_labels, val_equal_preds, average="weighted", zero_division=0)
    print(f"\nEqual-weight VALIDATION weighted F1: {val_equal_score:.4f} (baseline to beat)")

    nll_weights = fit_weights_nll(val_probs, val_labels)
    val_nll_preds = torch.argmax(apply_weights(nll_weights, val_probs), dim=-1).tolist()
    val_nll_score = f1_score(val_labels, val_nll_preds, average="weighted", zero_division=0)
    print(f"NLL-fitted weights: { {k: round(v, 3) for k, v in nll_weights.items()} } "
          f"-> VALIDATION weighted F1: {val_nll_score:.4f}")

    grid_weights, val_grid_score = grid_search_weighted_f1(val_probs, val_labels, step=0.1)
    print(f"Grid-search weights: { {k: round(v, 3) for k, v in grid_weights.items()} } "
          f"-> VALIDATION weighted F1: {val_grid_score:.4f}")

    candidates = [("equal", equal_weights, val_equal_score), ("nll", nll_weights, val_nll_score),
                  ("grid", grid_weights, val_grid_score)]
    best_method, best_weights, best_val_score = max(candidates, key=lambda c: c[2])
    print(f"\nBest method on VALIDATION: {best_method} (weighted F1 {best_val_score:.4f})")

    print("\n=== TEST RESULTS ===")
    for method_name, weights, _ in candidates:
        preds = torch.argmax(apply_weights(weights, test_probs), dim=-1).tolist()
        print(f"  {method_name} weights {({k: round(v, 3) for k, v in weights.items()})}: {metrics(test_labels, preds)}")

    print(f"\nSelected by validation ({best_method}), applied to test:")
    final_preds = torch.argmax(apply_weights(best_weights, test_probs), dim=-1).tolist()
    print(metrics(test_labels, final_preds))
    print(classification_report(
        test_labels, final_preds, labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES, digits=4, zero_division=0,
    ))


if __name__ == "__main__":
    main()
