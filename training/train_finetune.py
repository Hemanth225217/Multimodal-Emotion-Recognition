"""End-to-end text fine-tuning: the one lever every stronger 2024-2026
system in the literature comparison uses that this project never has.

Every previous text upgrade in this project (DistilBERT -> RoBERTa-base ->
RoBERTa-large) was a frozen feature swap -- extract once, never update
during training. This script instead runs a trainable text encoder through
a live forward+backward pass every training step
(models/finetune_text_encoder.py), feeding its output into the exact same
downstream architecture as the best-known recipe (checkpoint B/F): the
adaptive fusion model with auxiliary unimodal losses, per-batch adaptive
modality dropout, and dialogue-relative speaker embeddings. Graph fusion
and the sentiment loss stay off, matching the current best configuration --
fine-tuning is the one new variable being tested here.

First attempt (run17) fine-tuned all of roberta-base and tied the frozen
baseline while collapsing Fear to 0.0 F1. Freezing the bottom 8 of its 12
layers (FREEZE_LAYERS below) fixed both problems -- see README. This run
applies that same finding to roberta-large (AMB-DSGDN's actual encoder),
which this project had only ever used frozen (checkpoint F) until now.

Made practical by the Kaggle GPU pipeline: fine-tuning a transformer on CPU
would take hours per run, which is why every previous text upgrade in this
project stayed frozen. On a T4, a live encoder pass adds real but tolerable
overhead per step.

Uses a two-parameter-group optimizer: a small learning rate (2e-5, the
standard range for fine-tuning a pretrained transformer without destroying
its pretrained weights) for the RoBERTa encoder, and the existing recipe's
learning rate (3e-4) for everything else (LSTMs, attention, classifiers),
which starts randomly initialized and needs much larger updates.

Produces models/final_model_finetuned_roberta_large.pt (kept separate from
both models/final_model.pt and the roberta-base fine-tune's
models/final_model_finetuned.pt so this experiment can't silently clobber
either existing checkpoint if something goes wrong).
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    AUDIO_DIM, VIDEO_DIM, NUM_CLASSES, NUM_SPEAKER_SLOTS,
    NUM_SENTIMENT_CLASSES, EMOTION_NAMES, PROJECT_ROOT, SEED as _CONFIG_SEED, DEVICE,
)
from training.dataset_finetune import FinetuneMELDDataset
from training.train_final import (
    FocalLoss, compute_class_weights, apply_modality_dropout, update_dropout_probs,
    modality_weight_entropy, modality_weight_cap_penalty,
    FOCAL_GAMMA, MAX_MODALITY_WEIGHT, CAP_PENALTY_WEIGHT, GRAD_CLIP_NORM,
    INITIAL_DROPOUT_PROBS, BATCH_DROPOUT_EMA_DECAY, AUX_LOSS_WEIGHT,
    EARLY_STOP_PATIENCE, LR_PATIENCE, FEAR, DISGUST,
)

# Optional overrides for sweeping this recipe's two most-suspect
# hyperparameters (how much of the encoder is frozen, and how hard Fear/
# Disgust are up-weighted) without another edit-commit-push cycle per
# variant. Unset, every one of these reproduces prior behavior exactly.
import os
FREEZE_LAYERS_OVERRIDE = os.environ.get("TRAIN_FREEZE_LAYERS")
OUTPUT_PATH_OVERRIDE = os.environ.get("TRAIN_OUTPUT_PATH")
FEAR_WEIGHT_BOOST = float(os.environ.get("TRAIN_FEAR_WEIGHT_BOOST", "1.0"))
DISGUST_WEIGHT_BOOST = float(os.environ.get("TRAIN_DISGUST_WEIGHT_BOOST", "1.0"))
# GPU training is deterministic for a fixed seed (verified: repeated retrains
# of train_final.py recipes were bit-identical), so a different trained
# instance needs a different seed.
SEED = int(os.environ.get("TRAIN_SEED", _CONFIG_SEED))
from models.fusion_model import MultimodalFusionModel
from models.finetune_text_encoder import build_finetune_encoder

# Dialogue context fed to the transformer itself (see
# models/finetune_text_encoder.py's ContextFinetuneTextEncoder). 0/0 keeps
# the original isolated-utterance behaviour exactly.
CONTEXT_PAST = int(os.environ.get("TRAIN_CONTEXT_PAST", "0"))
CONTEXT_FUTURE = int(os.environ.get("TRAIN_CONTEXT_FUTURE", "0"))
MAX_TOTAL_LENGTH = int(os.environ.get("TRAIN_MAX_TOTAL_LENGTH", "192"))

FINETUNE_MODEL_PATH = Path(OUTPUT_PATH_OVERRIDE) if OUTPUT_PATH_OVERRIDE else (
    PROJECT_ROOT / "models" / "final_model_finetuned_roberta_large.pt"
)

# roberta-large: AMB-DSGDN's actual text encoder (this project only ever
# fine-tuned roberta-base until now; roberta-large was previously used
# frozen only, as checkpoint F). Output path above is deliberately distinct
# from final_model_finetuned.pt (the roberta-base layer-frozen result,
# 62.57%/61.66%/42.02% -- see README) so this run can't clobber it.
# TRAIN_TEXT_MODEL selects another encoder (e.g. roberta-base for the
# faster context-window experiments).
TEXT_HIDDEN_SIZES = {
    "roberta-base": 768, "roberta-large": 1024,
    "microsoft/deberta-v3-large": 1024,   # a different pretraining recipe and tokenizer: diversity
    "microsoft/deberta-v3-xsmall": 384,   # smoke tests only
}
TEXT_MODEL_NAME = os.environ.get("TRAIN_TEXT_MODEL", "roberta-large")
if TEXT_MODEL_NAME not in TEXT_HIDDEN_SIZES:
    raise SystemExit(f"TRAIN_TEXT_MODEL must be one of {sorted(TEXT_HIDDEN_SIZES)}, got {TEXT_MODEL_NAME!r}")
TEXT_DIM = TEXT_HIDDEN_SIZES[TEXT_MODEL_NAME]

EPOCHS = 10  # fine-tuned transformers converge (and overfit) faster than frozen-feature training
TEXT_ENCODER_LR = 2e-5
MODEL_LR = 3e-4

# Bottom N transformer layers (plus embeddings) to keep frozen. Freezing 8
# of roberta-base's 12 layers (leaving the top 4 trainable, 23.2% of the
# encoder) fixed run17's Fear collapse and beat the frozen baseline on
# weighted F1 for the first time (62.57%/61.66%/42.02%, see README) --
# confirming the overfitting hypothesis from that run. Scaling that same
# ~2/3-frozen ratio to roberta-large's 24 layers: freeze 16, leave the top 8
# (32.3% of the encoder, 116M of 355M params) trainable. That run's real
# result (64.18%/63.32%/43.57%, best solo model yet) still collapsed Fear
# to 0.0 F1, unlike the roberta-base run -- TRAIN_FREEZE_LAYERS lets a
# heavier-freeze variant be tried without editing this file.
FREEZE_LAYERS = int(FREEZE_LAYERS_OVERRIDE) if FREEZE_LAYERS_OVERRIDE else 16
WEIGHT_DECAY = 1e-4

import random
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def run_epoch(text_encoder, model, loader, criterion, optimizer=None, dropout_probs=None, ema_aux_accuracy=None):
    train_mode = optimizer is not None
    text_encoder.train(train_mode)
    model.train(train_mode)

    total_loss = 0.0
    total_entropy = 0.0
    all_preds, all_labels = [], []
    aux_preds = {"text": [], "audio": [], "video": []}

    with torch.set_grad_enabled(train_mode):
        for sample in loader:
            texts = sample["texts"]
            audio = sample["audio"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            video = sample["video"].unsqueeze(0).to(DEVICE, dtype=torch.float32)
            labels = sample["labels"].to(DEVICE, dtype=torch.long).reshape(-1)
            speaker_slots = sample["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long)

            text = text_encoder(texts, DEVICE, speaker_slots=sample["speaker_slots"])  # [1, U, H], gradients flow into text_encoder

            if train_mode:
                text, audio, video = apply_modality_dropout(text, audio, video, dropout_probs)

            logits, modality_weights, aux_logits, _sentiment_logits = model(
                text, audio, video, speaker_slots=speaker_slots, return_aux=True
            )  # sentiment logits unused -- this recipe keeps the sentiment loss off, matching checkpoint B/F
            logits = logits.reshape(-1, NUM_CLASSES)

            classification_loss = criterion(logits, labels)
            entropy = modality_weight_entropy(modality_weights)
            cap_penalty = modality_weight_cap_penalty(modality_weights)

            aux_loss = 0.0
            step_aux_preds = {}
            for modality, modality_logits in aux_logits.items():
                modality_logits = modality_logits.reshape(-1, NUM_CLASSES)
                aux_loss = aux_loss + nn.functional.cross_entropy(modality_logits, labels)
                step_preds = torch.argmax(modality_logits, dim=1).detach().cpu().tolist()
                aux_preds[modality].extend(step_preds)
                step_aux_preds[modality] = step_preds
            aux_loss = aux_loss / len(aux_logits)

            if train_mode and ema_aux_accuracy is not None:
                step_labels = labels.detach().cpu().tolist()
                for modality, preds in step_aux_preds.items():
                    step_acc = accuracy_score(step_labels, preds)
                    ema_aux_accuracy[modality] = (
                        BATCH_DROPOUT_EMA_DECAY * ema_aux_accuracy[modality]
                        + (1 - BATCH_DROPOUT_EMA_DECAY) * step_acc
                    )
                dropout_probs.update(update_dropout_probs(dropout_probs, ema_aux_accuracy))

            loss = classification_loss + CAP_PENALTY_WEIGHT * cap_penalty + AUX_LOSS_WEIGHT * aux_loss
            if not torch.isfinite(loss):
                # A diverged run otherwise trains on for epochs, 'completes', and looks like
                # a result (a DeBERTa run did, with NaN loss and majority-class predictions).
                raise FloatingPointError(
                    f"non-finite loss ({loss.item()}) -- training has diverged; aborting "
                    f"(check encoder dtype/learning rate)"
                )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(text_encoder.parameters()) + list(model.parameters()), GRAD_CLIP_NORM
                )
                optimizer.step()

            total_loss += classification_loss.item()
            total_entropy += entropy.item()
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    aux_accuracy = {
        modality: accuracy_score(all_labels, preds) if preds else 0.0
        for modality, preds in aux_preds.items()
    }
    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "entropy": total_entropy / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "aux_accuracy": aux_accuracy,
    }
    return metrics, all_labels, all_preds


def main():
    print("=" * 70)
    print(f"FINE-TUNING TRAINING RUN -- live {TEXT_MODEL_NAME} encoder, bottom {FREEZE_LAYERS} layers frozen")
    print(f"Seed: {SEED}   Dialogue context window: past={CONTEXT_PAST} future={CONTEXT_FUTURE} "
          f"(max {MAX_TOTAL_LENGTH} tokens)")
    print(f"Device: {DEVICE}   Text encoder: {TEXT_MODEL_NAME} (trainable)")
    print(f"Text encoder LR: {TEXT_ENCODER_LR}   Rest-of-model LR: {MODEL_LR}")
    print("=" * 70)

    train_dataset = FinetuneMELDDataset(split="train")
    val_dataset = FinetuneMELDDataset(split="val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    class_weights = compute_class_weights(train_dataset)
    if FEAR_WEIGHT_BOOST != 1.0 or DISGUST_WEIGHT_BOOST != 1.0:
        class_weights[FEAR] *= FEAR_WEIGHT_BOOST
        class_weights[DISGUST] *= DISGUST_WEIGHT_BOOST
        print(f"Extra class-weight boost applied: Fear x{FEAR_WEIGHT_BOOST}, Disgust x{DISGUST_WEIGHT_BOOST}")
    class_weights = class_weights.to(DEVICE)

    text_encoder = build_finetune_encoder(
        TEXT_MODEL_NAME, freeze_layers=FREEZE_LAYERS, context_past=CONTEXT_PAST,
        context_future=CONTEXT_FUTURE, max_total_length=MAX_TOTAL_LENGTH,
    ).to(DEVICE)
    model = MultimodalFusionModel(
        text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM, num_classes=NUM_CLASSES,
        num_speaker_slots=NUM_SPEAKER_SLOTS, num_sentiment_classes=NUM_SENTIMENT_CLASSES,
        use_graph_fusion=False,
    ).to(DEVICE)
    trainable_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_encoder_params) + sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable parameters (encoder + model): {total_params:,}")

    criterion = FocalLoss(class_weights, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW([
        {"params": trainable_encoder_params, "lr": TEXT_ENCODER_LR},
        {"params": model.parameters(), "lr": MODEL_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=LR_PATIENCE)

    best_weighted_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    dropout_probs = dict(INITIAL_DROPOUT_PROBS)
    ema_aux_accuracy = {"text": 0.5, "audio": 0.5, "video": 0.5}

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'=' * 70}\nEPOCH {epoch}/{EPOCHS}\n{'=' * 70}")

        train_metrics, _, _ = run_epoch(
            text_encoder, model, train_loader, criterion, optimizer, dropout_probs, ema_aux_accuracy
        )
        val_metrics, val_labels, val_preds = run_epoch(text_encoder, model, val_loader, criterion)
        scheduler.step(val_metrics["weighted_f1"])

        epoch_time = time.time() - epoch_start
        print(
            f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"macroF1={train_metrics['macro_f1']:.4f} weightedF1={train_metrics['weighted_f1']:.4f}"
        )
        print(
            f"val:   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} weightedF1={val_metrics['weighted_f1']:.4f}  ({epoch_time:.0f}s)"
        )

        if val_metrics["weighted_f1"] > best_weighted_f1:
            best_weighted_f1 = val_metrics["weighted_f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "text_encoder_state_dict": text_encoder.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "text_dim": TEXT_DIM,
                    "audio_dim": AUDIO_DIM,
                    "video_dim": VIDEO_DIM,
                    "num_classes": NUM_CLASSES,
                    "num_speaker_slots": NUM_SPEAKER_SLOTS,
                    "num_sentiment_classes": NUM_SENTIMENT_CLASSES,
                    "use_graph_fusion": False,
                    "text_model_name": TEXT_MODEL_NAME,
                    "finetuned": True,
                    "seed": SEED,
                    "freeze_layers": FREEZE_LAYERS,
                    "context_past": CONTEXT_PAST,
                    "context_future": CONTEXT_FUTURE,
                    "max_total_length": MAX_TOTAL_LENGTH,
                    "emotion_names": EMOTION_NAMES,
                },
                FINETUNE_MODEL_PATH,
            )
            print(f"  -> new best model saved to {FINETUNE_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}\nTRAINING COMPLETE ({total_time / 60:.1f} min)")
    print(f"Best epoch: {best_epoch}   Best val weighted F1: {best_weighted_f1:.4f}")
    print(f"Checkpoint: {FINETUNE_MODEL_PATH}\n{'=' * 70}")


if __name__ == "__main__":
    main()
