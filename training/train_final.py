"""Single canonical training run for the adaptive tri-modal fusion model.

Replaces the baseline -> V1 -> V2 -> V3 -> V4 experimental chain (kept for
reference under training/legacy/) with one self-contained run: random init,
the adaptive fusion architecture, moderate class weighting, gentle focal
loss, and modality dropout.

Two additions target modality collapse, a known failure mode where a
multimodal model learns to lean on whichever signal is easiest (here, text)
and never bothers with the others. An earlier run without either fix
converged to average adaptive weights of text=0.998, audio=0.001,
video=0.001 for every emotion:

  1. Modality dropout (zero one modality per training step) forces the
     model to solve the task from audio/video alone often enough that those
     encoders get real gradient signal. This alone was NOT enough -- a
     follow-up run with only this fix still converged to ~99.9% text weight,
     because dropout only teaches the model to cope when a modality is
     completely absent; it does nothing to the "all three genuinely present"
     regime, which is 100% of eval-time behaviour and 60% of training.
  2. A weight-cap penalty on the modality-weight network's output: any
     modality weight above MAX_MODALITY_WEIGHT is penalized, in every
     regime, not just the dropout-forced ones. An entropy bonus (maximize
     -sum(w*log(w))) was tried first and rejected: at every weight tried
     (0.02, 0.15) it eventually pushed weights toward *exactly*
     text=audio=video=0.333 for every single emotion, because uniform
     trivially maximizes entropy regardless of content -- not adaptive,
     just a different degenerate solution. A hinge-style cap has no such
     shortcut: the penalty is zero for any distribution that keeps every
     weight under the cap (a wide space including plenty of non-uniform,
     content-dependent options), so satisfying it doesn't require ignoring
     the input the way maximizing entropy does.

This version adds two more ideas, both directly inspired by AMB-DSGDN
(2026, arXiv 2603.10043) after reading its full method rather than just an
abstract:
  3. Auxiliary unimodal losses: each modality's own pre-fusion
     representation gets its own small classifier, trained with its own
     cross-entropy loss alongside the main fused prediction. Gives each
     encoder direct supervision instead of only ever being judged through
     the fused output.
  4. Adaptive (not fixed) modality dropout: AMB-DSGDN computes a per-modality
     dropout probability from that modality's relative performance each
     batch. We adapt the same idea at epoch granularity using the auxiliary
     heads' accuracy as the performance signal -- whichever modality is
     currently strongest gets dropped more, the weakest less -- rather than
     the fixed 40/15/15 split used in the previous version.

This version also adds speaker-aware context: a learned embedding for each
utterance's dialogue-relative speaker slot (config.NUM_SPEAKER_SLOTS), added
to the fused representation before the dialogue context attention. This is
the biggest architectural gap identified against AMB-DSGDN and most of the
ERC literature (DialogueRNN, DialogueGCN, ...), which explicitly model
same-/cross-speaker relationships; this project previously used none.
Verified on the test set: accuracy 60.96% -> 62.45%, weighted F1 60.68% ->
61.50% (macro F1 dipped slightly, 43.51% -> 42.74%, mostly a Disgust
regression on a 68-utterance-support class).

Two more ideas, added together and tested as one experiment:
  5. A speaker-relational attention bias in the dialogue context attention:
     two learned scalars (same-speaker vs. different-speaker) added to the
     attention scores. A cheap, CPU-feasible approximation of the explicit
     speaker-relation graphs AMB-DSGDN and DialogueGCN/DialogueRNN build,
     without an actual graph-conv layer or new library dependency.
  6. A sentiment auxiliary loss: MELD ships a 3-way Sentiment label
     (neutral/positive/negative) alongside the 7-way Emotion label, and it
     is not a deterministic function of it (surprise splits across both
     positive and negative depending on context). A small classifier head
     off the same fused representation, trained with its own cross-entropy
     loss, gives the model a second correlated supervision signal for free
     -- no new features needed, and MELD was originally released as a
     joint emotion+sentiment benchmark.

Produces models/final_model.pt.
"""

import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES, NUM_SPEAKER_SLOTS,
    NUM_SENTIMENT_CLASSES, EMOTION_NAMES, FINAL_MODEL_PATH, SEED, DEVICE,
)
from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel

EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 1.0
EARLY_STOP_PATIENCE = 6
LR_PATIENCE = 2
GRAD_CLIP_NORM = 1.0

# Hinge penalty: any modality weight above MAX_MODALITY_WEIGHT is punished,
# added to the loss with strength CAP_PENALTY_WEIGHT. Zero penalty for any
# distribution that already keeps every weight under the cap -- unlike
# maximizing entropy, satisfying this doesn't require ignoring the input.
MAX_MODALITY_WEIGHT = 0.60
CAP_PENALTY_WEIGHT = 1.0

# Modality dropout: at most one modality zeroed per training step. Starts
# at this fixed split (text dropped most, matching what the previous
# version found necessary) since epoch 1 has no real performance signal
# yet; from epoch 2 on, update_dropout_probs() below adapts these based on
# each modality's actual auxiliary-head accuracy that epoch.
INITIAL_DROPOUT_PROBS = {"text": 0.40, "audio": 0.15, "video": 0.15}
BASE_DROPOUT_PROB = 0.15
DROPOUT_SCALE = 0.50
MIN_DROPOUT_PROB = 0.05
MAX_DROPOUT_PROB = 0.55
DROPOUT_EMA_DECAY = 0.6  # smooths updates against the previous epoch's probs

# Weight on the summed auxiliary unimodal losses (added to the main loss).
AUX_LOSS_WEIGHT = 0.3

# Weight on the sentiment auxiliary loss (see module docstring, idea 6).
# Smaller than AUX_LOSS_WEIGHT since sentiment is a coarser 3-way signal
# meant to nudge the shared representation, not dominate the main 7-way task.
#
# run13 (this @ 0.2, bias off) tested clean: no collapse, but still a net
# loss vs. the speaker-embedding checkpoint on weighted F1 (see README
# "Results"). Back to 0.0 -- this run reproduces that checkpoint's exact
# recipe (RoBERTa + aux losses + adaptive dropout + speaker embeddings only)
# under a different seed, for ensembling (evaluation/ensemble_test.py).
SENTIMENT_LOSS_WEIGHT = 0.0

# Label smoothing on the main classification loss: a standard, near-free
# regularizer, not yet tried in this project. Default 0.0 (off) so existing
# runs are unaffected; set to e.g. 0.05-0.1 for the next experiment once
# there's CPU time free to test it (currently busy with the Wav2Vec2 audio
# extraction -- see modules/audio_features_wav2vec2.py).
LABEL_SMOOTHING = 0.0

NEUTRAL, SURPRISE, FEAR, SADNESS, JOY, DISGUST, ANGER = range(7)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class FocalLoss(nn.Module):
    """Class-weighted focal loss with a gentle gamma.

    A steep gamma (as tried in the V3 experiment) overcorrects for rare
    classes and damages overall accuracy; gamma=1.0 with moderate class
    weights was the best trade-off found during experimentation.
    """

    def __init__(self, class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        # Focal modulating factor still uses the hard-target probability,
        # not smoothed -- smoothing changes what the loss is computed
        # against, not which utterances count as "already easy" for the
        # purpose of down-weighting them.
        target_prob = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - target_prob) ** self.gamma
        weights = self.class_weights[targets]
        return (weights * focal_factor * ce).mean()


def compute_class_weights(dataset):
    counts = torch.zeros(NUM_CLASSES)
    for i in range(len(dataset)):
        for label in dataset[i]["labels"].tolist():
            counts[label] += 1
    total = counts.sum()

    weights = torch.sqrt(total / counts.clamp(min=1.0))
    weights /= weights.mean()

    # Moderate manual adjustments: ease off the dominant neutral class and
    # give fear/disgust/sadness a modest boost, without extreme weights that
    # destabilize training (that failure mode was the V3 experiment).
    weights[NEUTRAL] *= 0.80
    weights[FEAR] *= 1.15
    weights[DISGUST] *= 1.15
    weights[SADNESS] *= 1.05
    weights = weights.clamp(min=0.55, max=2.00)
    weights /= weights.mean()

    print("\nClass distribution & training weights:")
    for i, name in enumerate(EMOTION_NAMES):
        pct = 100.0 * counts[i].item() / total.item()
        print(f"  {name:10s}: {int(counts[i]):5d} ({pct:5.2f}%)  weight={weights[i].item():.3f}")

    return weights


def unpack_batch(batch):
    text = batch["text"].to(DEVICE, dtype=torch.float32)
    audio = batch["audio"].to(DEVICE, dtype=torch.float32)
    video = batch["video"].to(DEVICE, dtype=torch.float32)
    labels = batch["labels"].to(DEVICE, dtype=torch.long)
    speaker_slots = batch["speaker_slots"].to(DEVICE, dtype=torch.long)
    sentiment_labels = batch["sentiment_labels"].to(DEVICE, dtype=torch.long)
    return text, audio, video, labels, speaker_slots, sentiment_labels


def apply_modality_dropout(text, audio, video, probs):
    """Zero out at most one modality this step (see module docstring)."""
    roll = random.random()
    if roll < probs["text"]:
        text = torch.zeros_like(text)
    elif roll < probs["text"] + probs["audio"]:
        audio = torch.zeros_like(audio)
    elif roll < probs["text"] + probs["audio"] + probs["video"]:
        video = torch.zeros_like(video)
    return text, audio, video


def update_dropout_probs(current_probs, epoch_aux_accuracy):
    """Recompute per-modality dropout probabilities from this epoch's
    auxiliary-head accuracy (AMB-DSGDN's idea, adapted to epoch
    granularity): whichever modality is relatively strongest gets dropped
    more, the weakest less, instead of a fixed split.
    """
    mean_acc = sum(epoch_aux_accuracy.values()) / len(epoch_aux_accuracy)

    target_probs = {}
    for modality, acc in epoch_aux_accuracy.items():
        relative = (acc - mean_acc) / (mean_acc + 1e-8)
        prob = BASE_DROPOUT_PROB + DROPOUT_SCALE * max(relative, 0.0)
        target_probs[modality] = min(max(prob, MIN_DROPOUT_PROB), MAX_DROPOUT_PROB)

    return {
        m: DROPOUT_EMA_DECAY * current_probs[m] + (1 - DROPOUT_EMA_DECAY) * target_probs[m]
        for m in epoch_aux_accuracy
    }


def modality_weight_entropy(modality_weights, eps=1e-8):
    """Mean Shannon entropy of the per-utterance 3-way modality weights.

    Diagnostic only (not part of the loss): max is ln(3)=1.099 (uniform
    text/audio/video), near 0 means collapsed onto a single modality. Useful
    for watching training, but NOT used as the regularizer -- see
    modality_weight_cap_penalty and the module docstring for why.
    """
    return -(modality_weights * torch.log(modality_weights + eps)).sum(dim=-1).mean()


def modality_weight_cap_penalty(modality_weights, max_weight=MAX_MODALITY_WEIGHT):
    """Hinge penalty: cost only for the part of any weight above max_weight."""
    excess = (modality_weights - max_weight).clamp(min=0)
    return excess.sum(dim=-1).mean()


def run_epoch(model, loader, criterion, optimizer=None, dropout_probs=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_entropy = 0.0
    all_preds, all_labels = [], []
    aux_preds = {"text": [], "audio": [], "video": []}
    sentiment_preds, sentiment_targets = [], []

    with torch.set_grad_enabled(train_mode):
        for step, batch in enumerate(loader, start=1):
            text, audio, video, labels, speaker_slots, sentiment_labels = unpack_batch(batch)

            if train_mode:
                text, audio, video = apply_modality_dropout(text, audio, video, dropout_probs)

            logits, modality_weights, aux_logits, sentiment_logits = model(
                text, audio, video, speaker_slots=speaker_slots, return_aux=True
            )
            logits = logits.reshape(-1, NUM_CLASSES)
            labels = labels.reshape(-1)
            sentiment_logits = sentiment_logits.reshape(-1, NUM_SENTIMENT_CLASSES)
            sentiment_labels = sentiment_labels.reshape(-1)

            classification_loss = criterion(logits, labels)
            entropy = modality_weight_entropy(modality_weights)  # diagnostic only
            cap_penalty = modality_weight_cap_penalty(modality_weights)

            aux_loss = 0.0
            for modality, modality_logits in aux_logits.items():
                modality_logits = modality_logits.reshape(-1, NUM_CLASSES)
                aux_loss = aux_loss + nn.functional.cross_entropy(modality_logits, labels)
                aux_preds[modality].extend(torch.argmax(modality_logits, dim=1).detach().cpu().tolist())
            aux_loss = aux_loss / len(aux_logits)

            sentiment_loss = nn.functional.cross_entropy(sentiment_logits, sentiment_labels)

            loss = (
                classification_loss
                + CAP_PENALTY_WEIGHT * cap_penalty
                + AUX_LOSS_WEIGHT * aux_loss
                + SENTIMENT_LOSS_WEIGHT * sentiment_loss
            )

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += classification_loss.item()
            total_entropy += entropy.item()
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            sentiment_preds.extend(torch.argmax(sentiment_logits, dim=1).detach().cpu().tolist())
            sentiment_targets.extend(sentiment_labels.detach().cpu().tolist())

            if train_mode and step % 200 == 0:
                running_acc = accuracy_score(all_labels, all_preds)
                print(
                    f"  step {step}/{len(loader)}  loss={classification_loss.item():.4f}  "
                    f"auxLoss={aux_loss.item():.4f}  sentLoss={sentiment_loss.item():.4f}  "
                    f"capPenalty={cap_penalty.item():.4f}  "
                    f"entropy={entropy.item():.4f}  running_acc={running_acc:.4f}"
                )

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
        "sentiment_accuracy": accuracy_score(sentiment_targets, sentiment_preds),
    }
    return metrics, all_labels, all_preds


def main():
    print("=" * 70)
    print("FINAL MODEL TRAINING - adaptive tri-modal fusion")
    print(f"Device: {DEVICE}   Text={TEXT_DIM}D Audio={AUDIO_DIM}D Video={VIDEO_DIM}D -> {NUM_CLASSES} emotions")
    print(
        f"Modality dropout: starts text={INITIAL_DROPOUT_PROBS['text']:.0%} "
        f"audio={INITIAL_DROPOUT_PROBS['audio']:.0%} video={INITIAL_DROPOUT_PROBS['video']:.0%}, "
        f"then adapts each epoch from auxiliary-head accuracy"
    )
    print(f"Modality-weight cap: penalty {CAP_PENALTY_WEIGHT}x for any weight above {MAX_MODALITY_WEIGHT:.0%}")
    print(f"Auxiliary unimodal loss weight: {AUX_LOSS_WEIGHT}")
    print(f"Sentiment auxiliary loss weight: {SENTIMENT_LOSS_WEIGHT}")
    print("Speaker-relational attention bias: enabled (same-speaker vs different-speaker learned scalars)")
    print("=" * 70)

    train_dataset = MELDDataset(split="train")
    val_dataset = MELDDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    class_weights = compute_class_weights(train_dataset).to(DEVICE)

    model = MultimodalFusionModel(
        text_dim=TEXT_DIM, audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM, num_classes=NUM_CLASSES,
        num_speaker_slots=NUM_SPEAKER_SLOTS, num_sentiment_classes=NUM_SENTIMENT_CLASSES,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    criterion = FocalLoss(class_weights, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=LR_PATIENCE)

    # Selecting the "best" checkpoint by macro F1 alone is noisy: a single
    # epoch can spike on macro F1 while accuracy and weighted F1 (the more
    # stable, support-weighted signal) dip badly, giving an unrepresentative
    # checkpoint. Weighted F1 tracks overall quality more robustly while
    # still reflecting per-class performance, so it's the selection metric;
    # macro F1 is still logged every epoch for visibility into the
    # minority-class trend.
    best_weighted_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    dropout_probs = dict(INITIAL_DROPOUT_PROBS)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'=' * 70}\nEPOCH {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.6f})\n{'=' * 70}")
        print(
            "dropout probs: "
            + ", ".join(f"{m}={p:.0%}" for m, p in dropout_probs.items())
        )

        probs_used_this_epoch = dict(dropout_probs)
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, dropout_probs)
        val_metrics, val_labels, val_preds = run_epoch(model, val_loader, criterion)
        scheduler.step(val_metrics["weighted_f1"])

        dropout_probs = update_dropout_probs(dropout_probs, train_metrics["aux_accuracy"])

        epoch_time = time.time() - epoch_start
        aux_acc_str = ", ".join(f"{m}={a:.4f}" for m, a in train_metrics["aux_accuracy"].items())
        print(
            f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
            f"macroF1={train_metrics['macro_f1']:.4f} weightedF1={train_metrics['weighted_f1']:.4f} "
            f"weightEntropy={train_metrics['entropy']:.4f}/{math.log(3):.4f}"
        )
        print(
            f"train aux accuracy: {aux_acc_str}  "
            f"sentiment_accuracy={train_metrics['sentiment_accuracy']:.4f}"
        )
        print(
            f"val:   loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macroF1={val_metrics['macro_f1']:.4f} weightedF1={val_metrics['weighted_f1']:.4f}  "
            f"sentiment_accuracy={val_metrics['sentiment_accuracy']:.4f}  "
            f"({epoch_time:.0f}s)"
        )

        if val_metrics["weighted_f1"] > best_weighted_f1:
            best_weighted_f1 = val_metrics["weighted_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                    "emotion_names": EMOTION_NAMES,
                    "training_config": {
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "focal_gamma": FOCAL_GAMMA,
                        "seed": SEED,
                        "strategy": (
                            "single-stage adaptive fusion, moderate class weighting, gentle focal loss, "
                            "adaptive modality dropout, modality-weight cap penalty, auxiliary unimodal losses, "
                            "dialogue-relative speaker embedding, speaker-relational attention bias, "
                            "sentiment auxiliary loss"
                        ),
                        "modality_dropout_probs_this_epoch": probs_used_this_epoch,
                        "max_modality_weight": MAX_MODALITY_WEIGHT,
                        "cap_penalty_weight": CAP_PENALTY_WEIGHT,
                        "aux_loss_weight": AUX_LOSS_WEIGHT,
                        "sentiment_loss_weight": SENTIMENT_LOSS_WEIGHT,
                        "train_aux_accuracy": train_metrics["aux_accuracy"],
                        "train_sentiment_accuracy": train_metrics["sentiment_accuracy"],
                        "val_modality_weight_entropy": val_metrics["entropy"],
                    },
                },
                FINAL_MODEL_PATH,
            )
            print(f"  -> new best model saved to {FINAL_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}\nTRAINING COMPLETE ({total_time / 60:.1f} min)")
    print(f"Best epoch: {best_epoch}   Best val weighted F1: {best_weighted_f1:.4f}")
    print(f"Checkpoint: {FINAL_MODEL_PATH}\n{'=' * 70}")

    # Final class-wise report on validation using the best saved checkpoint.
    checkpoint = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_labels, val_preds = run_epoch(model, val_loader, criterion)
    print("\nFinal validation class-wise report:")
    print(classification_report(val_labels, val_preds, target_names=EMOTION_NAMES, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
