import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA / MODEL CONFIGURATION
# ============================================================

TEXT_DIM = 600
AUDIO_DIM = 300
VIDEO_DIM = 512

NUM_CLASSES = 7

BATCH_SIZE = 1

# V3 training
EPOCHS = 10

LEARNING_RATE = 0.0002

WEIGHT_DECAY = 1e-4

FOCAL_GAMMA = 2.0

PATIENCE = 4

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v3.pt"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# EMOTION CONFIGURATION
# ============================================================

EMOTION_NAMES = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger"
]


NEUTRAL = 0
SURPRISE = 1
FEAR = 2
SADNESS = 3
JOY = 4
DISGUST = 5
ANGER = 6


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():

    print()
    print("=" * 70)
    print("LOADING MELD TRI-MODAL DATASETS")
    print("=" * 70)

    train_dataset = MELDDataset(
        split="train"
    )

    val_dataset = MELDDataset(
        split="val"
    )

    print()

    print(
        f"Training dialogues   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation dialogues : "
        f"{len(val_dataset)}"
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# GET CLASS COUNTS
# ============================================================

def get_class_counts(dataset):

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float32
    )

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        labels = sample["labels"]

        for label in labels.tolist():

            label = int(label)

            if 0 <= label < NUM_CLASSES:

                counts[label] += 1

    return counts


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    counts,
    title
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    total = counts.sum().item()

    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            counts[class_id].item()
        )

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"{EMOTION_NAMES[class_id]:10s}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 70)

    print(
        f"Total utterances: "
        f"{int(total)}"
    )


# ============================================================
# V3 CLASS WEIGHTS
#
# Purpose:
# - Reduce neutral dominance
# - Increase rare-class learning
# - Give additional emphasis to fear/disgust
# - Avoid extreme weights
# ============================================================

def create_v3_class_weights(
    dataset
):

    counts = get_class_counts(
        dataset
    )

    print_distribution(
        counts,
        "V3 TRAINING CLASS DISTRIBUTION"
    )

    safe_counts = torch.clamp(
        counts,
        min=1.0
    )

    total = counts.sum()

    # Inverse square-root frequency.
    #
    # This is deliberately less aggressive
    # than pure inverse frequency.
    weights = torch.sqrt(
        total / safe_counts
    )

    # Normalize.
    weights = (
        weights /
        weights.mean()
    )

    # Targeted minority emphasis.
    weights[FEAR] *= 1.35

    weights[DISGUST] *= 1.35

    weights[SADNESS] *= 1.10

    # Reduce neutral dominance.
    weights[NEUTRAL] *= 0.70

    # Safety limits.
    weights = torch.clamp(
        weights,
        min=0.40,
        max=3.00
    )

    # Final normalization.
    weights = (
        weights /
        weights.mean()
    )

    print()
    print("=" * 70)
    print("V3 CLASS WEIGHTS")
    print("=" * 70)

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{EMOTION_NAMES[class_id]:10s}: "
            f"{weights[class_id].item():.4f}"
        )

    print("=" * 70)

    return weights


# ============================================================
# MINORITY-AWARE DIALOGUE SAMPLER
#
# IMPORTANT:
# We sample complete dialogues.
#
# We NEVER separate text/audio/video.
# We NEVER duplicate individual modality features.
#
# This keeps the multimodal sequence synchronized.
# ============================================================

def create_v3_sampler(
    dataset
):

    print()
    print("=" * 70)
    print(
        "CREATING V3 MINORITY-AWARE "
        "DIALOGUE SAMPLER"
    )
    print("=" * 70)

    counts = get_class_counts(
        dataset
    )

    safe_counts = torch.clamp(
        counts,
        min=1.0
    )

    # Base class priority.
    class_priority = (
        1.0 /
        torch.sqrt(safe_counts)
    )

    # Additional emphasis.
    class_priority[FEAR] *= 1.35

    class_priority[DISGUST] *= 1.35

    class_priority[SADNESS] *= 1.10

    # Reduce neutral sampling.
    class_priority[NEUTRAL] *= 0.45

    # Normalize.
    class_priority = (
        class_priority /
        class_priority.mean()
    )

    dialogue_weights = []

    for index in range(
        len(dataset)
    ):

        sample = dataset[index]

        labels = [
            int(label)
            for label
            in sample["labels"].tolist()
        ]

        if len(labels) == 0:

            dialogue_weights.append(
                1.0
            )

            continue

        # Determine which emotions
        # are present in this dialogue.
        present_classes = set(
            label
            for label in labels
            if 0 <= label < NUM_CLASSES
        )

        if not present_classes:

            dialogue_weights.append(
                1.0
            )

            continue

        present_weights = [
            class_priority[class_id].item()
            for class_id
            in present_classes
        ]

        # Use the strongest class represented
        # in the dialogue.
        dialogue_weight = max(
            present_weights
        )

        dialogue_weights.append(
            dialogue_weight
        )

    dialogue_weights = torch.tensor(
        dialogue_weights,
        dtype=torch.double
    )

    # Prevent extreme probabilities.
    dialogue_weights = torch.clamp(
        dialogue_weights,
        min=0.30,
        max=3.00
    )

    print()

    print(
        f"Number of dialogue samples : "
        f"{len(dialogue_weights)}"
    )

    print(
        f"Minimum dialogue weight    : "
        f"{dialogue_weights.min().item():.4f}"
    )

    print(
        f"Maximum dialogue weight    : "
        f"{dialogue_weights.max().item():.4f}"
    )

    sampler = WeightedRandomSampler(
        weights=dialogue_weights,
        num_samples=len(dataset),
        replacement=True
    )

    print()

    print(
        "✓ V3 minority-aware sampler created"
    )

    return sampler


# ============================================================
# V3 FOCAL LOSS
#
# Correct formulation:
#
#   CE = unweighted cross entropy
#   pt = probability of correct class
#   focal = (1 - pt)^gamma
#   class weight applied separately
#
# This avoids calculating pt from an already
# class-weighted cross entropy value.
# ============================================================

class V3FocalLoss(
    nn.Module
):

    def __init__(
        self,
        class_weights,
        gamma=2.0
    ):

        super().__init__()

        self.gamma = gamma

        self.register_buffer(
            "class_weights",
            class_weights
        )

    def forward(
        self,
        logits,
        targets
    ):

        # ----------------------------------------------------
        # Unweighted CE
        # ----------------------------------------------------

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Correct-class probability
        # ----------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        target_probability = probabilities.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        # ----------------------------------------------------
        # Focal factor
        # ----------------------------------------------------

        focal_factor = (
            1.0 -
            target_probability
        ) ** self.gamma

        # ----------------------------------------------------
        # Class-specific weights
        # ----------------------------------------------------

        target_weights = (
            self.class_weights[
                targets
            ]
        )

        # ----------------------------------------------------
        # Final loss
        # ----------------------------------------------------

        loss = (
            target_weights
            * focal_factor
            * ce_loss
        )

        return loss.mean()


# ============================================================
# CREATE MODEL
#
# IMPORTANT:
# The current MultimodalFusionModel constructor accepts:
#
#   text_dim
#   audio_dim
#   video_dim
#   num_classes
#
# It does NOT accept hidden_dim/fusion_dim.
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING V3 TRI-MODAL MODEL")
    print("=" * 70)

    model = MultimodalFusionModel(
        text_dim=TEXT_DIM,
        audio_dim=AUDIO_DIM,
        video_dim=VIDEO_DIM,
        num_classes=NUM_CLASSES
    )

    model = model.to(
        DEVICE
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print()

    print(
        f"Device                : "
        f"{DEVICE}"
    )

    print(
        f"Total parameters      : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters  : "
        f"{trainable_parameters:,}"
    )

    return model


# ============================================================
# PREPARE BATCH
# ============================================================

def prepare_batch(
    batch
):

    text = batch["text"]

    audio = batch["audio"]

    video = batch["video"]

    labels = batch["labels"]

    # --------------------------------------------------------
    # Batch size = 1
    # --------------------------------------------------------

    if isinstance(
        text,
        list
    ):

        text = text[0]

    if isinstance(
        audio,
        list
    ):

        audio = audio[0]

    if isinstance(
        video,
        list
    ):

        video = video[0]

    if isinstance(
        labels,
        list
    ):

        labels = labels[0]

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    text = text.float().to(
        DEVICE
    )

    audio = audio.float().to(
        DEVICE
    )

    video = video.float().to(
        DEVICE
    )

    labels = labels.long().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Ensure batch dimension
    # --------------------------------------------------------

    if text.dim() == 2:

        text = text.unsqueeze(0)

    if audio.dim() == 2:

        audio = audio.unsqueeze(0)

    if video.dim() == 2:

        video = video.unsqueeze(0)

    if labels.dim() == 1:

        labels = labels.unsqueeze(0)

    return (
        text,
        audio,
        video,
        labels
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch
):

    model.train()

    total_loss = 0.0

    all_predictions = []

    all_targets = []

    total_steps = len(
        loader
    )

    for step, batch in enumerate(
        loader,
        start=1
    ):

        (
            text,
            audio,
            video,
            labels
        ) = prepare_batch(
            batch
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            text,
            audio,
            video
        )

        # ----------------------------------------------------
        # Flatten utterances
        # ----------------------------------------------------

        logits_flat = logits.reshape(
            -1,
            NUM_CLASSES
        )

        labels_flat = labels.reshape(
            -1
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits_flat,
            labels_flat
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits_flat,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_targets.extend(
            labels_flat.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if step % 100 == 0:

            running_accuracy = accuracy_score(
                all_targets,
                all_predictions
            )

            running_macro_f1 = f1_score(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0
            )

            print(
                f"Step {step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {running_accuracy:.4f} | "
                f"Macro F1: {running_macro_f1:.4f}"
            )

    # --------------------------------------------------------
    # Epoch metrics
    # --------------------------------------------------------

    epoch_loss = (
        total_loss /
        max(total_steps, 1)
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    epoch_macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    epoch_weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1,
        epoch_weighted_f1
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_predictions = []

    all_targets = []

    total_steps = len(
        loader
    )

    for batch in loader:

        (
            text,
            audio,
            video,
            labels
        ) = prepare_batch(
            batch
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            text,
            audio,
            video
        )

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        logits_flat = logits.reshape(
            -1,
            NUM_CLASSES
        )

        labels_flat = labels.reshape(
            -1
        )

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        loss = criterion(
            logits_flat,
            labels_flat
        )

        total_loss += loss.item()

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits_flat,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_targets.extend(
            labels_flat.cpu()
            .numpy()
            .tolist()
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    validation_loss = (
        total_loss /
        max(total_steps, 1)
    )

    validation_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    validation_macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    validation_weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_macro_f1,
        validation_weighted_f1
    )


# ============================================================
# DETAILED VALIDATION REPORT
# ============================================================

@torch.no_grad()
def validation_report(
    model,
    loader
):

    model.eval()

    all_predictions = []

    all_targets = []

    for batch in loader:

        (
            text,
            audio,
            video,
            labels
        ) = prepare_batch(
            batch
        )

        logits = model(
            text,
            audio,
            video
        )

        logits_flat = logits.reshape(
            -1,
            NUM_CLASSES
        )

        labels_flat = labels.reshape(
            -1
        )

        predictions = torch.argmax(
            logits_flat,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        all_targets.extend(
            labels_flat.cpu()
            .numpy()
            .tolist()
        )

    print()
    print("=" * 70)
    print("V3 VALIDATION CLASS-WISE PERFORMANCE")
    print("=" * 70)

    precision = precision_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{EMOTION_NAMES[class_id]:10s} | "
            f"Precision: {precision[class_id]:.4f} | "
            f"Recall: {recall[class_id]:.4f} | "
            f"F1: {f1[class_id]:.4f}"
        )

    print("=" * 70)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_accuracy,
    val_macro_f1,
    val_weighted_f1
):

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "epoch":
            epoch,

        "val_accuracy":
            val_accuracy,

        "val_macro_f1":
            val_macro_f1,

        "val_weighted_f1":
            val_weighted_f1,

        "text_dim":
            TEXT_DIM,

        "audio_dim":
            AUDIO_DIM,

        "video_dim":
            VIDEO_DIM,

        "num_classes":
            NUM_CLASSES,

        "emotion_names":
            EMOTION_NAMES,

        "v3_config": {

            "focal_gamma":
                FOCAL_GAMMA,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "seed":
                SEED
        }
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MELD TRI-MODAL PROPOSED MODEL — V3"
    )
    print(
        "MINORITY-CLASS FOCUSED TRAINING"
    )
    print("=" * 70)

    print()

    print(
        f"Device : {DEVICE}"
    )

    print(
        "Text   : 600-D"
    )

    print(
        "Audio  : 300-D"
    )

    print(
        "Video  : 512-D"
    )

    print(
        "Output : 7 emotions"
    )

    print()

    print(
        "V3 improvements:"
    )

    print(
        "  ✓ Adaptive tri-modal fusion"
    )

    print(
        "  ✓ Minority-aware dialogue sampling"
    )

    print(
        "  ✓ Stronger Fear/Disgust weighting"
    )

    print(
        "  ✓ Correct weighted focal loss"
    )

    print(
        "  ✓ Gradient clipping"
    )

    print(
        "  ✓ ReduceLROnPlateau scheduler"
    )

    print(
        "  ✓ Early stopping"
    )

    print(
        "  ✓ Separate V3 checkpoint"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    (
        train_dataset,
        val_dataset
    ) = load_datasets()

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        create_v3_class_weights(
            train_dataset
        )
    )

    class_weights = (
        class_weights.to(
            DEVICE
        )
    )

    # ========================================================
    # SAMPLER
    # ========================================================

    sampler = create_v3_sampler(
        train_dataset
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model()

    # ========================================================
    # LOSS
    # ========================================================

    criterion = V3FocalLoss(
        class_weights=class_weights,
        gamma=FOCAL_GAMMA
    )

    print()
    print("=" * 70)
    print("V3 LOSS")
    print("=" * 70)

    print(
        "Weighted Focal Loss"
    )

    print(
        f"Gamma: {FOCAL_GAMMA}"
    )

    print("=" * 70)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-5
        )
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    best_macro_f1 = -1.0

    best_accuracy = 0.0

    best_weighted_f1 = 0.0

    best_epoch = 0

    patience_counter = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print("=" * 70)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("=" * 70)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Learning rate: "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            train_macro_f1,
            train_weighted_f1
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print()

        print(
            "Running validation..."
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1
        ) = validate(
            model,
            val_loader,
            criterion
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()
        print("-" * 70)
        print("EPOCH RESULTS")
        print("-" * 70)

        print(
            f"Train Loss        : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy    : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1    : "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted F1 : "
            f"{train_weighted_f1:.4f}"
        )

        print()

        print(
            f"Val Loss          : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy      : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1      : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Val Weighted F1   : "
            f"{val_weighted_f1:.4f}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )

        # ----------------------------------------------------
        # BEST MODEL
        #
        # Macro F1 is primary because V3 is
        # specifically intended to improve minority
        # emotion recognition.
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = (
                val_macro_f1
            )

            best_accuracy = (
                val_accuracy
            )

            best_weighted_f1 = (
                val_weighted_f1
            )

            best_epoch = epoch

            patience_counter = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_accuracy,
                val_macro_f1,
                val_weighted_f1
            )

            print()

            print(
                "✓ NEW BEST V3 MODEL SAVED"
            )

            print(
                f"  Macro F1: "
                f"{val_macro_f1:.4f}"
            )

            print(
                f"  Accuracy: "
                f"{val_accuracy:.4f}"
            )

            print(
                f"  Weighted F1: "
                f"{val_weighted_f1:.4f}"
            )

            print(
                f"  Path: "
                f"{MODEL_PATH}"
            )

        else:

            patience_counter += 1

            print()

            print(
                f"No improvement. "
                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if MODEL_PATH.exists():

        print()
        print(
            "Loading best V3 checkpoint..."
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    # ========================================================
    # CLASS-WISE VALIDATION
    # ========================================================

    validation_report(
        model,
        val_loader
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V3 TRI-MODAL TRAINING COMPLETE"
    )
    print("=" * 70)

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best validation Weighted F1: "
        f"{best_weighted_f1:.4f}"
    )

    print()

    print(
        "Best V3 model saved at:"
    )

    print(
        MODEL_PATH
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()