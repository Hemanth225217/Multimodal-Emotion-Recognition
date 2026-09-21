import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# CONFIGURATION
# ============================================================

TEXT_DIM = 600
AUDIO_DIM = 300
VIDEO_DIM = 512

NUM_CLASSES = 7

BATCH_SIZE = 1

EPOCHS = 8

# Low LR because V4 starts from the trained V2 model.
LEARNING_RATE = 0.0001

WEIGHT_DECAY = 1e-4

# Gentler focal loss than V3.
FOCAL_GAMMA = 1.0

PATIENCE = 3


# ============================================================
# MODEL PATHS
# ============================================================

V2_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v2.pt"
)

V4_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v4.pt"
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
# EMOTION NAMES
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
# CLASS COUNTS
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
# PRINT DISTRIBUTION
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
# V4 MODERATE CLASS WEIGHTS
#
# V3 used aggressive minority handling.
#
# V4 deliberately uses moderate weighting:
#
# - reduce neutral dominance
# - modestly increase fear/disgust
# - retain enough emphasis on sadness
# - avoid extreme rare-class weights
# ============================================================

def create_v4_class_weights(
    dataset
):

    counts = get_class_counts(
        dataset
    )

    print_distribution(
        counts,
        "V4 TRAINING CLASS DISTRIBUTION"
    )

    safe_counts = torch.clamp(
        counts,
        min=1.0
    )

    total = counts.sum()

    # Inverse square-root frequency.
    weights = torch.sqrt(
        total / safe_counts
    )

    # Normalize.
    weights = (
        weights /
        weights.mean()
    )

    # Moderate adjustments.
    #
    # These are intentionally much gentler
    # than the V3 sampling strategy.

    weights[NEUTRAL] *= 0.80

    weights[FEAR] *= 1.15

    weights[DISGUST] *= 1.15

    weights[SADNESS] *= 1.05

    # Keep weights in a stable range.
    weights = torch.clamp(
        weights,
        min=0.55,
        max=2.00
    )

    # Final normalization.
    weights = (
        weights /
        weights.mean()
    )

    print()
    print("=" * 70)
    print("V4 MODERATE CLASS WEIGHTS")
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
# V4 FOCAL LOSS
#
# V4 uses gamma=1.0.
#
# This gives some focus to difficult examples without
# aggressively changing the V2 decision boundary.
# ============================================================

class V4FocalLoss(
    nn.Module
):

    def __init__(
        self,
        class_weights,
        gamma=1.0
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
        # Standard CE
        # ----------------------------------------------------

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        # ----------------------------------------------------
        # Probability assigned to correct class
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
        # Gentle focal factor
        # ----------------------------------------------------

        focal_factor = (
            1.0 -
            target_probability
        ) ** self.gamma

        # ----------------------------------------------------
        # Class weighting
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
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING V4 MODEL")
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
        f"Device               : "
        f"{DEVICE}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    return model


# ============================================================
# LOAD V2 INITIALIZATION
#
# V4 starts from V2 rather than random initialization.
# ============================================================

def load_v2_initialization(
    model
):

    print()
    print("=" * 70)
    print("LOADING V2 WEIGHTS FOR V4 FINE-TUNING")
    print("=" * 70)

    print(
        f"Source: {V2_MODEL_PATH}"
    )

    if not V2_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nV2 checkpoint was not found:\n"
            f"{V2_MODEL_PATH}\n\n"
            "V4 requires the trained V2 model."
        )

    checkpoint = torch.load(
        V2_MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Standard checkpoint format
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict
    )

    print()

    print(
        "✓ V2 weights loaded successfully"
    )

    if (
        isinstance(checkpoint, dict)
        and
        "epoch" in checkpoint
    ):

        print(
            f"V2 source epoch: "
            f"{checkpoint['epoch']}"
        )

    if (
        isinstance(checkpoint, dict)
        and
        "val_accuracy" in checkpoint
    ):

        print(
            f"V2 validation accuracy: "
            f"{checkpoint['val_accuracy']:.4f}"
        )

    if (
        isinstance(checkpoint, dict)
        and
        "val_macro_f1" in checkpoint
    ):

        print(
            f"V2 validation Macro F1: "
            f"{checkpoint['val_macro_f1']:.4f}"
        )

    if (
        isinstance(checkpoint, dict)
        and
        "val_weighted_f1" in checkpoint
    ):

        print(
            f"V2 validation Weighted F1: "
            f"{checkpoint['val_weighted_f1']:.4f}"
        )


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
    # Batch dimension
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
    criterion
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
        # Optimizer
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
    # Metrics
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
        # Loss
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
# CLASS-WISE VALIDATION
# ============================================================

@torch.no_grad()
def validation_class_report(
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

    precision = precision_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        average=None,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    print()
    print("=" * 70)
    print("V4 VALIDATION CLASS-WISE PERFORMANCE")
    print("=" * 70)

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

        "v4_config": {

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "focal_gamma":
                FOCAL_GAMMA,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "seed":
                SEED,

            "initialized_from":
                "proposed_model_v2.pt",

            "minority_sampler":
                False,

            "training_strategy":
                "moderate_class_weighted_focal_finetuning"
        }
    }

    torch.save(
        checkpoint,
        V4_MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MELD TRI-MODAL PROPOSED MODEL — V4"
    )
    print(
        "V2 FINE-TUNING WITH MODERATE "
        "MINORITY BALANCING"
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
        "V4 strategy:"
    )

    print(
        "  ✓ Start from trained V2 model"
    )

    print(
        "  ✓ Keep adaptive tri-modal fusion"
    )

    print(
        "  ✓ No aggressive dialogue sampler"
    )

    print(
        "  ✓ Moderate minority class weights"
    )

    print(
        "  ✓ Gentle focal loss"
    )

    print(
        "  ✓ Low learning rate fine-tuning"
    )

    print(
        "  ✓ Gradient clipping"
    )

    print(
        "  ✓ Separate V4 checkpoint"
    )

    # ========================================================
    # DATA
    # ========================================================

    (
        train_dataset,
        val_dataset
    ) = load_datasets()

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        create_v4_class_weights(
            train_dataset
        )
    )

    class_weights = (
        class_weights.to(
            DEVICE
        )
    )

    # ========================================================
    # DATA LOADERS
    #
    # IMPORTANT:
    # No WeightedRandomSampler.
    #
    # We use the original MELD training distribution.
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()

    # ========================================================
    # INITIALIZE FROM V2
    # ========================================================

    load_v2_initialization(
        model
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = V4FocalLoss(
        class_weights=class_weights,
        gamma=FOCAL_GAMMA
    )

    print()
    print("=" * 70)
    print("V4 LOSS")
    print("=" * 70)

    print(
        "Moderately Weighted Focal Loss"
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
    # SCHEDULER
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
    # BEST MODEL TRACKING
    # ========================================================

    best_macro_f1 = -1.0

    best_accuracy = 0.0

    best_weighted_f1 = 0.0

    best_epoch = 0

    patience_counter = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print("=" * 70)

        print(
            f"V4 EPOCH {epoch}/{EPOCHS}"
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
            criterion
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
        # RESULTS
        # ----------------------------------------------------

        print()
        print("-" * 70)
        print("V4 EPOCH RESULTS")
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
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )

        # ----------------------------------------------------
        # BEST MODEL
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
                "✓ NEW BEST V4 MODEL SAVED"
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
                f"{V4_MODEL_PATH}"
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
    # LOAD BEST V4 CHECKPOINT
    # ========================================================

    if V4_MODEL_PATH.exists():

        print()
        print(
            "Loading best V4 checkpoint..."
        )

        checkpoint = torch.load(
            V4_MODEL_PATH,
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

    validation_class_report(
        model,
        val_loader
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V4 TRAINING COMPLETE"
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
        "V4 checkpoint:"
    )

    print(
        V4_MODEL_PATH
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()