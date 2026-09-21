import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEXT_DIM = 600
AUDIO_DIM = 300
VIDEO_DIM = 512

NUM_CLASSES = 7

BATCH_SIZE = 1

EPOCHS = 8

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-4

FOCAL_GAMMA = 1.5

LABEL_SMOOTHING = 0.0

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v2.pt"
)


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


# ============================================================
# WEIGHTED FOCAL LOSS
# ============================================================

class WeightedFocalLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        gamma=1.5,
        label_smoothing=0.0
    ):

        super().__init__()

        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        target_probability = probabilities.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        focal_factor = (
            1.0 - target_probability
        ) ** self.gamma

        loss = focal_factor * ce_loss

        return loss.mean()


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataset):

    print()
    print("=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float32
    )

    for index in range(len(dataset)):

        sample = dataset[index]

        labels = sample["labels"]

        for label in labels:

            label_id = int(label)

            if 0 <= label_id < NUM_CLASSES:

                counts[label_id] += 1


    total = counts.sum().item()


    print()

    for i in range(NUM_CLASSES):

        percentage = (
            counts[i].item()
            / total
            * 100
        )

        print(
            f"{EMOTION_NAMES[i]:10s} : "
            f"{int(counts[i].item()):5d} "
            f"({percentage:6.2f}%)"
        )


    print("-" * 70)

    print(
        f"Total training utterances: "
        f"{int(total)}"
    )


    # --------------------------------------------------------
    # SQRT INVERSE-FREQUENCY WEIGHTING
    #
    # This is intentionally moderate.
    # Extremely large weights can make the model unstable.
    # --------------------------------------------------------

    safe_counts = torch.clamp(
        counts,
        min=1.0
    )

    weights = torch.sqrt(
        total / (
            NUM_CLASSES * safe_counts
        )
    )


    # Normalize mean to 1

    weights = (
        weights
        / weights.mean()
    )


    # Prevent extreme weighting

    weights = torch.clamp(
        weights,
        min=0.50,
        max=2.00
    )


    print()
    print("=" * 70)
    print("MODERATE CLASS WEIGHTS")
    print("=" * 70)

    for i in range(NUM_CLASSES):

        print(
            f"{EMOTION_NAMES[i]:10s} : "
            f"{weights[i].item():.4f}"
        )

    print("=" * 70)

    return weights


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []


    for step, batch in enumerate(
        loader,
        start=1
    ):

        # ----------------------------------------------------
        # LOAD THREE MODALITIES
        # ----------------------------------------------------

        text = batch["text"].to(
            DEVICE,
            dtype=torch.float32
        )

        audio = batch["audio"].to(
            DEVICE,
            dtype=torch.float32
        )

        video = batch["video"].to(
            DEVICE,
            dtype=torch.float32
        )

        labels = batch["labels"].to(
            DEVICE,
            dtype=torch.long
        )


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        output = model(
            text,
            audio,
            video
        )


        # ----------------------------------------------------
        # RESHAPE
        #
        # Output:
        # [batch, utterances, 7]
        #
        # Labels:
        # [batch, utterances]
        #
        # Convert to:
        # [N, 7]
        # [N]
        # ----------------------------------------------------

        output = output.reshape(
            -1,
            NUM_CLASSES
        )

        labels = labels.reshape(
            -1
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            output,
            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()


        # Prevent exploding gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_loss += loss.item()


        predictions = torch.argmax(
            output,
            dim=1
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if step % 100 == 0:

            accuracy = accuracy_score(
                all_labels,
                all_predictions
            )

            macro_f1 = f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            )

            print(
                f"Step {step}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f}"
            )


    epoch_loss = (
        total_loss
        / len(loader)
    )


    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    epoch_weighted_f1 = f1_score(
        all_labels,
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

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []


    with torch.no_grad():

        for batch in loader:

            # ------------------------------------------------
            # THREE MODALITIES
            # ------------------------------------------------

            text = batch["text"].to(
                DEVICE,
                dtype=torch.float32
            )

            audio = batch["audio"].to(
                DEVICE,
                dtype=torch.float32
            )

            video = batch["video"].to(
                DEVICE,
                dtype=torch.float32
            )

            labels = batch["labels"].to(
                DEVICE,
                dtype=torch.long
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            output = model(
                text,
                audio,
                video
            )


            # ------------------------------------------------
            # RESHAPE
            # ------------------------------------------------

            output = output.reshape(
                -1,
                NUM_CLASSES
            )

            labels = labels.reshape(
                -1
            )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = criterion(
                output,
                labels
            )

            total_loss += loss.item()


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = torch.argmax(
                output,
                dim=1
            )


            all_predictions.extend(
                predictions
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels
                .cpu()
                .tolist()
            )


    loss = (
        total_loss
        / len(loader)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    return (
        loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "MELD TRI-MODAL PROPOSED MODEL"
    )

    print(
        "TEXT + AUDIO + VIDEO"
    )

    print("=" * 70)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print()

    print(
        "Architecture:"
    )

    print(
        "  Text  : 600-D"
    )

    print(
        "  Audio : 300-D"
    )

    print(
        "  Video : 512-D"
    )

    print(
        "  Output: 7 emotions"
    )

    print()


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    print(
        "Loading datasets..."
    )

    print()


    train_dataset = MELDDataset(
        "train"
    )

    val_dataset = MELDDataset(
        "val"
    )


    print()

    print(
        f"Training dialogues: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation dialogues: "
        f"{len(val_dataset)}"
    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = calculate_class_weights(
        train_dataset
    )

    class_weights = class_weights.to(
        DEVICE
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "CREATING TRI-MODAL FUSION MODEL"
    )

    print("=" * 70)

    print()


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
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print()


    # ========================================================
    # LOSS
    # ========================================================

    criterion = WeightedFocalLoss(
        class_weights=class_weights,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING
    )


    print(
        "Loss:"
    )

    print(
        f"  Weighted Focal Loss"
    )

    print(
        f"  Gamma: {FOCAL_GAMMA}"
    )

    print()


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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )


    # ========================================================
    # TRAINING
    # ========================================================

    best_macro_f1 = -1.0

    best_epoch = 0

    patience_counter = 0

    EARLY_STOPPING_PATIENCE = 3


    for epoch in range(
        EPOCHS
    ):

        print()

        print("=" * 70)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 70)

        print(
            f"Learning rate: "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        print()


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
            criterion,
            optimizer
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
        ) = evaluate(
            model,
            val_loader,
            criterion
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()

        print("-" * 70)

        print(
            "EPOCH RESULTS"
        )

        print("-" * 70)

        print(
            f"Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1   : "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted F1: "
            f"{train_weighted_f1:.4f}"
        )

        print()

        print(
            f"Val Loss         : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy     : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1     : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Val Weighted F1  : "
            f"{val_weighted_f1:.4f}"
        )

        print("-" * 70)


        # ----------------------------------------------------
        # LEARNING RATE UPDATE
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            best_epoch = epoch + 1

            patience_counter = 0


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_macro_f1":
                        val_macro_f1,

                    "val_accuracy":
                        val_accuracy,

                    "val_weighted_f1":
                        val_weighted_f1,

                    "text_dim":
                        TEXT_DIM,

                    "audio_dim":
                        AUDIO_DIM,

                    "video_dim":
                        VIDEO_DIM,

                    "num_classes":
                        NUM_CLASSES
                },
                MODEL_PATH
            )


            print()

            print(
                "✓ NEW BEST MODEL SAVED"
            )

            print(
                f"  Validation Macro F1: "
                f"{val_macro_f1:.4f}"
            )

            print(
                f"  Validation Accuracy: "
                f"{val_accuracy:.4f}"
            )

            print(
                f"  Path: {MODEL_PATH}"
            )


        else:

            patience_counter += 1

            print()

            print(
                f"No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TRI-MODAL TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print()

    print(
        "Best model saved at:"
    )

    print(
        MODEL_PATH
    )

    print()

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()