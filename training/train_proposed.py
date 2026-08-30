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

from training.dataset import MELDDataset


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEXT_DIM = 600
AUDIO_DIM = 300

HIDDEN_DIM = 256
FUSION_DIM = 256

NUM_CLASSES = 7

BATCH_SIZE = 1

EPOCHS = 12

LEARNING_RATE = 0.0002

WEIGHT_DECAY = 1e-4

DROPOUT = 0.25

# Focal-loss parameters
FOCAL_GAMMA = 1.5

# Slightly stronger minority emphasis
MAX_CLASS_WEIGHT = 3.0

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model.pt"
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
# PROPOSED MULTIMODAL MODEL
# ============================================================

class ProposedMultimodalModel(nn.Module):

    def __init__(
        self,
        text_dim=TEXT_DIM,
        audio_dim=AUDIO_DIM,
        hidden_dim=HIDDEN_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------

        self.text_encoder = nn.LSTM(
            input_size=text_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # ----------------------------------------------------
        # AUDIO ENCODER
        # ----------------------------------------------------

        self.audio_encoder = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # ----------------------------------------------------
        # CONTEXT ATTENTION
        # ----------------------------------------------------

        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # ----------------------------------------------------
        # MULTIMODAL GATE
        # ----------------------------------------------------

        combined_dim = hidden_dim * 4

        self.gate = nn.Sequential(
            nn.Linear(
                combined_dim,
                combined_dim
            ),
            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                combined_dim,
                fusion_dim
            ),

            nn.LayerNorm(
                fusion_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            )
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                fusion_dim // 2
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                fusion_dim // 2,
                num_classes
            )
        )


    def forward(
        self,
        text,
        audio
    ):

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_features, _ = self.text_encoder(
            text
        )

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        audio_features, _ = self.audio_encoder(
            audio
        )

        # ----------------------------------------------------
        # CONTEXT ATTENTION
        # ----------------------------------------------------

        text_context, _ = self.context_attention(
            text_features,
            text_features,
            text_features
        )

        text_features = (
            text_features
            + text_context
        )

        # ----------------------------------------------------
        # MULTIMODAL FUSION
        # ----------------------------------------------------

        combined = torch.cat(
            (
                text_features,
                audio_features
            ),
            dim=2
        )

        # ----------------------------------------------------
        # GATED FUSION
        # ----------------------------------------------------

        gate = self.gate(
            combined
        )

        gated_features = (
            combined * gate
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fused = self.fusion(
            gated_features
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        output = self.classifier(
            fused
        )

        return output


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
# CLASS WEIGHTS
# ============================================================

def create_class_weights(dataset):

    counts = get_class_counts(
        dataset
    )

    print()
    print("=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
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
        )

        print(
            f"{EMOTION_NAMES[class_id]:10s}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 70)

    print(
        f"Total training utterances: "
        f"{int(total)}"
    )

    # --------------------------------------------------------
    # EFFECTIVE MODERATE WEIGHTING
    #
    # Square-root inverse frequency gives minority classes
    # more importance without allowing fear/disgust to
    # completely dominate training.
    # --------------------------------------------------------

    safe_counts = torch.clamp(
        counts,
        min=1.0
    )

    weights = torch.sqrt(
        total / safe_counts
    )

    # Normalize
    weights = (
        weights /
        weights.mean()
    )

    # Slightly stronger ceiling than previous experiment
    weights = torch.clamp(
        weights,
        min=0.50,
        max=MAX_CLASS_WEIGHT
    )

    print()
    print("=" * 70)
    print("MINORITY-AWARE CLASS WEIGHTS")
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
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        weight=None,
        gamma=1.5
    ):

        super().__init__()

        self.weight = weight
        self.gamma = gamma


    def forward(
        self,
        logits,
        targets
    ):

        # ----------------------------------------------------
        # Cross entropy for each sample
        # ----------------------------------------------------

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )

        # ----------------------------------------------------
        # Convert CE to probability
        #
        # High probability → easy example
        # Low probability  → hard example
        # ----------------------------------------------------

        pt = torch.exp(
            -ce_loss
        )

        # ----------------------------------------------------
        # Focal weighting
        # ----------------------------------------------------

        focal_loss = (
            (1.0 - pt)
            ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# ============================================================
# TRAIN ONE EPOCH
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

        text = batch[
            "text"
        ].to(
            DEVICE
        )

        audio = batch[
            "audio"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        output = model(
            text,
            audio
        )

        # ----------------------------------------------------
        # FLATTEN
        # ----------------------------------------------------

        output_flat = output.reshape(
            -1,
            NUM_CLASSES
        )

        labels_flat = labels.reshape(
            -1
        )

        # ----------------------------------------------------
        # FOCAL LOSS
        # ----------------------------------------------------

        loss = criterion(
            output_flat,
            labels_flat
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

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
            output_flat,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            labels_flat.detach()
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
        total_loss /
        len(loader)
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

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1
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

            text = batch[
                "text"
            ].to(
                DEVICE
            )

            audio = batch[
                "audio"
            ].to(
                DEVICE
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            output = model(
                text,
                audio
            )

            # ------------------------------------------------
            # FLATTEN
            # ------------------------------------------------

            output_flat = output.reshape(
                -1,
                NUM_CLASSES
            )

            labels_flat = labels.reshape(
                -1
            )

            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = criterion(
                output_flat,
                labels_flat
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            predictions = torch.argmax(
                output_flat,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu()
                .tolist()
            )

            all_labels.extend(
                labels_flat.cpu()
                .tolist()
            )

    loss = (
        total_loss /
        len(loader)
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
        "MELD PROPOSED MODEL"
    )

    print(
        "EXPERIMENT 3 - MINORITY-AWARE FOCAL LOSS"
    )

    print("=" * 70)

    print()

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print()

    print(
        "Loading datasets..."
    )

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
    # CLASS WEIGHTS
    # ========================================================

    class_weights = create_class_weights(
        train_dataset
    )

    class_weights = class_weights.to(
        DEVICE
    )

    # ========================================================
    # DATA LOADERS
    #
    # NO OVERSAMPLING.
    # NO DUPLICATION.
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

    print()
    print("=" * 70)
    print("DATA SAMPLING")
    print("=" * 70)

    print(
        "Training sampling : NORMAL SHUFFLE"
    )

    print(
        "Physical duplicates: 0"
    )

    print(
        "Validation modified: NO"
    )

    print(
        "Test modified      : NO"
    )

    print("=" * 70)

    # ========================================================
    # MODEL
    # ========================================================

    print()

    print(
        "Creating proposed multimodal model..."
    )

    model = ProposedMultimodalModel()

    model = model.to(
        DEVICE
    )

    print()

    print(
        "Model created successfully."
    )

    # ========================================================
    # FOCAL LOSS
    # ========================================================

    criterion = FocalLoss(
        weight=class_weights,
        gamma=FOCAL_GAMMA
    )

    print()

    print("=" * 70)
    print("LOSS FUNCTION")
    print("=" * 70)

    print(
        "Loss              : Weighted Focal Loss"
    )

    print(
        f"Focal gamma       : {FOCAL_GAMMA}"
    )

    print(
        "Label smoothing   : 0.0"
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    best_macro_f1 = -1.0

    patience = 4

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        print()

        print("=" * 70)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            train_macro_f1
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
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1  : "
            f"{train_macro_f1:.4f}"
        )

        print()

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1    : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Val Weighted F1 : "
            f"{val_weighted_f1:.4f}"
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_macro_f1
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print()

        print(
            f"Learning Rate   : "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            epochs_without_improvement = 0

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print()

            print(
                "✓ BEST MODEL SAVED"
            )

            print(
                f"Best Validation Macro F1: "
                f"{best_macro_f1:.4f}"
            )

            print(
                f"Saved to: "
                f"{MODEL_PATH}"
            )

        else:

            epochs_without_improvement += 1

            print()

            print(
                "No improvement."
            )

            print(
                f"Early stopping: "
                f"{epochs_without_improvement}/"
                f"{patience}"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
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
        "EXPERIMENT 3 TRAINING COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Best Validation Macro F1: "
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