import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v2.pt"
)

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MELD TRI-MODAL V2 TEST EVALUATION")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")
    print()

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("Loading untouched MELD test dataset...")
    print()

    test_dataset = MELDDataset("test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print()
    print(
        f"Test dialogues: {len(test_dataset)}"
    )

    print(
        "Test data remains completely untouched."
    )

    print()

    # ========================================================
    # CREATE V2 MODEL
    # ========================================================

    print("=" * 70)
    print("CREATING V2 ADAPTIVE FUSION MODEL")
    print("=" * 70)

    print()

    model = MultimodalFusionModel(
        text_dim=TEXT_DIM,
        audio_dim=AUDIO_DIM,
        video_dim=VIDEO_DIM,
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # ========================================================
    # LOAD V2 CHECKPOINT
    # ========================================================

    print("Loading V2 checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "V2 checkpoint loaded successfully."
        )

        if "epoch" in checkpoint:

            print(
                f"Saved epoch: {checkpoint['epoch']}"
            )

        if "val_accuracy" in checkpoint:

            print(
                f"Saved validation accuracy: "
                f"{checkpoint['val_accuracy']:.4f}"
            )

        if "val_macro_f1" in checkpoint:

            print(
                f"Saved validation Macro F1: "
                f"{checkpoint['val_macro_f1']:.4f}"
            )

        if "val_weighted_f1" in checkpoint:

            print(
                f"Saved validation Weighted F1: "
                f"{checkpoint['val_weighted_f1']:.4f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "V2 model state dictionary loaded."
        )

    model.eval()

    print()

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    print("=" * 70)
    print("RUNNING V2 TEST EVALUATION")
    print("=" * 70)

    print()

    all_predictions = []
    all_labels = []

    total_utterances = 0

    dialogue_count = 0

    with torch.no_grad():

        for batch in test_loader:

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
            # TRI-MODAL V2 FORWARD
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

            predictions = torch.argmax(
                output,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            dialogue_count += 1

            total_utterances += labels.numel()

    print(
        "Evaluation complete."
    )

    print()

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # MAIN RESULTS
    # ========================================================

    print("=" * 70)
    print("V2 TRI-MODAL TEST RESULTS")
    print("=" * 70)

    print()

    print(
        f"Test dialogues     : "
        f"{dialogue_count}"
    )

    print(
        f"Test utterances    : "
        f"{total_utterances}"
    )

    print()

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("=" * 70)
    print("V2 CLASSIFICATION REPORT")
    print("=" * 70)

    print()

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0
    )

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("=" * 70)
    print("V2 CONFUSION MATRIX")
    print("=" * 70)

    print()

    print("Rows = Actual")
    print("Columns = Predicted")
    print()

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    print(
        "             "
        + " ".join(
            f"{name[:8]:>9}"
            for name in EMOTION_NAMES
        )
    )

    for i, row in enumerate(matrix):

        print(
            f"{EMOTION_NAMES[i]:10s} "
            + " ".join(
                f"{value:9d}"
                for value in row
            )
        )

    print()

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print("=" * 70)
    print("V2 PREDICTION DISTRIBUTION")
    print("=" * 70)

    print()

    for class_id, emotion in enumerate(
        EMOTION_NAMES
    ):

        count = sum(
            1
            for prediction in all_predictions
            if prediction == class_id
        )

        percentage = (
            count
            / len(all_predictions)
            * 100
        )

        print(
            f"{emotion:10s}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    print()

    # ========================================================
    # PER-CLASS F1
    # ========================================================

    print("=" * 70)
    print("V2 PER-CLASS F1")
    print("=" * 70)

    print()

    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0
    )

    for i, emotion in enumerate(
        EMOTION_NAMES
    ):

        print(
            f"{emotion:10s}: "
            f"F1 = {per_class_f1[i]:.4f}"
        )

    print()

    # ========================================================
    # V1 VS V2 COMPARISON
    # ========================================================

    V1_ACCURACY = 0.5556
    V1_WEIGHTED_F1 = 0.5537
    V1_MACRO_F1 = 0.3380

    print("=" * 70)
    print("V1 VS V2 TEST COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':20s}"
        f"{'V1':>12s}"
        f"{'V2':>12s}"
        f"{'Change':>12s}"
    )

    print("-" * 56)

    print(
        f"{'Accuracy':20s}"
        f"{V1_ACCURACY:>12.4f}"
        f"{accuracy:>12.4f}"
        f"{accuracy - V1_ACCURACY:>+12.4f}"
    )

    print(
        f"{'Weighted F1':20s}"
        f"{V1_WEIGHTED_F1:>12.4f}"
        f"{weighted_f1:>12.4f}"
        f"{weighted_f1 - V1_WEIGHTED_F1:>+12.4f}"
    )

    print(
        f"{'Macro F1':20s}"
        f"{V1_MACRO_F1:>12.4f}"
        f"{macro_f1:>12.4f}"
        f"{macro_f1 - V1_MACRO_F1:>+12.4f}"
    )

    print()

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    if macro_f1 > V1_MACRO_F1:

        print(
            "✓ V2 improves Macro F1 over V1."
        )

    elif macro_f1 < V1_MACRO_F1:

        print(
            "⚠ V2 has lower Macro F1 than V1."
        )

    else:

        print(
            "V2 has the same Macro F1 as V1."
        )

    print()

    if accuracy > V1_ACCURACY:

        print(
            "✓ V2 improves test accuracy over V1."
        )

    elif accuracy < V1_ACCURACY:

        print(
            "⚠ V2 has lower test accuracy than V1."
        )

    else:

        print(
            "V2 has the same test accuracy as V1."
        )

    print()

    print("=" * 70)
    print("V2 TEST EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()