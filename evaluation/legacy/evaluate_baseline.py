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
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline_model.pt"
)

BATCH_SIZE = 1


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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MELD BASELINE MODEL - FINAL TEST EVALUATION")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")
    print()


    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print("Loading untouched MELD test dataset...")
    print()

    test_dataset = MELDDataset(
        "test"
    )

    print(
        f"Test dialogues: {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print()


    # ========================================================
    # LOAD BASELINE MODEL
    # ========================================================

    print("Loading trained baseline model...")

    model = MultimodalFusionModel(
        text_dim=600,
        audio_dim=300,
        hidden_dim=256,
        num_classes=7
    )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # train_baseline.py saves model.state_dict()
    model.load_state_dict(
        checkpoint
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Baseline model loaded successfully!"
    )

    print()


    # ========================================================
    # RUN TEST EVALUATION
    # ========================================================

    print(
        "Running FINAL BASELINE TEST evaluation..."
    )

    print()

    all_predictions = []
    all_labels = []


    with torch.no_grad():

        for batch in test_loader:

            # ------------------------------------------------
            # Get inputs
            # ------------------------------------------------

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
            # Add batch dimension if necessary
            # ------------------------------------------------

            if text.dim() == 2:

                text = text.unsqueeze(0)


            if audio.dim() == 2:

                audio = audio.unsqueeze(0)


            if labels.dim() == 1:

                labels = labels.unsqueeze(0)


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                text,
                audio
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = torch.argmax(
                output,
                dim=-1
            )


            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_predictions.extend(
                predictions
                .reshape(-1)
                .cpu()
                .tolist()
            )

            all_labels.extend(
                labels
                .reshape(-1)
                .cpu()
                .tolist()
            )


    print(
        "Baseline test evaluation complete!"
    )

    print()


    # ========================================================
    # TEST SET SANITY CHECK
    # ========================================================

    print("=" * 70)
    print("BASELINE TEST SET SANITY CHECK")
    print("=" * 70)

    print(
        f"Test dialogues : {len(test_dataset)}"
    )

    print(
        f"Test utterances: {len(all_labels)}"
    )

    if len(all_labels) == 2610:

        print(
            "✓ Correct MELD test utterance count: 2610"
        )

    else:

        print(
            "⚠ WARNING: Expected 2610 test utterances."
        )


    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print()
    print("TEST LABEL DISTRIBUTION")
    print("-" * 70)


    label_counts = [
        all_labels.count(i)
        for i in range(
            len(EMOTION_NAMES)
        )
    ]


    for emotion, count in zip(
        EMOTION_NAMES,
        label_counts
    ):

        print(
            f"{emotion:10s}: {count:6d}"
        )


    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("BASELINE PREDICTION DISTRIBUTION")
    print("-" * 70)


    prediction_counts = [
        all_predictions.count(i)
        for i in range(
            len(EMOTION_NAMES)
        )
    ]


    for emotion, count in zip(
        EMOTION_NAMES,
        prediction_counts
    ):

        print(
            f"{emotion:10s}: {count:6d}"
        )


    # ========================================================
    # OVERALL METRICS
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
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL BASELINE TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Precision       : {precision:.4f}"
    )

    print(
        f"Recall          : {recall:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE TEST CLASSIFICATION REPORT")
    print("=" * 70)


    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(EMOTION_NAMES)
            )
        ),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0
    )


    print(
        report
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("=" * 70)
    print("BASELINE TEST CONFUSION MATRIX")
    print("=" * 70)


    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(EMOTION_NAMES)
            )
        )
    )


    print(
        matrix
    )


    # ========================================================
    # PER-EMOTION F1
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE PER-EMOTION F1 SUMMARY")
    print("=" * 70)


    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(EMOTION_NAMES)
            )
        ),
        average=None,
        zero_division=0
    )


    for emotion, score in zip(
        EMOTION_NAMES,
        per_class_f1
    ):

        print(
            f"{emotion:10s}: "
            f"F1 = {score:.4f}"
        )


    # ========================================================
    # TEST SET INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE TEST SET INFORMATION")
    print("=" * 70)


    print(
        f"Test dialogues  : {len(test_dataset)}"
    )

    print(
        f"Test utterances : {len(all_labels)}"
    )

    print()

    print(
        "Test set oversampling : NO"
    )

    print(
        "Test set duplication   : NO"
    )

    print(
        "Test set balancing     : NO"
    )

    print(
        "Test set modification  : NO"
    )

    print()

    print(
        "✓ Evaluation performed on untouched MELD test data."
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL BASELINE TEST EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()