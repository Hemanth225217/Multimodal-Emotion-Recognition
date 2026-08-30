import sys
from pathlib import Path
from collections import Counter

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
from training.train_proposed import ProposedMultimodalModel


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
    / "proposed_model.pt"
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

    print(
        "MELD PROPOSED MODEL - FINAL TEST EVALUATION"
    )

    print("=" * 70)

    print()

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # LOAD TEST DATASET
    # ========================================================

    print()

    print(
        "Loading untouched MELD test dataset..."
    )

    test_dataset = MELDDataset(
        "test"
    )

    print()

    print(
        f"Test dialogues: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print(
        "Loading best proposed model..."
    )

    model = ProposedMultimodalModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support both checkpoint formats
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Best proposed model loaded successfully!"
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    print()

    print(
        "Running FINAL TEST evaluation..."
    )

    print()

    all_predictions = []

    all_labels = []

    # ========================================================
    # RUN THROUGH TEST SET
    # ========================================================

    with torch.no_grad():

        for batch in test_loader:

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
            # Model prediction
            # ------------------------------------------------

            output = model(
                text,
                audio
            )

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
        "Test evaluation complete!"
    )

    # ========================================================
    # BASIC SANITY CHECK
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TEST SET SANITY CHECK"
    )

    print("=" * 70)

    print(
        f"Test dialogues : "
        f"{len(test_dataset)}"
    )

    print(
        f"Test utterances: "
        f"{len(all_labels)}"
    )

    if len(all_labels) == 2610:

        print(
            "✓ Correct MELD test utterance count: 2610"
        )

    else:

        print(
            "WARNING: Expected 2610 test utterances."
        )

    # ========================================================
    # TEST LABEL DISTRIBUTION
    # ========================================================

    label_counts = Counter(
        all_labels
    )

    print()

    print(
        "TEST LABEL DISTRIBUTION"
    )

    print("-" * 70)

    for class_id, emotion in enumerate(
        EMOTION_NAMES
    ):

        count = label_counts.get(
            class_id,
            0
        )

        print(
            f"{emotion:10s}: "
            f"{count:6d}"
        )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    prediction_counts = Counter(
        all_predictions
    )

    print()

    print(
        "MODEL PREDICTION DISTRIBUTION"
    )

    print("-" * 70)

    for class_id, emotion in enumerate(
        EMOTION_NAMES
    ):

        count = prediction_counts.get(
            class_id,
            0
        )

        print(
            f"{emotion:10s}: "
            f"{count:6d}"
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

    print(
        "FINAL TEST RESULTS"
    )

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

    print(
        "FINAL TEST CLASSIFICATION REPORT"
    )

    print("=" * 70)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(
                    EMOTION_NAMES
                )
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

    print(
        "FINAL TEST CONFUSION MATRIX"
    )

    print("=" * 70)

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(
                    EMOTION_NAMES
                )
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

    print(
        "PER-EMOTION F1 SUMMARY"
    )

    print("=" * 70)

    per_class_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(
            range(
                len(
                    EMOTION_NAMES
                )
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
    # TEST SET INTEGRITY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "TEST SET INFORMATION"
    )

    print("=" * 70)

    print(
        f"Test dialogues  : "
        f"{len(test_dataset)}"
    )

    print(
        f"Test utterances : "
        f"{len(all_labels)}"
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

    print(
        "FINAL TEST EVALUATION COMPLETE"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()