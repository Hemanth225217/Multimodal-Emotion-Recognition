import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# IMPORTS
# ============================================================

from training.dataset import MELDDataset
from models.fusion_model import MultimodalFusionModel


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_DIM = 600
AUDIO_DIM = 300
VIDEO_DIM = 512
NUM_CLASSES = 7

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "proposed_model_v4.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    print()
    print("=" * 70)
    print("LOADING MELD TEST DATASET")
    print("=" * 70)

    test_dataset = MELDDataset(
        split="test"
    )

    print()

    print(
        f"Test dialogues   : "
        f"{len(test_dataset)}"
    )

    return test_dataset


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print()
    print("=" * 70)
    print("CREATING V4 MODEL FOR EVALUATION")
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
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    return model


# ============================================================
# LOAD V4 CHECKPOINT
# ============================================================

def load_checkpoint(model):

    print()
    print("=" * 70)
    print("LOADING V4 CHECKPOINT")
    print("=" * 70)

    print(
        f"Path: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nV4 checkpoint not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print()

        print(
            "Checkpoint information:"
        )

        if "epoch" in checkpoint:

            print(
                f"  Epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_accuracy" in checkpoint:

            print(
                f"  Validation Accuracy: "
                f"{checkpoint['val_accuracy']:.4f}"
            )

        if "val_macro_f1" in checkpoint:

            print(
                f"  Validation Macro F1: "
                f"{checkpoint['val_macro_f1']:.4f}"
            )

        if "val_weighted_f1" in checkpoint:

            print(
                f"  Validation Weighted F1: "
                f"{checkpoint['val_weighted_f1']:.4f}"
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    print()

    print(
        "✓ V4 checkpoint loaded successfully"
    )


# ============================================================
# PREPARE BATCH
# ============================================================

def prepare_batch(batch):

    text = batch["text"]
    audio = batch["audio"]
    video = batch["video"]
    labels = batch["labels"]

    if isinstance(text, list):
        text = text[0]

    if isinstance(audio, list):
        audio = audio[0]

    if isinstance(video, list):
        video = video[0]

    if isinstance(labels, list):
        labels = labels[0]

    if not torch.is_tensor(text):

        text = torch.tensor(
            text,
            dtype=torch.float32
        )

    if not torch.is_tensor(audio):

        audio = torch.tensor(
            audio,
            dtype=torch.float32
        )

    if not torch.is_tensor(video):

        video = torch.tensor(
            video,
            dtype=torch.float32
        )

    if not torch.is_tensor(labels):

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

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
# EVALUATE V4
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader
):

    model.eval()

    all_predictions = []

    all_targets = []

    print()
    print("=" * 70)
    print("RUNNING V4 TEST EVALUATION")
    print("=" * 70)

    for dialogue_number, batch in enumerate(
        test_loader,
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

        if dialogue_number % 50 == 0:

            print(
                f"Evaluated "
                f"{dialogue_number}/"
                f"{len(test_loader)} dialogues..."
            )

    return (
        all_targets,
        all_predictions
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    targets,
    predictions
):

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision = precision_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("V4 TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Test Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Weighted Precision  : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall     : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.4f}"
    )

    print("=" * 70)

    print()
    print("=" * 70)
    print("V4 TEST CLASS-WISE PERFORMANCE")
    print("=" * 70)

    report = classification_report(
        targets,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0
    )

    print(report)

    return (
        accuracy,
        weighted_f1,
        macro_f1
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    targets,
    predictions
):

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    print()
    print("=" * 70)
    print("V4 TEST CONFUSION MATRIX")
    print("=" * 70)

    print()

    print(
        f"{'Actual / Pred':15s}",
        end=""
    )

    for emotion in EMOTION_NAMES:

        print(
            f"{emotion[:8]:>10s}",
            end=""
        )

    print()

    print("-" * 85)

    for row_index, row in enumerate(cm):

        print(
            f"{EMOTION_NAMES[row_index]:15s}",
            end=""
        )

        for value in row:

            print(
                f"{value:10d}",
                end=""
            )

        print()

    print()


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def print_prediction_distribution(
    predictions
):

    print()
    print("=" * 70)
    print("V4 TEST PREDICTION DISTRIBUTION")
    print("=" * 70)

    total = len(
        predictions
    )

    for class_id in range(
        NUM_CLASSES
    ):

        count = predictions.count(
            class_id
        )

        percentage = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

        print(
            f"{EMOTION_NAMES[class_id]:10s}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 70)

    print(
        f"Total predictions: "
        f"{total}"
    )


# ============================================================
# V2 / V3 / V4 COMPARISON
# ============================================================

def print_comparison(
    v4_accuracy,
    v4_weighted_f1,
    v4_macro_f1
):

    # Established test results.
    v2_accuracy = 0.5728
    v2_weighted_f1 = 0.5622
    v2_macro_f1 = 0.3378

    v3_accuracy = 0.5188
    v3_weighted_f1 = 0.5347
    v3_macro_f1 = 0.3434

    print()
    print("=" * 70)
    print("V2 vs V3 vs V4 TEST COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':25s}"
        f"{'V2':>12s}"
        f"{'V3':>12s}"
        f"{'V4':>12s}"
    )

    print("-" * 61)

    print(
        f"{'Accuracy':25s}"
        f"{v2_accuracy:>12.4f}"
        f"{v3_accuracy:>12.4f}"
        f"{v4_accuracy:>12.4f}"
    )

    print(
        f"{'Weighted F1':25s}"
        f"{v2_weighted_f1:>12.4f}"
        f"{v3_weighted_f1:>12.4f}"
        f"{v4_weighted_f1:>12.4f}"
    )

    print(
        f"{'Macro F1':25s}"
        f"{v2_macro_f1:>12.4f}"
        f"{v3_macro_f1:>12.4f}"
        f"{v4_macro_f1:>12.4f}"
    )

    print()

    print(
        "V4 change compared with V2:"
    )

    print(
        f"  Accuracy: "
        f"{v4_accuracy - v2_accuracy:+.4f}"
    )

    print(
        f"  Weighted F1: "
        f"{v4_weighted_f1 - v2_weighted_f1:+.4f}"
    )

    print(
        f"  Macro F1: "
        f"{v4_macro_f1 - v2_macro_f1:+.4f}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MELD TRI-MODAL PROPOSED MODEL — V4 TEST"
    )
    print(
        "V2 FINE-TUNED MODEL"
    )
    print("=" * 70)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Model: proposed_model_v4.pt"
    )

    print(
        "Dataset: MELD TEST"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset()

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # V4 checkpoint
    # --------------------------------------------------------

    load_checkpoint(
        model
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    (
        targets,
        predictions
    ) = evaluate(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        v4_accuracy,
        v4_weighted_f1,
        v4_macro_f1
    ) = print_metrics(
        targets,
        predictions
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        targets,
        predictions
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print_prediction_distribution(
        predictions
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print_comparison(
        v4_accuracy,
        v4_weighted_f1,
        v4_macro_f1
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V4 TEST EVALUATION COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()