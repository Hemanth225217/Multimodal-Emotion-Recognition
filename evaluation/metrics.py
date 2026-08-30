import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
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


def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(7))
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=EMOTION_NAMES,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }