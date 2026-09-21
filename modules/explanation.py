"""Human-readable explanation generation for a single prediction.

Turns a prediction's confidence, adaptive modality weights, and modality
agreement into the plain-language explanation shown in the web app.
"""

from collections import Counter

from config import EMOTION_NAMES

MODALITY_NAMES = ["text", "audio", "video"]


def build_explanation(predicted_class, confidence, modality_weights, disagreement, modality_predictions):
    """
    predicted_class: int class id chosen by the fusion model
    confidence: float in [0, 1], fusion's probability for predicted_class
    modality_weights: (text_weight, audio_weight, video_weight), sums to ~1
    disagreement: "LOW" | "MEDIUM" | "HIGH"
    modality_predictions: (text_pred, audio_pred, video_pred) class ids, each
        from running that modality alone through the model
    """
    emotion = EMOTION_NAMES[predicted_class].upper()
    ranked = sorted(zip(MODALITY_NAMES, modality_weights), key=lambda pair: pair[1], reverse=True)
    dominant = " and ".join(name for name, _ in ranked[:2])

    lines = [
        f"Predicted emotion: {emotion} ({confidence * 100:.0f}% confidence).",
        "Modal evidence -- " + ", ".join(
            f"{name} {weight * 100:.0f}%" for name, weight in zip(MODALITY_NAMES, modality_weights)
        ) + ".",
    ]

    if disagreement == "LOW":
        lines.append("All three modalities independently agree on this emotion.")
    elif disagreement == "MEDIUM":
        counts = Counter(modality_predictions)
        majority_class, _ = counts.most_common(1)[0]
        agreeing = [name for name, pred in zip(MODALITY_NAMES, modality_predictions) if pred == majority_class]
        lines.append(f"{' and '.join(agreeing)} agree; one modality reads this utterance differently.")
    else:
        lines.append("The three modalities disagree with each other on this utterance.")

    lines.append(f"{dominant.capitalize()} contributed most strongly to this prediction.")

    return " ".join(lines)
