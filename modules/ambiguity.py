"""Modality agreement / disagreement scoring.

Given what each modality would predict on its own for one utterance,
classifies how much they agree. Used by the disagreement analysis script
and by the web app to show a live ambiguity indicator.
"""


def disagreement_level(text_pred, audio_pred, video_pred):
    """Classify agreement between the three single-modality predictions.

    Returns (level, score). score is (distinct predictions - 1) / 2:
      0.0 -> LOW    all three modalities agree
      0.5 -> MEDIUM exactly one modality disagrees with the other two
      1.0 -> HIGH   all three modalities predict a different emotion
    """
    distinct = len({text_pred, audio_pred, video_pred})
    score = (distinct - 1) / 2.0
    level = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}[distinct]
    return level, score
