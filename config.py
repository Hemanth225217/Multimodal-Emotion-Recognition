"""Central configuration for the multimodal emotion recognition project.

Every training/evaluation/inference script should import from here instead
of redefining dimensions, paths, or the emotion label set.
"""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Data locations
# --------------------------------------------------------------------------

MELD_FEATURES_DIR = PROJECT_ROOT / "meld_features"
VISUAL_FEATURES_DIR = MELD_FEATURES_DIR / "visual"
TEXT_AUDIO_FEATURES_DIR = MELD_FEATURES_DIR / "MELD.Features.Models" / "features"
MELD_RAW_DIR = PROJECT_ROOT / "meld_dataset" / "raw"
MELD_ANNOTATIONS_DIR = PROJECT_ROOT / "meld_dataset" / "raw"

# --------------------------------------------------------------------------
# Model dimensions (fixed by the pre-extracted MELD baseline features)
# --------------------------------------------------------------------------

# TEXT_DIM is 768 -- frozen RoBERTa-base (modules/text_features_roberta.py), the
# features checkpoint B (models/final_model.pt) was trained on. B is the model
# the demo app, the evaluation scripts and `python training/train_final.py`
# all target, so the defaults here must match it. Other text encoders are
# used through explicit overrides rather than by changing this default:
# RoBERTa-large (1024-D, modules/text_features_roberta_large.py) via
# TRAIN_TEXT_DIM=1024 TRAIN_TEXT_PATH=<its .pkl>, DistilBERT (768-D) via
# TRAIN_TEXT_PATH. (The default was briefly 1024 while RoBERTa-large was
# being explored, which silently broke the demo and every default-configured
# evaluation script against checkpoint B with a 768-vs-1024 shape error.)
TEXT_DIM = 768
# AUDIO_DIM is 300 -- the original MELD paper's openSMILE-style features.
# Frozen Wav2Vec2-base (768-D, modules/audio_features_wav2vec2.py) was tried
# as an upgrade and tested thoroughly (solo and as a 5th ensemble member in
# every combination) but did not beat this: see README "Results". Kept
# available via modules.data_loader.load_features(use_legacy_audio=False).
AUDIO_DIM = 300
VIDEO_DIM = 512
HIDDEN_DIM = 256
NUM_CLASSES = 7

# Dialogue-relative speaker slots (1st distinct speaker in a dialogue = slot
# 0, 2nd = slot 1, ...), not a global per-character embedding -- see
# modules/data_loader.load_speaker_lookup for why. Observed max distinct
# speakers in one MELD dialogue is 9 (train); this leaves headroom.
NUM_SPEAKER_SLOTS = 10

EMOTION_NAMES = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
]
EMOTION_TO_ID = {name: idx for idx, name in enumerate(EMOTION_NAMES)}

# MELD also ships a 3-way Sentiment column (neutral/positive/negative)
# alongside Emotion -- not a deterministic function of it (surprise splits
# across both positive and negative depending on context), so it's a
# genuinely separate signal used as an auxiliary multi-task loss (see
# modules/data_loader.load_sentiment_lookup and training/train_final.py).
SENTIMENT_NAMES = ["neutral", "positive", "negative"]
NUM_SENTIMENT_CLASSES = len(SENTIMENT_NAMES)

# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pt"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pt"
LEGACY_MODELS_DIR = MODELS_DIR / "legacy"

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
