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

# TEXT_DIM is 768 (frozen DistilBERT, see modules/text_features_distilbert.py)
# rather than the original MELD paper's 600-D task-specific CNN features.
TEXT_DIM = 768
AUDIO_DIM = 300
VIDEO_DIM = 512
HIDDEN_DIM = 256
NUM_CLASSES = 7

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
