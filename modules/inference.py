"""Shared model loading and prediction helpers.

This is the one place that knows how to load the fusion-model checkpoint and
run a forward pass. Evaluation scripts and the FastAPI backend both import
from here instead of re-implementing checkpoint loading.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES, NUM_SPEAKER_SLOTS, DEVICE
from models.fusion_model import MultimodalFusionModel


def load_model(checkpoint_path, device=DEVICE):
    """Build a MultimodalFusionModel and load weights from checkpoint_path.

    Returns (model, metadata) where metadata is the checkpoint's extra
    fields (val_accuracy, epoch, ...) if it was saved as a dict, else {}.
    """
    model = MultimodalFusionModel(
        text_dim=TEXT_DIM,
        audio_dim=AUDIO_DIM,
        video_dim=VIDEO_DIM,
        num_classes=NUM_CLASSES,
        num_speaker_slots=NUM_SPEAKER_SLOTS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"} if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else {}

    # strict=False: checkpoints saved before speaker_embedding was added to
    # the architecture won't have that key. It's safe to leave it at random
    # init in that case, since it's only ever used when the caller passes
    # speaker_slots -- and callers correctly don't do that for a checkpoint
    # that never learned to use it.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_model: missing keys {missing}, unexpected keys {unexpected}")

    model.eval()
    return model, metadata


@torch.no_grad()
def predict_dialogue(model, text, audio, video, speaker_slots=None):
    """Run one already-batched dialogue ([1, T, dim] tensors) through the model.

    Returns per-utterance predicted class ids, full class probabilities, and
    the three adaptive modality weights (text/audio/video), each [T, ...].
    """
    logits, weights = model(text, audio, video, speaker_slots=speaker_slots, return_weights=True)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    return {
        "predictions": predictions.squeeze(0).cpu(),
        "probabilities": probabilities.squeeze(0).cpu(),
        "modality_weights": weights.squeeze(0).cpu(),
    }
