"""FastAPI backend for the multimodal emotion recognition demo.

Design note: text uses frozen DistilBERT embeddings (768-D, publicly
available -- could in principle embed arbitrary new sentences, though that
isn't wired up here yet). Audio still uses the original MELD paper's 300-D
openSMILE-style features, whose extractor config was never released, so
there's no way to reproduce that exact feature space for a brand-new audio
clip. Given audio is the constraint, this API serves predictions over real
MELD test-set dialogues (for which we do have correct features for every
modality) rather than accepting partial uploads. Every /predict call is a
genuine forward pass through the trained model -- nothing here is mocked.
The missing-modality toggle is real too: setting use_audio=false actually
zeroes the audio tensor and re-runs the model.
"""

import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import PROJECT_ROOT, EMOTION_NAMES, NUM_CLASSES, FINAL_MODEL_PATH, DEVICE
from training.dataset import MELDDataset
from modules.data_loader import ROBERTA_TEXT_PATH
from modules.inference import load_model, predict_dialogue
from modules.ambiguity import disagreement_level
from modules.explanation import build_explanation

TEST_VIDEO_DIR = PROJECT_ROOT / "meld_dataset" / "raw" / "output_repeated_splits_test"

app = FastAPI(title="Multimodal Emotion Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_test_dataset = None
_checkpoint_meta = None


def get_model():
    global _model, _checkpoint_meta
    if _model is None:
        _model, _checkpoint_meta = load_model(FINAL_MODEL_PATH)
    return _model


def get_test_dataset():
    global _test_dataset
    if _test_dataset is None:
        # The demo serves final_model.pt (checkpoint B), which was trained on
        # frozen RoBERTa-base (768-D) text features. Pin them explicitly: the
        # global default text features were later switched to RoBERTa-large
        # (1024-D), which made /predict crash with a shape mismatch until this
        # was pinned.
        _test_dataset = MELDDataset(split="test", text_path=ROBERTA_TEXT_PATH, use_legacy_audio=True)
        get_model()
        sample_dim = _test_dataset[0]["text"].shape[-1]
        if sample_dim != _checkpoint_meta["text_dim"]:
            raise RuntimeError(
                f"Demo dataset text features are {sample_dim}-D but the served checkpoint "
                f"expects {_checkpoint_meta['text_dim']}-D"
            )
    return _test_dataset


class PredictRequest(BaseModel):
    dialogue_index: int
    utterance_index: int
    use_text: bool = True
    use_audio: bool = True
    use_video: bool = True


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": str(FINAL_MODEL_PATH.name),
        "checkpoint_val_accuracy": _checkpoint_meta.get("val_accuracy"),
        "checkpoint_val_macro_f1": _checkpoint_meta.get("val_macro_f1"),
    }


@app.get("/examples")
def list_examples():
    dataset = get_test_dataset()
    examples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        examples.append({
            "dialogue_index": i,
            "dialogue_id": sample["dialogue_id"],
            "num_utterances": sample["length"],
            "labels": [EMOTION_NAMES[label] for label in sample["labels"].tolist()],
            "utterance_ids": sample["utterance_ids"],
        })
    return examples


@app.get("/video/{dialogue_id}/{utterance_id}")
def get_video(dialogue_id: str, utterance_id: str):
    path = TEST_VIDEO_DIR / f"dia{dialogue_id}_utt{utterance_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")


@torch.no_grad()
def _single_modality_predictions(model, text, audio, video, speaker_slots):
    zero_text, zero_audio, zero_video = torch.zeros_like(text), torch.zeros_like(audio), torch.zeros_like(video)
    text_pred = torch.argmax(
        model(text, zero_audio, zero_video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES), dim=1
    ).tolist()
    audio_pred = torch.argmax(
        model(zero_text, audio, zero_video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES), dim=1
    ).tolist()
    video_pred = torch.argmax(
        model(zero_text, zero_audio, video, speaker_slots=speaker_slots).reshape(-1, NUM_CLASSES), dim=1
    ).tolist()
    return text_pred, audio_pred, video_pred


@app.post("/predict")
def predict(req: PredictRequest):
    dataset = get_test_dataset()
    if not (0 <= req.dialogue_index < len(dataset)):
        raise HTTPException(404, "dialogue_index out of range")

    sample = dataset[req.dialogue_index]
    if not (0 <= req.utterance_index < sample["length"]):
        raise HTTPException(404, "utterance_index out of range")

    model = get_model()
    text = sample["text"].unsqueeze(0).to(DEVICE)
    audio = sample["audio"].unsqueeze(0).to(DEVICE)
    video = sample["video"].unsqueeze(0).to(DEVICE)
    speaker_slots = (
        sample["speaker_slots"].unsqueeze(0).to(DEVICE)
        if "num_speaker_slots" in _checkpoint_meta else None
    )

    if not req.use_text:
        text = torch.zeros_like(text)
    if not req.use_audio:
        audio = torch.zeros_like(audio)
    if not req.use_video:
        video = torch.zeros_like(video)

    result = predict_dialogue(model, text, audio, video, speaker_slots=speaker_slots)
    i = req.utterance_index

    pred_class = int(result["predictions"][i])
    confidence = float(result["probabilities"][i, pred_class])
    weights = result["modality_weights"][i].tolist()
    probabilities = {EMOTION_NAMES[c]: float(result["probabilities"][i, c]) for c in range(NUM_CLASSES)}

    text_preds, audio_preds, video_preds = _single_modality_predictions(model, text, audio, video, speaker_slots)
    level, score = disagreement_level(text_preds[i], audio_preds[i], video_preds[i])

    explanation = build_explanation(
        pred_class, confidence, weights, level, (text_preds[i], audio_preds[i], video_preds[i])
    )

    return {
        "dialogue_id": sample["dialogue_id"],
        "utterance_index": i,
        "utterance_id": sample["utterance_ids"][i],
        "ground_truth": EMOTION_NAMES[int(sample["labels"][i])],
        "predicted_emotion": EMOTION_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": probabilities,
        "modality_weights": {"text": weights[0], "audio": weights[1], "video": weights[2]},
        "modality_predictions": {
            "text": EMOTION_NAMES[text_preds[i]],
            "audio": EMOTION_NAMES[audio_preds[i]],
            "video": EMOTION_NAMES[video_preds[i]],
        },
        "disagreement_level": level,
        "disagreement_score": score,
        "explanation": explanation,
        "modalities_used": {"text": req.use_text, "audio": req.use_audio, "video": req.use_video},
    }


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
