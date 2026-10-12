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


# ---------------------------------------------------------------- final-review additions (12 Oct 2026)
# The frozen final ensemble (validation-chosen families, first and official test look; see final_review/JOURNAL.md):
# its cached test probabilities are averaged per family, then across families. Nothing is recomputed or tuned here.
FINAL_ENSEMBLE = {
    "H3c": ["logs/prob_cache/CTX_large_s42.pt", "logs/prob_cache/CTX_large_s1.pt"],
    "DBc": ["logs/prob_cache/DEB_ctx_s42.pt", "logs/prob_cache/DEB_ctx_s1.pt"],
    "ctx_dropout": [f"final_review/results/meld_ctx_crct/output/prob_cache/meld_ctx_s{s}.pt" for s in (42, 1, 2)],
    "H3": ["logs/prob_cache/H3_large.pt"],
}
_ens = None
_lookups = None


def get_ensemble():
    global _ens
    if _ens is None:
        fams = {}
        sizes = None
        for name, paths in FINAL_ENSEMBLE.items():
            caches = [torch.load(PROJECT_ROOT / p, map_location="cpu", weights_only=False) for p in paths]
            fams[name] = torch.stack([torch.as_tensor(c["test"], dtype=torch.float32) for c in caches]).mean(0)
            sizes = list(caches[0]["test_sizes"])
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        _ens = {"families": fams, "offsets": offsets, "sizes": sizes}
    return _ens


def get_lookups():
    global _lookups
    if _lookups is None:
        from modules.data_loader import load_speaker_lookup, load_utterance_lookup
        _lookups = (load_speaker_lookup("test"), load_utterance_lookup("test"))
    return _lookups


def _inputs(sample):
    text = sample["text"].unsqueeze(0).to(DEVICE)
    audio = sample["audio"].unsqueeze(0).to(DEVICE)
    video = sample["video"].unsqueeze(0).to(DEVICE)
    slots = sample["speaker_slots"].unsqueeze(0).to(DEVICE) if "num_speaker_slots" in _checkpoint_meta else None
    return text, audio, video, slots


@app.get("/dialogue/{dialogue_index}")
@torch.no_grad()
def dialogue(dialogue_index: int):
    """Whole dialogue: speakers, lines, gold labels, model B predictions and the final ensemble's predictions."""
    dataset = get_test_dataset()
    if not (0 <= dialogue_index < len(dataset)):
        raise HTTPException(404, "dialogue_index out of range")
    sample = dataset[dialogue_index]
    model = get_model()
    text, audio, video, slots = _inputs(sample)
    res = predict_dialogue(model, text, audio, video, speaker_slots=slots)
    ens = get_ensemble()
    assert ens["sizes"][dialogue_index] == sample["length"], "ensemble cache and dataset disagree on dialogue length"
    off = ens["offsets"][dialogue_index]
    speakers, lines = get_lookups()
    out = []
    d = int(sample["dialogue_id"])
    for i, u in enumerate(sample["utterance_ids"]):
        fam_probs = {f: p[off + i] for f, p in ens["families"].items()}
        avg = torch.stack(list(fam_probs.values())).mean(0)
        ens_pred = int(avg.argmax())
        agree = sum(int(p.argmax()) == ens_pred for p in fam_probs.values())
        out.append({
            "utterance_index": i, "utterance_id": u, "speaker": speakers.get((d, int(u)), "?"),
            "text": lines.get((d, int(u)), ""), "gold": EMOTION_NAMES[int(sample["labels"][i])],
            "model_b": EMOTION_NAMES[int(res["predictions"][i])], "model_b_conf": float(res["probabilities"][i].max()),
            "ensemble": EMOTION_NAMES[ens_pred], "ensemble_conf": float(avg[ens_pred]),
            "ensemble_agree": agree, "ensemble_size": len(fam_probs),
            "ensemble_votes": {f: EMOTION_NAMES[int(p.argmax())] for f, p in fam_probs.items()},
        })
    return {"dialogue_id": sample["dialogue_id"], "utterances": out}


class AttributionRequest(BaseModel):
    dialogue_index: int
    utterance_index: int


@app.post("/attribution")
@torch.no_grad()
def attribution(req: AttributionRequest):
    """Context attribution for one target, as in final_review/analysis/context_attribution.py: hide one context cell
    (speaker role x direction) in one modality and report the change in the probability of the originally predicted
    class and whether the prediction changes. The target utterance itself is never modified."""
    dataset = get_test_dataset()
    sample = dataset[req.dialogue_index]
    n, t = sample["length"], req.utterance_index
    if not (0 <= t < n):
        raise HTTPException(404, "utterance_index out of range")
    model = get_model()
    text, audio, video, slots = _inputs(sample)
    base = torch.softmax(model(text, audio, video, speaker_slots=slots).reshape(n, NUM_CLASSES)[t], -1)
    pred = int(base.argmax())
    sl = sample["speaker_slots"].tolist()
    cells = {
        "Same speaker, before": [i for i in range(n) if i < t and sl[i] == sl[t]],
        "Other speakers, before": [i for i in range(n) if i < t and sl[i] != sl[t]],
        "Same speaker, after": [i for i in range(n) if i > t and sl[i] == sl[t]],
        "Other speakers, after": [i for i in range(n) if i > t and sl[i] != sl[t]],
        "All context": [i for i in range(n) if i != t],
    }
    mods = {"Words": (1, 0, 0), "Voice": (0, 1, 0), "Face": (0, 0, 1), "All channels": (1, 1, 1)}
    grid = []
    for cname, idx in cells.items():
        row = {"cell": cname, "count": len(idx), "values": []}
        for mname, (mt, ma, mv) in mods.items():
            if not idx:
                row["values"].append(None)
                continue
            mask = torch.zeros(1, n, 1, device=DEVICE)
            mask[0, idx] = 1.0
            tt, aa, vv = text * (1 - mask * mt), audio * (1 - mask * ma), video * (1 - mask * mv)
            p = torch.softmax(model(tt, aa, vv, speaker_slots=slots).reshape(n, NUM_CLASSES)[t], -1)
            row["values"].append({"modality": mname, "delta": float(p[pred] - base[pred]),
                                  "new_prediction": EMOTION_NAMES[int(p.argmax())]})
        grid.append(row)
    return {"prediction": EMOTION_NAMES[pred], "confidence": float(base[pred]),
            "gold": EMOTION_NAMES[int(sample["labels"][t])], "modalities": list(mods), "grid": grid}


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
