"""Extract frozen Wav2Vec2 audio embeddings for MELD utterances.

Replaces the original MELD paper's 300-D openSMILE-style audio features
(2018-era) with 768-D frozen Wav2Vec2 embeddings (mean-pooled over time),
extracted directly from the raw MELD video clips' audio track via ffmpeg (no
fine-tuning -- still pre-extracted features, just from a much stronger
encoder). Uses the plain self-supervised "wav2vec2-base" rather than an
ASR-fine-tuned variant: ASR fine-tuning pushes representations toward
phonetic/word content and tends to wash out the prosodic cues (tone, pitch,
energy) that actually carry emotional signal.

Saved in the same [train_dict, dev_dict, test_dict] pickle structure as the
other feature files, keyed by str(Dialogue_ID) -> embeddings sorted by
Utterance_ID, so training/dataset.py needs no changes beyond config.AUDIO_DIM.
"""

import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT

MELD_CSV_DIR = PROJECT_ROOT / "meld_dataset" / "data" / "MELD"
VIDEO_ROOT = PROJECT_ROOT / "meld_dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "meld_features" / "audio_wav2vec2"
OUTPUT_PATH = OUTPUT_DIR / "audio_wav2vec2.pkl"

MODEL_NAME = "facebook/wav2vec2-base"
DEVICE = torch.device("cpu")
SAMPLE_RATE = 16000

SPLIT_CONFIG = {
    "train": {"csv": "train_sent_emo.csv", "folder": "train_splits"},
    "dev": {"csv": "dev_sent_emo.csv", "folder": "dev_splits_complete"},
    "test": {"csv": "test_sent_emo.csv", "folder": "output_repeated_splits_test"},
}


def find_video(folder, dialogue_id, utterance_id):
    filename = f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"
    path = VIDEO_ROOT / folder / filename
    return path if path.exists() else None


def extract_audio_array(video_path):
    """Extract mono 16kHz audio from a video file via ffmpeg, in memory."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-f", "wav", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None

    audio, _ = sf.read(io.BytesIO(result.stdout))
    if audio.size == 0:
        return None
    return audio.astype(np.float32)


BATCH_SIZE = 16
CHUNK_SIZE = 128  # local length-sort window (bounds memory, cuts padding waste)
MAX_SECONDS = 15  # MELD utterances are short; caps rare long outliers for speed

# facebook/wav2vec2-base is a full transformer (95M params); one-at-a-time
# CPU inference was measured at ~0.7s/clip, which would take ~2.7 hours for
# the full dataset. Batching gets real CPU vectorization/throughput gains.
# Padding must be masked out correctly: the model's conv feature extractor
# downsamples time steps, so the raw-sample attention mask has to be
# downsampled the same way before mean-pooling, or padding would bias the
# embeddings of shorter clips in a batch. _get_feature_vector_attention_mask
# is the model's own helper for exactly this.


@torch.no_grad()
def embed_audio_batch(audio_arrays, feature_extractor, model):
    trimmed = [a[: MAX_SECONDS * SAMPLE_RATE] for a in audio_arrays]

    inputs = feature_extractor(
        trimmed,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    ).to(DEVICE)

    output = model(**inputs)

    feature_mask = model._get_feature_vector_attention_mask(
        output.last_hidden_state.shape[1], inputs["attention_mask"]
    ).unsqueeze(-1).float()

    summed = (output.last_hidden_state * feature_mask).sum(dim=1)
    counts = feature_mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def build_split_dict(split_name, config, feature_extractor, model):
    df = pd.read_csv(MELD_CSV_DIR / config["csv"])
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    rows = list(df.itertuples())

    # Keyed by (dialogue_id, utterance_id) while processing in length-bucketed
    # order for speed; regrouped into the required dialogue->sorted-list shape
    # at the end, so processing order never has to match output order.
    embeddings_by_key = {}
    missing = 0
    failed = 0

    for chunk_start in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[chunk_start:chunk_start + CHUNK_SIZE]

        chunk_items = []
        for row in chunk:
            dialogue_id = int(row.Dialogue_ID)
            utterance_id = int(row.Utterance_ID)

            video_path = find_video(config["folder"], dialogue_id, utterance_id)
            if video_path is None:
                missing += 1
                continue

            audio = extract_audio_array(video_path)
            if audio is None:
                failed += 1
                continue

            chunk_items.append((dialogue_id, utterance_id, audio))

        # Local length-sort: minimizes padding waste within each sub-batch
        # without holding more than CHUNK_SIZE clips' audio in memory at once.
        chunk_items.sort(key=lambda item: len(item[2]))

        for sub_start in range(0, len(chunk_items), BATCH_SIZE):
            sub_batch = chunk_items[sub_start:sub_start + BATCH_SIZE]
            audios = [item[2] for item in sub_batch]

            try:
                embeddings = embed_audio_batch(audios, feature_extractor, model)
                for (dialogue_id, utterance_id, _), embedding in zip(sub_batch, embeddings):
                    embeddings_by_key[(dialogue_id, utterance_id)] = embedding
            except Exception as exc:
                print(f"  batch failed ({exc}); skipping {len(sub_batch)} clips")

        done = chunk_start + len(chunk)
        print(f"  {done}/{len(rows)}  (missing={missing} failed={failed})")

    by_dialogue = {}
    for (dialogue_id, utterance_id), embedding in embeddings_by_key.items():
        by_dialogue.setdefault(dialogue_id, []).append((utterance_id, embedding))

    result = {}
    for dialogue_id, items in by_dialogue.items():
        items.sort(key=lambda pair: pair[0])
        result[str(dialogue_id)] = [embedding for _, embedding in items]

    total_utterances = sum(len(v) for v in result.values())
    print(f"{split_name}: {len(result)} dialogues, {total_utterances} utterances, missing={missing}, failed={failed}")
    return result


def main():
    print(f"Loading {MODEL_NAME} (frozen -- inference only, no fine-tuning)...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print(f"Embedding dimension: {model.config.hidden_size}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_dicts = []

    for split_name, config in SPLIT_CONFIG.items():
        print(f"\n=== {split_name} ({config['csv']}) ===")
        result = build_split_dict(split_name, config, feature_extractor, model)
        split_dicts.append(result)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(split_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
