import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the downloaded MELD feature files (original 2018 paper release)
BASE_DIR = PROJECT_ROOT / "meld_features" / "MELD.Features.Models" / "features"

# Speaker names aren't in the original MELD.Features.Models pickles, so
# they're read directly from the official annotation CSVs when needed.
MELD_CSV_DIR = PROJECT_ROOT / "meld_dataset" / "data" / "MELD"
SPEAKER_CSV_FILES = {
    "train": "train_sent_emo.csv",
    "val": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

# Frozen text embeddings, tried in this order (both replace the original
# 600-D task-specific CNN text features from the MELD paper's release):
DISTILBERT_TEXT_PATH = PROJECT_ROOT / "meld_features" / "text_distilbert" / "text_distilbert.pkl"
ROBERTA_TEXT_PATH = PROJECT_ROOT / "meld_features" / "text_roberta" / "text_roberta.pkl"

# Frozen audio embeddings (see modules/audio_features_wav2vec2.py), replacing
# the original MELD paper's 300-D openSMILE-style features.
WAV2VEC2_AUDIO_PATH = PROJECT_ROOT / "meld_features" / "audio_wav2vec2" / "audio_wav2vec2.pkl"


# Emotion mapping used by MELD
EMOTION_MAP = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}


def load_pickle(filename):
    """Load a MELD pickle feature file (the original files are Python-2 pickles)."""
    file_path = BASE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find feature file: {file_path}"
        )

    with open(file_path, "rb") as file:
        return pickle.load(file, encoding="latin1")


def load_speaker_lookup(split):
    """Maps (dialogue_id, utterance_id) -> speaker name for one split.

    Used to build dialogue-relative speaker slots (see training/dataset.py)
    rather than a global per-character embedding: MELD has 260 unique
    speaker names in train alone, dominated by the 6 main cast members with
    a long tail of one-off guest characters, so a name-keyed embedding
    would overfit and wouldn't generalize to unseen test speakers.
    """
    df = pd.read_csv(MELD_CSV_DIR / SPEAKER_CSV_FILES[split])
    return {
        (int(row.Dialogue_ID), int(row.Utterance_ID)): str(row.Speaker)
        for row in df.itertuples()
    }


def load_sentiment_lookup(split):
    """Maps (dialogue_id, utterance_id) -> sentiment string for one split.

    Used as an auxiliary multi-task loss (see training/train_final.py):
    MELD's Sentiment column isn't a deterministic function of Emotion
    (surprise appears under both positive and negative sentiment depending
    on context), so it carries real extra signal rather than just relabeling
    the main task.
    """
    df = pd.read_csv(MELD_CSV_DIR / SPEAKER_CSV_FILES[split])
    return {
        (int(row.Dialogue_ID), int(row.Utterance_ID)): str(row.Sentiment)
        for row in df.itertuples()
    }


def load_original_text_features():
    """The original MELD paper's 600-D task-specific CNN text features.

    Kept available (not deleted) for the report's feature-comparison
    ablation, but load_features() below uses the DistilBERT embeddings.
    """
    return load_pickle("text_emotion.pkl")


def load_features(text_path=None, use_legacy_audio=False):
    """Load text, audio and emotion information from MELD.

    Text uses frozen RoBERTa-base embeddings by default (768-D, see
    modules/text_features_roberta.py) -- tried after DistilBERT
    (modules/text_features_distilbert.py, still available via
    DISTILBERT_TEXT_PATH) as a stronger pretraining recipe at a similar
    size, in pursuit of matching AMB-DSGDN's RoBERTa-large text encoder as
    closely as CPU-only extraction allows. Still pre-extracted, not
    fine-tuned end-to-end.

    Audio uses frozen Wav2Vec2-base embeddings by default (768-D, see
    modules/audio_features_wav2vec2.py), replacing the original MELD paper's
    300-D openSMILE-style features.

    text_path overrides which text pickle to load -- used to build a
    second, DistilBERT-aligned dataset instance for ensembling with the
    older DistilBERT-trained checkpoint (see evaluation/ensemble_test.py),
    without disturbing the default RoBERTa path every other caller relies on.

    use_legacy_audio=True loads the original 300-D features instead --
    needed to build a dataset instance for ensembling with checkpoints
    trained before the Wav2Vec2 upgrade existed (B, C, D all used the
    300-D audio; only a checkpoint trained after this upgrade needs the
    768-D features).
    """

    with open(text_path or ROBERTA_TEXT_PATH, "rb") as file:
        text_features = pickle.load(file)

    if use_legacy_audio:
        audio_features = load_pickle("audio_emotion.pkl")
    else:
        with open(WAV2VEC2_AUDIO_PATH, "rb") as file:
            audio_features = pickle.load(file)

    emotion_data = load_pickle("data_emotion.p")

    return text_features, audio_features, emotion_data


if __name__ == "__main__":
    text, audio, emotions = load_features()

    print("MELD data loaded successfully!")
    print("Text partitions:", len(text))
    print("Audio partitions:", len(audio))
    print("Emotion data items:", len(emotions))
    print("Emotion mapping:", EMOTION_MAP)