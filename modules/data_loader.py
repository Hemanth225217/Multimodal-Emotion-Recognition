import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the downloaded MELD feature files (original 2018 paper release)
BASE_DIR = PROJECT_ROOT / "meld_features" / "MELD.Features.Models" / "features"

# Frozen DistilBERT text embeddings (modules/text_features_distilbert.py),
# used in place of the original 600-D task-specific CNN text features.
DISTILBERT_TEXT_PATH = PROJECT_ROOT / "meld_features" / "text_distilbert" / "text_distilbert.pkl"


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


def load_original_text_features():
    """The original MELD paper's 600-D task-specific CNN text features.

    Kept available (not deleted) for the report's feature-comparison
    ablation, but load_features() below uses the DistilBERT embeddings.
    """
    return load_pickle("text_emotion.pkl")


def load_features():
    """Load text, audio and emotion information from MELD.

    Text uses frozen DistilBERT embeddings (768-D, see
    modules/text_features_distilbert.py) instead of the original paper's
    600-D task-specific CNN features -- a stronger modern encoder, while
    still pre-extracted rather than fine-tuned end-to-end here.
    """

    with open(DISTILBERT_TEXT_PATH, "rb") as file:
        text_features = pickle.load(file)

    audio_features = load_pickle("audio_emotion.pkl")
    emotion_data = load_pickle("data_emotion.p")

    return text_features, audio_features, emotion_data


if __name__ == "__main__":
    text, audio, emotions = load_features()

    print("MELD data loaded successfully!")
    print("Text partitions:", len(text))
    print("Audio partitions:", len(audio))
    print("Emotion data items:", len(emotions))
    print("Emotion mapping:", EMOTION_MAP)