import pickle
from pathlib import Path


# Location of the downloaded MELD feature files
BASE_DIR = (
    Path(__file__).resolve().parent.parent
    / "meld_features"
    / "MELD.Features.Models"
    / "features"
)


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
    """Load a MELD pickle feature file."""
    file_path = BASE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find feature file: {file_path}"
        )

    with open(file_path, "rb") as file:
        return pickle.load(file, encoding="latin1")


def load_features():
    """Load text, audio and emotion information from MELD."""

    text_features = load_pickle("text_emotion.pkl")
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