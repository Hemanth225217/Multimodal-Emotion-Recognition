"""Dataset variant for end-to-end text fine-tuning (training/train_finetune.py).

Every other dataset path in this project (training/dataset.py) reads a
frozen, pre-extracted text embedding. Fine-tuning needs the actual sentence
instead, so it can be tokenized and run through a trainable encoder each
step. Audio and video stay frozen (see module docstring in
train_finetune.py for why) -- only text changes.

A separate class rather than adding a flag to MELDDataset: the alignment
logic is identical, but text is fetched from a different source entirely
(raw strings, not an embedding array), and MELDDataset's __getitem__ isn't
structured with hooks to override just that piece cleanly. Kept parallel
and independently simple rather than risk destabilizing the dataset class
every other working part of this project depends on.
"""

import sys
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import (
    load_pickle, load_speaker_lookup, load_sentiment_lookup, load_utterance_lookup,
)
from config import NUM_SPEAKER_SLOTS

VISUAL_ROOT = PROJECT_ROOT / "meld_features" / "visual"


class FinetuneMELDDataset(Dataset):
    """Tri-modal MELD dataset for fine-tuning: raw utterance text (list of
    strings, one per utterance) instead of a frozen text embedding tensor.
    Audio (300-D legacy) and video (512-D) are frozen, same as everywhere
    else in this project.
    """

    def __init__(self, split="train"):
        super().__init__()

        split_index = {"train": 0, "val": 1, "test": 2}
        if split not in split_index:
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.split = split
        self.split_index = split_index[split]

        visual_split = {"train": "train", "val": "dev", "test": "test"}[split]
        visual_path = VISUAL_ROOT / f"{visual_split}_visual.pkl"
        if not visual_path.exists():
            raise FileNotFoundError(f"Visual feature file not found:\n{visual_path}")
        with open(visual_path, "rb") as f:
            visual_data = pickle.load(f)
        self.visual_features = visual_data["features"]

        audio_data = load_pickle("audio_emotion.pkl")
        emotion_data = load_pickle("data_emotion.p")
        self.audio_data = audio_data[self.split_index]

        self.speaker_lookup = load_speaker_lookup(split)
        self.sentiment_lookup = load_sentiment_lookup(split)
        self.utterance_lookup = load_utterance_lookup(split)
        self.sentiment_map = {"neutral": 0, "positive": 1, "negative": 2}
        self.emotion_map = {
            "neutral": 0, "surprise": 1, "fear": 2, "sadness": 3,
            "joy": 4, "disgust": 5, "anger": 6,
        }

        records = emotion_data[0]
        self.labels = {}
        for record in records:
            if record["split"] != split:
                continue
            dialogue_id = str(record["dialog"])
            utterance_id = int(record["utterance"])
            self.labels.setdefault(dialogue_id, {})[utterance_id] = record["y"]

        self.dialogue_ids = sorted(
            (d for d in (str(d) for d in self.audio_data.keys()) if d in self.labels),
            key=lambda x: int(x),
        )

        print(f"{split.upper()} FINE-TUNE DATASET: {len(self.dialogue_ids)} dialogues")

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, index):
        dialogue_id = self.dialogue_ids[index]
        audio = self.audio_data[dialogue_id]
        label_dict = self.labels[dialogue_id]
        sorted_utterance_ids = sorted(label_dict.keys())
        feature_count = len(audio)

        aligned_texts = []
        aligned_audio = []
        aligned_video = []
        aligned_labels = []
        aligned_utterance_ids = []
        aligned_speaker_slots = []
        aligned_sentiments = []
        seen_speakers = {}

        for position, utterance_id in enumerate(sorted_utterance_ids):
            if position >= feature_count:
                break

            visual_key = f"{dialogue_id}_{utterance_id}"
            if visual_key not in self.visual_features:
                continue

            text = self.utterance_lookup.get((int(dialogue_id), utterance_id), "")
            aligned_texts.append(text)
            aligned_audio.append(audio[position])
            aligned_video.append(self.visual_features[visual_key])

            emotion = label_dict[utterance_id]
            if emotion not in self.emotion_map:
                raise ValueError(f"Unknown emotion '{emotion}' in dialogue {dialogue_id}, utterance {utterance_id}")
            aligned_labels.append(self.emotion_map[emotion])
            aligned_utterance_ids.append(utterance_id)

            speaker_name = self.speaker_lookup.get((int(dialogue_id), utterance_id), "UNKNOWN")
            if speaker_name not in seen_speakers:
                seen_speakers[speaker_name] = min(len(seen_speakers), NUM_SPEAKER_SLOTS - 1)
            aligned_speaker_slots.append(seen_speakers[speaker_name])

            sentiment_name = self.sentiment_lookup.get((int(dialogue_id), utterance_id), "neutral")
            aligned_sentiments.append(self.sentiment_map[sentiment_name])

        if not aligned_texts:
            raise RuntimeError(f"No usable tri-modal utterances in dialogue {dialogue_id}")

        return {
            "dialogue_id": dialogue_id,
            "texts": aligned_texts,  # list[str], length U -- tokenized in the training loop
            "audio": torch.tensor(np.array(aligned_audio), dtype=torch.float32),
            "video": torch.tensor(np.array(aligned_video), dtype=torch.float32),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "speaker_slots": torch.tensor(aligned_speaker_slots, dtype=torch.long),
            "sentiment_labels": torch.tensor(aligned_sentiments, dtype=torch.long),
            "utterance_ids": aligned_utterance_ids,
            "length": len(aligned_labels),
        }
