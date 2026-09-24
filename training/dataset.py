import sys
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from modules.data_loader import load_features, load_speaker_lookup, load_sentiment_lookup
from config import NUM_SPEAKER_SLOTS


# ============================================================
# PATHS
# ============================================================

VISUAL_ROOT = (
    PROJECT_ROOT
    / "meld_features"
    / "visual"
)


# ============================================================
# MELD DATASET
# ============================================================

class MELDDataset(Dataset):
    """
    Tri-modal MELD dataset.

    Each sample represents one complete MELD dialogue.

    Text:
        [utterances, 768]

    Audio:
        [utterances, 300]

    Video:
        [utterances, 512]

    Labels:
        [utterances]

    Important:
        The original MELD text/audio feature files contain
        padding up to 33 utterances per dialogue.

        We therefore use the actual MELD emotion records to
        determine which utterances are real.

        Visual features are aligned using:
            Dialogue_ID + Utterance_ID

        Missing individual visual utterances are removed from
        ALL modalities so that text/audio/video/labels remain
        perfectly aligned.
    """

    def __init__(self, split="train", text_path=None, use_legacy_audio=False):

        super().__init__()

        # ====================================================
        # LOAD TEXT, AUDIO AND EMOTION DATA
        #
        # text_path overrides the default RoBERTa features -- used to build
        # a DistilBERT-aligned instance for ensembling with the older
        # DistilBERT-trained checkpoint (see modules.data_loader.load_features
        # and evaluation/ensemble_test.py). use_legacy_audio=True loads the
        # original 300-D audio instead of the Wav2Vec2 768-D features, for
        # ensembling with checkpoints trained before that upgrade. Video/
        # labels/speaker/sentiment alignment is identical in every case.
        # ====================================================

        text_data, audio_data, emotion_data = load_features(text_path, use_legacy_audio)

        # ====================================================
        # SPLIT MAPPING
        # ====================================================

        split_index = {
            "train": 0,
            "val": 1,
            "test": 2
        }

        if split not in split_index:

            raise ValueError(
                "split must be 'train', 'val', or 'test'"
            )

        self.split = split

        self.split_index = split_index[split]

        # ====================================================
        # VISUAL FILE MAPPING
        #
        # Dataset calls validation "val".
        # Visual extraction calls it "dev".
        # ====================================================

        visual_split = {
            "train": "train",
            "val": "dev",
            "test": "test"
        }[split]

        visual_path = (
            VISUAL_ROOT
            / f"{visual_split}_visual.pkl"
        )

        if not visual_path.exists():

            raise FileNotFoundError(
                f"Visual feature file not found:\n"
                f"{visual_path}"
            )

        with open(
            visual_path,
            "rb"
        ) as f:

            visual_data = pickle.load(f)

        self.visual_features = visual_data["features"]

        self.visual_missing = set(
            visual_data.get(
                "missing_videos",
                []
            )
        )

        self.visual_failed = set(
            visual_data.get(
                "failed_videos",
                []
            )
        )

        # ====================================================
        # FEATURES FOR CURRENT SPLIT
        # ====================================================

        self.text_data = text_data[
            self.split_index
        ]

        self.audio_data = audio_data[
            self.split_index
        ]

        # ====================================================
        # SPEAKER LOOKUP (for dialogue-relative speaker slots)
        # ====================================================

        self.speaker_lookup = load_speaker_lookup(split)

        # ====================================================
        # SENTIMENT LOOKUP (auxiliary multi-task loss)
        # ====================================================

        self.sentiment_lookup = load_sentiment_lookup(split)

        self.sentiment_map = {
            "neutral": 0,
            "positive": 1,
            "negative": 2,
        }

        # ====================================================
        # EMOTION MAPPING
        # ====================================================

        self.emotion_map = {

            "neutral": 0,

            "surprise": 1,

            "fear": 2,

            "sadness": 3,

            "joy": 4,

            "disgust": 5,

            "anger": 6
        }

        # ====================================================
        # BUILD LABEL DICTIONARY
        # ====================================================

        records = emotion_data[0]

        self.labels = {}

        for record in records:

            if record["split"] != split:
                continue

            dialogue_id = str(
                record["dialog"]
            )

            utterance_id = int(
                record["utterance"]
            )

            emotion = record["y"]

            if dialogue_id not in self.labels:

                self.labels[
                    dialogue_id
                ] = {}

            self.labels[
                dialogue_id
            ][utterance_id] = emotion

        # ====================================================
        # KEEP ONLY DIALOGUES AVAILABLE IN:
        #
        # 1. Text
        # 2. Audio
        # 3. Emotion labels
        # 4. Visual features
        # ====================================================

        self.dialogue_ids = []

        for dialogue_id in self.text_data.keys():

            dialogue_id = str(
                dialogue_id
            )

            if dialogue_id not in self.audio_data:
                continue

            if dialogue_id not in self.labels:
                continue

            self.dialogue_ids.append(
                dialogue_id
            )

        # ====================================================
        # SORT DIALOGUES
        # ====================================================

        self.dialogue_ids.sort(
            key=lambda x: int(x)
        )

        # ====================================================
        # ALIGNMENT STATISTICS
        # ====================================================

        self.total_utterances = 0

        self.missing_feature_dialogues = 0

        self.truncated_dialogues = 0

        self.missing_visual_utterances = 0

        self.usable_dialogue_count = 0

        # ====================================================
        # VERIFY DIALOGUES
        # ====================================================

        for dialogue_id in self.dialogue_ids:

            label_dict = self.labels[
                dialogue_id
            ]

            sorted_utterance_ids = sorted(
                label_dict.keys()
            )

            label_count = len(
                sorted_utterance_ids
            )

            text_count = len(
                self.text_data[
                    dialogue_id
                ]
            )

            audio_count = len(
                self.audio_data[
                    dialogue_id
                ]
            )

            feature_count = min(
                text_count,
                audio_count
            )

            if feature_count < label_count:

                self.missing_feature_dialogues += 1

            if feature_count > label_count:

                self.truncated_dialogues += 1

            # ------------------------------------------------
            # Check visual availability per utterance.
            # ------------------------------------------------

            usable_count = 0

            for position, utterance_id in enumerate(
                sorted_utterance_ids
            ):

                if position >= feature_count:
                    break

                visual_key = (
                    f"{dialogue_id}_{utterance_id}"
                )

                if visual_key in self.visual_features:

                    usable_count += 1

                else:

                    self.missing_visual_utterances += 1

            if usable_count > 0:

                self.usable_dialogue_count += 1

                self.total_utterances += usable_count

        # ====================================================
        # PRINT DATASET INFORMATION
        # ====================================================

        print()
        print("=" * 70)
        print(
            f"{split.upper()} TRI-MODAL DATASET"
        )
        print("=" * 70)

        print(
            f"{split.upper()} dialogues: "
            f"{len(self.dialogue_ids)}"
        )

        print(
            f"{split.upper()} usable utterances: "
            f"{self.total_utterances}"
        )

        print(
            f"{split.upper()} feature vectors before alignment: "
            f"{sum(len(self.text_data[d]) for d in self.dialogue_ids)}"
        )

        print(
            f"{split.upper()} dialogues requiring feature truncation: "
            f"{self.truncated_dialogues}"
        )

        print(
            f"{split.upper()} dialogues with insufficient features: "
            f"{self.missing_feature_dialogues}"
        )

        print(
            f"{split.upper()} missing visual utterances: "
            f"{self.missing_visual_utterances}"
        )

        print(
            f"{split.upper()} visual features loaded: "
            f"{len(self.visual_features)}"
        )

        print(
            f"{split.upper()} visual feature dimension: 512"
        )

        print("=" * 70)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.dialogue_ids
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        dialogue_id = self.dialogue_ids[
            index
        ]

        # ====================================================
        # GET RAW FEATURES
        # ====================================================

        text = self.text_data[
            dialogue_id
        ]

        audio = self.audio_data[
            dialogue_id
        ]

        label_dict = self.labels[
            dialogue_id
        ]

        # ====================================================
        # SORT ACTUAL UTTERANCE IDS
        # ====================================================

        sorted_utterance_ids = sorted(
            label_dict.keys()
        )

        label_count = len(
            sorted_utterance_ids
        )

        # ====================================================
        # NUMBER OF AVAILABLE TEXT/AUDIO FEATURES
        # ====================================================

        feature_count = min(
            len(text),
            len(audio)
        )

        # ====================================================
        # BUILD ALIGNED TRI-MODAL ARRAYS
        # ====================================================

        aligned_text = []

        aligned_audio = []

        aligned_video = []

        aligned_labels = []

        aligned_utterance_ids = []

        aligned_speaker_slots = []

        aligned_sentiments = []

        # Dialogue-relative: the first distinct speaker encountered in this
        # dialogue gets slot 0, the second gets slot 1, and so on -- see
        # config.NUM_SPEAKER_SLOTS and modules/data_loader.load_speaker_lookup.
        seen_speakers = {}

        for position, utterance_id in enumerate(
            sorted_utterance_ids
        ):

            # ------------------------------------------------
            # Never exceed available text/audio features.
            # ------------------------------------------------

            if position >= feature_count:
                break

            # ------------------------------------------------
            # Visual feature key.
            # ------------------------------------------------

            visual_key = (
                f"{dialogue_id}_{utterance_id}"
            )

            # ------------------------------------------------
            # If visual feature is unavailable, skip the
            # SAME utterance from every modality.
            # ------------------------------------------------

            if visual_key not in self.visual_features:

                continue

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            aligned_text.append(
                text[position]
            )

            # ------------------------------------------------
            # Audio
            # ------------------------------------------------

            aligned_audio.append(
                audio[position]
            )

            # ------------------------------------------------
            # Video
            # ------------------------------------------------

            aligned_video.append(
                self.visual_features[
                    visual_key
                ]
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            emotion = label_dict[
                utterance_id
            ]

            if emotion not in self.emotion_map:

                raise ValueError(
                    f"Unknown emotion '{emotion}' "
                    f"in dialogue {dialogue_id}, "
                    f"utterance {utterance_id}"
                )

            aligned_labels.append(
                self.emotion_map[
                    emotion
                ]
            )

            aligned_utterance_ids.append(
                utterance_id
            )

            # ------------------------------------------------
            # Speaker slot (dialogue-relative, see __init__)
            # ------------------------------------------------

            speaker_name = self.speaker_lookup.get(
                (int(dialogue_id), utterance_id),
                "UNKNOWN"
            )

            if speaker_name not in seen_speakers:

                seen_speakers[speaker_name] = min(
                    len(seen_speakers),
                    NUM_SPEAKER_SLOTS - 1
                )

            aligned_speaker_slots.append(
                seen_speakers[speaker_name]
            )

            # ------------------------------------------------
            # Sentiment (auxiliary multi-task loss, see __init__)
            # ------------------------------------------------

            sentiment_name = self.sentiment_lookup.get(
                (int(dialogue_id), utterance_id),
                "neutral"
            )

            aligned_sentiments.append(
                self.sentiment_map[sentiment_name]
            )

        # ====================================================
        # CONVERT TO TENSORS
        # ====================================================

        if len(aligned_text) == 0:

            raise RuntimeError(
                f"No usable tri-modal utterances "
                f"in dialogue {dialogue_id}"
            )

        text = torch.tensor(
            np.array(aligned_text),
            dtype=torch.float32
        )

        audio = torch.tensor(
            np.array(aligned_audio),
            dtype=torch.float32
        )

        video = torch.tensor(
            np.array(aligned_video),
            dtype=torch.float32
        )

        labels = torch.tensor(
            aligned_labels,
            dtype=torch.long
        )

        speaker_slots = torch.tensor(
            aligned_speaker_slots,
            dtype=torch.long
        )

        sentiment_labels = torch.tensor(
            aligned_sentiments,
            dtype=torch.long
        )

        # ====================================================
        # FINAL SAFETY CHECKS
        # ====================================================

        if len(text) != len(audio):

            raise RuntimeError(
                f"Text/audio alignment error in "
                f"dialogue {dialogue_id}: "
                f"text={len(text)}, "
                f"audio={len(audio)}"
            )

        if len(text) != len(video):

            raise RuntimeError(
                f"Text/video alignment error in "
                f"dialogue {dialogue_id}: "
                f"text={len(text)}, "
                f"video={len(video)}"
            )

        if len(text) != len(labels):

            raise RuntimeError(
                f"Text/label alignment error in "
                f"dialogue {dialogue_id}: "
                f"text={len(text)}, "
                f"labels={len(labels)}"
            )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "dialogue_id": dialogue_id,

            "text": text,

            "audio": audio,

            "video": video,

            "labels": labels,

            "speaker_slots": speaker_slots,

            "sentiment_labels": sentiment_labels,

            "utterance_ids": aligned_utterance_ids,

            "length": len(labels)
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TESTING TRI-MODAL MELD DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()

    train_dataset = MELDDataset(
        "train"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()

    val_dataset = MELDDataset(
        "val"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()

    test_dataset = MELDDataset(
        "test"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRI-MODAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Train dialogues      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Train utterances     : "
        f"{train_dataset.total_utterances}"
    )

    print(
        f"Validation dialogues : "
        f"{len(val_dataset)}"
    )

    print(
        f"Validation utterances: "
        f"{val_dataset.total_utterances}"
    )

    print(
        f"Test dialogues       : "
        f"{len(test_dataset)}"
    )

    print(
        f"Test utterances      : "
        f"{test_dataset.total_utterances}"
    )

    # ========================================================
    # TEST FIRST TEST DIALOGUE
    # ========================================================

    sample = test_dataset[0]

    print()
    print("=" * 70)
    print("FIRST TEST DIALOGUE CHECK")
    print("=" * 70)

    print(
        f"Dialogue ID : "
        f"{sample['dialogue_id']}"
    )

    print(
        f"Text shape  : "
        f"{sample['text'].shape}"
    )

    print(
        f"Audio shape : "
        f"{sample['audio'].shape}"
    )

    print(
        f"Video shape : "
        f"{sample['video'].shape}"
    )

    print(
        f"Labels shape: "
        f"{sample['labels'].shape}"
    )

    print(
        f"Length      : "
        f"{sample['length']}"
    )

    print(
        f"Utterance IDs: "
        f"{sample['utterance_ids']}"
    )

    print(
        f"Labels      : "
        f"{sample['labels'].tolist()}"
    )

    print()
    print("=" * 70)
    print("TRI-MODAL DATASET TEST COMPLETE")
    print("=" * 70)