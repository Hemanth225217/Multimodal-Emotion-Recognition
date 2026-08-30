import sys
from pathlib import Path

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

from modules.data_loader import load_features


# ============================================================
# MELD DATASET
# ============================================================

class MELDDataset(Dataset):
    """
    Dataset for multimodal conversational emotion recognition.

    Each sample represents one complete MELD dialogue.

    Text:
        [utterances, 600]

    Audio:
        [utterances, 300]

    Labels:
        [utterances]

    IMPORTANT:
    The number of utterances is determined from the actual
    MELD emotion records, NOT from the length of the feature
    arrays.

    This prevents padded/extra feature vectors from being
    incorrectly treated as real utterances.
    """

    def __init__(self, split="train"):

        super().__init__()

        # ====================================================
        # LOAD RAW MELD FEATURES
        # ====================================================

        text_data, audio_data, emotion_data = load_features()

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

        self.split_index = split_index[
            split
        ]

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
        # VERIFY ALIGNMENT
        # ====================================================

        self.total_utterances = 0

        self.missing_feature_dialogues = 0

        self.truncated_dialogues = 0

        for dialogue_id in self.dialogue_ids:

            label_count = len(
                self.labels[
                    dialogue_id
                ]
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

            usable_count = min(
                label_count,
                feature_count
            )

            self.total_utterances += (
                usable_count
            )

        # ====================================================
        # PRINT DATASET INFORMATION
        # ====================================================

        print(
            f"{split.upper()} dataset: "
            f"{len(self.dialogue_ids)} dialogues"
        )

        print(
            f"{split.upper()} actual utterances: "
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
        # GET FEATURES
        # ====================================================

        text = self.text_data[
            dialogue_id
        ]

        audio = self.audio_data[
            dialogue_id
        ]

        # ====================================================
        # CONVERT TO TENSORS
        # ====================================================

        text = torch.tensor(
            text,
            dtype=torch.float32
        )

        audio = torch.tensor(
            audio,
            dtype=torch.float32
        )

        # ====================================================
        # GET ACTUAL LABELS
        # ====================================================

        label_dict = self.labels[
            dialogue_id
        ]

        # ----------------------------------------------------
        # Sort labels according to utterance ID
        # ----------------------------------------------------

        sorted_utterance_ids = sorted(
            label_dict.keys()
        )

        # Number of actual labeled utterances
        label_count = len(
            sorted_utterance_ids
        )

        # Number of available feature vectors
        feature_count = min(
            len(text),
            len(audio)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use only the number of ACTUAL MELD utterances.
        #
        # Never create missing labels as Neutral.
        # ----------------------------------------------------

        num_utterances = min(
            label_count,
            feature_count
        )

        text = text[
            :num_utterances
        ]

        audio = audio[
            :num_utterances
        ]

        # ====================================================
        # CREATE LABEL TENSOR
        # ====================================================

        labels = []

        for utterance_id in sorted_utterance_ids[
            :num_utterances
        ]:

            emotion = label_dict[
                utterance_id
            ]

            if emotion not in self.emotion_map:

                raise ValueError(
                    f"Unknown emotion '{emotion}' "
                    f"in dialogue {dialogue_id}, "
                    f"utterance {utterance_id}"
                )

            labels.append(
                self.emotion_map[
                    emotion
                ]
            )

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        # ====================================================
        # FINAL SAFETY CHECK
        # ====================================================

        if len(text) != len(labels):

            raise RuntimeError(
                f"Alignment error in dialogue "
                f"{dialogue_id}: "
                f"text={len(text)}, "
                f"labels={len(labels)}"
            )

        if len(audio) != len(labels):

            raise RuntimeError(
                f"Alignment error in dialogue "
                f"{dialogue_id}: "
                f"audio={len(audio)}, "
                f"labels={len(labels)}"
            )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "dialogue_id": dialogue_id,
            "text": text,
            "audio": audio,
            "labels": labels,
            "length": num_utterances
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TESTING CORRECTED MELD DATASET")
    print("=" * 70)

    print()

    train_dataset = MELDDataset(
        "train"
    )

    print()

    val_dataset = MELDDataset(
        "val"
    )

    print()

    test_dataset = MELDDataset(
        "test"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
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
        f"Labels shape: "
        f"{sample['labels'].shape}"
    )

    print(
        f"Length      : "
        f"{sample['length']}"
    )

    print(
        f"Labels      : "
        f"{sample['labels'].tolist()}"
    )

    print()
    print("=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)