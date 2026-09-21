import os
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MELD_DATA = os.path.join(
    PROJECT_ROOT,
    "meld_dataset",
    "data",
    "MELD"
)

VIDEO_ROOT = os.path.join(
    PROJECT_ROOT,
    "meld_dataset",
    "raw"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "meld_features",
    "visual"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

NUM_FRAMES = 8

FEATURE_DIM = 512


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

print()
print("=" * 70)
print("LOADING VISUAL MODEL")
print("=" * 70)

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

# Remove ImageNet classification layer.
# ResNet-18 now outputs a 512-dimensional visual representation.
model.fc = nn.Identity()

model = model.to(DEVICE)

model.eval()

preprocess = weights.transforms()

print("Visual model: ResNet-18")
print("Device:", DEVICE)
print("Feature dimension:", FEATURE_DIM)
print("Frames per video:", NUM_FRAMES)
print("Visual model ready.")


# ============================================================
# DATASET SPLITS
# ============================================================

SPLIT_CONFIG = {

    "train": {
        "csv": "train_sent_emo.csv",
        "folder": "train_splits"
    },

    "dev": {
        "csv": "dev_sent_emo.csv",
        "folder": "dev_splits_complete"
    },

    "test": {
        "csv": "test_sent_emo.csv",
        "folder": "output_repeated_splits_test"
    }
}


# ============================================================
# FIND VIDEO
# ============================================================

def find_video(folder, dialogue_id, utterance_id):

    filename = (
        f"dia{int(dialogue_id)}_utt{int(utterance_id)}.mp4"
    )

    path = os.path.join(
        VIDEO_ROOT,
        folder,
        filename
    )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

def sample_frames(video_path, num_frames=NUM_FRAMES):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return []

    # Select frames evenly across the complete video.
    positions = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for position in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position)
        )

        success, frame = cap.read()

        if not success:
            continue

        # OpenCV: BGR
        # PIL/torchvision: RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# EXTRACT ONE VIDEO FEATURE
# ============================================================

@torch.no_grad()
def extract_visual_feature(video_path):

    frames = sample_frames(
        video_path,
        NUM_FRAMES
    )

    if len(frames) == 0:
        return None

    tensors = []

    for frame in frames:

        # Convert NumPy array → PIL Image.
        image = Image.fromarray(frame)

        # Apply ResNet preprocessing.
        tensor = preprocess(image)

        tensors.append(tensor)

    # [number_of_frames, 3, 224, 224]
    batch = torch.stack(tensors).to(DEVICE)

    # [number_of_frames, 512]
    features = model(batch)

    # Average frame representations.
    feature = features.mean(dim=0)

    # Convert to NumPy.
    feature = feature.cpu().numpy().astype(
        np.float32
    )

    return feature


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(split_name, config):

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("=" * 70)

    csv_path = os.path.join(
        MELD_DATA,
        config["csv"]
    )

    df = pd.read_csv(csv_path)

    print("Annotations:", len(df))

    features = {}
    labels = {}
    metadata = {}

    missing_videos = []
    failed_videos = []

    total = len(df)

    for index, row in df.iterrows():

        dialogue_id = int(row["Dialogue_ID"])

        utterance_id = int(row["Utterance_ID"])

        key = f"{dialogue_id}_{utterance_id}"

        # ----------------------------------------------------
        # Find corresponding video.
        # ----------------------------------------------------

        video_path = find_video(
            config["folder"],
            dialogue_id,
            utterance_id
        )

        if video_path is None:

            missing_videos.append(key)

            continue

        # ----------------------------------------------------
        # Extract visual representation.
        # ----------------------------------------------------

        feature = extract_visual_feature(
            video_path
        )

        if feature is None:

            failed_videos.append(key)

            continue

        # ----------------------------------------------------
        # Store feature.
        # ----------------------------------------------------

        features[key] = feature

        labels[key] = row["Emotion"]

        metadata[key] = {

            "Dialogue_ID": dialogue_id,

            "Utterance_ID": utterance_id,

            "Speaker": row["Speaker"],

            "Emotion": row["Emotion"],

            "Utterance": row["Utterance"]
        }

        # ----------------------------------------------------
        # Progress.
        # ----------------------------------------------------

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{total} "
                f"| features={len(features)} "
                f"| missing={len(missing_videos)} "
                f"| failed={len(failed_videos)}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "features": features,

        "labels": labels,

        "metadata": metadata,

        "feature_dim": FEATURE_DIM,

        "num_frames": NUM_FRAMES,

        "missing_videos": missing_videos,

        "failed_videos": failed_videos
    }

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{split_name}_visual.pkl"
    )

    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            output,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print()
    print(f"{split_name.upper()} COMPLETE")
    print("-" * 50)
    print("Annotations:", len(df))
    print("Features:", len(features))
    print("Missing videos:", len(missing_videos))
    print("Failed videos:", len(failed_videos))
    print("Saved:", output_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("MELD VISUAL FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("This script will process train/dev/test only.")
    print("It does NOT modify or delete your videos.")
    print()

    for split_name, config in SPLIT_CONFIG.items():

        process_split(
            split_name,
            config
        )

    print()
    print("=" * 70)
    print("ALL VISUAL FEATURE EXTRACTION COMPLETE")
    print("=" * 70)