"""Write MC-EIU (English) features into exactly the files and layouts the MELD pipeline reads, under a given project root,
so training/dataset.py, train_final.py, cache_member_probs.py and the analysis scripts run on MC-EIU unchanged.

Input: the output folder of the Kaggle kernel `mc-eiu-extract` (utterances.csv, text_roberta.npy,
wavlm_large_all_layers.npy, video_resnet18.npy, has_audio.npy, has_video.npy).
Output under ROOT (normally a fresh clone of the repository; it must not contain MELD data):
  meld_features/MELD.Features.Models/features/data_emotion.p   [records, None, {}, {}, 0, emotion_map]; records use
                                                                 MELD's keys and split names (train / val / test)
  meld_features/MELD.Features.Models/features/audio_emotion.pkl the WavLM pickle again, so no code path can silently
                                                                 pick up MELD's 300-D audio
  meld_features/text_roberta/text_roberta.pkl                    [train, dev, test] dict dialogue -> list of 768-D
  meld_features/visual/{train,dev,test}_visual.pkl               {"features": {"<dia>_<utt>": 512-D}, ...}
  meld_dataset/data/MELD/{train,dev,test}_sent_emo.csv           MELD columns; Speaker = "0"/"1"; Sentiment = "neutral"
                                                                 (MC-EIU has no sentiment labels; the sentiment loss
                                                                 weight is 0 in train_final.py)
  AUDIO_OUT (argument)                                           legacy-layout audio pickle for TRAIN_AUDIO_PATH
Utterances without decodable video are left out of the visual features, so the datasets skip them, exactly as for
MELD's two missing clips; utterances without audio get a zero audio row.

Usage: python mceiu_to_meld_layout.py EXTRACT_DIR ROOT AUDIO_OUT [--layers 22]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

EMOTION_RENAME = {"happy": "joy", "sad": "sadness"}  # MC-EIU names -> MELD names; the other five are identical
MELD_EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
SPLITS = [("train", "train", "train_sent_emo.csv", "train_visual.pkl"),
          ("dev", "val", "dev_sent_emo.csv", "dev_visual.pkl"),
          ("test", "test", "test_sent_emo.csv", "test_visual.pkl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extract_dir")
    ap.add_argument("root")
    ap.add_argument("audio_out")
    ap.add_argument("--layers", default="22", help="WavLM layers to average (comma-separated)")
    args = ap.parse_args()
    src, root = Path(args.extract_dir), Path(args.root)
    layers = [int(x) for x in args.layers.split(",")]

    df = pd.read_csv(src / "utterances.csv")
    df["emotion"] = df["emotion"].str.lower().map(lambda e: EMOTION_RENAME.get(e, e))
    assert set(df["emotion"]) <= set(MELD_EMOTIONS), set(df["emotion"]) - set(MELD_EMOTIONS)
    n = len(df)
    text = np.load(src / "text_roberta.npy", mmap_mode="r")
    wavlm = np.load(src / "wavlm_large_all_layers.npy", mmap_mode="r")
    video = np.load(src / "video_resnet18.npy", mmap_mode="r")
    has_a = np.load(src / "has_audio.npy")
    has_v = np.load(src / "has_video.npy")
    assert text.shape[0] == wavlm.shape[0] == video.shape[0] == len(has_a) == len(has_v) == n
    assert (df.sort_values(["Dia_No", "Utt_No"]).index == df.index).all(), "utterances.csv must be sorted"
    audio = np.zeros((n, wavlm.shape[2]), dtype=np.float32)
    for s in range(0, n, 2000):
        audio[s:s + 2000] = np.asarray(wavlm[s:s + 2000][:, layers, :], dtype=np.float32).mean(1)
    audio[~has_a] = 0.0

    feat_dir = root / "meld_features" / "MELD.Features.Models" / "features"
    csv_dir = root / "meld_dataset" / "data" / "MELD"
    vis_dir = root / "meld_features" / "visual"
    txt_dir = root / "meld_features" / "text_roberta"
    for d in (feat_dir, csv_dir, vis_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    records, text_splits, audio_splits = [], [], []
    for split, rec_split, csv_name, vis_name in SPLITS:
        part = df[df["split"] == split]
        t_dict, a_dict, feats, labels, meta, missing = {}, {}, {}, {}, {}, []
        for dia, g in part.groupby("Dia_No", sort=True):
            g = g.sort_values("Utt_No")
            key = str(int(dia))
            t_dict[key] = [np.asarray(text[i], dtype=np.float32) for i in g.index]
            a_dict[key] = audio[g.index]
            for i, row in g.iterrows():
                vk = f"{int(dia)}_{int(row.Utt_No)}"
                if has_v[i]:
                    feats[vk] = np.asarray(video[i], dtype=np.float32)
                else:
                    missing.append(vk)
                labels[vk] = row.emotion
                meta[vk] = {"Dialogue_ID": int(dia), "Utterance_ID": int(row.Utt_No), "Speaker": str(row.speaker),
                            "Emotion": row.emotion, "Utterance": str(row.Subtitle)}
                records.append({"y": row.emotion, "dialog": key, "utterance": str(int(row.Utt_No)),
                                "text": str(row.Subtitle), "num_words": len(str(row.Subtitle).split()), "split": rec_split})
        text_splits.append(t_dict)
        audio_splits.append(a_dict)
        with open(vis_dir / vis_name, "wb") as f:
            pickle.dump({"features": feats, "labels": labels, "metadata": meta, "feature_dim": 512, "num_frames": 8,
                         "missing_videos": missing, "failed_videos": []}, f, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame({
            "Sr No.": range(1, len(part) + 1), "Utterance": part["Subtitle"].fillna("").astype(str).values,
            "Speaker": part["speaker"].astype(str).values, "Emotion": part["emotion"].values,
            "Sentiment": "neutral", "Dialogue_ID": part["Dia_No"].values, "Utterance_ID": part["Utt_No"].values,
            "Season": part["Season"].values, "Episode": part["Episode"].values,
        }).to_csv(csv_dir / csv_name, index=False)
        print(f"{split}: {part['Dia_No'].nunique()} dialogues, {len(part)} utterances, video {len(feats)}, "
              f"audio {int(has_a[part.index].sum())}, missing video {len(missing)}")

    emotion_map = {e: i for i, e in enumerate(MELD_EMOTIONS)}
    with open(feat_dir / "data_emotion.p", "wb") as f:
        pickle.dump([records, None, {}, {}, 0, emotion_map], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(txt_dir / "text_roberta.pkl", "wb") as f:
        pickle.dump(text_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    for path in (Path(args.audio_out), feat_dir / "audio_emotion.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(audio_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"written under {root}; audio pickle {args.audio_out} (WavLM layers {layers}, D={audio.shape[1]})")


if __name__ == "__main__":
    main()
