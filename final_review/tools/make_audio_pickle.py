"""Turn the all-layer audio feature arrays of the Kaggle extraction kernels into the pickle format the datasets read.

Input, per split (train / dev / test): feat_<model>_<split>.npy (float16 [n, layers, hidden]) and index_<split>.json
([[dialogue_id, utterance_id], ...] in the row order of the array). The CSVs give every utterance of every dialogue.

Output: a pickle [train, dev, test] of dialogue_id (str) -> float32 array [n_utterances_in_csv, D], rows in utterance-id
order (the "legacy layout" that position-based alignment in training/dataset.py expects). A clip that could not be decoded
gets a zero row; such clips have no video features either and are skipped by the datasets.

Usage: python make_audio_pickle.py OUT.pkl --model wavlm-large --layers 22,14,21 --feat-dir DIR [--feat-dir DIR2 ...] --csv-dir DIR
The layers are averaged. Feature and index files are found by name anywhere under the --feat-dir folders, so train/dev can
come from one kernel's output and test from another's.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "dev", "test")
CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


def find(name, dirs):
    for d in dirs:
        hits = sorted(Path(d).rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {[str(d) for d in dirs]}")


def build_split(split, model, layers, feat_dirs, csv_dir):
    feats = np.load(find(f"feat_{model}_{split}.npy", feat_dirs), mmap_mode="r")
    index = json.loads(find(f"index_{split}.json", feat_dirs).read_text())
    assert len(index) == feats.shape[0], (split, len(index), feats.shape)
    rows = np.asarray(feats[:, layers, :], dtype=np.float32).mean(axis=1)  # [n, D]
    df = pd.read_csv(Path(csv_dir) / CSV[split]).sort_values(["Dialogue_ID", "Utterance_ID"])
    rank, sizes = {}, {}
    for d, g in df.groupby("Dialogue_ID"):
        utts = sorted(int(u) for u in g["Utterance_ID"])
        sizes[int(d)] = len(utts)
        rank.update({(int(d), u): i for i, u in enumerate(utts)})
    out = {d: np.zeros((n, rows.shape[1]), dtype=np.float32) for d, n in sizes.items()}
    placed = 0
    for (d, u), vec in zip(index, rows):
        out[int(d)][rank[(int(d), int(u))]] = vec
        placed += 1
    total = sum(sizes.values())
    print(f"{split}: {placed} of {total} utterances have audio features ({total - placed} zero rows), D={rows.shape[1]}")
    return {str(d): arr for d, arr in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="wavlm-large")
    ap.add_argument("--layers", default="22,14,21")
    ap.add_argument("--feat-dir", action="append", required=True)
    ap.add_argument("--csv-dir", required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    result = [build_split(sp, args.model, layers, args.feat_dir, args.csv_dir) for sp in SPLITS]
    with open(args.out, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved", args.out)


if __name__ == "__main__":
    main()
