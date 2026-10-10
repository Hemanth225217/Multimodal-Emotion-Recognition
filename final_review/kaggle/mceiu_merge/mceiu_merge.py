"""Kaggle kernel (CPU, no GPU quota, no token): join the outputs of mc-eiu-extract-a1 / -a2 / -a3 (one archive each)
into the files the training and analysis kernels read: utterances.csv, text_roberta.npy, wavlm_large_all_layers.npy,
video_resnet18.npy, has_audio.npy, has_video.npy, split.json, report.json (same names and layouts as the all-in-one
extraction). Checks that the three runs used the same table and split and that no utterance appears twice.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bases = sorted(INPUT.rglob("utterances_base.csv"))
    print("tables:", [str(b) for b in bases], flush=True)
    assert len(bases) == 3, bases
    dfs = [pd.read_csv(b) for b in bases]
    for d in dfs[1:]:
        assert d.equals(dfs[0]), "the three runs disagree on the utterance table or split"
    df = dfs[0]
    n = len(df)
    texts = sorted(INPUT.rglob("text_roberta.npy"))
    assert len(texts) == 1, texts
    T = np.load(texts[0])
    assert T.shape == (n, 768)
    parts = sorted(INPUT.rglob("part*.npz"))
    print("parts:", [str(p) for p in parts], flush=True)
    assert sorted(p.name for p in parts) == ["part1.npz", "part2.npz", "part3.npz"], parts
    aud = np.lib.format.open_memmap(OUT / "wavlm_large_all_layers.npy", mode="w+", dtype=np.float16, shape=(n, 25, 1024))
    vid = np.zeros((n, 512), np.float32)
    ha = np.zeros(n, bool)
    hv = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    report = {"parts": {}}
    for p in parts:
        z = np.load(p)
        rows = z["rows"]
        assert not seen[rows].any(), f"{p.name}: utterances already filled by another part"
        seen[rows] = True
        aud[rows] = z["audio"]; vid[rows] = z["video"]; ha[rows] = z["has_audio"]; hv[rows] = z["has_video"]
        report["parts"][p.name] = {"rows": int(len(rows)), "audio_ok": int(z["has_audio"].sum()),
                                   "video_ok": int(z["has_video"].sum())}
    aud.flush()
    np.save(OUT / "text_roberta.npy", T)
    np.save(OUT / "video_resnet18.npy", vid)
    np.save(OUT / "has_audio.npy", ha)
    np.save(OUT / "has_video.npy", hv)
    df["has_audio"], df["has_video"] = ha, hv
    df.to_csv(OUT / "utterances.csv", index=False)
    split = sorted(INPUT.rglob("split.json"))[0]
    shutil.copy(split, OUT / "split.json")
    report["utterances"] = n
    report["no_clip_found"] = int((~seen).sum())
    report["coverage"] = {sp: {"utterances": int((df.split == sp).sum()), "audio": int(df[df.split == sp].has_audio.sum()),
                               "video": int(df[df.split == sp].has_video.sum())} for sp in ("train", "dev", "test")}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
