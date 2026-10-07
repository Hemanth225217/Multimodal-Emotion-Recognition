"""Kaggle kernel for final-review work package W1 (part 2): extract the audio features of the TEST split.

The first kernel (meld-audio-probe) decoded only 1,595 of the 2,610 test clips: when a video name exists in several folders of
the public dataset it kept the first copy it found, and for the test split that copy often could not be decoded. Train and
validation were unaffected (9,988 and 1,108 usable clips, the same counts as the local pipeline), so the probe results hold.

This kernel keeps EVERY candidate path of each clip, tries the canonical folder first (`output_repeated_splits_test`, the one
the local pipeline used) and falls back to the other copies, and logs which copy worked and why the failures failed. It then
extracts the all-layer mean-pooled hidden states with exactly the same code as the probe kernel, so the arrays can be joined to
the train / validation arrays of `meld-audio-probe`.

Outputs (in /kaggle/working): feat_<model>_test.npy (float16 [n, layers, hidden]), index_test.json, labels_test.json,
decode_report.json.
Environment variables (optional): INPUT_ROOT, FEATURES_ROOT, OUT_DIR, LIMIT, MODELS, SPLITS, DEVICE.
"""
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
FEATURES_ROOT = Path(os.environ.get("FEATURES_ROOT", str(INPUT_ROOT)))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SPLITS = [s for s in os.environ.get("SPLITS", "test").split(",") if s]
MODELS = [m.strip() for m in os.environ.get("MODELS", "microsoft/wavlm-large,facebook/hubert-large-ll60k").split(",") if m.strip()]
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 15 * SAMPLE_RATE
EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMO_ID = {e: i for i, e in enumerate(EMOTIONS)}
CSV_NAMES = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
CANONICAL = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split_of(rel_path):
    parts = [p.lower() for p in rel_path.parts]
    if any("dev" in p for p in parts):
        return "dev"
    if any("test" in p for p in parts):
        return "test"
    if any("train" in p for p in parts):
        return "train"
    return None


def index_videos():
    """(split, dialogue, utterance) -> every candidate path, canonical folder first."""
    pat = re.compile(r"dia(\d+)_utt(\d+)\.mp4$", re.I)
    idx = {}
    for root in sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir()):
        for p in root.rglob("*.mp4"):
            m = pat.search(p.name)
            sp = split_of(p.relative_to(root)) if m else None
            if sp is None:
                continue
            idx.setdefault((sp, int(m.group(1)), int(m.group(2))), []).append(p)
    for key, paths in idx.items():
        canon = CANONICAL[key[0]]
        paths.sort(key=lambda p: (canon not in str(p).replace("\\", "/"), str(p)))
    multi = sum(1 for v in idx.values() if len(v) > 1)
    log(f"indexed {len(idx)} clips, {multi} with more than one copy; per split:",
        {s: sum(1 for k in idx if k[0] == s) for s in ("train", "dev", "test")})
    folders = Counter(str(v[0].parent.relative_to(INPUT_ROOT)) for v in idx.values())
    log("first-choice folders:", dict(folders.most_common(8)))
    return idx


def find_file(name, roots):
    for root in roots:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def load_split_table(split):
    csv = find_file(CSV_NAMES[split], [INPUT_ROOT, FEATURES_ROOT])
    df = pd.read_csv(csv).sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    rows = [(int(r.Dialogue_ID), int(r.Utterance_ID), EMO_ID[str(r.Emotion).lower()]) for r in df.itertuples()]
    return rows[:LIMIT] if LIMIT else rows


def decode_one(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE),
           "-ac", "1", "-f", "wav", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None, (r.stderr.decode("utf8", "replace").strip()[:200] or f"empty output, rc={r.returncode}")
    try:
        audio, _ = sf.read(io.BytesIO(r.stdout), dtype="float32")
    except Exception as exc:  # truncated wav header etc.
        return None, f"soundfile: {exc!r}"[:200]
    if audio.ndim > 1:
        audio = audio.mean(1)
    if audio.size == 0:
        return None, "no audio samples"
    return audio[:MAX_SAMPLES], None


def decode_any(paths):
    """Try each copy in turn; returns (audio or None, index of the copy that worked, list of errors)."""
    errors = []
    for i, p in enumerate(paths):
        audio, err = decode_one(p)
        if audio is not None:
            return audio, i, errors
        errors.append((str(p), err))
    return None, -1, errors


@torch.no_grad()
def extract(model_name, audios):
    """Identical to the probe kernel: masked-mean hidden state of every layer, float16 [N, layers, hidden]."""
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    layer_norm_extractor = getattr(model.config, "feat_extract_norm", "group") == "layer"
    batch = 12 if hidden >= 1024 else 24
    order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
    feats = np.zeros((len(audios), n_layers, hidden), dtype=np.float16)
    use_amp = DEVICE == "cuda"
    t0 = time.time()
    for start in range(0, len(order), batch):
        idxs = order[start:start + batch]
        arrays = [audios[i] for i in idxs]
        inputs = fe(arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
                    return_attention_mask=layer_norm_extractor)
        x = inputs["input_values"].to(DEVICE)
        kwargs = {"attention_mask": inputs["attention_mask"].to(DEVICE)} if layer_norm_extractor else {}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x, output_hidden_states=True, **kwargs)
        hs = torch.stack(out.hidden_states, dim=1).float()
        t_frames = hs.shape[2]
        lengths = torch.tensor([len(a) for a in arrays])
        flen = model._get_feat_extract_output_lengths(lengths).to(DEVICE).clamp(min=1, max=t_frames)
        mask = (torch.arange(t_frames, device=DEVICE)[None, :] < flen[:, None]).float()
        pooled = (hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
        feats[idxs] = pooled.cpu().numpy().astype(np.float16)
        if (start // batch) % 50 == 0:
            log(f"  {model_name}: {start + len(idxs)}/{len(order)} ({time.time() - t0:.0f}s)")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return feats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("device", DEVICE, "| splits", SPLITS, "| models", MODELS, "| LIMIT", LIMIT)
    index = index_videos()
    report = {}
    audio = {}
    for sp in SPLITS:
        t0 = time.time()
        table = load_split_table(sp)
        todo = [(d, u, y, index.get((sp, d, u), [])) for d, u, y in table]
        no_video = [(d, u) for d, u, _, ps in todo if not ps]
        todo = [t for t in todo if t[3]]
        with cf.ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 4)) as pool:
            results = list(pool.map(lambda t: decode_any(t[3]), todo))
        keep = [(t, r[0]) for t, r in zip(todo, results) if r[0] is not None]
        used_copy = Counter(r[1] for r in results if r[0] is not None)
        failures = [(t[0], t[1], r[2]) for t, r in zip(todo, results) if r[0] is None]
        err_kinds = Counter((e[1][:60] if e[1] else "") for _, _, errs in failures for e in errs[:1])
        report[sp] = {"rows_in_csv": len(table), "usable": len(keep), "no_video": len(no_video), "failed": len(failures),
                      "copy_used": dict(used_copy), "error_kinds": dict(err_kinds.most_common(5)),
                      "failed_examples": [{"dialogue": d, "utterance": u, "errors": errs[:2]} for d, u, errs in failures[:8]],
                      "no_video_examples": no_video[:8]}
        log(f"{sp}: usable {len(keep)}/{len(table)} | no video {len(no_video)} | failed {len(failures)} | "
            f"copy index used {dict(used_copy)} | {time.time() - t0:.0f}s")
        for d, u, errs in failures[:3]:
            log(f"  FAILED dia{d}_utt{u}: {errs[:2]}")
        audio[sp] = ([a for _, a in keep], [(t[0], t[1]) for t, _ in keep], [t[2] for t, _ in keep])
        (OUT / f"index_{sp}.json").write_text(json.dumps([[d, u] for d, u in audio[sp][1]]))
        (OUT / f"labels_{sp}.json").write_text(json.dumps(audio[sp][2]))
        (OUT / "decode_report.json").write_text(json.dumps(report, indent=1))

    for name in MODELS:
        short = name.split("/")[-1]
        for sp in SPLITS:
            t0 = time.time()
            feats = extract(name, audio[sp][0])
            np.save(OUT / f"feat_{short}_{sp}.npy", feats)
            log(f"{name} {sp}: saved {feats.shape} in {time.time() - t0:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
