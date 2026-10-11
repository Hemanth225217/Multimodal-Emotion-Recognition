"""Kaggle kernel (GPU): face-expression features for MELD, to test whether the non-verbal context becomes used once the
visual input carries emotion information (the ResNet-18 scene features do not: the video-only prediction is constant).

Per utterance: the same 8 evenly spaced frames as modules/visual_features.py; in each frame the LARGEST face found by
MTCNN (a proxy for the speaker; MELD has no face annotations) is cropped with a 20 % margin and passed through the public
facial-expression model trpakov/vit-face-expression (ViT, FER-2013 classes); the feature is the mean over frames of
[ViT CLS embedding (768) | softmax over its 7 expression classes (7)] = 775-D. Utterances without any detected face get a
zero vector, so the set of usable utterances is exactly that of the original visual pickles.
Inputs: zaber666/meld-dataset (raw clips; several copies of some clips, the canonical folder is tried first, as in the
audio kernels) and hemanths0411/meld-emotion-features (the original visual pickles define which keys exist).
Outputs (/kaggle/working/output): train_visual.pkl, dev_visual.pkl, test_visual.pkl (same layout as the originals,
feature_dim 775), face_report.json. A self-test on 32 clips runs first and stops the kernel if face detection fails.
"""
import json
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")
NUM_FRAMES = 8
FER_MODEL = "trpakov/vit-face-expression"
CANONICAL = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
VIS_FILES = {"train": "train_visual.pkl", "dev": "dev_visual.pkl", "test": "test_visual.pkl"}
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


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
    pat = re.compile(r"dia(\d+)_utt(\d+)\.mp4$", re.I)
    idx = {}
    for p in INPUT_ROOT.rglob("*.mp4"):
        m = pat.search(p.name)
        sp = split_of(p.relative_to(INPUT_ROOT)) if m else None
        if sp is None:
            continue
        idx.setdefault((sp, int(m.group(1)), int(m.group(2))), []).append(p)
    for key, paths in idx.items():
        paths.sort(key=lambda p: (CANONICAL[key[0]] not in str(p), str(p)))
    log("indexed", len(idx), "clips; per split", {s: sum(1 for k in idx if k[0] == s) for s in CANONICAL})
    return idx


def sample_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for pos in np.linspace(0, total - 1, NUM_FRAMES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def frames_any(paths):
    for p in paths:
        f = sample_frames(p)
        if f:
            return f
    return []


class FaceFeatures:
    def __init__(self, device):
        import torch
        from facenet_pytorch import MTCNN
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch, self.device = torch, device
        self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
        self.proc = AutoImageProcessor.from_pretrained(FER_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(FER_MODEL).to(device).eval()
        log("FER labels:", self.model.config.id2label)

    def __call__(self, frames):
        """775-D feature (zeros if no face in any frame) and the number of frames with a face."""
        from PIL import Image
        crops = []
        imgs = [Image.fromarray(f) for f in frames]
        if not imgs:
            return np.zeros(775, np.float32), 0
        try:  # one batched detection per clip (all frames of a clip have the same size)
            all_boxes, _ = self.mtcnn.detect(imgs)
        except Exception:
            all_boxes = [self.mtcnn.detect(im)[0] for im in imgs]
        for img, boxes in zip(imgs, all_boxes):
            if boxes is None or len(boxes) == 0:
                continue
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            x0, y0, x1, y1 = boxes[int(np.argmax(areas))]
            w, h = x1 - x0, y1 - y0
            box = (max(0, x0 - 0.2 * w), max(0, y0 - 0.2 * h), min(img.width, x1 + 0.2 * w), min(img.height, y1 + 0.2 * h))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            crops.append(img.crop(box).convert("RGB"))
        if not crops:
            return np.zeros(775, np.float32), 0
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=crops, return_tensors="pt").to(self.device)
            out = self.model(**inp, output_hidden_states=True)
            cls = out.hidden_states[-1][:, 0, :].float()
            probs = torch.softmax(out.logits.float(), dim=-1)
        feat = torch.cat([cls, probs], dim=-1).mean(0).cpu().numpy().astype(np.float32)
        return feat, len(crops)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run("pip install -q --no-deps facenet-pytorch", shell=True, capture_output=True, text=True)
    log("facenet-pytorch install:", r.returncode, r.stderr[-300:])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ff = FaceFeatures(device)
    idx = index_videos()
    originals = {}
    for sp, name in VIS_FILES.items():
        f = sorted(INPUT_ROOT.rglob(name))[0]
        originals[sp] = pickle.load(open(f, "rb"))
        log(sp, "original visual keys:", len(originals[sp]["features"]))

    # self-test
    test_keys = list(originals["dev"]["features"].keys())[:32]
    ok = 0
    for k in test_keys:
        d, u = map(int, k.split("_"))
        feat, n = ff(frames_any(idx.get(("dev", d, u), [])))
        ok += n > 0
    log(f"SELF-TEST: {ok} of {len(test_keys)} dev clips have a detected face")
    if ok < len(test_keys) // 2:
        log("SELF-TEST FAILED: face detection too rare; stopping.")
        sys.exit(1)

    report = {"model": FER_MODEL, "frames": NUM_FRAMES}
    for sp in ("dev", "test", "train"):
        keys = list(originals[sp]["features"].keys())
        t = time.time()
        feats, nfaces, nodecode = {}, Counter(), 0
        for s in range(0, len(keys), 64):
            chunk = keys[s:s + 64]
            with ThreadPoolExecutor(max_workers=8) as pool:
                frames = list(pool.map(lambda k: frames_any(idx.get((sp, *map(int, k.split("_"))), [])), chunk))
            for k, fr in zip(chunk, frames):
                if not fr:
                    nodecode += 1
                feat, n = ff(fr)
                feats[k] = feat
                nfaces[min(n, NUM_FRAMES)] += 1
            if (s // 64) % 20 == 0:
                log(f"  {sp}: {s + len(chunk)}/{len(keys)}")
        with_face = sum(v for n, v in nfaces.items() if n > 0)
        report[sp] = {"utterances": len(keys), "with_face": with_face, "share_with_face": round(with_face / len(keys), 4),
                      "frames_with_face_histogram": dict(sorted(nfaces.items())), "not_decoded": nodecode,
                      "minutes": round((time.time() - t) / 60, 1)}
        log(sp, report[sp])
        new = dict(originals[sp])
        new["features"] = feats
        new["feature_dim"] = 775
        new["face_model"] = FER_MODEL
        with open(OUT / VIS_FILES[sp], "wb") as f:
            pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
        (OUT / "face_report.json").write_text(json.dumps(report, indent=1))
    report["total_minutes"] = round((time.time() - T0) / 60, 1)
    (OUT / "face_report.json").write_text(json.dumps(report, indent=1))
    log("DONE", report)


if __name__ == "__main__":
    main()
