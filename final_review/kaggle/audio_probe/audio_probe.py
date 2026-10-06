"""Kaggle kernel for final-review work package W1: a layer-by-layer audio probe for MELD.

For each pretrained speech encoder this extracts the mean-pooled hidden state of EVERY layer for every MELD utterance
(audio decoded from the raw videos), then fits a linear probe per layer on the TRAIN split and scores it on the
VALIDATION split only. Test-split features are extracted and saved for later training runs but are never scored here
(protocol: every choice is made on validation; the test set is scored once, later, for configurations fixed in advance).

Why: the earlier wav2vec 2.0 test used only the last layer, mean-pooled, from a base model. Middle layers and WavLM-style
models are normally much better for emotion, so that test does not rule out stronger audio features.

Outputs (in /kaggle/working):
  probe_results.json          every (model, layer) probe on validation, the legacy openSMILE-style baseline, timings
  feat_<model>_<split>.npy    float16 [n_utterances, n_layers, hidden] mean-pooled hidden states, per split
  index_<split>.json          [[dialogue_id, utterance_id], ...] in the row order of the feature arrays
  labels_<split>.json         emotion id per row (neutral 0, surprise 1, fear 2, sadness 3, joy 4, disgust 5, anger 6)

Environment variables (all optional): INPUT_ROOT, FEATURES_ROOT, OUT_DIR, LIMIT (smoke test: first N utterances per
split), MODELS (comma-separated Hugging Face ids), DEVICE.
"""
import concurrent.futures as cf
import io
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
FEATURES_ROOT = Path(os.environ.get("FEATURES_ROOT", str(INPUT_ROOT)))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
LIMIT = int(os.environ.get("LIMIT", "0"))
DEFAULT_MODELS = "microsoft/wavlm-base-plus,microsoft/wavlm-large,facebook/hubert-large-ll60k,facebook/wav2vec2-base"
MODELS = [m.strip() for m in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if m.strip()]
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
MAX_SAMPLES = 15 * SAMPLE_RATE  # MELD utterances are short; caps rare long outliers
EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMO_ID = {e: i for i, e in enumerate(EMOTIONS)}
SPLITS = ("train", "dev", "test")
CSV_NAMES = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data discovery
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
    idx, dup = {}, 0
    for root in sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir()):
        for p in root.rglob("*.mp4"):
            m = pat.search(p.name)
            if not m:
                continue
            sp = split_of(p.relative_to(root))
            if sp is None:
                continue
            key = (sp, int(m.group(1)), int(m.group(2)))
            if key in idx:
                dup += 1
                if "complete" in str(p).lower():  # dev has both dev_splits and dev_splits_complete in some copies
                    idx[key] = p
                continue
            idx[key] = p
    log(f"indexed {len(idx)} videos ({dup} duplicate names); per split:",
        {s: sum(1 for k in idx if k[0] == s) for s in SPLITS})
    return idx


def find_file(name, roots):
    for root in roots:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under {[str(r) for r in roots]}")


def load_split_table(split):
    csv = find_file(CSV_NAMES[split], [INPUT_ROOT, FEATURES_ROOT])
    df = pd.read_csv(csv).sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    rows = [(int(r.Dialogue_ID), int(r.Utterance_ID), EMO_ID[str(r.Emotion).lower()]) for r in df.itertuples()]
    if LIMIT:
        rows = rows[:LIMIT]
    return rows


def decode(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE),
           "-ac", "1", "-f", "wav", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    audio, _ = sf.read(io.BytesIO(r.stdout), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    return audio[:MAX_SAMPLES] if audio.size else None


# ------------------------------------------------------------------ extraction
@torch.no_grad()
def extract(model_name, audios):
    """audios: list of float32 arrays. Returns float16 [N, n_layers, hidden] of masked-mean hidden states."""
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    layer_norm_extractor = getattr(model.config, "feat_extract_norm", "group") == "layer"
    big = hidden >= 1024
    batch = 12 if big else 24
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
        hs = torch.stack(out.hidden_states, dim=1).float()  # [B, L, T, D]
        t_frames = hs.shape[2]
        lengths = torch.tensor([len(a) for a in arrays])
        flen = model._get_feat_extract_output_lengths(lengths).to(DEVICE).clamp(min=1, max=t_frames)
        mask = (torch.arange(t_frames, device=DEVICE)[None, :] < flen[:, None]).float()
        pooled = (hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
        feats[idxs] = pooled.cpu().numpy().astype(np.float16)
        if (start // batch) % 100 == 0:
            log(f"  {model_name}: {start + len(idxs)}/{len(order)} ({time.time() - t0:.0f}s)")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return feats


# ------------------------------------------------------------------ probing
def macro_weighted(y_true, y_pred):
    return (float((y_true == y_pred).mean()),
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def probe(xtr, ytr, xva, yva, steps=300, lr=2e-3, wd=1e-2, seed=0):
    """Linear softmax probe, class weights = (inverse frequency)^0.5, fixed hyper-parameters for every layer."""
    torch.manual_seed(seed)
    xtr = torch.as_tensor(xtr, dtype=torch.float32, device=DEVICE)
    xva = torch.as_tensor(xva, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=DEVICE)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-5)
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    counts = torch.bincount(ytr_t, minlength=7).float().clamp(min=1)
    cw = (counts.sum() / counts) ** 0.5
    cw = cw / cw.mean()
    lin = torch.nn.Linear(xtr.shape[1], 7).to(DEVICE)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(xtr), ytr_t, weight=cw).backward()
        opt.step()
    with torch.no_grad():
        pred = lin(xva).argmax(1).cpu().numpy()
    return macro_weighted(np.asarray(yva), pred)


def legacy_features(tables, usable):
    """The original 300-D openSMILE-style features, aligned to the same (dialogue, utterance) rows."""
    path = find_file("audio_emotion.pkl", [FEATURES_ROOT, INPUT_ROOT])
    with open(path, "rb") as f:
        parts = pickle.load(f, encoding="latin1")
    out = {}
    for sp, part in zip(SPLITS, parts):
        rank = {}
        for d in sorted({d for d, _, _ in tables[sp]}):
            utts = sorted(u for dd, u, _ in tables[sp] if dd == d)
            rank.update({(d, u): i for i, u in enumerate(utts)})
        rows = []
        for d, u in usable[sp]:
            arr = part[str(d)]
            rows.append(arr[rank[(d, u)]])
        out[sp] = np.stack(rows).astype(np.float32)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("device", DEVICE, "| models", MODELS, "| LIMIT", LIMIT)
    log("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    video_index = index_videos()
    tables = {sp: load_split_table(sp) for sp in SPLITS}

    audio, usable, labels = {}, {}, {}
    for sp in SPLITS:
        t0 = time.time()
        paths = [(d, u, y, video_index.get((sp, d, u))) for d, u, y in tables[sp]]
        missing = [(d, u) for d, u, _, p in paths if p is None]
        todo = [(d, u, y, p) for d, u, y, p in paths if p is not None]
        with cf.ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4))) as pool:
            decoded = list(pool.map(lambda t: decode(t[3]), todo))
        keep = [(t, a) for t, a in zip(todo, decoded) if a is not None]
        failed = len(todo) - len(keep)
        audio[sp] = [a for _, a in keep]
        usable[sp] = [(t[0], t[1]) for t, _ in keep]
        labels[sp] = [t[2] for t, _ in keep]
        log(f"{sp}: {len(keep)} usable of {len(tables[sp])} (no video: {len(missing)}, decode failed: {failed}) "
            f"in {time.time() - t0:.0f}s; mean {np.mean([len(a) for a in audio[sp]]) / SAMPLE_RATE:.2f}s")
        (OUT / f"index_{sp}.json").write_text(json.dumps([[d, u] for d, u in usable[sp]]))
        (OUT / f"labels_{sp}.json").write_text(json.dumps(labels[sp]))

    ytr, yva = np.array(labels["train"]), np.array(labels["dev"])
    results = {"emotions": EMOTIONS, "n": {sp: len(usable[sp]) for sp in SPLITS},
               "probe": "linear softmax, AdamW lr 2e-3 wd 1e-2, 300 full-batch steps, class weight=(1/freq)^0.5, "
                        "train -> validation only", "baseline_legacy": None, "models": {}}

    try:
        leg = legacy_features(tables, usable)
        a, w, m = probe(leg["train"], ytr, leg["dev"], yva)
        results["baseline_legacy"] = {"acc": a, "weighted_f1": w, "macro_f1": m, "dim": 300}
        log(f"legacy openSMILE-style 300-D probe: val acc {a:.4f} wF1 {w:.4f} mF1 {m:.4f}")
    except Exception as exc:  # the baseline is a comparison, not a requirement
        log("legacy baseline skipped:", repr(exc))

    for name in MODELS:
        short = name.split("/")[-1]
        t0 = time.time()
        log(f"=== {name} ===")
        feats = {}
        for sp in SPLITS:
            feats[sp] = extract(name, audio[sp])
            np.save(OUT / f"feat_{short}_{sp}.npy", feats[sp])
        t_extract = time.time() - t0
        n_layers = feats["train"].shape[1]
        rows = []
        for layer in range(n_layers):
            a, w, m = probe(feats["train"][:, layer], ytr, feats["dev"][:, layer], yva)
            rows.append({"layer": layer, "acc": a, "weighted_f1": w, "macro_f1": m})
            log(f"  layer {layer:2d}: val acc {a:.4f} wF1 {w:.4f} mF1 {m:.4f}")
        order = sorted(rows, key=lambda r: -r["weighted_f1"])
        top3 = [r["layer"] for r in order[:3]]
        avg_feats = {sp: feats[sp][:, top3].astype(np.float32).mean(1) for sp in ("train", "dev")}
        a, w, m = probe(avg_feats["train"], ytr, avg_feats["dev"], yva)
        mean_all = {sp: feats[sp].astype(np.float32).mean(1) for sp in ("train", "dev")}
        a2, w2, m2 = probe(mean_all["train"], ytr, mean_all["dev"], yva)
        results["models"][name] = {
            "n_layers": n_layers, "hidden": int(feats["train"].shape[2]), "extract_seconds": round(t_extract),
            "layers": rows, "best_by_weighted_f1": order[0], "top3_layers": top3,
            "top3_mean": {"acc": a, "weighted_f1": w, "macro_f1": m},
            "all_layer_mean": {"acc": a2, "weighted_f1": w2, "macro_f1": m2},
        }
        log(f"  best layer {order[0]['layer']} (wF1 {order[0]['weighted_f1']:.4f}); top-3 mean {top3}: wF1 {w:.4f}; "
            f"all-layer mean wF1 {w2:.4f}; {time.time() - t0:.0f}s")
        (OUT / "probe_results.json").write_text(json.dumps(results, indent=1))
        del feats

    (OUT / "probe_results.json").write_text(json.dumps(results, indent=1))
    log("done")


if __name__ == "__main__":
    main()
