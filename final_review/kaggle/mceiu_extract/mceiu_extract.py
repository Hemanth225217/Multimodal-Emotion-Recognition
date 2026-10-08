"""Kaggle kernel (GPU): extract MELD-identical features for MC-EIU English (Hugging Face YulangZhuo/MC-EIU, gated).

Must be started from the Kaggle editor (Save Version > Save & Run All) with the HF_TOKEN secret attached: Kaggle does not
pass secrets to runs pushed from the command line. The token is read from Kaggle's secret store and never printed.

Per utterance, with exactly the settings used for MELD in this project:
  text   RoBERTa-base, attention-masked mean of the last hidden layer, max_length 64 (modules/text_features_roberta.py),
         on the "Subtitle" column
  audio  WavLM-large, masked mean over time of every hidden layer (25 x 1024, float16), mono 16 kHz, first 15 s
         (final_review/kaggle/audio_extract_test/audio_extract.py)
  video  ResNet-18 (ImageNet weights, fc removed), 8 frames evenly spaced, mean over frames, 512-D
         (modules/visual_features.py)
Split: our own, because the official MC-EIU split is not public (GitHub MC-EIU/MC-EIU issue #4). Dialogue-level, seed 42,
stratified by show, sizes 2,807 / 400 / 806 dialogues as in the MC-EIU paper.

Stages: (1) SELF-TEST on 64 clips read from zip 1 by HTTP range requests (nothing large downloaded); stops the run if
anything is wrong. (2) Each archive in turn: download, process clip by clip, save a part file, delete the archive.
(3) Text features, merge, report.
Outputs (/kaggle/working/output): utterances.csv (row order of every array), split.json, text_roberta.npy [N,768] f32,
wavlm_large_all_layers.npy [N,25,1024] f16, video_resnet18.npy [N,512] f32, has_audio.npy, has_video.npy, report.json
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = "YulangZhuo/MC-EIU"
OUT = Path("/kaggle/working/output")
PARTS = OUT / "parts"
ARCHIVES = ["English_Dialogues_1.zip", "English_Dialogues_2.zip", "English_Dialogues_3.rar"]
SPLIT_SIZES = {"train": 2807, "dev": 400, "test": 806}
SEED = 42
SAMPLE_RATE = 16000
MAX_SAMPLES = 15 * SAMPLE_RATE
NUM_FRAMES = 8
CHUNK = 256
CLIP_RE = re.compile(r"dia_(\d+)_utt_(\d+)\.mp4$", re.I)
SHOWS = {"摩登家庭": "modern_family", "老友记": "friends", "生活大爆炸": "big_bang_theory"}
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)


def token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        log("No HF_TOKEN secret available. Start this run from the editor with the secret attached.", type(exc).__name__)
        sys.exit(1)


# ---------------------------------------------------------------- split
def make_split(df):
    """Dialogue-level split, stratified by show (largest-remainder allocation), fixed seed."""
    rng = np.random.default_rng(SEED)
    dia_show = df.groupby("Dia_No")["show"].first()
    total = len(dia_show)
    assert total == sum(SPLIT_SIZES.values()), (total, SPLIT_SIZES)
    shows = sorted(dia_show.unique())
    alloc = {}
    for sp, n in SPLIT_SIZES.items():
        raw = {s: n * (dia_show == s).sum() / total for s in shows}
        base = {s: int(np.floor(v)) for s, v in raw.items()}
        for s in sorted(shows, key=lambda s: raw[s] - base[s], reverse=True)[: n - sum(base.values())]:
            base[s] += 1
        alloc[sp] = base
    split = {}
    for s in shows:
        dias = sorted(int(d) for d in dia_show[dia_show == s].index)
        rng.shuffle(dias)
        start = 0
        for sp in ("train", "dev", "test"):
            k = alloc[sp][s]
            if sp == "train":  # absorb any per-show rounding difference in train
                k = len(dias) - alloc["dev"][s] - alloc["test"][s]
            for d in dias[start:start + k]:
                split[d] = sp
            start += k
    counts = {sp: sum(1 for v in split.values() if v == sp) for sp in SPLIT_SIZES}
    assert counts == SPLIT_SIZES, counts
    return split


# ---------------------------------------------------------------- per-clip decoding (CPU threads)
def decode_audio(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
           "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio[:MAX_SAMPLES] if audio.size else None


def sample_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frames = []
    for pos in np.linspace(0, total - 1, NUM_FRAMES, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def decode_clip(path, preprocess):
    """Audio array (or None) and preprocessed frame tensor [k,3,224,224] (or None)."""
    from PIL import Image
    import torch
    audio = decode_audio(path)
    frames = sample_frames(path)
    video = torch.stack([preprocess(Image.fromarray(f)) for f in frames]) if frames else None
    return audio, video


# ---------------------------------------------------------------- models (GPU)
class Models:
    def __init__(self, device):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights
        from transformers import AutoFeatureExtractor, AutoModel
        self.torch, self.device = torch, device
        w = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=w)
        self.resnet.fc = nn.Identity()
        self.resnet = self.resnet.to(device).eval()
        self.preprocess = w.transforms()
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.wavlm = AutoModel.from_pretrained("microsoft/wavlm-large").to(device).eval()
        self.ln = getattr(self.wavlm.config, "feat_extract_norm", "group") == "layer"

    def video(self, tensors):
        torch = self.torch
        out = np.zeros((len(tensors), 512), dtype=np.float32)
        with torch.no_grad():
            for i, t in enumerate(tensors):
                if t is not None:
                    out[i] = self.resnet(t.to(self.device)).mean(0).float().cpu().numpy()
        return out

    def audio(self, arrays):
        """Same as the MELD audio kernels: masked mean of every hidden state, float16 [n, 25, 1024]."""
        torch = self.torch
        feats = np.zeros((len(arrays), 25, 1024), dtype=np.float16)
        idx_ok = [i for i, a in enumerate(arrays) if a is not None and len(a) >= 400]
        order = sorted(idx_ok, key=lambda i: len(arrays[i]))
        with torch.no_grad():
            for s in range(0, len(order), 12):
                idxs = order[s:s + 12]
                batch = [arrays[i] for i in idxs]
                inp = self.fe(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
                              return_attention_mask=self.ln)
                kw = {"attention_mask": inp["attention_mask"].to(self.device)} if self.ln else {}
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    out = self.wavlm(inp["input_values"].to(self.device), output_hidden_states=True, **kw)
                hs = torch.stack(out.hidden_states, dim=1).float()
                tf = hs.shape[2]
                flen = self.wavlm._get_feat_extract_output_lengths(torch.tensor([len(a) for a in batch]))
                flen = flen.to(self.device).clamp(min=1, max=tf)
                mask = (torch.arange(tf, device=self.device)[None, :] < flen[:, None]).float()
                pooled = (hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
                feats[idxs] = pooled.cpu().numpy().astype(np.float16)
        return feats, np.array([i in set(idx_ok) for i in range(len(arrays))])


def process_clips(clip_paths, models, workers=8):
    """clip_paths: list of local mp4 paths. Returns audio feats, video feats, has_audio, has_video."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decoded = list(pool.map(lambda p: decode_clip(p, models.preprocess), clip_paths))
    a_feats, has_a = models.audio([d[0] for d in decoded])
    v_feats = models.video([d[1] for d in decoded])
    has_v = np.array([d[1] is not None for d in decoded])
    return a_feats, v_feats, has_a, has_v


# ---------------------------------------------------------------- archives
def rar_tool():
    """First working RAR extractor: unrar, 7z with the Rar codec, unar, bsdtar (installed with apt-get if absent)."""
    def ok():
        if sh("unrar", check=False).returncode in (0, 7) and "UNRAR" in (sh("unrar", check=False).stdout or "").upper():
            return "unrar"
        if "Rar" in (sh("7z i", check=False).stdout or ""):
            return "7z"
        if sh("unar -v", check=False).returncode == 0:
            return "unar"
        if sh("bsdtar --version", check=False).returncode == 0:
            return "bsdtar"
        return None
    tool = ok()
    if tool is None:
        sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y unrar unar p7zip-full p7zip-rar libarchive-tools "
           "> /dev/null 2>&1", check=False)
        tool = ok()
    return tool or "missing"


def unpack_rar(path, dest, tool):
    """Unpack the whole archive once (solid RAR archives make per-member extraction quadratic)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = {"unrar": f"unrar x -o+ -idq '{path}' '{dest}/'", "7z": f"7z x -y -bd -o'{dest}' '{path}'",
           "unar": f"unar -q -f -o '{dest}' '{path}'", "bsdtar": f"bsdtar -xf '{path}' -C '{dest}'"}[tool]
    r = sh(cmd, check=False)
    files = sorted(p for p in dest.rglob("*.mp4") if CLIP_RE.search(p.name))
    return files, (r.stderr or "")[-300:]


def zip_members(path):
    with zipfile.ZipFile(path) as z:
        return [n for n in z.namelist() if CLIP_RE.search(n)]


def extract_zip_members(path, names, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for n in names:
            with z.open(n) as src, open(dest / Path(n).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return [dest / Path(n).name for n in names]


def work_dir(need_gb):
    best = max(["/kaggle/temp", "/tmp", "/kaggle/working"],
               key=lambda d: shutil.disk_usage(d if os.path.exists(d) else "/").free)
    free = shutil.disk_usage(best).free / 1e9
    log(f"work dir {best}: {free:.1f} GB free (need about {need_gb:.0f} GB)")
    return Path(best), free


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    tok = token()
    sh("pip install -q remotezip hf_xet", check=False)
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
    from remotezip import RemoteZip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device", device, "| disk:", {d: round(shutil.disk_usage(d).free / 1e9, 1) for d in ("/kaggle/working", "/tmp")
                                       if os.path.exists(d)})
    report = {"seed": SEED, "split_sizes": SPLIT_SIZES}

    # table and split
    csv = hf_hub_download(REPO, "EnglishDialogues.csv", repo_type="dataset", token=tok, local_dir="/tmp/mceiu")
    df = pd.read_csv(csv)
    df["show"] = df["video_name"].map(SHOWS)
    assert df["show"].notna().all(), df["video_name"].unique()
    df = df.sort_values(["Dia_No", "Utt_No"]).reset_index(drop=True)
    assert not df.duplicated(["Dia_No", "Utt_No"]).any()
    split = make_split(df)
    df["split"] = df["Dia_No"].map(split)
    (OUT / "split.json").write_text(json.dumps({"seed": SEED, "method": "dialogue-level, stratified by show",
                                                "sizes": SPLIT_SIZES, "dialogue_split": {str(k): v for k, v in sorted(split.items())}}))
    row_of = {(int(d), int(u)): i for i, (d, u) in enumerate(zip(df["Dia_No"], df["Utt_No"]))}
    report["utterances"] = len(df)
    report["split_utterances"] = df["split"].value_counts().to_dict()
    report["split_by_show"] = df.groupby(["split", "show"])["Dia_No"].nunique().unstack().to_dict()
    log("split:", report["split_utterances"])

    models = Models(device)
    log("models loaded")

    # ---- stage 1: self-test on 64 clips via HTTP range requests
    headers = {"Authorization": f"Bearer {tok}"}
    url = hf_hub_url(REPO, ARCHIVES[0], repo_type="dataset")
    test_dir = Path("/tmp/selftest")
    test_dir.mkdir(exist_ok=True)
    with RemoteZip(url, headers=headers) as z:
        names = [n for n in z.namelist() if CLIP_RE.search(n)][:64]
        for n in names:
            (test_dir / Path(n).name).write_bytes(z.read(n))
    paths = sorted(test_dir.glob("*.mp4"))
    a, v, ha, hv = process_clips(paths, models)
    keys = [tuple(int(x) for x in CLIP_RE.search(p.name).groups()) for p in paths]
    st = {"clips": len(paths), "in_csv": sum(k in row_of for k in keys), "audio_ok": int(ha.sum()), "video_ok": int(hv.sum()),
          "audio_finite": bool(np.isfinite(a[ha].astype(np.float32)).all()), "video_finite": bool(np.isfinite(v[hv]).all()),
          "audio_layer22_norm_mean": float(np.linalg.norm(a[ha][:, 22].astype(np.float32), axis=1).mean()),
          "video_norm_mean": float(np.linalg.norm(v[hv], axis=1).mean())}
    report["self_test"] = st
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log("SELF-TEST", st)
    if not (st["clips"] == 64 and st["in_csv"] == 64 and st["audio_ok"] >= 58 and st["video_ok"] >= 58
            and st["audio_finite"] and st["video_finite"]):
        log("SELF-TEST FAILED: stopping before the full run.")
        sys.exit(1)
    shutil.rmtree(test_dir)

    # ---- stage 2: archives
    N = len(df)
    for k, name in enumerate(ARCHIVES, 1):
        part = PARTS / f"part{k}.npz"
        if part.exists():
            continue
        size_gb = get_hf_file_metadata(hf_hub_url(REPO, name, repo_type="dataset"), token=tok).size / 1e9
        need = size_gb * (2 if name.endswith(".rar") else 1) + 3  # a RAR is unpacked whole next to itself
        wdir, free = work_dir(need)
        if free < need:
            report[f"archive_{k}"] = {"error": f"not enough disk: {free:.1f} GB free, {size_gb:.1f} GB needed"}
            log(report[f"archive_{k}"])
            continue
        t = time.time()
        local = Path(hf_hub_download(REPO, name, repo_type="dataset", token=tok, local_dir=str(wdir / "mceiu_dl")))
        log(f"{name}: downloaded {size_gb:.1f} GB in {(time.time() - t) / 60:.1f} min")
        rows, A, V, HA, HV = [], [], [], [], []
        clip_dir = wdir / "mceiu_clips"

        def consume(names_or_paths, paths):
            for n, ai, vi, hai, hvi in zip(names_or_paths, *paths):
                key = tuple(int(x) for x in CLIP_RE.search(str(n)).groups())
                if key in row_of:
                    rows.append(row_of[key]); A.append(ai); V.append(vi); HA.append(hai); HV.append(hvi)

        if local.suffix == ".zip":
            members = zip_members(local)
            log(f"{name}: {len(members)} clips")
            for s in range(0, len(members), CHUNK):
                chunk = members[s:s + CHUNK]
                try:
                    consume(chunk, process_clips(extract_zip_members(local, chunk, clip_dir), models))
                except Exception as exc:
                    log(f"  chunk {s} failed: {type(exc).__name__}: {exc}"[:300])
                finally:
                    shutil.rmtree(clip_dir, ignore_errors=True)
                if (s // CHUNK) % 10 == 0:
                    log(f"  {name}: {s + len(chunk)}/{len(members)}")
        else:
            tool = rar_tool()
            log(f"{name}: rar tool {tool}")
            if tool == "missing":
                report[f"archive_{k}"] = {"error": "no RAR extractor available"}
                continue
            if shutil.disk_usage(wdir).free / 1e9 < size_gb + 2:
                report[f"archive_{k}"] = {"error": "not enough disk to unpack the RAR"}
                continue
            files, err = unpack_rar(local, clip_dir, tool)
            local.unlink(missing_ok=True)
            members = files
            log(f"{name}: unpacked {len(files)} clips {err}")
            for s in range(0, len(files), CHUNK):
                chunk = files[s:s + CHUNK]
                try:
                    consume([p.name for p in chunk], process_clips(chunk, models))
                except Exception as exc:
                    log(f"  chunk {s} failed: {type(exc).__name__}: {exc}"[:300])
                if (s // CHUNK) % 10 == 0:
                    log(f"  {name}: {s + len(chunk)}/{len(files)}")
            shutil.rmtree(clip_dir, ignore_errors=True)
        np.savez(part, rows=np.array(rows, dtype=np.int64), audio=np.array(A, dtype=np.float16).reshape(-1, 25, 1024),
                 video=np.array(V, dtype=np.float32).reshape(-1, 512), has_audio=np.array(HA, bool), has_video=np.array(HV, bool))
        report[f"archive_{k}"] = {"name": name, "clips": len(members), "matched_rows": len(rows),
                                  "audio_ok": int(np.sum(HA)), "video_ok": int(np.sum(HV)),
                                  "minutes": round((time.time() - t) / 60, 1)}
        log(report[f"archive_{k}"])
        (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
        local.unlink(missing_ok=True)
        shutil.rmtree(wdir / "mceiu_dl", ignore_errors=True)

    # ---- stage 3: text, merge, report
    from transformers import AutoModel, AutoTokenizer
    tz = AutoTokenizer.from_pretrained("roberta-base")
    rb = AutoModel.from_pretrained("roberta-base").to(device).eval()
    texts = df["Subtitle"].fillna("").astype(str).tolist()
    T = np.zeros((N, 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, 64):
            enc = tz(texts[s:s + 64], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
            hs = rb(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            T[s:s + 64] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
    np.save(OUT / "text_roberta.npy", T)
    AUD = np.lib.format.open_memmap(OUT / "wavlm_large_all_layers.npy", mode="w+", dtype=np.float16, shape=(N, 25, 1024))
    VID = np.zeros((N, 512), dtype=np.float32)
    HA = np.zeros(N, bool)
    HV = np.zeros(N, bool)
    for part in sorted(PARTS.glob("part*.npz")):
        p = np.load(part)
        AUD[p["rows"]] = p["audio"]; VID[p["rows"]] = p["video"]; HA[p["rows"]] = p["has_audio"]; HV[p["rows"]] = p["has_video"]
    AUD.flush()
    np.save(OUT / "video_resnet18.npy", VID); np.save(OUT / "has_audio.npy", HA); np.save(OUT / "has_video.npy", HV)
    df["has_audio"], df["has_video"] = HA, HV
    df[["Dia_No", "Utt_No", "speaker", "show", "Season", "Episode", "split", "emotion", "intent", "Subtitle",
        "has_audio", "has_video"]].to_csv(OUT / "utterances.csv", index=False)
    report["coverage"] = {sp: {"utterances": int((df.split == sp).sum()), "audio": int(df[df.split == sp].has_audio.sum()),
                               "video": int(df[df.split == sp].has_video.sum())} for sp in SPLIT_SIZES}
    report["total_minutes"] = round((time.time() - T0) / 60, 1)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    shutil.rmtree(PARTS, ignore_errors=True)
    log("DONE", json.dumps(report["coverage"]))


if __name__ == "__main__":
    main()
