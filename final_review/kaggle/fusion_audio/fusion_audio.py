"""Kaggle kernel, final-review lever L1: does WavLM-large audio help the fusion model (model-B recipe)?

Variants, each at seeds 42, 1, 2 (same script, same GPU, so the control is comparable):
  control     the model-B recipe with the original 300-D MELD audio features
  wavlm_mean  WavLM-large, mean of layers 22, 14, 21 (best on the W1 probe), 1024-D
  wavlm_l22   WavLM-large, layer 22 only, 1024-D

Audio features come from two earlier kernels (kernel_sources): meld-audio-test (complete test split, 2,610 clips)
and meld-audio-probe (train and dev; its own test files are INCOMPLETE and must not be used). The test kernel's folder
is passed to the pickle builder first, and the built test split is checked to have no missing clips.

The training script only sees train and validation. Test probabilities are written by evaluation.cache_member_probs,
which prints no test metrics; nothing in this kernel scores the test set. summary.json holds validation numbers only.

Outputs (/kaggle/working/output): summary.json, prob_cache/<run>.pt, logs/<run>.out, checkpoints/<run>.pt.
MODE = "smoke" runs one variant at one seed for one epoch, to check the pipeline before spending GPU hours.
"""

import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

MODE = "smoke"  # "smoke" or "full"

REPO_URL = "https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git"
BRANCH = "final-review"
REPO_DIR = Path("/kaggle/temp/repo")
INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")
SEEDS = [42, 1, 2]
VARIANTS = {
    "control": None,
    "wavlm_mean": "22,14,21",
    "wavlm_l22": "22",
}

REQUIRED_FILES = {
    "audio_emotion.pkl": "meld_features/MELD.Features.Models/features/audio_emotion.pkl",
    "data_emotion.p": "meld_features/MELD.Features.Models/features/data_emotion.p",
    "text_roberta.pkl": "meld_features/text_roberta/text_roberta.pkl",
    "text_distilbert.pkl": "meld_features/text_distilbert/text_distilbert.pkl",
    "text_roberta_large.pkl": "meld_features/text_roberta_large/text_roberta_large.pkl",
    "audio_wav2vec2.pkl": "meld_features/audio_wav2vec2/audio_wav2vec2.pkl",
    "train_visual.pkl": "meld_features/visual/train_visual.pkl",
    "dev_visual.pkl": "meld_features/visual/dev_visual.pkl",
    "test_visual.pkl": "meld_features/visual/test_visual.pkl",
    "train_sent_emo.csv": "meld_dataset/data/MELD/train_sent_emo.csv",
    "dev_sent_emo.csv": "meld_dataset/data/MELD/dev_sent_emo.csv",
    "test_sent_emo.csv": "meld_dataset/data/MELD/test_sent_emo.csv",
}


def run(cmd, cwd=None, env=None, log=None):
    print(f"$ {cmd}", flush=True)
    full_env = {**os.environ, **env} if env else None
    if log is None:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full_env)
        return
    with open(log, "w") as f:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
    print(Path(log).read_text()[-1500:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {cmd}")


def find_dir(marker, exclude=()):
    """The attached input folder that contains a file named `marker` (sources mount in no fixed order)."""
    for d in sorted(INPUT_ROOT.rglob(marker)):
        root = d.parent
        if not any(str(root).startswith(str(e)) for e in exclude):
            return root
    raise SystemExit(f"no attached input contains {marker}")


def build_audio(layers, test_dir, probe_dir, csv_dir, out_path):
    run(
        f"python -u final_review/tools/make_audio_pickle.py {out_path} --model wavlm-large --layers {layers} "
        f"--feat-dir {test_dir} --feat-dir {probe_dir} --csv-dir {csv_dir}",
        cwd=REPO_DIR,
    )
    with open(out_path, "rb") as f:
        splits = pickle.load(f)
    zeros = [sum(int(not a[i].any()) for a in s.values() for i in range(len(a))) for s in splits]
    print(f"zero (missing) rows train/dev/test: {zeros}", flush=True)
    # 1 missing clip each in train and dev (no video either); the test kernel decoded all 2,610.
    if zeros[2] != 0 or zeros[0] > 1 or zeros[1] > 1:
        raise SystemExit(f"unexpected missing audio rows {zeros}: wrong feature folder picked up?")


def main():
    t0 = time.time()
    run(f"ls -la {INPUT_ROOT}")
    test_dir = find_dir("decode_report.json")
    probe_dir = find_dir("probe_results.json")
    # The features dataset: climb from data_emotion.p to the highest folder that holds neither audio kernel output.
    dataset_dir = find_dir("data_emotion.p")
    while dataset_dir.parent != INPUT_ROOT and not any(
        str(p).startswith(str(dataset_dir.parent)) for p in (test_dir, probe_dir)
    ):
        dataset_dir = dataset_dir.parent
    print(f"features file: {dataset_dir}\ntest audio dir: {test_dir}\nprobe audio dir: {probe_dir}", flush=True)
    report = json.loads((test_dir / "decode_report.json").read_text())
    print("test decode report:", report, flush=True)

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {REPO_DIR}")
    commit = subprocess.run("git rev-parse HEAD", shell=True, cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
    print("repo commit:", commit, flush=True)
    run("pip install -q torch_geometric", cwd=REPO_DIR)

    missing = []
    for filename, relative_target in REQUIRED_FILES.items():
        matches = sorted(dataset_dir.rglob(filename))
        if not matches:
            missing.append(filename)
            continue
        target = REPO_DIR / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(matches[0], target)
    if missing:
        raise SystemExit(f"Cannot proceed, missing files: {missing}")
    run("python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"", cwd=REPO_DIR)

    for sub in ("prob_cache", "logs", "checkpoints", "audio"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    csv_dir = REPO_DIR / "meld_dataset/data/MELD"

    if MODE == "smoke":
        jobs = [("wavlm_mean", 42)]
        epochs = "1"
    else:
        jobs = [(v, s) for s in SEEDS for v in VARIANTS]
        epochs = "25"

    audio_paths = {}
    for variant in sorted({v for v, _ in jobs}):
        if VARIANTS[variant] is not None:
            path = Path("/kaggle/temp") / f"audio_{variant}.pkl"
            build_audio(VARIANTS[variant], test_dir, probe_dir, csv_dir, path)
            audio_paths[variant] = path

    summary = {"mode": MODE, "repo_commit": commit, "decode_report": report, "runs": []}
    for variant, seed in jobs:
        name = f"L1_{variant}_s{seed}" + ("_smoke" if MODE == "smoke" else "")
        ckpt = REPO_DIR / "models" / f"ft_{name}.pt"
        env = {
            "TRAIN_TEXT_PATH": "meld_features/text_roberta/text_roberta.pkl",
            "TRAIN_TEXT_DIM": "768",
            "TRAIN_SEED": str(seed),
            "TRAIN_EPOCHS": epochs,
            "TRAIN_OUTPUT_PATH": str(ckpt),
        }
        cache_cmd = f"python -u -m evaluation.cache_member_probs {name} frozen {ckpt} --features roberta"
        if variant in audio_paths:
            env.update({"TRAIN_AUDIO_PATH": str(audio_paths[variant]), "TRAIN_AUDIO_DIM": "1024"})
            cache_cmd += f" --audio-path {audio_paths[variant]}"
        entry = {"name": name, "variant": variant, "seed": seed}
        start = time.time()
        try:
            run("python -u training/train_final.py", cwd=REPO_DIR, env=env, log=OUT / "logs" / f"{name}.out")
            import torch
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            entry.update({
                "best_epoch": meta["epoch"], "val_accuracy": meta["val_accuracy"],
                "val_weighted_f1": meta["val_weighted_f1"], "val_macro_f1": meta["val_macro_f1"],
                "audio_dim": meta["audio_dim"],
            })
            # Test probabilities are cached for the later, single scoring run; no test metric is printed here.
            run(cache_cmd, cwd=REPO_DIR, log=OUT / "logs" / f"{name}_cache.out")
            shutil.copy(REPO_DIR / "logs" / "prob_cache" / f"{name}.pt", OUT / "prob_cache" / f"{name}.pt")
            shutil.copy(ckpt, OUT / "checkpoints" / f"{name}.pt")
            entry["status"] = "ok"
        except Exception as exc:  # keep going: one failed run must not lose the others
            entry.update({"status": "failed", "error": str(exc)})
        entry["minutes"] = round((time.time() - start) / 60, 1)
        summary["runs"].append(entry)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        print("RUN DONE", json.dumps(entry), flush=True)

    by_variant = {}
    for r in summary["runs"]:
        if r["status"] == "ok":
            by_variant.setdefault(r["variant"], []).append(r["val_weighted_f1"])
    summary["val_weighted_f1_mean"] = {v: float(np.mean(x)) for v, x in by_variant.items()}
    summary["total_minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
