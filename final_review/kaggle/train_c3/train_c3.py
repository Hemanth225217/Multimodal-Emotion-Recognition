"""Kaggle kernel (GPU): training runs for the paper's two-dataset analysis (fixed before any result is seen).

Jobs, each at seeds 42, 1, 2, model-B recipe (train_final.py), RoBERTa-base text:
  meld_ctx      MELD, original 300-D audio, context dropout 0.15   (its no-dropout control = L1_control_s*, already run)
  mceiu_base    MC-EIU English, WavLM-large layer 22 audio, no context dropout
  mceiu_ctx     MC-EIU English, WavLM-large layer 22 audio, context dropout 0.15
Audio for MC-EIU: WavLM layer 22, fixed in advance (MC-EIU has no openSMILE features; on MELD layer 22 was within seed
noise of the 300-D control, see the journal for 8 Oct). The MC-EIU split is our own (seed 42, see mc-eiu-extract).

MELD and MC-EIU each get their own clone of the repository, so their data never share a folder. Training sees only
train and validation; val and test probabilities are cached without printing test metrics; summary.json is validation-only.
Inputs: dataset hemanths0411/meld-emotion-features; kernel outputs of mc-eiu-extract (MC-EIU features).
Outputs (/kaggle/working/output): summary.json, prob_cache/<run>.pt, checkpoints/<run>.pt, logs/<run>.out,
mceiu_audio_l22.pkl is NOT kept (rebuilt by the analysis kernels from the extract output).
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_URL = "https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git"
INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")
SEEDS = [42, 1, 2]
CTX = "0.15"
JOBS = [("meld_ctx", "meld", CTX), ("mceiu_base", "mceiu", "0"), ("mceiu_ctx", "mceiu", CTX)]
MELD_FILES = {
    "audio_emotion.pkl": "meld_features/MELD.Features.Models/features/audio_emotion.pkl",
    "data_emotion.p": "meld_features/MELD.Features.Models/features/data_emotion.p",
    "text_roberta.pkl": "meld_features/text_roberta/text_roberta.pkl",
    "train_visual.pkl": "meld_features/visual/train_visual.pkl",
    "dev_visual.pkl": "meld_features/visual/dev_visual.pkl",
    "test_visual.pkl": "meld_features/visual/test_visual.pkl",
    "train_sent_emo.csv": "meld_dataset/data/MELD/train_sent_emo.csv",
    "dev_sent_emo.csv": "meld_dataset/data/MELD/dev_sent_emo.csv",
    "test_sent_emo.csv": "meld_dataset/data/MELD/test_sent_emo.csv",
}


def run(cmd, cwd=None, env=None, log=None):
    print(f"$ {cmd}", flush=True)
    full_env = {**os.environ, **(env or {})}
    if log is None:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full_env)
        return
    with open(log, "w") as f:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
    print(Path(log).read_text()[-1200:], flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {cmd}")


def find_one(name):
    hits = sorted(INPUT_ROOT.rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under {INPUT_ROOT}")
    return hits[0]


def clone(dest):
    if dest.exists():
        shutil.rmtree(dest)
    run(f"git clone --depth 1 --branch final-review {REPO_URL} {dest}")
    return subprocess.run("git rev-parse HEAD", shell=True, cwd=dest, capture_output=True, text=True).stdout.strip()


def main():
    t0 = time.time()
    run(f"ls -la {INPUT_ROOT}")
    run("pip install -q torch_geometric")
    repos = {"meld": Path("/kaggle/temp/repo_meld"), "mceiu": Path("/kaggle/temp/repo_mceiu")}
    commit = clone(repos["meld"])
    clone(repos["mceiu"])
    # MELD files: from the dataset folder that holds data_emotion.p (not from the MC-EIU extract output)
    meld_dir = find_one("data_emotion.p").parent
    while meld_dir.parent != INPUT_ROOT and not list(meld_dir.rglob("train_sent_emo.csv")):
        meld_dir = meld_dir.parent
    for name, rel in MELD_FILES.items():
        target = repos["meld"] / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sorted(meld_dir.rglob(name))[0], target)
    # MC-EIU in MELD layout
    extract_dir = find_one("utterances.csv").parent
    report = json.loads((extract_dir / "report.json").read_text()) if (extract_dir / "report.json").exists() else {}
    print("mc-eiu-extract coverage:", report.get("coverage"), flush=True)
    mceiu_audio = Path("/kaggle/temp/mceiu_audio_l22.pkl")
    run(f"python -u final_review/tools/mceiu_to_meld_layout.py {extract_dir} . {mceiu_audio} --layers 22", cwd=repos["mceiu"])

    for sub in ("prob_cache", "checkpoints", "logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    summary = {"repo_commit": commit, "mceiu_coverage": report.get("coverage"), "runs": []}
    for seed in SEEDS:
        for job, data, ctx in JOBS:
            name = f"{job}_s{seed}"
            repo = repos[data]
            ckpt = repo / "models" / f"ft_{name}.pt"
            env = {"TRAIN_TEXT_PATH": "meld_features/text_roberta/text_roberta.pkl", "TRAIN_TEXT_DIM": "768",
                   "TRAIN_SEED": str(seed), "TRAIN_OUTPUT_PATH": str(ckpt), "TRAIN_CTX_DROPOUT": ctx}
            cache = f"python -u -m evaluation.cache_member_probs {name} frozen {ckpt} --features roberta"
            if data == "mceiu":
                env.update({"TRAIN_AUDIO_PATH": str(mceiu_audio), "TRAIN_AUDIO_DIM": "1024"})
                cache += f" --audio-path {mceiu_audio}"
            entry = {"name": name, "job": job, "data": data, "seed": seed, "ctx_dropout": float(ctx)}
            start = time.time()
            try:
                run("python -u training/train_final.py", cwd=repo, env=env, log=OUT / "logs" / f"{name}.out")
                import torch
                meta = torch.load(ckpt, map_location="cpu", weights_only=False)
                entry.update({k: meta[k] for k in ("epoch", "val_accuracy", "val_weighted_f1", "val_macro_f1")})
                run(cache, cwd=repo, log=OUT / "logs" / f"{name}_cache.out")
                shutil.copy(repo / "logs" / "prob_cache" / f"{name}.pt", OUT / "prob_cache" / f"{name}.pt")
                shutil.copy(ckpt, OUT / "checkpoints" / f"{name}.pt")
                entry["status"] = "ok"
            except Exception as exc:
                entry.update({"status": "failed", "error": str(exc)[:300]})
            entry["minutes"] = round((time.time() - start) / 60, 1)
            summary["runs"].append(entry)
            (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
            print("RUN DONE", json.dumps(entry), flush=True)
    summary["total_minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
