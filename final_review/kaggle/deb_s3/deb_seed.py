"""Kaggle kernel (GPU): one more seed of the DBc family (DeBERTa-v3-large + context 2/1, bottom 16 of 24 layers frozen),
trained with the code of branch main (the code that trained DBc seeds 42 and 1), then its validation and test
probabilities are cached with evaluation.cache_member_probs (no metric is printed for the test split).
Seed: 3. Outputs (/kaggle/working/output): prob_cache/DEB_ctx_s3.pt, train.out, cache.out.
Added on 12 Oct 2026 after the first test scoring, at the user's request; the extra seeds and the resulting second
test scoring are disclosed in the journal and the report.
"""
import os
import shutil
import subprocess
from pathlib import Path

SEED = "3"
NAME = f"DEB_ctx_s{SEED}"
REPO_URL = "https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git"
REPO = Path("/kaggle/temp/repo")
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")
REQUIRED = {
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
    print("$", cmd, flush=True)
    full = {**os.environ, **(env or {})}
    if log is None:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full)
        return
    with open(log, "w") as f:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=full, stdout=f, stderr=subprocess.STDOUT)
    print(Path(log).read_text()[-2500:], flush=True)
    if p.returncode:
        raise SystemExit(f"failed: {cmd}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prob_cache").mkdir(exist_ok=True)
    data_dir = sorted(INPUT.rglob("data_emotion.p"))[0].parent
    while data_dir.parent != INPUT and not list(data_dir.rglob("train_sent_emo.csv")):
        data_dir = data_dir.parent
    if REPO.exists():
        shutil.rmtree(REPO)
    run(f"git clone --depth 1 --branch main {REPO_URL} {REPO}")
    run("git log --oneline -1", cwd=REPO)
    run("pip install -q torch_geometric sentencepiece")
    for name, rel in REQUIRED.items():
        t = REPO / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sorted(data_dir.rglob(name))[0], t)
    ckpt = f"models/ft_deb_ctx21_s{SEED}.pt"
    env = {"TRAIN_TEXT_MODEL": "microsoft/deberta-v3-large", "TRAIN_CONTEXT_PAST": "2", "TRAIN_CONTEXT_FUTURE": "1",
           "TRAIN_SEED": SEED, "TRAIN_OUTPUT_PATH": ckpt}
    run("python -u -m training.train_finetune", cwd=REPO, env=env, log=OUT / "train.out")
    run(f"python -u -m evaluation.cache_member_probs {NAME} finetune {ckpt}", cwd=REPO, log=OUT / "cache.out")
    shutil.copy(REPO / "logs" / "prob_cache" / f"{NAME}.pt", OUT / "prob_cache" / f"{NAME}.pt")
    print("DONE", NAME, flush=True)


if __name__ == "__main__":
    main()
