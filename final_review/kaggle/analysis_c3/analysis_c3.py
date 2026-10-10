"""Kaggle kernel (CPU only, no GPU quota): modality audit + context attribution (C3), on VALIDATION, for every
checkpoint produced by meld-mceiu-train (meld_ctx, mceiu_base, mceiu_ctx; seeds 42, 1, 2).

MELD and MC-EIU each get their own clone of the repository (their data never share a folder); MC-EIU inputs are rebuilt
from the mc-eiu-extract output with tools/mceiu_to_meld_layout.py exactly as in the training kernel (WavLM layer 22).
Outputs (/kaggle/working/output): audit/<run>.json, c3/<run>.json, summary.json.
"""
import json
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git"
INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/output")
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


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def clone(dest):
    if dest.exists():
        shutil.rmtree(dest)
    r = sh(f"git clone --depth 1 --branch final-review {REPO_URL} {dest}")
    if r.returncode:
        raise SystemExit(r.stderr)


def main():
    sh("pip install -q torch_geometric")
    repos = {"meld": Path("/kaggle/temp/repo_meld"), "mceiu": Path("/kaggle/temp/repo_mceiu")}
    for r in repos.values():
        clone(r)
    meld_dir = sorted(INPUT_ROOT.rglob("data_emotion.p"))[0].parent
    while meld_dir.parent != INPUT_ROOT and not list(meld_dir.rglob("train_sent_emo.csv")):
        meld_dir = meld_dir.parent
    for name, rel in MELD_FILES.items():
        t = repos["meld"] / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sorted(meld_dir.rglob(name))[0], t)
    extract_dir = sorted(INPUT_ROOT.rglob("utterances.csv"))[0].parent
    audio = Path("/kaggle/temp/mceiu_audio_l22.pkl")
    r = sh(f"python -u final_review/tools/mceiu_to_meld_layout.py {extract_dir} . {audio} --layers 22", cwd=repos["mceiu"])
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)

    ckpts = sorted(p for p in INPUT_ROOT.rglob("*.pt") if p.parent.name == "checkpoints"
                   and p.stem.startswith(("meld_ctx_", "meld_crct_", "mceiu_base_", "mceiu_ctx_", "mceiu_crct_")))
    print("checkpoints:", [c.name for c in ckpts], flush=True)
    for sub in ("audit", "c3"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    summary = {}
    for ckpt in ckpts:
        name = ckpt.stem
        data = "mceiu" if name.startswith("mceiu") else "meld"
        extra = f" --audio-path {audio}" if data == "mceiu" else ""
        entry = {}
        for tool, sub in (("modality_audit.py", "audit"), ("context_attribution.py", "c3")):
            out = OUT / sub / f"{name}.json"
            r = sh(f"python -u final_review/analysis/{tool} {ckpt} --split val --out {out}{extra}", cwd=repos[data])
            print(name, tool, r.stdout[-1800:], r.stderr[-1200:] if r.returncode else "", flush=True)
            entry[sub] = "ok" if r.returncode == 0 else "failed"
        summary[name] = entry
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
