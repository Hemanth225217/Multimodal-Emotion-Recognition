"""Kaggle kernel (CPU only, no GPU quota): context attribution (speaker role x direction x modality, contribution C3) on
VALIDATION for the L1 control and WavLM-layer-22 checkpoints (seeds 42, 1, 2) of meld-fusion-audio, with
final_review/analysis/context_attribution.py. Same input discovery and audio pickles as the audit kernel.
Outputs: /kaggle/working/output/c3/<run>.json and c3_summary.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/temp/repo")
OUT = Path("/kaggle/working/output/c3")
RUNS = ("L1_control_", "L1_wavlm_l22_")


def main():
    if REPO.exists():
        shutil.rmtree(REPO)
    subprocess.run(f"git clone --depth 1 --branch final-review "
                   f"https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git {REPO}", shell=True, check=True)
    subprocess.run("pip install -q torch_geometric", shell=True, check=True)
    sys.path.insert(0, str(REPO / "final_review/kaggle/fusion_audio"))
    import fusion_audio as fa
    fa.REPO_DIR = REPO
    test_dir, probe_dir = fa.find_dir("decode_report.json"), fa.find_dir("probe_results.json")
    ckpts = sorted(p for p in fa.INPUT_ROOT.rglob("L1_*.pt") if p.parent.name == "checkpoints" and p.stem.startswith(RUNS))
    print("checkpoints:", [c.name for c in ckpts], flush=True)
    dataset_dir = fa.find_dir("data_emotion.p")
    while dataset_dir.parent != fa.INPUT_ROOT and not any(
            str(p).startswith(str(dataset_dir.parent)) for p in (test_dir, probe_dir)):
        dataset_dir = dataset_dir.parent
    for filename, rel in fa.REQUIRED_FILES.items():
        target = REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sorted(dataset_dir.rglob(filename))[0], target)
    l22 = Path("/kaggle/temp/audio_wavlm_l22.pkl")
    fa.build_audio("22", test_dir, probe_dir, REPO / "meld_dataset/data/MELD", l22)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ckpt in ckpts:
        name = ckpt.stem
        cmd = f"python -u final_review/analysis/context_attribution.py {ckpt} --split val --out {OUT / (name + '.json')}"
        if "wavlm_l22" in name:
            cmd += f" --audio-path {l22}"
        proc = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
        print(name, proc.stdout[-2500:], proc.stderr[-1500:] if proc.returncode else "", flush=True)
        summary[name] = json.loads((OUT / (name + ".json")).read_text()) if proc.returncode == 0 else {"status": "failed"}
    (OUT / "c3_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
