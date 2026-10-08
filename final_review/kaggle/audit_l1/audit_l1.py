"""Kaggle kernel (CPU only, no GPU quota): modality-use audit, on VALIDATION, of every L1 checkpoint produced by
meld-fusion-audio (control = 300-D audio, wavlm_mean, wavlm_l22; seeds 42, 1, 2), with
final_review/analysis/modality_audit.py. The WavLM pickles are rebuilt exactly as in the training kernel.
Outputs: /kaggle/working/output/audit/<run>.json and audit_summary.json.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/kaggle/temp/repo/final_review/kaggle/fusion_audio")
REPO = Path("/kaggle/temp/repo")
OUT = Path("/kaggle/working/output/audit")


def main():
    import shutil
    if REPO.exists():
        shutil.rmtree(REPO)
    subprocess.run(f"git clone --depth 1 --branch final-review "
                   f"https://github.com/Hemanth225217/Multimodal-Emotion-Recognition.git {REPO}", shell=True, check=True)
    subprocess.run("pip install -q torch_geometric", shell=True, check=True)
    import fusion_audio as fa  # reuse input discovery, file placement and the checked pickle builder
    fa.REPO_DIR = REPO
    test_dir, probe_dir = fa.find_dir("decode_report.json"), fa.find_dir("probe_results.json")
    ckpts = sorted(p for p in fa.INPUT_ROOT.rglob("L1_*.pt") if p.parent.name == "checkpoints")
    print("checkpoints:", [c.name for c in ckpts], flush=True)
    if not ckpts:
        raise SystemExit("no L1 checkpoints found in the meld-fusion-audio output")
    dataset_dir = fa.find_dir("data_emotion.p")
    while dataset_dir.parent != fa.INPUT_ROOT and not any(
            str(p).startswith(str(dataset_dir.parent)) for p in (test_dir, probe_dir)):
        dataset_dir = dataset_dir.parent
    for filename, rel in fa.REQUIRED_FILES.items():
        target = REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sorted(dataset_dir.rglob(filename))[0], target)
    csv_dir = REPO / "meld_dataset/data/MELD"
    audio = {}
    for variant, layers in fa.VARIANTS.items():
        if layers is not None:
            audio[variant] = Path("/kaggle/temp") / f"audio_{variant}.pkl"
            fa.build_audio(layers, test_dir, probe_dir, csv_dir, audio[variant])
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ckpt in ckpts:
        name = ckpt.stem
        variant = next(v for v in fa.VARIANTS if f"L1_{v}_" in name)
        cmd = f"python -u final_review/analysis/modality_audit.py {ckpt} --split val --out {OUT / (name + '.json')}"
        if variant in audio:
            cmd += f" --audio-path {audio[variant]}"
        proc = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:], proc.stderr[-3000:], flush=True)
            summary[name] = {"status": "failed"}
            continue
        r = json.loads((OUT / (name + ".json")).read_text())
        summary[name] = {
            "variant": variant, "val_fused": r["fused"], "weights": r["weights"],
            "text_largest_weight_share": r["text_largest_weight_share"],
            "single_modality": {m: {k: v[k] for k in ("accuracy", "classes_predicted", "share_neutral")}
                                for m, v in r["single_modality"].items()},
            "ablation_accuracy": {k: v["accuracy"] for k, v in r["ablation"].items()},
            "audio_right_fused_wrong_non_neutral": r["audio_right_fused_wrong_non_neutral"],
            "fused_follows_text_when_text_audio_disagree": r["fused_follows_text_when_text_audio_disagree"],
            "single_modality_agreement": r["single_modality_agreement"],
        }
        print(name, json.dumps(summary[name]), flush=True)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
