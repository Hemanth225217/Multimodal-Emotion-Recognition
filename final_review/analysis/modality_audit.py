"""Modality-use audit of one fusion checkpoint (the pre-final audit of model B, made repeatable for any checkpoint).

Measures what the pre-final report found for model B (HANDOFF 5.6), so the WavLM "repair" can be judged on the same terms:
  - adaptive modality weights: mean / SD / range per modality, share of utterances where text has the largest weight,
    spread of the mean text weight across gold emotions;
  - single-modality predictions (the full model with the other two inputs zeroed, as the demo does): accuracy, which
    classes they ever predict, share predicted neutral;
  - modality ablation over all 7 input combinations;
  - unused audio signal: on non-neutral utterances, audio-only right while the fused prediction is wrong; and how often
    the fused prediction follows text when text-only and audio-only disagree;
  - agreement of the three single-modality predictions (all agree / two agree / all differ) with fused accuracy.

Default split is VALIDATION, so auditing new models never looks at the test set. Usage (repository root):
  python final_review/analysis/modality_audit.py CHECKPOINT [--audio-path PKL] [--text-path PKL] [--split val] [--out JSON]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import NUM_CLASSES, EMOTION_NAMES, DEVICE  # noqa: E402
from training.dataset import MELDDataset  # noqa: E402
from modules.inference import load_model  # noqa: E402

COMBOS = {
    "text_only": (1, 0, 0), "audio_only": (0, 1, 0), "video_only": (0, 0, 1),
    "text+audio": (1, 1, 0), "text+video": (1, 0, 1), "audio+video": (0, 1, 1), "all": (1, 1, 1),
}
NEUTRAL = EMOTION_NAMES.index("neutral")


@torch.no_grad()
def run(model, dataset, use_speaker):
    preds = {k: [] for k in COMBOS}
    weights, labels = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        t, a, v = (s[k].unsqueeze(0).to(DEVICE, dtype=torch.float32) for k in ("text", "audio", "video"))
        spk = s["speaker_slots"].unsqueeze(0).to(DEVICE, dtype=torch.long) if use_speaker else None
        for name, (ut, ua, uv) in COMBOS.items():
            args = (t if ut else torch.zeros_like(t), a if ua else torch.zeros_like(a), v if uv else torch.zeros_like(v))
            if name == "all":
                logits, w = model(*args, speaker_slots=spk, return_weights=True)
                weights.append(w.reshape(-1, 3).cpu().numpy())
            else:
                logits = model(*args, speaker_slots=spk)
            preds[name].append(logits.reshape(-1, NUM_CLASSES).argmax(1).cpu().numpy())
        labels.append(np.asarray(s["labels"]).reshape(-1))
    return {k: np.concatenate(v) for k, v in preds.items()}, np.concatenate(weights), np.concatenate(labels)


def scores(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0))}


def audit(preds, w, y):
    fused, tp, ap, vp = preds["all"], preds["text_only"], preds["audio_only"], preds["video_only"]
    res = {"n": int(len(y)), "fused": scores(y, fused)}
    res["weights"] = {m: {"mean": float(w[:, k].mean()), "sd": float(w[:, k].std()),
                          "min": float(w[:, k].min()), "max": float(w[:, k].max())}
                      for k, m in enumerate(("text", "audio", "video"))}
    res["text_largest_weight_share"] = float((w.argmax(1) == 0).mean())
    per_emotion = {EMOTION_NAMES[c]: float(w[y == c, 0].mean()) for c in range(NUM_CLASSES) if (y == c).any()}
    res["mean_text_weight_by_gold_emotion"] = per_emotion
    res["text_weight_range_across_emotions"] = float(max(per_emotion.values()) - min(per_emotion.values()))
    res["single_modality"] = {}
    for m, p in (("text", tp), ("audio", ap), ("video", vp)):
        counts = Counter(EMOTION_NAMES[c] for c in p)
        res["single_modality"][m] = {"accuracy": float((p == y).mean()), "classes_predicted": len(counts),
                                     "share_neutral": float((p == NEUTRAL).mean()), "counts": dict(counts.most_common())}
    res["ablation"] = {k: scores(y, p) for k, p in preds.items()}
    nn = y != NEUTRAL
    res["non_neutral_n"] = int(nn.sum())
    res["audio_right_on_non_neutral"] = int(((ap == y) & nn).sum())
    res["audio_right_fused_wrong_non_neutral"] = int(((ap == y) & (fused != y) & nn).sum())
    dis = tp != ap
    res["text_audio_disagree_n"] = int(dis.sum())
    res["fused_follows_text_when_text_audio_disagree"] = float((fused[dis] == tp[dis]).mean()) if dis.any() else None
    res["fused_follows_audio_when_text_audio_disagree"] = float((fused[dis] == ap[dis]).mean()) if dis.any() else None
    distinct = np.array([len({a, b, c}) for a, b, c in zip(tp, ap, vp)])
    res["single_modality_agreement"] = {
        name: {"n": int((distinct == k).sum()), "fused_accuracy": float((fused[distinct == k] == y[distinct == k]).mean())
               if (distinct == k).any() else None}
        for k, name in ((1, "all_three_agree"), (2, "two_agree"), (3, "all_differ"))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--audio-path", default=None)
    ap.add_argument("--text-path", default="meld_features/text_roberta/text_roberta.pkl")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test":
        print("WARNING: auditing on TEST. Only do this for frozen, already-scored models.", flush=True)
    model, meta = load_model(Path(args.checkpoint))
    model.eval()
    use_speaker = "num_speaker_slots" in meta
    dataset = MELDDataset(split=args.split, text_path=args.text_path, use_legacy_audio=True, audio_path=args.audio_path)
    preds, w, y = run(model, dataset, use_speaker)
    res = {"checkpoint": str(args.checkpoint), "audio_path": args.audio_path, "split": args.split, **audit(preds, w, y)}
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
