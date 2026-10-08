"""Context attribution by speaker role x direction x modality (contribution C3 / N1), for one fusion checkpoint.

For every target utterance in a dialogue, one context CELL is hidden at a time, in one MODALITY, and the target's
prediction is compared with the unmasked one:
  cells       same-speaker past, other-speaker past, same-speaker future, other-speaker future, all context
  modalities  text, audio, video, all three
"Hidden" = that modality's input vector set to zero for the utterances of the cell (the target itself is never masked),
exactly the way the project's modality ablation hides a modality. Speaker identity = the dataset's dialogue-relative
speaker slot.

Reported per (cell, modality): number of targets that have such context, change in accuracy and weighted F1 over those
targets, mean change in the log-probability of the gold class, and how often the predicted class flips.
Caveat to report: a model never trained with hidden context sees out-of-distribution inputs; the planned remedy is a
model trained with context-modality dropout (same masks at training time).

Default split is VALIDATION. Usage (repository root):
  python final_review/analysis/context_attribution.py CHECKPOINT [--audio-path PKL] [--split val] [--out JSON]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import NUM_CLASSES, DEVICE  # noqa: E402
from training.dataset import MELDDataset  # noqa: E402
from modules.inference import load_model  # noqa: E402

CELLS = ["same_past", "other_past", "same_future", "other_future", "all_context"]
MODALITIES = {"text": (1, 0, 0), "audio": (0, 1, 0), "video": (0, 0, 1), "all": (1, 1, 1)}


def cell_mask(slots, t, cell):
    n = len(slots)
    idx = np.arange(n)
    same = slots == slots[t]
    sel = {
        "same_past": (idx < t) & same, "other_past": (idx < t) & ~same,
        "same_future": (idx > t) & same, "other_future": (idx > t) & ~same,
        "all_context": idx != t,
    }[cell]
    return sel


@torch.no_grad()
def run(model, dataset, use_speaker):
    base_logp, base_pred, golds, dia = [], [], [], []
    rec = {(c, m): {"has": [], "logp": [], "pred": []} for c in CELLS for m in MODALITIES}
    for i in range(len(dataset)):
        s = dataset[i]
        t_in, a_in, v_in = (s[k].to(DEVICE, dtype=torch.float32) for k in ("text", "audio", "video"))
        slots_t = s["speaker_slots"].to(DEVICE, dtype=torch.long)
        slots = s["speaker_slots"].cpu().numpy()
        y = np.asarray(s["labels"]).reshape(-1)
        n = len(y)
        spk = slots_t.unsqueeze(0) if use_speaker else None
        lp = torch.log_softmax(model(t_in[None], a_in[None], v_in[None], speaker_slots=spk).reshape(n, NUM_CLASSES), -1)
        base_logp.extend(lp[torch.arange(n), torch.as_tensor(y)].cpu().numpy())
        base_pred.extend(lp.argmax(1).cpu().numpy())
        golds.extend(y)
        dia.extend([i] * n)
        # one batch per (cell, modality): row t of the batch hides the cell of target t
        for c in CELLS:
            masks = np.stack([cell_mask(slots, t, c) for t in range(n)])  # [targets, utterances]
            has = masks.any(1)
            mk = torch.as_tensor(masks, device=DEVICE, dtype=torch.float32).unsqueeze(-1)  # 1 = hide
            for m, (mt, ma, mv) in MODALITIES.items():
                tt = t_in[None].repeat(n, 1, 1) * (1 - mk * mt)
                aa = a_in[None].repeat(n, 1, 1) * (1 - mk * ma)
                vv = v_in[None].repeat(n, 1, 1) * (1 - mk * mv)
                sp = slots_t[None].repeat(n, 1) if use_speaker else None
                out = torch.log_softmax(model(tt, aa, vv, speaker_slots=sp), -1)  # [n targets, n utts, C]
                diag = out[torch.arange(n), torch.arange(n)]  # target t's own prediction in row t
                r = rec[(c, m)]
                r["has"].extend(has)
                r["logp"].extend(diag[torch.arange(n), torch.as_tensor(y)].cpu().numpy())
                r["pred"].extend(diag.argmax(1).cpu().numpy())
    return np.array(golds), np.array(base_pred), np.array(base_logp), rec, np.array(dia)


def boot_ci(y, p, b, dia, samples=2000, seed=0):
    """Paired dialogue-level bootstrap of (masked - unmasked) accuracy and weighted F1 over the same targets."""
    groups = [np.flatnonzero(dia == d) for d in np.unique(dia)]
    rng = np.random.default_rng(seed)
    da, dw = [], []
    for _ in range(samples):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        da.append((p[idx] == y[idx]).mean() - (b[idx] == y[idx]).mean())
        dw.append(f1_score(y[idx], p[idx], average="weighted") - f1_score(y[idx], b[idx], average="weighted"))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(da), q(dw)


def summarize(y, bp, blp, rec, dia):
    out = {"n_targets": int(len(y)),
           "base": {"accuracy": float((bp == y).mean()), "weighted_f1": float(f1_score(y, bp, average="weighted"))}}
    for (c, m), r in rec.items():
        has = np.array(r["has"], bool)
        if not has.any():
            continue
        p, lp = np.array(r["pred"])[has], np.array(r["logp"])[has]
        yy, bb, bl = y[has], bp[has], blp[has]
        out[f"{c}|{m}"] = {
            "n": int(has.sum()),
            "d_accuracy": float((p == yy).mean() - (bb == yy).mean()),
            "d_weighted_f1": float(f1_score(yy, p, average="weighted") - f1_score(yy, bb, average="weighted")),
            "d_gold_logprob": float((lp - bl).mean()),
            "flip_rate": float((p != bb).mean()),
        }
        ca, cw = boot_ci(yy, p, bb, dia[has])
        out[f"{c}|{m}"].update({"d_accuracy_ci95": ca, "d_weighted_f1_ci95": cw})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--audio-path", default=None)
    ap.add_argument("--text-path", default="meld_features/text_roberta/text_roberta.pkl")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.split == "test":
        print("WARNING: TEST split. Only for frozen, already-scored models.", flush=True)
    t0 = time.time()
    model, meta = load_model(Path(args.checkpoint))
    model.eval()
    use_speaker = "num_speaker_slots" in meta
    ds = MELDDataset(split=args.split, text_path=args.text_path, use_legacy_audio=True, audio_path=args.audio_path)
    y, bp, blp, rec, dia = run(model, ds, use_speaker)
    res = {"checkpoint": str(args.checkpoint), "split": args.split, "use_speaker": use_speaker, **summarize(y, bp, blp, rec, dia),
           "seconds": round(time.time() - t0)}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"base acc {res['base']['accuracy']*100:.2f}  wF1 {res['base']['weighted_f1']*100:.2f}  ({res['seconds']} s)")
    print(f"{'cell':13s}{'modality':>9s}{'n':>6s}{'dAcc':>8s}{'95% CI':>16s}{'dWF1':>8s}{'95% CI':>16s}{'flip%':>7s}")
    for c in CELLS:
        for m in MODALITIES:
            r = res.get(f"{c}|{m}")
            if r:
                ca, cw = r["d_accuracy_ci95"], r["d_weighted_f1_ci95"]
                print(f"{c:13s}{m:>9s}{r['n']:6d}{r['d_accuracy']*100:8.2f}  [{ca[0]*100:5.2f},{ca[1]*100:5.2f}]"
                      f"{r['d_weighted_f1']*100:8.2f}  [{cw[0]*100:5.2f},{cw[1]*100:5.2f}]{r['flip_rate']*100:7.1f}")


if __name__ == "__main__":
    main()
