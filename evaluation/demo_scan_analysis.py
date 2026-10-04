"""Scan the RUNNING demo API over the whole test set and analyse what model B's single-modality heads, adaptive
weights and disagreement levels actually do.

Start the demo first (python -m uvicorn app.backend.main:app --port 8000, or start_demo.bat), then:

    python evaluation/demo_scan_analysis.py            # about 2 minutes on a laptop CPU

It only needs the standard library: every number comes from the same /predict endpoint the web page uses, so it also
checks that the demo reproduces the offline analyses (fused accuracy, the disagreement table, the ablation accuracies).
Written 2026-10-04; output kept in logs/demo_scan_analysis.out.
"""
import argparse
import json
import statistics
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def call(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf8"))


def predict(base, d, i, text=True, audio=True, video=True):
    return call(base, "/predict", {"dialogue_index": d, "utterance_index": i,
                                   "use_text": text, "use_audio": audio, "use_video": video})


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--toggle-every", type=int, default=5, help="missing-modality check on every k-th utterance")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    health = call(base, "/health")
    examples = call(base, "/examples")
    jobs = [(d["dialogue_index"], i) for d in examples for i in range(d["num_utterances"])]
    print(f"demo: {health}")
    print(f"test utterances: {len(jobs)} in {len(examples)} dialogues\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: predict(base, *j), jobs))
    print(f"scanned in {time.time() - t0:.0f} s\n")

    n = len(rows)
    correct = [r["predicted_emotion"] == r["ground_truth"] for r in rows]
    print(f"fused accuracy: {sum(correct)}/{n} = {pct(sum(correct), n):.3f}%")

    print("\n=== disagreement levels (report Table 12.7) ===")
    for level in ("LOW", "MEDIUM", "HIGH"):
        sel = [k for k, r in enumerate(rows) if r["disagreement_level"] == level]
        ok = sum(correct[k] for k in sel)
        conf = statistics.mean(rows[k]["confidence"] for k in sel)
        print(f"{level:7s} n={len(sel):5d} share={pct(len(sel), n):5.1f}%  accuracy={ok}/{len(sel)} = {pct(ok, len(sel)):6.3f}%  mean confidence={100 * conf:.1f}%")

    print("\n=== single-modality heads ===")
    for m in ("text", "audio", "video"):
        preds = Counter(r["modality_predictions"][m] for r in rows)
        acc = pct(sum(r["modality_predictions"][m] == r["ground_truth"] for r in rows), n)
        print(f"{m:5s}-only accuracy {acc:6.2f}%  predicted classes: {dict(preds.most_common())}")

    print("\n=== what LOW / MEDIUM / HIGH consist of ===")
    low = [r for r in rows if r["disagreement_level"] == "LOW"]
    print(f"LOW predictions by class: {dict(Counter(r['predicted_emotion'] for r in low))}")
    neutral = [k for k, r in enumerate(rows) if r["predicted_emotion"] == "neutral"]
    other = [k for k, r in enumerate(rows) if r["predicted_emotion"] != "neutral"]
    print(f"neutral predictions: {len(neutral)}, {pct(sum(correct[k] for k in neutral), len(neutral)):.1f}% correct; "
          f"non-neutral predictions: {len(other)}, {pct(sum(correct[k] for k in other), len(other)):.1f}% correct")
    for label, idx in (("neutral predictions", neutral), ("non-neutral predictions", other)):
        parts = []
        for level in ("LOW", "MEDIUM", "HIGH"):
            sel = [k for k in idx if rows[k]["disagreement_level"] == level]
            parts.append(f"{level} n={len(sel)} acc={pct(sum(correct[k] for k in sel), len(sel)):.1f}%")
        print(f"  {label} by level: " + "; ".join(parts))

    print("\n=== adaptive weights over the whole test set ===")
    for m in ("text", "audio", "video"):
        v = [r["modality_weights"][m] for r in rows]
        print(f"{m:5s} mean={statistics.mean(v):.3f} sd={statistics.pstdev(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    top = Counter(max(r["modality_weights"], key=r["modality_weights"].get) for r in rows)
    print(f"modality with the largest weight: {dict(top)}")
    dis = [r for r in rows if r["disagreement_level"] != "LOW"]
    follows_text = sum(r["predicted_emotion"] == r["modality_predictions"]["text"] for r in dis)
    print(f"when the modalities disagree ({len(dis)} utterances) the fused prediction equals the text-only prediction in "
          f"{pct(follows_text, len(dis)):.1f}% of cases")

    print("\n=== does audio ever matter? (non-neutral utterances) ===")
    nn = [r for r in rows if r["ground_truth"] != "neutral"]
    audio_ok = [r for r in nn if r["modality_predictions"]["audio"] == r["ground_truth"]]
    rescue = [r for r in audio_ok if r["predicted_emotion"] == r["ground_truth"] and r["modality_predictions"]["text"] != r["ground_truth"]]
    missed = [r for r in audio_ok if r["predicted_emotion"] != r["ground_truth"]]
    print(f"audio-only right on {len(audio_ok)} of {len(nn)}; of these the fused prediction is right while text-only is wrong in "
          f"{len(rescue)} cases, and the fused prediction is wrong in {len(missed)} cases")

    print(f"\n=== weights when a modality is missing (every {args.toggle_every}th utterance) ===")
    sample = jobs[:: args.toggle_every]
    conds = {"all present": (True, True, True), "text zeroed": (False, True, True),
             "audio zeroed": (True, False, True), "video zeroed": (True, True, False)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, (t, a, v) in conds.items():
            out = list(pool.map(lambda j: predict(base, j[0], j[1], t, a, v), sample))
            w = {m: statistics.mean(r["modality_weights"][m] for r in out) for m in ("text", "audio", "video")}
            acc = pct(sum(r["predicted_emotion"] == r["ground_truth"] for r in out), len(out))
            print(f"{name:13s} (n={len(out)}) mean weights text/audio/video = {w['text']:.3f}/{w['audio']:.3f}/{w['video']:.3f}   accuracy on this sample {acc:.1f}%")

    lat = []
    for k in range(10):
        s = time.time()
        predict(base, 30 + k, 0)
        lat.append(1000 * (time.time() - s))
    lat.sort()
    print(f"\nsingle /predict latency on this machine (after warm-up): median {lat[len(lat) // 2]:.0f} ms, max {lat[-1]:.0f} ms")


if __name__ == "__main__":
    main()
