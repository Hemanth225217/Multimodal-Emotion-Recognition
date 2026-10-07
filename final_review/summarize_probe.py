"""Summarise results/audio_probe/probe_results.json (read-only): the layer profile of every model against the legacy baseline."""
import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parent / "results" / "audio_probe" / "probe_results.json"
r = json.loads(path.read_text())
print("utterances used:", r["n"])
print("probe:", r["probe"])
b = r.get("baseline_legacy")
if b:
    print(f"\nlegacy openSMILE-style 300-D baseline : val acc {b['acc']:.4f}  weighted F1 {b['weighted_f1']:.4f}  macro F1 {b['macro_f1']:.4f}")
for name, m in r["models"].items():
    best = m["best_by_weighted_f1"]
    print(f"\n{name}  ({m['n_layers']} layers x {m['hidden']}-D, extraction {m['extract_seconds']} s)")
    print("  layer: " + " ".join(f"{x['layer']:5d}" for x in m["layers"]))
    print("  wF1  : " + " ".join(f"{x['weighted_f1'] * 100:5.1f}" for x in m["layers"]))
    print("  mF1  : " + " ".join(f"{x['macro_f1'] * 100:5.1f}" for x in m["layers"]))
    print(f"  best layer {best['layer']}: acc {best['acc']:.4f} wF1 {best['weighted_f1']:.4f} mF1 {best['macro_f1']:.4f}")
    t3, am = m["top3_mean"], m["all_layer_mean"]
    print(f"  mean of top-3 layers {m['top3_layers']}: wF1 {t3['weighted_f1']:.4f} mF1 {t3['macro_f1']:.4f} | mean of all layers: wF1 {am['weighted_f1']:.4f} mF1 {am['macro_f1']:.4f}")
