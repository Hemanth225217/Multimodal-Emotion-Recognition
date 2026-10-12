"""Build app/frontend/summary.json for the demo's Results tab from the committed result files (no number typed by hand
except the published systems, which are the values verified in their papers; see the README positioning table).
Run from the repository root:  python final_review/tools/make_demo_summary.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "final_review" / "results"


def load(pat):
    return {os.path.splitext(os.path.basename(f))[0]: json.load(open(f))
            for f in glob.glob(str(R / pat), recursive=True) if "summary" not in os.path.basename(f)}


def pct(x):
    return round(100 * x, 2)


def main():
    first = json.load(open(R / "final_scoring" / "TEST_SCORED_meld.json"))
    second_p = R / "final_scoring" / "TEST_SCORED_meld_SECOND_LOOK.json"
    third = json.load(open(R / "final_scoring" / "TEST_SCORED_meld_THIRD_LOOK.json"))
    mceiu = json.load(open(R / "final_scoring" / "TEST_SCORED_mceiu.json"))
    m, t = first["test"]["selected"], third["test"]
    out = {
        "final": {
            # headline = third, disclosed look (stacking chosen by CV for accuracy); all looks are listed in "looks"
            "meld": {"accuracy": pct(t["accuracy"]), "weighted_f1": pct(t["weighted_f1"]), "macro_f1": pct(t["macro_f1"]),
                     "ci_weighted_f1": [pct(x) for x in third["ci95"]["weighted_f1"]],
                     "ci_accuracy": [pct(x) for x in third["ci95"]["accuracy"]],
                     "procedure": "stacking of 17 families (third, disclosed test look)"},
            "first_look": {"accuracy": pct(m["accuracy"]), "weighted_f1": pct(m["weighted_f1"]), "macro_f1": pct(m["macro_f1"]),
                           "members": first["selected"]},
            "prefinal": {k: pct(v) for k, v in first["test"]["reference"]["metrics"].items()},
            "mceiu": {"accuracy": pct(mceiu["test"]["selected"]["accuracy"]), "weighted_f1": pct(mceiu["test"]["selected"]["weighted_f1"]),
                      "macro_f1": pct(mceiu["test"]["selected"]["macro_f1"]),
                      "gain_vs_base_weighted_f1": pct(mceiu["test"]["selected_minus_reference"]["weighted_f1"]["mean"]),
                      "gain_ci": [pct(x) for x in mceiu["test"]["selected_minus_reference"]["weighted_f1"]["ci95"]]},
        },
        "published": [
            ["MCN-CL (2025)", 73.1, None], ["AM²-EmoJE (2024)", 71.98, None], ["DialogueLLM (2023)", 71.90, None],
            ["AMuSE (2024)", 71.32, 73.28], ["InstructERC (2023)", 69.15, None], ["M2FNet (2022)", 67.85, None],
            ["TelME (2024)", 67.37, None], ["GraphSmile (2024)", 66.71, None], ["AMB-DSGDN (2026, base paper)", 66.18, 66.07],
        ],
    }
    if second_p.exists():
        s = json.load(open(second_p))["test"]["selected"]
        out["final"]["second_look"] = {k: pct(v) for k, v in s.items()}
    c3 = {**load("C3_L1/**/c3/*.json"), **load("meld_mceiu_analysis/**/c3/*.json")}
    au = {**load("L1_audit/**/audit/*.json"), **load("meld_mceiu_analysis/**/audit/*.json"), **load("meld_face_analysis/**/audit/*.json")}
    groups = {"MELD (multi-party)": ["L1_control_s42", "L1_control_s1", "L1_control_s2"],
              "MC-EIU (dyadic)": ["mceiu_base_s42", "mceiu_base_s1", "mceiu_base_s2"]}
    cells = [("all_context|all", "All context"), ("other_past|text", "Partner's earlier words"),
             ("all_context|video", "Neighbours' faces"), ("all_context|audio", "Neighbours' voices")]
    out["context"] = {"cells": [c[1] for c in cells],
                      "groups": {g: [round(float(np.mean([c3[r][k]["d_accuracy"] * 100 for r in runs])), 2) for k, _ in cells]
                                 for g, runs in groups.items()}}
    aud = {"MELD, scene video": ["L1_control_s42", "L1_control_s1", "L1_control_s2"],
           "MELD, face video + CRCT": ["meld_face_crct_s42", "meld_face_crct_s1", "meld_face_crct_s2"],
           "MC-EIU": ["mceiu_base_s42", "mceiu_base_s1", "mceiu_base_s2"]}
    out["audit"] = {g: {"text_largest_share": round(float(np.mean([au[r]["text_largest_weight_share"] for r in rs])) * 100, 1),
                        "video_adds": round(float(np.mean([(au[r]["ablation"]["all"]["accuracy"] - au[r]["ablation"]["text+audio"]["accuracy"]) * 100 for r in rs])), 2)}
                    for g, rs in aud.items()}
    hs = json.load(open(R / "analysis" / "headroom_selective.json"))["selective_test"]
    out["reliability"] = {k: {"n": v["n"], "accuracy": round(v["accuracy"] * 100, 1)} for k, v in hs["agreement_levels"].items()}
    (ROOT / "app" / "frontend" / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["final"], indent=1))


if __name__ == "__main__":
    main()
