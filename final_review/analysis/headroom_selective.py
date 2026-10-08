"""Final-review CPU analyses on the committed probability caches (no training, no GPU).

1. Headroom (lever L15), VALIDATION ONLY: how often at least one ensemble family is right (oracle), how often all are, and
   per class. It bounds what any combination rule (stacking, L11) could gain, so it must not look at test.
2. Selective prediction for the PRE-FINAL HEADLINE ensemble (D + H3 + H3c + H2c + DBc, equal-weight average, the system
   whose test score 67.89 / 66.30 / 46.00 is already reported): risk-coverage curves and AURC on validation and test, for
   two confidence scores fixed in advance: (a) the ensemble's top probability, (b) agreement = share of the five families
   whose own prediction equals the ensemble's. Dialogue-level bootstrap intervals (2,000 resamples).

Run from the repository root:  python final_review/analysis/headroom_selective.py
Writes final_review/results/analysis/headroom_selective.json and risk_coverage.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.ensemble_from_cache import load_cache, metrics, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED  # noqa: E402
from config import EMOTION_NAMES  # noqa: E402

OUT_DIR = ROOT / "final_review" / "results" / "analysis"
FAMILIES = {  # the nine families of the pre-final pool (README / HANDOFF table 5.3)
    "B": ["B"], "D": ["D_orig"], "F": ["F_orig"], "H2fam": ["H2_base", "H2b_s1"], "H3": ["H3_large"],
    "H3c": ["CTX_large_s42", "CTX_large_s1"], "H2c": ["CTXb_s42", "CTXb_s1"], "H17": ["H_run17_regen"],
    "DBc": ["DEB_ctx_s42", "DEB_ctx_s1"],
}
HEADLINE = ["D", "H3", "H3c", "H2c", "DBc"]
EXPECTED_HEADLINE_TEST = (0.6789, 0.6630, 0.4600)
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def family_probs(cache, split):
    out = {}
    for fam, members in FAMILIES.items():
        out[fam] = np.mean([np.asarray(cache[m][split], dtype=np.float64) for m in members], axis=0)
    return out


def split_data(cache, split):
    labels = np.asarray(cache["B"][f"{split}_labels"])
    for name, c in cache.items():
        assert np.array_equal(np.asarray(c[f"{split}_labels"]), labels), f"label order differs in {name} ({split})"
    sizes = list(cache["B"][f"{split}_sizes"])
    assert sum(sizes) == len(labels)
    return labels, sizes, family_probs(cache, split)


def headroom(labels, fams):
    correct = np.stack([fams[f].argmax(1) == labels for f in fams])  # [families, n]
    ens = np.mean([fams[f] for f in fams], axis=0).argmax(1) == labels
    res = {
        "n": int(len(labels)),
        "oracle_any_family_correct": float(correct.any(0).mean()),
        "all_families_correct": float(correct.all(0).mean()),
        "no_family_correct": float((~correct.any(0)).mean()),
        "all_family_average_accuracy": float(ens.mean()),
        "best_single_family_accuracy": float(correct.mean(1).max()),
        "per_class": {},
    }
    for k, name in enumerate(EMOTION_NAMES):
        m = labels == k
        res["per_class"][name] = {
            "n": int(m.sum()),
            "oracle_any": float(correct[:, m].any(0).mean()),
            "all_family_average_recall": float(ens[m].mean()),
        }
    return res


def risk_coverage(correct, score):
    """Sort by score descending; selective risk at each coverage k/n. AURC = mean risk over all k."""
    order = np.argsort(-score, kind="stable")
    c = correct[order].astype(np.float64)
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return risk


def summarize(correct, score):
    risk = risk_coverage(correct, score)
    n = len(correct)
    out = {"aurc": float(risk.mean())}
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        out[f"acc_at_{int(cov * 100)}"] = float(1 - risk[k - 1])
    return out


def oracle_aurc(correct):
    return float(risk_coverage(correct, correct.astype(float)).mean())


def bootstrap(correct, scores, sizes):
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    per_dialogue = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(sizes))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {name: {"aurc": [], "acc_at_80": []} for name in scores}
    diff = []
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = np.concatenate([per_dialogue[i] for i in rng.integers(0, len(sizes), size=len(sizes))])
        a = {}
        for name, s in scores.items():
            r = summarize(correct[idx], s[idx])
            draws[name]["aurc"].append(r["aurc"])
            draws[name]["acc_at_80"].append(r["acc_at_80"])
            a[name] = r["aurc"]
        diff.append(a["agreement"] - a["max_prob"])
    ci = {name: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
          for name, d in draws.items()}
    d = np.asarray(diff)
    ci["aurc_agreement_minus_max_prob"] = {"mean": float(d.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return ci


def selective(labels, sizes, fams):
    probs = np.mean([fams[f] for f in HEADLINE], axis=0)
    pred = probs.argmax(1)
    correct = pred == labels
    member_preds = np.stack([fams[f].argmax(1) for f in HEADLINE])
    agreement = (member_preds == pred).mean(0)
    # ties in agreement (only 6 distinct values) are broken by the top probability, so both scores give a full ranking
    scores = {"max_prob": probs.max(1), "agreement": agreement + 1e-3 * probs.max(1)}
    res = {"metrics": metrics(labels, pred), "random_aurc": float(1 - correct.mean()), "oracle_aurc": oracle_aurc(correct)}
    for name, s in scores.items():
        res[name] = summarize(correct, s)
    res["bootstrap_95ci"] = bootstrap(correct, scores, sizes)
    res["agreement_levels"] = {
        f"{int(round(a * 5))}_of_5": {"n": int((np.isclose(agreement, a)).sum()), "accuracy": float(correct[np.isclose(agreement, a)].mean())}
        for a in sorted(set(np.round(agreement, 6)))
    }
    curves = {name: risk_coverage(correct, s) for name, s in scores.items()}
    return res, curves


def main():
    names = sorted({m for ms in FAMILIES.values() for m in ms})
    cache = load_cache(names)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    labels, sizes, fams = split_data(cache, "val")
    report["headroom_val"] = headroom(labels, fams)
    report["selective_val"], curves_val = selective(labels, sizes, fams)

    labels_t, sizes_t, fams_t = split_data(cache, "test")
    report["selective_test"], curves_test = selective(labels_t, sizes_t, fams_t)
    m = report["selective_test"]["metrics"]
    got = (round(m["accuracy"], 4), round(m["weighted_f1"], 4), round(m["macro_f1"], 4))
    assert got == EXPECTED_HEADLINE_TEST, f"headline not reproduced: {got} vs {EXPECTED_HEADLINE_TEST}"
    print("headline ensemble reproduced on test:", got)

    (OUT_DIR / "headroom_selective.json").write_text(json.dumps(report, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, curves, title in ((axes[0], curves_val, "Validation (1,108)"), (axes[1], curves_test, "Test (2,610)")):
        for name, risk in curves.items():
            cov = np.arange(1, len(risk) + 1) / len(risk)
            ax.plot(cov, 100 * (1 - risk), label={"max_prob": "top probability", "agreement": "member agreement"}[name])
        ax.set_title(title)
        ax.set_xlabel("coverage (share of utterances answered)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy on answered utterances (%)")
    axes[0].legend()
    fig.suptitle("Selective prediction, pre-final headline ensemble")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "risk_coverage.png", dpi=150)

    h = report["headroom_val"]
    print(f"\nHEADROOM (validation, 9 families): any family right {h['oracle_any_family_correct']*100:.2f}%, "
          f"all right {h['all_families_correct']*100:.2f}%, none right {h['no_family_correct']*100:.2f}%, "
          f"average of all {h['all_family_average_accuracy']*100:.2f}%, best single {h['best_single_family_accuracy']*100:.2f}%")
    for name, v in h["per_class"].items():
        print(f"  {name:9s} n={v['n']:4d}  any right {v['oracle_any']*100:6.2f}%  average-of-all recall {v['all_family_average_recall']*100:6.2f}%")
    for split in ("val", "test"):
        s = report[f"selective_{split}"]
        print(f"\nSELECTIVE ({split}): random AURC {s['random_aurc']:.4f}, oracle AURC {s['oracle_aurc']:.4f}")
        for name in ("max_prob", "agreement"):
            r, ci = s[name], s["bootstrap_95ci"][name]
            accs = "  ".join(f"{c}%:{r[f'acc_at_{c}']*100:.1f}" for c in (100, 90, 80, 70, 60, 50))
            print(f"  {name:10s} AURC {r['aurc']:.4f} [{ci['aurc'][0]:.4f}, {ci['aurc'][1]:.4f}]  acc at coverage {accs}")
        d = s["bootstrap_95ci"]["aurc_agreement_minus_max_prob"]
        print(f"  AURC agreement - max_prob: {d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
        print("  by agreement level:", {k: f"n={v['n']} acc={v['accuracy']*100:.1f}%" for k, v in s["agreement_levels"].items()})


if __name__ == "__main__":
    main()
