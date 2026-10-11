"""Final scoring for the final review and the paper (rules fixed in final_review/MASTER_PLAN.md before any result).

Input: a candidates file (JSON), frozen and committed before test scoring:
  {"datasets": {"meld": {"reference": {"name": "...", "members": {"FAMILY": ["cache.pt", ...], ...}},
                         "families": {"FAMILY": ["path/to/cache.pt", ...], ...}},
                "mceiu": {...}},
   "commit": "<git hash of the code that produced the caches>"}
A family = one recipe; its member caches (seeds) are averaged first. Ensembles are equal-weight averages of families.

Selection (validation only): greedy forward selection on weighted F1, starting from the best single family, adding the
family that raises validation weighted F1 most, stopping when the gain is below 0.1 points. The unselected average of all
families is reported beside it.

Without --score-test the script prints VALIDATION numbers only. With --score-test it scores the test set once per dataset:
selected ensemble, all-family average, each family, with dialogue-level bootstrap 95 % intervals and a paired bootstrap
against the reference system; it then writes final_review/results/final_scoring/TEST_SCORED_<dataset>.json and refuses to
score that dataset's test set again unless --i-know-this-is-a-second-look is given (which is recorded in the output).

Usage (repository root):
  python final_review/analysis/final_scoring.py CANDIDATES.json                 # validation only (safe)
  python final_review/analysis/final_scoring.py CANDIDATES.json --score-test    # the single test scoring
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.ensemble_from_cache import metrics, bootstrap_ci, paired_bootstrap_diff  # noqa: E402

OUT_DIR = ROOT / "final_review" / "results" / "final_scoring"
MIN_GAIN = 0.001  # 0.1 weighted-F1 points


def load(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return torch.load(p, map_location="cpu", weights_only=False)


def family_probs(paths, split, ref_labels=None):
    caches = [load(p) for p in paths]
    labels = np.asarray(caches[0][f"{split}_labels"])
    for c, p in zip(caches, paths):
        assert np.array_equal(np.asarray(c[f"{split}_labels"]), labels), f"label order differs: {p} ({split})"
    if ref_labels is not None:
        assert np.array_equal(labels, ref_labels), f"label order differs from the other families ({split})"
    probs = np.mean([np.asarray(c[split], dtype=np.float64) for c in caches], axis=0)
    return probs, labels, list(caches[0][f"{split}_sizes"])


def wf1(labels, probs):
    return metrics(labels, probs.argmax(1))["weighted_f1"]


def greedy(fams, labels):
    order = sorted(fams, key=lambda f: wf1(labels, fams[f]), reverse=True)
    chosen = [order[0]]
    best = wf1(labels, fams[order[0]])
    trace = [{"added": order[0], "val_weighted_f1": best}]
    while True:
        cands = [f for f in fams if f not in chosen]
        if not cands:
            break
        scores = {f: wf1(labels, np.mean([fams[g] for g in chosen + [f]], axis=0)) for f in cands}
        f, s = max(scores.items(), key=lambda kv: kv[1])
        if s - best < MIN_GAIN:
            break
        chosen.append(f)
        best = s
        trace.append({"added": f, "val_weighted_f1": s})
    return chosen, trace


def fmt(m):
    return f"{m['accuracy']*100:6.2f} / {m['weighted_f1']*100:6.2f} / {m['macro_f1']*100:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--score-test", action="store_true")
    ap.add_argument("--i-know-this-is-a-second-look", action="store_true")
    args = ap.parse_args()
    raw = Path(args.candidates).read_bytes()
    cand = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()[:16]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"candidates {args.candidates} sha256:{digest} commit {cand.get('commit')}")

    for ds, spec in cand["datasets"].items():
        print(f"\n================ {ds.upper()} ================")
        fams_val, labels_val, sizes_val = {}, None, None
        for f, paths in spec["families"].items():
            fams_val[f], labels_val, sizes_val = family_probs(paths, "val", labels_val)
        chosen, trace = greedy(fams_val, labels_val)
        all_val = np.mean(list(fams_val.values()), axis=0)
        sel_val = np.mean([fams_val[f] for f in chosen], axis=0)
        print("VALIDATION (acc / wF1 / mF1):")
        for f in sorted(fams_val, key=lambda f: -wf1(labels_val, fams_val[f])):
            print(f"  {f:28s} {fmt(metrics(labels_val, fams_val[f].argmax(1)))}  ({len(spec['families'][f])} caches)")
        print(f"  greedy selection: {chosen}")
        print(f"  selected ensemble            {fmt(metrics(labels_val, sel_val.argmax(1)))}")
        print(f"  all-family average           {fmt(metrics(labels_val, all_val.argmax(1)))}")
        result = {"dataset": ds, "candidates_sha256": digest, "commit": cand.get("commit"), "selection_trace": trace,
                  "selected": chosen, "val": {f: metrics(labels_val, p.argmax(1)) for f, p in fams_val.items()},
                  "val_selected": metrics(labels_val, sel_val.argmax(1)), "val_all": metrics(labels_val, all_val.argmax(1))}
        if not args.score_test:
            (OUT_DIR / f"validation_{ds}.json").write_text(json.dumps(result, indent=1, default=float))
            print("  (test not scored: run with --score-test once the candidate list is frozen and committed)")
            continue

        marker = OUT_DIR / f"TEST_SCORED_{ds}.json"
        if marker.exists() and not args.i_know_this_is_a_second_look:
            print(f"  REFUSED: the {ds} test set was already scored ({marker}). Not scoring it again.")
            continue
        fams_test, labels_test, sizes_test = {}, None, None
        for f, paths in spec["families"].items():
            fams_test[f], labels_test, sizes_test = family_probs(paths, "test", labels_test)
        sel = np.mean([fams_test[f] for f in chosen], axis=0).argmax(1)
        allf = np.mean(list(fams_test.values()), axis=0).argmax(1)
        test = {"selected": metrics(labels_test, sel), "all_families": metrics(labels_test, allf),
                "families": {f: metrics(labels_test, p.argmax(1)) for f, p in fams_test.items()}}
        ci_a, ci_w = bootstrap_ci(labels_test, sel, sizes_test)
        test["selected_ci95"] = {"accuracy": [float(x) for x in ci_a], "weighted_f1": [float(x) for x in ci_w]}
        ref = spec.get("reference")
        if ref:
            ref_probs = np.mean([family_probs(p, "test", labels_test)[0] for p in ref["members"].values()], axis=0)
            ref_pred = ref_probs.argmax(1)
            test["reference"] = {"name": ref["name"], "metrics": metrics(labels_test, ref_pred)}
            d = paired_bootstrap_diff(labels_test, sel, ref_pred, sizes_test)
            test["selected_minus_reference"] = {k: {"mean": float(v[0]), "ci95": [float(x) for x in v[1]],
                                                    "p_greater": v[2]} for k, v in d.items()}
        result["test"] = test
        result["scored_at"] = datetime.now().isoformat(timespec="seconds")
        result["second_look"] = bool(marker.exists())
        marker.write_text(json.dumps(result, indent=1, default=float))
        print("TEST (scored once; acc / wF1 / mF1):")
        print(f"  selected ensemble            {fmt(test['selected'])}   95% CI wF1 "
              f"[{test['selected_ci95']['weighted_f1'][0]*100:.2f}, {test['selected_ci95']['weighted_f1'][1]*100:.2f}]")
        print(f"  all-family average           {fmt(test['all_families'])}")
        if ref:
            print(f"  reference: {ref['name']:16s}{fmt(test['reference']['metrics'])}")
            for k, v in test["selected_minus_reference"].items():
                print(f"  selected - reference {k}: {v['mean']*100:+.2f} [{v['ci95'][0]*100:+.2f}, {v['ci95'][1]*100:+.2f}]"
                      f"  P(>0)={v['p_greater']:.3f}")
        print(f"  written: {marker}")


if __name__ == "__main__":
    main()
