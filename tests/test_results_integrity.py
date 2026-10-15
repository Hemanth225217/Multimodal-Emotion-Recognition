"""Consistency checks on the recorded MELD test results.

The MELD data and model checkpoints are not in the repository, so CI cannot re-score
the models. These tests check what can be checked from the committed files: the
exported headline predictions are well formed and match the scored third look, and
every number in the README results table matches the JSON file it was copied from.
Standard library only.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORING = ROOT / "final_review" / "results" / "final_scoring"
N_TEST = 2610  # MELD test utterances
N_CLASSES = 7


def load(name):
    return json.loads((SCORING / name).read_text(encoding="utf-8"))


def pct(x):
    return round(100 * x, 2)


def readme_row(label):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    line = next(l for l in readme.splitlines() if l.startswith(f"| {label} |"))
    return [float(v) for v in re.findall(r"-?\d+\.\d+", line.split("|", 3)[3])]


class HeadlinePredictionExport(unittest.TestCase):
    def setUp(self):
        self.export = load("third_look_test_predictions.json")
        self.scored = load("TEST_SCORED_meld_THIRD_LOOK.json")

    def test_shape(self):
        self.assertEqual(len(self.export["pred"]), N_TEST)
        self.assertEqual(len(self.export["proba"]), N_TEST)
        self.assertEqual(sum(self.export["sizes"]), N_TEST)

    def test_predictions_are_argmax_of_probabilities(self):
        for i, (pred, row) in enumerate(zip(self.export["pred"], self.export["proba"])):
            self.assertEqual(len(row), N_CLASSES)
            self.assertAlmostEqual(sum(row), 1.0, delta=1e-3, msg=f"row {i}")
            self.assertEqual(pred, max(range(N_CLASSES), key=row.__getitem__), msg=f"row {i}")

    def test_export_matches_scored_third_look(self):
        for metric, value in self.scored["test"].items():
            self.assertAlmostEqual(self.export["metrics"][metric], value, places=9, msg=metric)

    def test_confidence_intervals_contain_point_estimate(self):
        for metric, (low, high) in self.scored["ci95"].items():
            self.assertLess(low, self.scored["test"][metric], metric)
            self.assertLess(self.scored["test"][metric], high, metric)


class ReadmeResultsTable(unittest.TestCase):
    def assertRow(self, label, expected):
        self.assertEqual(readme_row(label), expected, label)

    def test_pre_final_row(self):
        m = load("TEST_SCORED_meld.json")["test"]["reference"]["metrics"]
        self.assertRow("pre-final", [pct(m["accuracy"]), pct(m["weighted_f1"]), pct(m["macro_f1"])])

    def test_first_look_row(self):
        m = load("TEST_SCORED_meld_FIRST_LOOK.json")["test"]["selected"]
        self.assertRow("1 (protocol)", [pct(m["accuracy"]), pct(m["weighted_f1"]), pct(m["macro_f1"])])

    def test_second_look_row(self):
        m = load("TEST_SCORED_meld_SECOND_LOOK.json")["test"]["selected"]
        self.assertRow("2 (disclosed)", [pct(m["accuracy"]), pct(m["weighted_f1"]), pct(m["macro_f1"])])

    def test_headline_row_with_intervals(self):
        s = load("TEST_SCORED_meld_THIRD_LOOK.json")
        m, ci = s["test"], s["ci95"]
        self.assertRow(
            "**3 (disclosed, headline)**",
            [pct(m["accuracy"]), *map(pct, ci["accuracy"]),
             pct(m["weighted_f1"]), *map(pct, ci["weighted_f1"]),
             pct(m["macro_f1"])],
        )

    def test_gain_over_pre_final_as_stated(self):
        d = load("TEST_SCORED_meld_THIRD_LOOK.json")["minus_reference"]["accuracy"]
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        stated = f"(+{pct(d['mean']):.2f} accuracy, interval {pct(d['ci95'][0]):.2f} to +{pct(d['ci95'][1]):.2f})"
        self.assertIn(stated, readme)


if __name__ == "__main__":
    unittest.main()
