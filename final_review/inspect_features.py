"""Print the structure of the feature pickles (read-only), so new feature files can be written in the same format."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "meld_features"


def describe(name, obj, depth=0, maxdepth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}{name}: dict with {len(keys)} keys, first keys {keys[:3]}")
        if depth < maxdepth and keys:
            describe(f"[{keys[0]!r}]", obj[keys[0]], depth + 1, maxdepth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} of {len(obj)}")
        if depth < maxdepth and len(obj):
            describe("[0]", obj[0], depth + 1, maxdepth)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape {obj.shape} dtype {obj.dtype}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} {str(obj)[:60]!r}")


for rel, enc in [
    ("MELD.Features.Models/features/audio_emotion.pkl", "latin1"),
    ("audio_wav2vec2/audio_wav2vec2.pkl", None),
    ("text_roberta/text_roberta.pkl", None),
    ("MELD.Features.Models/features/data_emotion.p", "latin1"),
]:
    p = F / rel
    print("=" * 100)
    print(rel, f"({p.stat().st_size / 1e6:.0f} MB)")
    with open(p, "rb") as f:
        obj = pickle.load(f, encoding=enc) if enc else pickle.load(f)
    describe("root", obj)
    if isinstance(obj, (list, tuple)):
        for i, part in enumerate(obj[:4]):
            if isinstance(part, dict):
                n_dia = len(part)
                n_utt = sum(len(v) for v in part.values()) if all(hasattr(v, "__len__") for v in part.values()) else "?"
                print(f"  part {i}: {n_dia} dialogues, {n_utt} utterance entries")
