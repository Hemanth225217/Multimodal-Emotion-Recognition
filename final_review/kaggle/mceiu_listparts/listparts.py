"""Kaggle kernel (CPU, seconds): list what the last mc-eiu-extract run left in its output (were the per-archive part
files saved when the run hit the 12-hour limit?) and check that each part file loads."""
import json
from pathlib import Path
import numpy as np

found = sorted(p for p in Path("/kaggle/input").rglob("*") if p.is_file())
report = {"files": [[str(p), round(p.stat().st_size / 1e6, 1)] for p in found]}
for p in found:
    if p.suffix == ".npz":
        z = np.load(p)
        report[p.name] = {k: list(z[k].shape) for k in z.files}
print(json.dumps(report, indent=1))
Path("/kaggle/working/listing.json").write_text(json.dumps(report, indent=1))
