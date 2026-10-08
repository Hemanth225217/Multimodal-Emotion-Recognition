"""Kaggle kernel (CPU): look inside the gated MC-EIU dataset (Hugging Face YulangZhuo/MC-EIU) without downloading the
60 GB of English video: download the English CSV, list the zip archives' members through HTTP range requests, and write
a small report. Needs a Kaggle secret named HF_TOKEN (a read token of an account that accepted the dataset terms);
the token is read from Kaggle's secret store and never printed.
Outputs (/kaggle/working/output): EnglishDialogues.csv, inspect_report.json, zip_members_<n>.txt
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = "YulangZhuo/MC-EIU"
OUT = Path("/kaggle/working/output")


def token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        print("No HF_TOKEN secret attached to this notebook (Add-ons > Secrets). Stopping.", type(exc).__name__)
        sys.exit(0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    subprocess.run("pip install -q remotezip", shell=True, check=True)
    import pandas as pd
    from huggingface_hub import hf_hub_download, hf_hub_url
    from remotezip import RemoteZip

    csv = hf_hub_download(REPO, "EnglishDialogues.csv", repo_type="dataset", token=tok, local_dir=str(OUT))
    df = pd.read_csv(csv)
    report = {"csv_rows": len(df), "columns": list(df.columns), "head": df.head(5).astype(str).to_dict("records")}
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() <= 40:
            report[f"values_{col}"] = df[col].value_counts().to_dict()
    for col in ("video_name", "Dia_No"):
        if col in df.columns:
            report[f"nunique_{col}"] = int(df[col].nunique())
    if {"Begin_timestamp", "End_timestamp"} <= set(df.columns):
        dur = df["End_timestamp"] - df["Begin_timestamp"]
        report["utterance_seconds"] = {"mean": float(dur.mean()), "max": float(dur.max()), "total_hours": float(dur.sum() / 3600)}

    headers = {"Authorization": f"Bearer {tok}"}
    for i, name in enumerate(["English_Dialogues_1.zip", "English_Dialogues_2.zip"], 1):
        try:
            url = hf_hub_url(REPO, name, repo_type="dataset")
            with RemoteZip(url, headers=headers) as z:
                infos = z.infolist()
            lines = [f"{x.file_size}\t{x.compress_size}\t{x.filename}" for x in infos]
            (OUT / f"zip_members_{i}.txt").write_text("\n".join(lines))
            exts = {}
            for x in infos:
                ext = Path(x.filename).suffix.lower()
                exts[ext] = exts.get(ext, 0) + 1
            report[name] = {"members": len(infos), "by_extension": exts, "first": [x.filename for x in infos[:15]],
                            "largest_mb": round(max(x.file_size for x in infos) / 1e6, 1)}
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    (OUT / "inspect_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
