import os
import pickle
import pandas as pd


PROJECT_ROOT = os.path.abspath(".")

MELD_DATA = os.path.join(
    PROJECT_ROOT,
    "meld_dataset",
    "data",
    "MELD"
)

VISUAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "meld_features",
    "visual"
)


SPLITS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv"
}


print()
print("=" * 70)
print("TRI-MODAL ALIGNMENT VERIFICATION")
print("=" * 70)


for split, csv_name in SPLITS.items():

    print()
    print("-" * 70)
    print(split.upper())

    # --------------------------------------------------------
    # Load official MELD annotations
    # --------------------------------------------------------

    csv_path = os.path.join(
        MELD_DATA,
        csv_name
    )

    df = pd.read_csv(csv_path)

    annotation_keys = {
        f"{int(row['Dialogue_ID'])}_{int(row['Utterance_ID'])}"
        for _, row in df.iterrows()
    }

    print("Annotations:", len(annotation_keys))

    # --------------------------------------------------------
    # Load visual features
    # --------------------------------------------------------

    visual_path = os.path.join(
        VISUAL_ROOT,
        f"{split}_visual.pkl"
    )

    with open(visual_path, "rb") as f:
        visual_data = pickle.load(f)

    visual_keys = set(
        visual_data["features"].keys()
    )

    print("Visual features:", len(visual_keys))

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    missing_visual = annotation_keys - visual_keys

    extra_visual = visual_keys - annotation_keys

    common = annotation_keys & visual_keys

    print("Common utterances:", len(common))
    print("Missing visual:", len(missing_visual))
    print("Extra visual:", len(extra_visual))

    if missing_visual:
        print()
        print("Missing visual keys:")

        for key in sorted(missing_visual):
            print(" ", key)

    if extra_visual:
        print()
        print("Unexpected visual keys:")

        for key in sorted(extra_visual)[:20]:
            print(" ", key)


print()
print("=" * 70)
print("ALIGNMENT VERIFICATION COMPLETE")
print("=" * 70)