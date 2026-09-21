import os
import pickle


ROOT = r".\meld_features\visual"

FILES = [
    "train_visual.pkl",
    "dev_visual.pkl",
    "test_visual.pkl"
]


print()
print("=" * 70)
print("VERIFYING VISUAL FEATURES")
print("=" * 70)


for filename in FILES:

    path = os.path.join(ROOT, filename)

    print()
    print("-" * 70)
    print(filename)

    if not os.path.exists(path):
        print("ERROR: File does not exist")
        continue

    print("File exists: YES")
    print("File size:", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")

    with open(path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    labels = data["labels"]
    metadata = data["metadata"]

    print("Features:", len(features))
    print("Labels:", len(labels))
    print("Metadata:", len(metadata))
    print("Feature dimension:", data["feature_dim"])
    print("Frames per video:", data["num_frames"])
    print("Missing videos:", len(data["missing_videos"]))
    print("Failed videos:", len(data["failed_videos"]))

    if len(features) > 0:

        first_key = next(iter(features))

        first_feature = features[first_key]

        print("Example key:", first_key)
        print("Example feature shape:", first_feature.shape)
        print("Example feature dtype:", first_feature.dtype)


print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)