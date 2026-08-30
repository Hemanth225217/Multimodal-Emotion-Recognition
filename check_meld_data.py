from modules.data_loader import load_features


text_data, audio_data, emotion_data = load_features()

records = emotion_data[0]


print("=" * 70)
print("MELD FEATURE / LABEL ALIGNMENT CHECK")
print("=" * 70)


# ============================================================
# FIND FIRST TEST DIALOGUE
# ============================================================

test_record_ids = []

for record in records:

    if record["split"] == "test":

        test_record_ids.append(
            (
                str(record["dialog"]),
                int(record["utterance"]),
                record["y"]
            )
        )


print()
print("FIRST 20 TEST LABEL RECORDS")
print("-" * 70)


for item in test_record_ids[:20]:

    print(
        f"Dialogue={item[0]:>4} | "
        f"Utterance={item[1]:>3} | "
        f"Emotion={item[2]}"
    )


# ============================================================
# CHECK FIRST TEST DIALOGUE
# ============================================================

dialogue_id = test_record_ids[0][0]


print()
print("=" * 70)
print(
    f"FIRST TEST DIALOGUE: {dialogue_id}"
)
print("=" * 70)


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

dialogue_labels = [
    item
    for item in test_record_ids
    if item[0] == dialogue_id
]


print()
print("LABEL RECORDS")
print("-" * 70)


for item in dialogue_labels:

    print(
        f"Utterance {item[1]:>3} "
        f"-> {item[2]}"
    )


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

text_features = text_data[2][dialogue_id]

audio_features = audio_data[2][dialogue_id]


print()
print("FEATURE INFORMATION")
print("-" * 70)


print(
    "Text shape:",
    text_features.shape
)


print(
    "Audio shape:",
    audio_features.shape
)


print(
    "Number of text feature vectors:",
    len(text_features)
)


print(
    "Number of audio feature vectors:",
    len(audio_features)
)


print()
print("=" * 70)
print("ALIGNMENT CHECK COMPLETE")
print("=" * 70)