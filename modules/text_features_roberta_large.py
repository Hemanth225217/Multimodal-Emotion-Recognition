"""Extract frozen RoBERTa-large sentence embeddings for MELD utterances.

Third text-feature attempt, after DistilBERT and RoBERTa-base. This matches
AMB-DSGDN's actual text encoder choice (RoBERTa-large, 1024-D, ~355M
parameters) rather than the lighter RoBERTa-base (768-D, ~125M) used
earlier for CPU-time reasons. Output dimension is 1024, not 768 --
config.TEXT_DIM must be updated to 1024 to use these features (the model's
text_encoder LSTM takes text_dim as its input_size).
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT

MELD_CSV_DIR = PROJECT_ROOT / "meld_dataset" / "data" / "MELD"
OUTPUT_DIR = PROJECT_ROOT / "meld_features" / "text_roberta_large"
OUTPUT_PATH = OUTPUT_DIR / "text_roberta_large.pkl"

MODEL_NAME = "roberta-large"
DEVICE = torch.device("cpu")
BATCH_SIZE = 16  # smaller than roberta-base's 32 -- large is ~3x the params
MAX_LENGTH = 64

SPLIT_FILES = [
    ("train", "train_sent_emo.csv"),
    ("dev", "dev_sent_emo.csv"),
    ("test", "test_sent_emo.csv"),
]


@torch.no_grad()
def embed_texts(texts, tokenizer, model):
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)

        output = model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).cpu().numpy()

        embeddings.extend(pooled)
        if start % (BATCH_SIZE * 20) == 0:
            print(f"  {start}/{len(texts)}")

    return embeddings


def build_split_dict(csv_path, tokenizer, model):
    df = pd.read_csv(csv_path)
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    texts = df["Utterance"].astype(str).tolist()
    embeddings = embed_texts(texts, tokenizer, model)

    result = {}
    for dialogue_id, embedding in zip(df["Dialogue_ID"], embeddings):
        key = str(int(dialogue_id))
        result.setdefault(key, []).append(embedding)

    return result


def main():
    print(f"Loading {MODEL_NAME} (frozen -- inference only, no fine-tuning)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"Embedding dimension: {hidden_size}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_dicts = []

    for split_name, csv_name in SPLIT_FILES:
        print(f"\n=== {split_name} ({csv_name}) ===")
        result = build_split_dict(MELD_CSV_DIR / csv_name, tokenizer, model)
        total_utterances = sum(len(v) for v in result.values())
        print(f"{split_name}: {len(result)} dialogues, {total_utterances} utterances")
        split_dicts.append(result)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(split_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
