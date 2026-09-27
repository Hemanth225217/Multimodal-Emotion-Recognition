"""Trainable (not frozen) text encoder for end-to-end fine-tuning.

Every text encoder used elsewhere in this project (DistilBERT, RoBERTa-
base, RoBERTa-large) is frozen: embeddings are extracted once, offline, and
never updated during training. The literature comparison in the README
notes that every 2024-2026 system reporting higher MELD numbers than this
project fine-tunes its text encoder end-to-end -- this module is the
project's first attempt at that, made practical now that GPU training
(via Kaggle) makes the extra compute cost of a live forward+backward pass
through a transformer, every training step, actually affordable.

Wraps a HuggingFace AutoModel with gradients enabled, tokenizes a
dialogue's utterances together, and mean-pools each utterance's real
(non-padding) tokens -- the same pooling convention used by every frozen
extraction script in modules/text_features_*.py, so this is a drop-in
replacement for that pipeline stage, not a different design.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FinetuneTextEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", max_length=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

    def forward(self, texts, device):
        """texts: list[str], one dialogue's utterances in order.
        Returns [1, U, hidden_size], gradients flow back into self.model.
        """
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(device)

        output = self.model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts  # [U, hidden_size]

        return pooled.unsqueeze(0)  # [1, U, hidden_size]
