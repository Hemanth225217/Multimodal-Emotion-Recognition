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
    def __init__(self, model_name="roberta-base", max_length=64, freeze_layers=0):
        """freeze_layers: number of bottom transformer layers (plus the
        embedding layer) to keep frozen, fine-tuning only the top
        (total_layers - freeze_layers) layers. 0 fine-tunes everything
        (the first attempt, run17: tied the frozen baseline on weighted F1
        and made macro F1 worse -- full fine-tuning of all 12 layers on
        ~10K training utterances plausibly overfits/destabilizes the
        minority-class protections faster than it helps). Freezing most of
        the network and adapting only the top layers is the standard fix
        for exactly this symptom: less capacity to overfit, while still
        letting the representation specialize for emotion classification.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size

        if freeze_layers > 0:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            for layer in self.model.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            total_layers = len(self.model.encoder.layer)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"FinetuneTextEncoder: froze embeddings + bottom {freeze_layers}/{total_layers} layers "
                f"({trainable:,}/{total:,} params trainable, {100*trainable/total:.1f}%)"
            )

    def forward(self, texts, device, speaker_slots=None):
        """texts: list[str], one dialogue's utterances in order.
        Returns [1, U, hidden_size], gradients flow back into self.model.
        speaker_slots is accepted (and ignored) so this class and
        ContextFinetuneTextEncoder are interchangeable at every call site.
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


SLOT_TAGS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ContextFinetuneTextEncoder(FinetuneTextEncoder):
    """Fine-tuned encoder that reads dialogue context *inside* the transformer.

    FinetuneTextEncoder embeds every utterance in isolation and leaves all
    cross-utterance reasoning to the BiLSTM/attention layers stacked on top.
    That is a real limitation on MELD, where many utterances are short and
    ambiguous on their own ("Yeah.", "What?", "Oh."): what a character feels
    is often only recoverable from the line before it or the reply after it.
    The strongest published MELD text models put the neighbouring
    utterances in the transformer's input for exactly this reason.

    For target utterance i this builds
        <s> ctx[i-P] </s> ... ctx[i-1] </s> TARGET </s> ctx[i+1] ... ctx[i+F] </s>
    where every utterance is prefixed with a dialogue-relative speaker tag
    ("A:", "B:", ...; the same relative identities the fusion model's speaker
    embeddings use -- never character names), and mean-pools ONLY the
    target utterance's tokens. Context is dropped farthest-first if the
    sequence would exceed max_total_length, so the target is never
    truncated by its own context. With P = F = 0 this reduces to the
    isolated-utterance encoder (use build_finetune_encoder, which returns
    the original class in that case).
    """

    def __init__(self, model_name="roberta-base", max_length=64, freeze_layers=0,
                 context_past=3, context_future=1, max_total_length=192):
        super().__init__(model_name, max_length=max_length, freeze_layers=freeze_layers)
        self.context_past = context_past
        self.context_future = context_future
        self.max_total_length = max_total_length

    def _utterance_ids(self, texts, slots):
        tokenizer = self.tokenizer
        ids = []
        for text, slot in zip(texts, slots):
            tag = SLOT_TAGS[min(int(slot), len(SLOT_TAGS) - 1)]
            ids.append(tokenizer(
                f"{tag}: {text}", add_special_tokens=False, truncation=True, max_length=self.max_length,
            )["input_ids"])
        return ids

    def _build_sequence(self, utt_ids, i):
        count = len(utt_ids)
        past = list(range(max(0, i - self.context_past), i))
        future = list(range(i + 1, min(count, i + 1 + self.context_future)))

        def total_length(p, f):
            return (1 + sum(len(utt_ids[j]) + 1 for j in p) + len(utt_ids[i]) + 1
                    + sum(len(utt_ids[j]) + 1 for j in f))

        while total_length(past, future) > self.max_total_length and (past or future):
            if past and (not future or len(past) >= len(future)):
                past = past[1:]      # drop the oldest context first
            else:
                future = future[:-1]  # then the most distant future

        ids = [self.tokenizer.cls_token_id]
        for j in past:
            ids += utt_ids[j] + [self.tokenizer.sep_token_id]
        start = len(ids)
        ids += utt_ids[i]
        end = len(ids)
        ids += [self.tokenizer.sep_token_id]
        for j in future:
            ids += utt_ids[j] + [self.tokenizer.sep_token_id]
        return ids, start, end

    def forward(self, texts, device, speaker_slots=None):
        count = len(texts)
        if speaker_slots is None:
            slots = [0] * count
        elif torch.is_tensor(speaker_slots):
            slots = speaker_slots.reshape(-1).tolist()
        else:
            slots = list(speaker_slots)

        utt_ids = self._utterance_ids(texts, slots)
        built = [self._build_sequence(utt_ids, i) for i in range(count)]
        longest = max(len(ids) for ids, _, _ in built)

        input_ids = torch.full((count, longest), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((count, longest), dtype=torch.long)
        target_mask = torch.zeros((count, longest), dtype=torch.float)
        for row, (ids, start, end) in enumerate(built):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :len(ids)] = 1
            target_mask[row, start:end] = 1.0

        output = self.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))

        mask = target_mask.to(device).unsqueeze(-1)
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).unsqueeze(0)  # [1, U, hidden_size]


def build_finetune_encoder(model_name, freeze_layers=0, context_past=0, context_future=0, max_total_length=192):
    """Isolated-utterance encoder when no context window is requested (the
    original behaviour, so every existing checkpoint loads unchanged),
    context-aware encoder otherwise."""
    if context_past > 0 or context_future > 0:
        return ContextFinetuneTextEncoder(
            model_name, freeze_layers=freeze_layers, context_past=context_past,
            context_future=context_future, max_total_length=max_total_length,
        )
    return FinetuneTextEncoder(model_name, freeze_layers=freeze_layers)


def build_encoder_from_checkpoint(checkpoint):
    """Rebuild the right encoder class for a saved fine-tuned checkpoint.
    Checkpoints that predate context support lack these keys and default to
    the isolated-utterance encoder."""
    return build_finetune_encoder(
        checkpoint["text_model_name"],
        context_past=checkpoint.get("context_past", 0),
        context_future=checkpoint.get("context_future", 0),
        max_total_length=checkpoint.get("max_total_length", 192),
    )
