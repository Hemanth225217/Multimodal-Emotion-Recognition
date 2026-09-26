"""Graph attention fusion layer -- our approximation of AMB-DSGDN's dynamic
semantic graph differential network, and a real (not scalar-hack) version of
relational speaker modeling.

Design (matches the graph pattern used by MMGCN and similar multimodal-ERC
graph papers, already cited in the literature survey): for one dialogue of
U utterances, build a graph with 3U nodes (one per utterance per modality)
and three edge types:
  1. Cross-modal, same-utterance: text_i <-> audio_i <-> video_i.
     Lets the three modalities of one utterance exchange information
     directly, richer than a single round of cross-attention.
  2. Same-modality, temporal: utterance_i <-> utterance_{i+1} (window),
     within each modality. Captures how each modality's signal evolves
     over the dialogue.
  3. Same-modality, same-speaker: utterance_i <-> the most recent prior
     utterance by the same speaker (using the existing dialogue-relative
     speaker_slots), within each modality. This is the real graph-based
     replacement for the speaker-relational attention bias that was tried
     earlier (two learned scalars) and failed, collapsing Fear/Disgust
     twice -- an actual graph edge carries far more information than a
     single learned offset added to attention scores.

Graph attention (GATConv) then lets each node aggregate information from
its neighbors with learned attention weights ("differential" in the sense
that attention differentiates which neighbors matter). Output is fed back
into the existing per-modality feature tensors, which then continue through
the model's existing cross-modal attention / adaptive weighting / fusion
pipeline unchanged -- this is additive, not a wholesale replacement, to
keep the already-validated downstream components intact.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphFusionLayer(nn.Module):
    def __init__(self, hidden_dim=256, heads=4, temporal_window=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window

        # GATConv with heads>1 concatenates by default (out_channels * heads);
        # divide so the output dimension matches hidden_dim, keeping this
        # layer a drop-in enrichment of the existing per-modality tensors.
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.gat = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_edge_index(self, num_utterances, speaker_ids, device):
        """Node index convention: node (u, m) -> u * 3 + m, m in {0:text, 1:audio, 2:video}."""
        edges = []

        def node(u, m):
            return u * 3 + m

        for u in range(num_utterances):
            # 1. Cross-modal, same-utterance (fully connected triangle, both directions).
            for m1 in range(3):
                for m2 in range(3):
                    if m1 != m2:
                        edges.append((node(u, m1), node(u, m2)))

            # 2. Same-modality, temporal window (both directions).
            for offset in range(1, self.temporal_window + 1):
                if u + offset < num_utterances:
                    for m in range(3):
                        edges.append((node(u, m), node(u + offset, m)))
                        edges.append((node(u + offset, m), node(u, m)))

        # 3. Same-modality, same-speaker: connect each utterance to the most
        # recent prior utterance by the same speaker (both directions).
        last_seen = {}
        for u in range(num_utterances):
            speaker = speaker_ids[u]
            if speaker in last_seen:
                prev_u = last_seen[speaker]
                for m in range(3):
                    edges.append((node(prev_u, m), node(u, m)))
                    edges.append((node(u, m), node(prev_u, m)))
            last_seen[speaker] = u

        if not edges:
            # Single-utterance dialogue: no edges to add beyond self-loops,
            # which GATConv adds internally by default.
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    def forward(self, text_feature, audio_feature, video_feature, speaker_slots):
        """All feature args: [1, U, hidden_dim]. speaker_slots: [1, U].

        Assumes batch_size=1 (true at every call site in this repo -- see
        the same assumption already documented in fusion_model.py's
        speaker-relational bias code).
        """
        device = text_feature.device
        num_utterances = text_feature.shape[1]
        speaker_ids = speaker_slots[0].detach().cpu().tolist()

        # Interleave the three modalities per utterance into one node
        # sequence: [text_0, audio_0, video_0, text_1, audio_1, video_1, ...]
        stacked = torch.stack(
            [text_feature[0], audio_feature[0], video_feature[0]], dim=1
        )  # [U, 3, hidden_dim]
        node_features = stacked.reshape(num_utterances * 3, self.hidden_dim)  # [3U, hidden_dim]

        edge_index = self._build_edge_index(num_utterances, speaker_ids, device)

        graph_output = self.gat(node_features, edge_index)  # [3U, hidden_dim]
        graph_output = self.norm(node_features + graph_output)  # residual

        graph_output = graph_output.reshape(num_utterances, 3, self.hidden_dim)
        text_out = graph_output[:, 0, :].unsqueeze(0)
        audio_out = graph_output[:, 1, :].unsqueeze(0)
        video_out = graph_output[:, 2, :].unsqueeze(0)

        return text_out, audio_out, video_out
