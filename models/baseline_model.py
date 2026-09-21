"""The original simple bimodal baseline: BiLSTM(text) + BiLSTM(audio) -> concat -> MLP.

fusion_model.py's MultimodalFusionModel evolved in place from this simple
architecture into the tri-modal adaptive-fusion model, so the original class
no longer exists in the repo -- only baseline_model.pt's trained weights do.
This definition was reconstructed by inspecting that checkpoint's exact
parameter names and shapes, so it loads with strict=True.
"""

import torch
import torch.nn as nn


class BaselineModel(nn.Module):

    def __init__(self, text_dim=600, audio_dim=300, hidden_dim=256, num_classes=7):
        super().__init__()

        self.text_encoder = nn.LSTM(text_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_encoder = nn.LSTM(audio_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio):
        text_out, _ = self.text_encoder(text)
        audio_out, _ = self.audio_encoder(audio)
        fused = self.fusion(torch.cat([text_out, audio_out], dim=-1))
        return self.classifier(fused)
