import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):

    def __init__(
        self,
        text_dim=600,
        audio_dim=300,
        hidden_dim=256,
        num_classes=7
    ):

        super().__init__()

        # ------------------------------------------
        # Text encoder
        # ------------------------------------------

        self.text_encoder = nn.LSTM(
            input_size=text_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # ------------------------------------------
        # Audio encoder
        # ------------------------------------------

        self.audio_encoder = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Because the LSTMs are bidirectional:
        #
        # hidden_dim = 256
        # forward  = 256
        # backward = 256
        #
        # total = 512
        #
        # Text = 512
        # Audio = 512
        # Combined = 1024

        # ------------------------------------------
        # Fusion layer
        # ------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                hidden_dim * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3)
        )

        # ------------------------------------------
        # Emotion classifier
        # ------------------------------------------

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, text, audio):

        # ------------------------------------------
        # Text features
        # ------------------------------------------

        text_output, _ = self.text_encoder(text)

        # IMPORTANT:
        # Keep ALL utterances.
        #
        # Shape:
        # [batch, utterances, 512]

        text_feature = text_output

        # ------------------------------------------
        # Audio features
        # ------------------------------------------

        audio_output, _ = self.audio_encoder(audio)

        # IMPORTANT:
        # Keep ALL utterances.
        #
        # Shape:
        # [batch, utterances, 512]

        audio_feature = audio_output

        # ------------------------------------------
        # Combine text + audio
        # ------------------------------------------

        combined = torch.cat(
            (
                text_feature,
                audio_feature
            ),
            dim=2
        )

        # Shape:
        # [batch, utterances, 1024]

        # ------------------------------------------
        # Fusion
        # ------------------------------------------

        fused = self.fusion(combined)

        # Shape:
        # [batch, utterances, 256]

        # ------------------------------------------
        # Emotion prediction
        # ------------------------------------------

        output = self.classifier(fused)

        # Shape:
        # [batch, utterances, 7]

        return output