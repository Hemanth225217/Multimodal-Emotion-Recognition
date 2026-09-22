import torch
import torch.nn as nn


class MultimodalFusionModel(nn.Module):

    def __init__(
        self,
        text_dim=768,
        audio_dim=300,
        video_dim=512,
        hidden_dim=256,
        num_classes=7,
        num_speaker_slots=10
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        # ============================================================
        # SPEAKER EMBEDDING (dialogue-relative slot, see
        # training/dataset.py and config.NUM_SPEAKER_SLOTS)
        #
        # Optional at the forward() call site: passing speaker_slots=None
        # skips this entirely, so older call sites keep working unchanged.
        # ============================================================

        self.speaker_embedding = nn.Embedding(num_speaker_slots, hidden_dim)

        # ============================================================
        # TEXT ENCODER
        # ============================================================

        self.text_encoder = nn.LSTM(
            input_size=text_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU()
        )

        # ============================================================
        # AUDIO ENCODER
        # ============================================================

        self.audio_encoder = nn.LSTM(
            input_size=audio_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU()
        )

        # ============================================================
        # VIDEO ENCODER
        # ============================================================

        self.video_encoder = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.video_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU()
        )

        # ============================================================
        # MODALITY NORMALIZATION
        # ============================================================

        self.text_norm = nn.LayerNorm(
            hidden_dim
        )

        self.audio_norm = nn.LayerNorm(
            hidden_dim
        )

        self.video_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # TEXT -> AUDIO CROSS-MODAL ATTENTION
        # ============================================================

        self.text_audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.text_audio_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # TEXT -> VIDEO CROSS-MODAL ATTENTION
        # ============================================================

        self.text_video_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.text_video_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # ADAPTIVE MODALITY WEIGHTING
        #
        # Instead of applying the same gate to all modalities,
        # this module predicts THREE independent modality weights:
        #
        #   Text
        #   Audio
        #   Video
        #
        # The weights are normalized using Softmax.
        #
        # Example:
        #
        #   Text  = 0.65
        #   Audio = 0.20
        #   Video = 0.15
        #
        # The weights are learned separately for every utterance.
        # ============================================================

        self.modality_weight_network = nn.Sequential(

            nn.Linear(
                hidden_dim * 3,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                3
            )
        )

        # ============================================================
        # ADAPTIVE FUSION
        #
        # We keep all three modalities after weighting rather than
        # collapsing them into one vector.
        #
        # Weighted Text  : 256
        # Weighted Audio : 256
        # Weighted Video : 256
        #
        # Total           : 768
        # ============================================================

        self.fusion = nn.Sequential(

            nn.Linear(
                hidden_dim * 3,
                hidden_dim * 2
            ),

            nn.LayerNorm(
                hidden_dim * 2
            ),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.25)
        )

        # ============================================================
        # FUSION RESIDUAL PROJECTION
        #
        # Provides a stable residual path from the weighted modalities.
        # ============================================================

        self.fusion_residual = nn.Linear(
            hidden_dim * 3,
            hidden_dim
        )

        self.fusion_residual_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # DIALOGUE CONTEXT ATTENTION
        #
        # Each utterance can attend to other utterances in the
        # same dialogue.
        # ============================================================

        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.context_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # CONTEXT FEED-FORWARD NETWORK
        # ============================================================

        self.context_ffn = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        )

        self.context_ffn_norm = nn.LayerNorm(
            hidden_dim
        )

        # ============================================================
        # AUXILIARY UNIMODAL CLASSIFIERS
        #
        # Each modality's own (pre-fusion) representation gets its own
        # classification head, trained with its own loss alongside the
        # main fused prediction (see training/train_final.py). This gives
        # each encoder a direct supervised signal instead of only ever
        # being judged through the fused output -- one more angle of
        # attack on modality collapse, on top of dropout and the weight
        # cap, following the same principle AMB-DSGDN (2026) uses.
        # Inference-only use of the model can ignore these entirely.
        # ============================================================

        self.text_aux_classifier = nn.Linear(hidden_dim, num_classes)
        self.audio_aux_classifier = nn.Linear(hidden_dim, num_classes)
        self.video_aux_classifier = nn.Linear(hidden_dim, num_classes)

        # ============================================================
        # FINAL CLASSIFIER
        # ============================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        text,
        audio,
        video,
        speaker_slots=None,
        return_weights=False,
        return_aux=False
    ):

        # ============================================================
        # TEXT ENCODING
        # ============================================================

        text_output, _ = self.text_encoder(
            text
        )

        text_feature = self.text_projection(
            text_output
        )

        text_feature = self.text_norm(
            text_feature
        )

        # ============================================================
        # AUDIO ENCODING
        # ============================================================

        audio_output, _ = self.audio_encoder(
            audio
        )

        audio_feature = self.audio_projection(
            audio_output
        )

        audio_feature = self.audio_norm(
            audio_feature
        )

        # ============================================================
        # VIDEO ENCODING
        # ============================================================

        video_output, _ = self.video_encoder(
            video
        )

        video_feature = self.video_projection(
            video_output
        )

        video_feature = self.video_norm(
            video_feature
        )

        # Saved before cross-attention mixes text with audio/video below,
        # so the auxiliary text head is judged on text alone.
        text_feature_pure = text_feature

        # ============================================================
        # TEXT <-> AUDIO CROSS-MODAL ATTENTION
        # ============================================================

        text_audio_feature, _ = self.text_audio_attention(
            query=text_feature,
            key=audio_feature,
            value=audio_feature
        )

        text_feature = self.text_audio_norm(
            text_feature + text_audio_feature
        )

        # ============================================================
        # TEXT <-> VIDEO CROSS-MODAL ATTENTION
        # ============================================================

        text_video_feature, _ = self.text_video_attention(
            query=text_feature,
            key=video_feature,
            value=video_feature
        )

        text_feature = self.text_video_norm(
            text_feature + text_video_feature
        )

        # ============================================================
        # COMBINE MODALITIES FOR WEIGHT PREDICTION
        # ============================================================

        modality_context = torch.cat(
            (
                text_feature,
                audio_feature,
                video_feature
            ),
            dim=-1
        )

        # ============================================================
        # LEARN THREE SEPARATE MODALITY WEIGHTS
        # ============================================================

        modality_scores = self.modality_weight_network(
            modality_context
        )

        modality_weights = torch.softmax(
            modality_scores,
            dim=-1
        )

        # ============================================================
        # EXTRACT INDIVIDUAL MODALITY WEIGHTS
        #
        # Shape:
        # [batch, utterances, 1]
        # ============================================================

        text_weight = modality_weights[
            :, :, 0:1
        ]

        audio_weight = modality_weights[
            :, :, 1:2
        ]

        video_weight = modality_weights[
            :, :, 2:3
        ]

        # ============================================================
        # APPLY ADAPTIVE MODALITY WEIGHTS
        # ============================================================

        weighted_text = (
            text_feature
            * text_weight
        )

        weighted_audio = (
            audio_feature
            * audio_weight
        )

        weighted_video = (
            video_feature
            * video_weight
        )

        # ============================================================
        # WEIGHTED MULTIMODAL REPRESENTATION
        # ============================================================

        weighted_modalities = torch.cat(
            (
                weighted_text,
                weighted_audio,
                weighted_video
            ),
            dim=-1
        )

        # ============================================================
        # MAIN FUSION PATH
        # ============================================================

        fused = self.fusion(
            weighted_modalities
        )

        # ============================================================
        # RESIDUAL FUSION PATH
        # ============================================================

        residual = self.fusion_residual(
            weighted_modalities
        )

        residual = self.fusion_residual_norm(
            residual
        )

        # ============================================================
        # COMBINE MAIN + RESIDUAL FUSION
        # ============================================================

        fused = fused + residual

        # ============================================================
        # SPEAKER EMBEDDING
        #
        # Added before context attention so the attention mechanism can
        # use speaker identity (dialogue-relative slot) as part of what it
        # compares utterances by -- e.g. learning to weigh same-speaker
        # utterances differently from a different speaker's, the way
        # DialogueRNN/DialogueGCN-style speaker states do, without a full
        # graph architecture. Skipped entirely if speaker_slots is None.
        # ============================================================

        if speaker_slots is not None:

            speaker_embed = self.speaker_embedding(
                speaker_slots
            )

            fused = fused + speaker_embed

        # ============================================================
        # DIALOGUE CONTEXT ATTENTION
        # ============================================================

        context_output, _ = self.context_attention(
            query=fused,
            key=fused,
            value=fused
        )

        fused = self.context_norm(
            fused + context_output
        )

        # ============================================================
        # CONTEXT FEED-FORWARD NETWORK
        # ============================================================

        context_refined = self.context_ffn(
            fused
        )

        fused = self.context_ffn_norm(
            fused + context_refined
        )

        # ============================================================
        # FINAL EMOTION CLASSIFICATION
        # ============================================================

        output = self.classifier(
            fused
        )

        # ============================================================
        # OUTPUT
        #
        # [batch, utterances, 7]
        # ============================================================

        if return_aux:

            modality_weights_out = torch.cat(
                (text_weight, audio_weight, video_weight),
                dim=-1
            )

            aux_logits = {
                "text": self.text_aux_classifier(text_feature_pure),
                "audio": self.audio_aux_classifier(audio_feature),
                "video": self.video_aux_classifier(video_feature),
            }

            return output, modality_weights_out, aux_logits

        if return_weights:

            # [batch, utterances, 3] -> (text, audio, video)
            modality_weights_out = torch.cat(
                (text_weight, audio_weight, video_weight),
                dim=-1
            )

            return output, modality_weights_out

        return output