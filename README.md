# Multimodal Emotion Recognition using Fusion Deep Learning

Context-aware **adaptive tri-modal fusion** for conversational emotion recognition on
[MELD](https://affective-meld.github.io/) (text + audio + video, 7 emotion classes).

```
   TEXT              AUDIO             VIDEO
    |                  |                 |
 BiLSTM             BiLSTM            BiLSTM
    |                  |                 |
 project+norm      project+norm      project+norm
    |                  |                 |
    +--- text<->audio cross-attention ---+
    +--- text<->video cross-attention ---+
                       |
          Modality Weight Network (softmax)
        [per-utterance text/audio/video weights]
                       |
          Weighted Fusion + residual projection
                       |
         Dialogue Context Self-Attention + FFN
                       |
                  Classifier (7 classes)
```

The model does not treat every modality equally: a small network predicts a
per-utterance softmax weight for text/audio/video, so (for example) a flat
"I'm fine" said in a trembling voice with a sad expression can be weighted
toward audio/video rather than text. A second research thread analyzes how
often the three modalities *disagree* with each other and how that affects
the fusion model's confidence and correctness (`evaluation/disagreement_analysis.py`).

## Honesty notes (please read before citing numbers from this repo)

- **Features are pre-extracted, not end-to-end.** Text (600-D) and audio
  (300-D) features come from the original MELD baseline paper's released
  `MELD.Features.Models` package, not from fine-tuning BERT/Wav2Vec2 here.
  Video features (512-D) are extracted locally with a frozen ImageNet
  ResNet-18 (8 sampled frames, mean-pooled) because the project's hardware
  (Intel Core Ultra 5 225U, no CUDA GPU) can't run a ViT/Wav2Vec2 extraction
  pipeline in reasonable time. Accurately: *"a multimodal deep-learning
  framework using pre-extracted textual/acoustic representations and locally
  extracted visual representations, followed by BiLSTM temporal encoding,
  cross-modal attention, adaptive fusion, and conversational context
  modelling."*
- **The web demo browses real MELD test examples, not arbitrary uploads.**
  The original text/audio feature extractors (a CNN text encoder and an
  openSMILE config) were never publicly released, so there's no way to map a
  brand-new sentence or audio clip into the same 600-D/300-D feature space
  the model was trained on. Every prediction in the demo is a genuine forward
  pass through the trained model on a real test utterance, including the
  missing-modality toggle (it actually zeroes that modality's input tensor
  and re-runs the model) -- nothing is mocked or pre-computed.
- **The model does not solve all seven emotions equally well.** Fear and
  disgust have very little training data (~2.7% each) and F1 for those
  classes is near zero. See `evaluation/evaluate_final.py` for the full
  per-class breakdown. The project's contribution is the fusion/context/
  adaptive-weighting/disagreement architecture and analysis, not a claim of
  uniformly solved emotion recognition.
- **The adaptive weighting network has collapsed onto text ("modality
  collapse" / "modality laziness").** `evaluation/disagreement_analysis.py`
  shows the learned weights average text=0.998, audio=0.001, video=0.001
  across every emotion -- i.e. the mechanism is not actually adapting
  per-utterance the way it was designed to. `evaluation/modality_ablation.py`
  shows why: text alone already reaches 54.9% accuracy (vs. 56.3% tri-modal),
  while audio-only and audio+video collapse to ~8% (near chance), and video's
  standalone contribution is small (+2 points accuracy over text+audio).
  This is a documented failure mode in multimodal learning, not a bug in this
  code -- gradient descent exploits the strongest modality (text, from the
  original paper's task-specific CNN features) and has no incentive to also
  learn from noisier ones (audio: generic openSMILE features; video: generic
  ImageNet ResNet-18, never fine-tuned for emotion). The standard fix is
  **modality dropout** during training (randomly zero 1-2 modalities on a
  fraction of steps so the model is forced to learn each pathway) -- not yet
  implemented here; flagged as the clear next experiment rather than another
  blind architecture change.

## Results

Ran `training/train_final.py` once (single clean run: random init, adaptive
fusion architecture, moderate class weighting, gentle focal loss -- the
combination that scored best across the baseline -> V1 -> V2 -> V3 -> V4
experimental history kept in `*/legacy/`). Checkpoint selection uses
validation **weighted F1**, not macro F1 -- macro F1 alone picked an unstable
one-epoch spike in early testing (see git history / commit message for
`train_final.py` if curious). Full run: `logs/train_final.log`.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (text+audio BiLSTM, no fusion) | 59.12% | 55.28% | 31.27% |
| Final adaptive tri-modal fusion | 56.32% | 55.55% | 33.84% |

Full per-class precision/recall/F1/support: `logs/evaluate_final_output.log`.
Confusion matrix: `evaluation/confusion_matrix_final.png`.

## Dataset

[MELD](https://github.com/declare-lab/MELD): 13,708 utterances across 1,433
dialogues from *Friends*, labelled with 7 emotions (neutral, surprise, fear,
sadness, joy, disgust, anger). Train/dev/test = 9,989 / 1,109 / 2,610
utterances. Raw video lives under `meld_dataset/raw/` (not committed --
see `.gitignore`); pre-extracted features live under `meld_features/`.

## Project structure

```
config.py                    Central config: dims, paths, emotion names, seed
models/
  fusion_model.py             Adaptive tri-modal fusion architecture (the model)
  baseline_model.py           Original bimodal baseline (reconstructed from checkpoint)
  final_model.pt               Trained final checkpoint
  baseline_model.pt            Trained baseline checkpoint
  legacy/                      Archived V1-V4 checkpoints + old model stub files
training/
  dataset.py                   MELDDataset: aligns text/audio/video/labels per dialogue
  train_final.py                Single canonical training script -> final_model.pt
  legacy/                      Archived baseline/V2/V3/V4 training scripts
evaluation/
  evaluate_final.py             Overall + per-class metrics + confusion matrix
  evaluate_baseline.py          Same, for the baseline model
  modality_ablation.py          All 7 modality-combination results
  disagreement_analysis.py      Modality agreement/disagreement + adaptive-weight analysis
  metrics.py                    Shared metric helpers
  legacy/                      Archived old evaluation scripts (one had a dead import)
modules/
  data_loader.py                Loads the original MELD.Features.Models pickles
  visual_features.py            ResNet-18 video feature extraction
  inference.py                  Shared checkpoint loading + forward pass (used by app + eval)
  ambiguity.py                  Modality agreement/disagreement scoring
  explanation.py                Human-readable explanation generation
app/
  backend/main.py               FastAPI app (see endpoints below)
  frontend/index.html           Single-page demo UI (dark theme, no build step)
meld_dataset/, meld_features/   Data + features (gitignored, not committed)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Requires MELD raw video under `meld_dataset/raw/` and features under
`meld_features/` (see Dataset section) -- these are large and gitignored.

## Running things

```bash
# Train the final model (~30-40 min on this hardware, no GPU)
python training/train_final.py

# Evaluate
python evaluation/evaluate_final.py
python evaluation/evaluate_baseline.py
python evaluation/modality_ablation.py
python evaluation/disagreement_analysis.py

# Run the web demo (http://127.0.0.1:8000)
python -m uvicorn app.backend.main:app --port 8000
```

## API

- `GET /health` -- model/device status
- `GET /examples` -- list of test-set dialogues available to browse
- `POST /predict` -- `{dialogue_index, utterance_index, use_text, use_audio, use_video}`
  -> prediction, confidence, per-emotion probabilities, adaptive modality
  weights, single-modality predictions, disagreement level, explanation text
- `GET /video/{dialogue_id}/{utterance_id}` -- streams the real MELD clip

## Research journey (kept for the report, not superseded silently)

| Version | Accuracy | Weighted F1 | Macro F1 | Note |
|---|---|---|---|---|
| Baseline (text+audio, no fusion) | 59.12% | 55.28% | 31.27% | Fear/disgust F1 = 0 |
| V1 (+ video, naive fusion) | 55.56% | 55.37% | 33.80% | Adding a modality isn't automatically an improvement |
| V2 (adaptive fusion + context attention) | 57.28% | 56.22% | 33.78% | Architecture in this repo |
| V3 (aggressive minority-class handling) | 51.88% | 53.47% | 34.34% | Forcing minority recall hurt overall accuracy |
| V4 (V2 + moderate class weighting, gentle focal loss) | 57.36% | 56.32% | 34.16% | Best of the historical runs |
| **Final (this repo, single clean run)** | 56.32% | 55.55% | 33.84% | Same recipe as V4, trained from scratch in one run; within ~1 point of V4 on every metric |

Full per-class numbers, confusion matrices, and the modality-ablation /
disagreement-analysis results are generated by the `evaluation/` scripts
above rather than pasted here, so they can't go stale.
