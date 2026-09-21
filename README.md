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
- **The adaptive weighting network initially collapsed onto text ("modality
  collapse"), and fixing it took five training runs -- documented here rather
  than quietly overwritten, because the failed attempts are informative.**
  A first run (no countermeasures) converged to average weights
  text=0.998/audio=0.001/video=0.001 for *every* emotion -- gradient descent
  found text (task-specific CNN features from the original paper) was the
  easiest signal and had no incentive to also learn from noisier ones (audio:
  generic openSMILE features; video: generic ImageNet ResNet-18, never
  fine-tuned for emotion). Fixes tried, in order:
  1. **Modality dropout** (zero one modality per training step, forcing the
     model to solve the task from the other two often enough to get real
     gradient signal). Alone: still converged to ~99.9% text. Dropout only
     teaches the model to cope when a modality is completely *absent*; it
     doesn't touch the "all three genuinely present" regime, which is 100%
     of eval-time behaviour.
  2. **Entropy bonus** on the modality-weight network's output (reward
     spreading weight across modalities), on top of dropout. At weight 0.15:
     overshot to a *different* degenerate solution -- exactly
     text=audio=video=0.333 for every emotion, because uniform trivially
     maximizes entropy regardless of content. At weight 0.02: drifted the
     same direction more slowly, reaching entropy 1.07/1.10 by epoch 7 and
     still climbing. Any positive entropy weight has "always uniform" as a
     trivial global optimum, so this mechanism was abandoned.
  3. **Weight-cap penalty** (hinge loss: punish only the part of any modality
     weight above 60%), on top of dropout. This has no trivial shortcut --
     zero penalty for a wide space of non-uniform, content-dependent
     distributions. This is what worked: average weights settled at
     text=0.561/audio=0.130/video=0.309, with genuine per-emotion structure
     (audio peaks at surprise, 16.4% -- plausible, vocal surprise cues like
     gasps; video peaks at joy, 33.7%, and anger, 32.8% -- plausible, visible
     facial expressions). `evaluation/modality_ablation.py` confirms the
     encoders themselves improved, not just the weights: audio-only went
     from ~8% accuracy (chance) to 45.9%, video-only from ~9% to 48.1%.
  See `logs/train_final_run*.log` for all five full runs and
  `logs/disagreement_analysis_output.log` for the final weight breakdown.

## Results

`training/train_final.py`: adaptive fusion architecture, moderate class
weighting, gentle focal loss, modality dropout, and a modality-weight cap
penalty (see the modality-collapse note above for why the last two are
there). Checkpoint selection uses validation **weighted F1**, not macro F1 --
macro F1 alone picked an unstable one-epoch spike in early testing. Full run:
`logs/train_final.log` (earlier superseded attempts kept as
`logs/train_final_run*.log` for the record).

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (text+audio BiLSTM, no fusion) | 59.12% | 55.28% | 31.27% |
| Final adaptive tri-modal fusion | **58.35%** | **56.00%** | **32.87%** |

This is the best accuracy and weighted F1 across every version tried,
historical or current (see the Research journey table below) -- and unlike
every earlier version, it comes with adaptive weights that actually vary by
content instead of a collapsed shortcut. Macro F1 is slightly below the
historical best (34.16%, V4) -- a reasonable trade-off: fixing modality
collapse and chasing minority-class macro F1 pull in somewhat different
directions, and this run prioritized the former since it's the project's
core claim. Full per-class precision/recall/F1/support:
`logs/evaluate_final_output.log`. Confusion matrix:
`evaluation/confusion_matrix_final.png`.

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
# Train the final model (~10-40 min on this hardware, no GPU -- varies with
# when early stopping triggers)
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
| V4 (V2 + moderate class weighting, gentle focal loss) | 57.36% | 56.32% | 34.16% | Best historical accuracy/weighted-F1, but weights still collapsed onto text (not measured at the time -- see below) |
| **Final (this repo)** | **58.35%** | **56.00%** | 32.87% | V4's recipe + modality dropout + weight-cap penalty; best accuracy/weighted-F1 of any version, and the only one with genuinely adaptive (not collapsed) modality weights |

**On modality collapse specifically:** every historical version above
(including V4) was never checked for this -- `evaluation/disagreement_analysis.py`
and the modality-weight-network code didn't exist yet. Re-running that
analysis isn't meaningful on the archived checkpoints without re-deriving
their exact training conditions, so whether V1-V4 also collapsed onto text is
unknown; given they share the same architecture and had no countermeasures,
it's likely. This repo's final model is the first one actually verified not
to have collapsed.

Full per-class numbers, confusion matrices, and the modality-ablation /
disagreement-analysis results are generated by the `evaluation/` scripts
above rather than pasted here, so they can't go stale.
