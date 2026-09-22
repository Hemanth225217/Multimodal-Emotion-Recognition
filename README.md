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

- **Features are pre-extracted, not end-to-end.** Text (768-D) comes from
  frozen DistilBERT (mean-pooled token embeddings, no fine-tuning --
  `modules/text_features_distilbert.py`). Audio (300-D) still comes from the
  original MELD baseline paper's released `MELD.Features.Models` package
  (openSMILE-style features) -- its extractor was never publicly released, so
  it can't be swapped the same way. Video features (512-D) are extracted
  locally with a frozen ImageNet ResNet-18 (8 sampled frames, mean-pooled)
  because the project's hardware (Intel Core Ultra 5 225U, no CUDA GPU) can't
  run a ViT extraction pipeline in reasonable time. Accurately: *"a
  multimodal deep-learning framework using frozen pretrained text embeddings,
  pre-extracted acoustic features, and locally extracted visual
  representations, followed by BiLSTM temporal encoding, cross-modal
  attention, adaptive fusion, and conversational context modelling."* Text
  was originally also a 2018-era task-specific CNN feature (600-D, same
  package as audio) -- swapping it for DistilBERT lifted accuracy from
  58.35% to 61.69% (see Results); `modules/data_loader.py` keeps the
  original loader available (`load_original_text_features()`) for comparison.
- **The web demo browses real MELD test examples, not arbitrary uploads --
  specifically because of audio, not text anymore.** DistilBERT is a public
  model, so arbitrary new text *could* now be embedded into a compatible
  feature space (not wired up in the API yet). Audio remains the blocker: the
  openSMILE-style extractor config was never published, so a brand-new audio
  clip can't be mapped into the same 300-D space the model was trained on.
  Every prediction in the demo is a genuine forward pass through the trained
  model on a real test utterance, including the missing-modality toggle (it
  actually zeroes that modality's input tensor and re-runs the model) --
  nothing is mocked or pre-computed.
- **The model does not solve all seven emotions equally well.** Fear and
  disgust have very little training data (~2.7% each). Disgust F1 is exactly
  0 across every version of this project, historical or current -- the model
  never once predicts it. Fear F1 was also exactly 0 in every version until
  the DistilBERT text upgrade, which got it to a still-poor-but-nonzero
  0.092 (3/50 correctly classified). See `evaluation/evaluate_final.py` for
  the full per-class breakdown. The project's contribution is the
  fusion/context/adaptive-weighting/disagreement architecture and analysis,
  not a claim of uniformly solved emotion recognition.
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
weighting, gentle focal loss, adaptive modality dropout, a modality-weight
cap penalty, and auxiliary unimodal losses (see the modality-collapse note
above for the dropout/cap history), trained on frozen RoBERTa-base text
features (see below). Checkpoint selection uses validation **weighted F1**,
not macro F1. Full run: `logs/train_final.log` (all eleven superseded
attempts kept as `logs/train_final_run*.log` for the record, regressions
included).

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (text+audio BiLSTM, no fusion) | 59.12% | 55.28% | 31.27% |
| Final, original 600-D text features | 58.35% | 56.00% | 32.87% |
| Final, DistilBERT text features | 61.69% | 59.97% | 38.29% |
| Final, + auxiliary unimodal losses + adaptive dropout | 61.23% | 60.43% | 41.44% |
| Final, + RoBERTa-base text features | 60.96% | 60.68% | **43.51%** |
| **Final, + dialogue-relative speaker embeddings** | **62.45%** | **61.50%** | 42.74% |
| ~~+ speaker-relational attn bias + sentiment loss~~ | ~~61.57%~~ | ~~60.23%~~ | ~~38.41%~~ |

The bolded row is the current model. Speaker-aware modeling is the first
change this session to move accuracy meaningfully (+1.49 points) rather than
trading it off, though it cost a little macro F1 (43.51% -> 42.74%): Disgust
F1 dropped from 0.144 to 0.050 (support is only 68 test utterances, so this
is a noisy class, but it's a real regression on this run, not omitted here).
Fear held steady at 0.222. Full per-class precision/recall/F1/support:
`logs/evaluate_final_output.log`. Confusion matrix:
`evaluation/confusion_matrix_final.png`.

**The struck-through row is a real, reported failure, not hidden.** Adding
the speaker-relational attention bias and sentiment auxiliary loss *together*
made every headline metric worse than speaker embeddings alone, and Fear and
Disgust both collapsed to **0.0 F1** (0 Disgust predictions and 1 Fear
prediction across the entire 2,610-utterance test set) -- worse than the
original un-fixed modality collapse this project spent real effort curing
earlier. Working hypothesis: the sentiment auxiliary loss is the likely
cause, not the attention bias. Fear/Sadness/Disgust/Anger all map to the
same "negative" sentiment label, so a loss term that rewards the *shared*
fused representation for being good at 3-way sentiment gives zero gradient
signal to keep those four classes separated from each other -- actively
working against the auxiliary unimodal losses and class weighting that were
trying to protect exactly those minority classes. This checkpoint was
**not** kept as `models/final_model.pt` (reverted to the speaker-embedding
checkpoint above); the failed run's log is kept as
`logs/train_final_run11_relbias_sentiment_REGRESSION_60.23wf1.log` for the
record, matching how V3's regression was kept earlier in this project's
history rather than deleted. Next step: rerun with the sentiment loss
disabled (weight 0) to isolate whether the attention bias is blameless, per
the same one-variable-at-a-time discipline used to diagnose modality
collapse originally.

**Base paper comparison -- AMB-DSGDN (2026, arXiv 2603.10043).** This is the
most recent closely-related paper found (adaptive per-modality dropout based
on relative performance, differential graph attention, auxiliary unimodal
losses) and the explicit target for this phase of work. Its reported MELD
numbers: 66.07% accuracy / 66.18% weighted F1 (IEMOCAP: 76.09%/75.64%).

**Honest verdict: we still do not beat it.** Best validated model remains
62.45% accuracy / 61.50% weighted F1 -- 3.62 / 4.68 points behind
respectively, down from ~5 points behind two versions ago. Four changes
adapted from AMB-DSGDN's method have helped so far (auxiliary unimodal
losses, adaptive dropout, the RoBERTa text upgrade, dialogue-relative speaker
embeddings); a fifth attempt (this section) did not. Not every idea inspired
by a stronger paper transfers cleanly, and reporting the failures is as much
part of the record as the successes.

**For broader context:** other 2024-2026 systems on this task report
weighted F1 in the 66-74% range (MCN-CL 73.1%, AMuSE ~74%, AM2-EmoJE 71.98%,
TelME 67.37%, AMB-DSGDN 66.18%) -- all fine-tune large pretrained
transformers end-to-end on GPUs, several with graph networks or contrastive
learning. This project's frozen-feature, ~7.1M-parameter architecture
trained entirely on a CPU laptop is closing that gap incrementally rather
than matching their scale outright.

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
  data_loader.py                Loads text (DistilBERT)/audio/label features
  text_features_distilbert.py   One-off: extracts frozen DistilBERT text embeddings
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
# One-time: extract frozen DistilBERT text embeddings (downloads ~270MB
# model weights on first run, then a few minutes of CPU inference)
python modules/text_features_distilbert.py

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
| Final, original text features | 58.35% | 56.00% | 32.87% | V4's recipe + modality dropout + weight-cap penalty (fixes modality collapse) |
| Final, DistilBERT text features | 61.69% | 59.97% | 38.29% | Same as above + frozen DistilBERT text embeddings instead of the 2018 CNN features |
| Final, + auxiliary unimodal losses + adaptive dropout | 61.23% | 60.43% | 41.44% | Same as above + two ideas adapted from AMB-DSGDN (2026); first version with a non-zero F1 on all seven classes (Disgust: 0.0 -> 0.125) |
| Final, + RoBERTa-base text features | 60.96% | 60.68% | 43.51% | Stronger frozen text encoder, in the spirit of AMB-DSGDN's RoBERTa-large; best macro F1 and per-class balance yet, small accuracy trade-off |
| **Final, + dialogue-relative speaker embeddings** | **62.45%** | **61.50%** | 42.74% | First change to move accuracy meaningfully rather than trade it off; cost some macro F1 (Disgust F1 0.144 -> 0.050, low-support class) |

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

## An honest note on what the ablation numbers actually show

`text_only` (60.50% acc / 59.84% weighted F1) and `text+audio` (61.26% /
60.50%) land within about a point of the full `text+audio+video` (61.23% /
60.43%) on these *aggregate* zero-ablation numbers. Read literally, that
could suggest video adds little on its own. That's not the full picture:

- The adaptive weight network uses audio and video meaningfully and
  *differently* depending on content -- e.g. video is weighted highest for
  joy (27.4%) and lowest for sadness (21.1%), a sensible pattern (visible
  facial expression carries more signal for joy than for sadness); audio
  peaks for surprise (28.2%). See `logs/disagreement_analysis_output.log`.
  Text/audio/video now average 50.6%/25.2%/24.1% -- audio and video are
  nearly equal contributors, up from a near-total text monopoly before the
  modality-collapse fix.
- Zero-ablation (feeding a modality all zeros) is a coarse way to measure
  "value" -- it tests whether the network can still function *without* a
  modality, not how much that modality helps on the specific utterances
  where it matters. Averaged across a mostly-text-decidable dataset, a
  modality that meaningfully helps on a minority of hard utterances can
  still show a near-zero or slightly negative aggregate delta.
- The missing-modality robustness itself (the app's toggle, and this
  ablation study existing at all) is part of the project's stated
  contribution, independent of whether it nudges the headline accuracy
  number up or down by a fraction of a point.

Reporting this straight rather than only showing the number that looks best
is more defensible in front of faculty than claiming tri-modal strictly
dominates every single metric, which it does not.
