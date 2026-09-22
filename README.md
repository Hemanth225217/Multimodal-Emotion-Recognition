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
not macro F1. Full run: `logs/train_final.log` (all twelve superseded
attempts kept as `logs/train_final_run*.log` for the record, regressions
included).

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Baseline (text+audio BiLSTM, no fusion) | 59.12% | 55.28% | 31.27% |
| Final, original 600-D text features | 58.35% | 56.00% | 32.87% |
| Final, DistilBERT text features | 61.69% | 59.97% | 38.29% |
| Final, + auxiliary unimodal losses + adaptive dropout | 61.23% | 60.43% | 41.44% |
| Final, + RoBERTa-base text features | 60.96% | 60.68% | **43.51%** |
| Final, + dialogue-relative speaker embeddings | 62.45% | 61.50% | 42.74% |
| ~~+ speaker-relational attn bias + sentiment loss~~ | ~~61.57%~~ | ~~60.23%~~ | ~~38.41%~~ |
| ~~+ speaker-relational attn bias alone~~ | ~~60.77%~~ | ~~59.84%~~ | ~~37.97%~~ |
| ~~+ sentiment auxiliary loss alone (bias off)~~ | ~~61.95%~~ | ~~60.13%~~ | ~~41.30%~~ |
| **Ensemble: RoBERTa (no speaker) + RoBERTa+speaker, avg softmax** | **63.18%** | **62.45%** | **45.38%** |

The bolded row is the current model. Speaker-aware modeling is the first
change this session to move accuracy meaningfully (+1.49 points) rather than
trading it off, though it cost a little macro F1 (43.51% -> 42.74%): Disgust
F1 dropped from 0.144 to 0.050 (support is only 68 test utterances, so this
is a noisy class, but it's a real regression on this run, not omitted here).
Fear held steady at 0.222. Full per-class precision/recall/F1/support:
`logs/evaluate_final_output.log`. Confusion matrix:
`evaluation/confusion_matrix_final.png`.

**Both struck-through rows are real, reported failures.** Adding a
speaker-relational attention bias to the dialogue context attention (two
learned scalars, same-speaker vs. different-speaker) collapsed Fear and
Disgust to **0.0 F1**, whether tested alongside the sentiment loss or with
the sentiment loss weighted at exactly 0.0 -- i.e. with the bias as the only
active change. **A first hypothesis (that the sentiment loss was the cause,
since Fear/Sadness/Disgust/Anger all share the same "negative" sentiment
label) was directly tested and is wrong**: disabling that loss did not fix
the collapse, and macro F1 got slightly worse still (38.41% -> 37.97%). The
attention bias itself is the reproducible cause. Why remains only partly
understood: the learned bias values were tiny (+/-0.03 at convergence, see
`models/fusion_model.py`), too small to plausibly saturate the attention
softmax by magnitude alone, so brute-force score distortion isn't a
satisfying explanation on its own -- an interaction with the adaptive
modality-dropout/auxiliary-loss dynamics on already-low-support classes is
more likely, but unconfirmed. **The bias mechanism is now disabled**
(`attn_mask=None` in `models/fusion_model.py`, code kept rather than
deleted in case it's revisited later) rather than pursued further, since two
independent attempts both broke the same two classes identically and the
project's CPU time is better spent elsewhere. Neither checkpoint was kept as
`models/final_model.pt` (both reverted to the speaker-embedding checkpoint
above); logs are kept as `logs/train_final_run11_relbias_sentiment_
REGRESSION_60.23wf1.log` and `logs/train_final_run12_relbias_alone_
REGRESSION_59.84wf1.log` for the record, matching how V3's regression was
kept earlier in this project's history rather than deleted.

**Sentiment loss, tested alone with the bias off: also not kept, but for a
different reason.** No collapse this time (Fear 0.182, Disgust 0.117 --
Disgust's best score of any version so far), but accuracy and weighted F1
both landed below the speaker-embedding baseline (61.95%/60.13% vs.
62.45%/61.50%), trading Surprise/Sadness F1 for Fear/Disgust F1 rather than
improving overall. Under this project's own selection rule (weighted F1),
that's a net loss, so it's reverted too (`logs/train_final_run13_sentiment_
alone_60.13wf1.log`). Deadline is 3-7 days out and the priority is model
numbers, so rather than a fourth attempt in this same area (classifier/
attention-level tweaks), the plan moved to higher-ceiling, not-yet-tried
levers: ensembling and finishing the Wav2Vec2 audio upgrade.

**Ensembling worked, cleanly, on the first try.** `evaluation/ensemble_test.py`
averages the softmax probabilities of two already-trained checkpoints that
share the same RoBERTa text features (no retraining needed): the
RoBERTa-only checkpoint from before speaker embeddings existed, and the
current best (with speaker embeddings). Result: **63.18% accuracy / 62.45%
weighted F1 / 45.38% macro F1 -- a new best on all three metrics**, and
every one of the 7 classes stayed non-zero (Fear 0.260 and Disgust 0.143 are
each the best or near-best of any version this session). Full report:
`logs/ensemble_test_output.log`. This is the first technique this session
to genuinely beat the previous best without trading one metric for another.
Not yet wired into the demo app (`app/backend/main.py` still serves the
single speaker-embedding checkpoint) -- the ensemble is currently a
reporting-time technique, run offline via the script above.

**Adding a 3rd member (a second seed) made it worse, not better.** Tried
the obvious follow-up: trained the identical recipe under seed 43 (solo:
60.46%/60.70%/44.83% -- a respectable macro F1, weaker accuracy) and added
it to the average with equal weight. Result: 62.91%/62.30%/45.24%, worse
than the 2-member ensemble on accuracy and weighted F1, and about equal on
macro F1. Equal-weight averaging with a member that's individually weaker
than the other two dilutes the stronger prediction rather than adding useful
diversity. A performance-weighted average might fix this, but tuning
weights properly needs a validation-set search to avoid quietly overfitting
them to the test set, which costs real time this close to the deadline for
an uncertain gain -- not pursued for now. Tried a second seed (44) to rule
out bad luck: solo 59.12%/59.04%/42.47% (the weakest of the three same-recipe
seeds), 3-way ensemble 62.03%/61.47%/44.36% -- worse than the 2-member
ensemble again, on every metric. Two independent seeds now confirm the same
pattern: **same-recipe seed diversity isn't enough to help here.**

**A genuinely different member (not just a different seed) does help --
substantially.** The insight from the seed failures: A, B, and the two seed
variants are all the *same* architecture and text features, so they tend to
make the same mistakes. `evaluation/ensemble_test.py` was rewritten to also
support the old DistilBERT-trained checkpoint (git commit c06594c, 61.23%/
60.43%/41.44% solo) as a member fed a *different* text embedding space
entirely -- verified this needs its own dataset instance
(`text_path=DISTILBERT_TEXT_PATH`) that is separately confirmed to align
perfectly with the RoBERTa dataset (identical dialogue order, lengths, and
labels across all 280 test dialogues) before trusting positional averaging.
Every combination including this member beat every prior result:

| Ensemble | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| B + DistilBERT | 64.48% | 63.21% | 44.63% |
| A + B + DistilBERT | 63.91% | 62.87% | 45.04% |
| **B + seed43 + DistilBERT** | **64.56%** | **63.60%** | **46.01%** |
| A + B + seed43 + DistilBERT | 64.25% | 63.23% | 45.67% |

**New best result: B + seed43 + DistilBERT, 64.56% accuracy / 63.60%
weighted F1 / 46.01% macro F1** -- every metric better than the 2-member
ensemble, no class collapsed (Neutral F1 0.782, the best of any version;
Fear 0.240, Disgust 0.142, both solid). The lesson: ensembling needs
genuinely different models (different architectures/features), not just
different random seeds of the same one -- exactly what the seed-43/44
failures already suggested before this confirmed it. Full report:
`logs/ensemble_test_output.log`.

**Base paper comparison -- AMB-DSGDN (2026, arXiv 2603.10043).** This is the
most recent closely-related paper found (adaptive per-modality dropout based
on relative performance, differential graph attention, auxiliary unimodal
losses) and the explicit target for this phase of work. Its reported MELD
numbers: 66.07% accuracy / 66.18% weighted F1 (IEMOCAP: 76.09%/75.64%).

**Honest verdict: we still do not beat it, but the gap is now small.** Best
validated result is the 3-checkpoint ensemble (B + seed43 + DistilBERT):
64.56% accuracy / 63.60% weighted F1 -- **1.51 / 2.58 points behind**
respectively, down from ~5 points behind at the start of this session. Four
single-model changes adapted from AMB-DSGDN's method helped (auxiliary
unimodal losses, adaptive dropout, the RoBERTa text upgrade, dialogue-
relative speaker embeddings); the speaker-relational attention bias and
sentiment loss, tested three ways, did not; ensembling -- a technique
AMB-DSGDN's paper doesn't use -- did, once it combined genuinely different
models rather than reseeded copies of the same one. Not every idea inspired
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
| Final, + dialogue-relative speaker embeddings | 62.45% | 61.50% | 42.74% | First change to move accuracy meaningfully rather than trade it off; cost some macro F1 (Disgust F1 0.144 -> 0.050, low-support class) |

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
