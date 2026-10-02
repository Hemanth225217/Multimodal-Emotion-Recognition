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

- **Most of the pipeline uses pre-extracted features; only the text encoder
  is ever fine-tuned.** Text comes from pretrained transformers. The
  single-model demo checkpoint (B) uses frozen RoBERTa-base mean-pooled
  embeddings (`modules/text_features_roberta.py`); other ensemble members use
  frozen DistilBERT or RoBERTa-large features; and two members fine-tune
  RoBERTa-base / RoBERTa-large end to end with their bottom layers frozen
  (`training/train_finetune.py`, trained on Kaggle's free T4 GPU). Audio
  (300-D) still comes from the original MELD baseline paper's released
  `MELD.Features.Models` package (openSMILE-style features) -- its extractor
  was never publicly released, so it can't be swapped the same way; a
  Wav2Vec2-base replacement was tried and did not beat it. Video features
  (512-D) are extracted locally with a frozen ImageNet ResNet-18 (8 sampled
  frames, mean-pooled). The laptop (Intel Core Ultra 5 225U, no CUDA GPU)
  trained the early models; fine-tuning and later retraining used Kaggle's
  GPU. Accurately: *"a multimodal deep-learning framework using pretrained
  text encoders (frozen or partially fine-tuned), pre-extracted acoustic
  features, and locally extracted visual representations, followed by
  BiLSTM temporal encoding, cross-modal attention, adaptive fusion, and
  conversational context modelling, with the best results coming from an
  ensemble of such models."* Text was originally also a 2018-era
  task-specific CNN feature (600-D, same package as audio) -- swapping it for
  DistilBERT lifted accuracy from 58.35% to 61.69% (see Results);
  `modules/data_loader.py` keeps the original loader available
  (`load_original_text_features()`) for comparison.
- **The web demo browses real MELD test examples, not arbitrary uploads --
  specifically because of audio, not text anymore.** DistilBERT is a public
  model, so arbitrary new text *could* now be embedded into a compatible
  feature space (not wired up in the API yet). Audio remains the blocker: the
  openSMILE-style extractor config was never published, so a brand-new audio
  clip can't be mapped into the same 300-D space the model was trained on.
  Every prediction in the demo is a genuine forward pass through the trained
  model on a real test utterance, including the missing-modality toggle (it
  actually zeroes that modality's input tensor and re-runs the model) --
  nothing is mocked or pre-computed. The demo serves the single model B
  (62.45% / 61.50%), not the ensemble reported in the Results section.
- **The model does not solve all seven emotions equally well.** Fear and
  disgust have very little training data (~2.7% each) and remain the weak
  classes. In the best ensemble Fear F1 is 0.07 (2 of 50 test utterances
  right) and Disgust F1 is 0.14; individual models sometimes score exactly 0
  on one of them (Fear in the RoBERTa-large fine-tune, Disgust in several
  earlier versions), and two attempts to fix this (a heavier layer freeze and
  stronger class weights) did not help. See `evaluation/evaluate_final.py`
  and the ensemble section for per-class breakdowns. The project's
  contribution is the fusion/context/adaptive-weighting/disagreement
  architecture and analysis, not a claim of uniformly solved emotion
  recognition.
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
not macro F1. Full run: `logs/train_final.log` (all twenty superseded
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
| Ensemble: RoBERTa (no speaker) + RoBERTa+speaker, avg softmax | 63.18% | 62.45% | 45.38% |
| Ensemble: + DistilBERT (B+C+D, seed variant + DistilBERT) | 64.56% | 63.60% | **46.01%** |
| Ensemble: B + DistilBERT + RoBERTa-large (B+D+F) | 65.56% | 63.90% | 44.57% |
| Ensemble: B + DistilBERT + fine-tuned RoBERTa (B+D+H) | 65.98% | 64.56% | 45.28% |
| Ensemble: B+D+F + fine-tuned RoBERTa (B+D+F+H), best of several subsets on test | 66.36% | 64.55% | 44.96% |
| Ensemble: B + fine-tuned RoBERTa-large (B+H3) | 64.98% | 63.76% | 43.83% |
| Ensemble: B+D'+F' (current-recipe retrains) + H3 | 65.56% | 64.16% | 45.27% |
| **Ensemble: B + D + F + H2 + H3, all five, equal weight, no selection** | **66.63%** | **64.83%** | 44.73% |
| Ensemble: D + F + H3, subset chosen on validation, test looked at once | 66.48% | 64.74% | 44.59% |

**Read this table with the confidence intervals in mind.** On 2,610 test
utterances the 95% dialogue-level bootstrap interval for any of the
ensemble rows is about +/-2.1 points on accuracy and +/-2.3 on weighted F1
(the validation-chosen row: accuracy [64.33%, 68.61%], weighted F1
[62.44%, 67.07%]). AMB-DSGDN's reported 66.07% / 66.18% sits *inside* both
intervals, so the honest summary is "statistically indistinguishable from
AMB-DSGDN, with point estimates of +0.4 points on accuracy and -1.4 on
weighted F1" -- not "beats it". What the data does show decisively is that
ensembling helps: the validation-chosen ensemble beats the single model B
by +4.0 accuracy points (95% CI +2.6 to +5.5) and +3.2 weighted-F1 points
(CI +1.8 to +4.8). The two bolded/bottom rows are the numbers to quote: the
five-member average involves no selection at all (every available member,
equal weight), and the last row picks its three members on *validation*
and looks at test once. The row marked "best of several subsets on test"
is the project's earlier headline (66.36%): it is a real measurement, but
it was the maximum over several ensembles scored on the very test set it
is reported on, which biases it upward, and its H member (the full
fine-tune of RoBERTa-base) is not retained as a file. See "Ensemble results
under an honest selection protocol" below for how the bottom rows were
produced.

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

**Per-member temperature calibration (`evaluation/calibrated_ensemble_test.py`),
tried on top of B+seed43+DistilBERT: also didn't help.** Fit a temperature
per member on the validation set (never the test set, to avoid overfitting
a calibration knob to the reported numbers) -- B=1.1, seed43=1.1,
DistilBERT=1.25, all mild -- then averaged the rescaled softmax outputs on
test. Result: 64.37%/63.40%/45.84%, marginally worse than the uncalibrated
average (64.56%/63.60%/46.01%) on every metric. The members were already
reasonably well-calibrated (temperatures close to 1), so there wasn't much
for this to fix. Not pursued further; plain equal-weight averaging of the
three members remains the best result.

**The Wav2Vec2 audio upgrade: a mixed result solo, not a clean win.**
Replaced the original MELD paper's 300-D openSMILE-style audio features
with 768-D frozen Wav2Vec2-base embeddings (`modules/audio_features_wav2vec2.py`)
-- extraction completed cleanly, 0 missing/failed across all 13,706
utterances -- then retrained the exact best recipe (RoBERTa + aux losses +
adaptive dropout + speaker embeddings) on it. Solo result: 62.49% accuracy /
60.85% weighted F1 / 39.41% macro F1. Accuracy is flat versus the 300-D-audio
version of this same recipe (62.45%), but weighted F1 and macro F1 are both
*worse* (61.50% -> 60.85%, 42.74% -> 39.41%) -- Disgust collapsed to 0.0 F1
and Fear dropped to 0.080. A genuinely stronger audio encoder did not
translate into a better solo model here; plausibly the richer 768-D
embedding needs more capacity/tuning in the audio encoder LSTM than a
straight dimension swap gives it, or is simply harder for this architecture
to use well on noisy, music-and-crosstalk-heavy TV-show audio. Reported
honestly rather than declared a win because the extraction itself succeeded.

**Tested as a 5th ensemble member (E) anyway -- it does not help, in any
combination.** DistilBERT's solo score was similarly unremarkable yet added
real ensemble value, so the same test was run for E: all 5 checkpoints
(A/B/C/D/E), every combination that includes B (15 total), full results in
`logs/ensemble_test_output.log`. The highest accuracy of any combination is
B+D+E at 64.71%, but its weighted F1 (63.26%) and macro F1 (44.86%) are both
below B+C+D. **B+C+D remains the best result overall (64.56%/63.60%/46.01%)
-- no combination that includes E beats it on weighted F1.** Unlike
DistilBERT, which was a different *text* embedding space, E's checkpoint
still uses the same text encoder and speaker embeddings as B and C; the only
difference is audio, and apparently that alone doesn't produce assessments
different enough from B/C's to be worth averaging in. Building this test
also surfaced a real bug: `modules/inference.load_model()` always
constructed models with `config.AUDIO_DIM` (768, the new default), so it
crashed loading any pre-upgrade checkpoint (300-D audio) with a state_dict
size mismatch. Fixed by reading dimensions from each checkpoint's own saved
metadata first, falling back to config only when absent.

**Label smoothing (0.1, the standard default): another regression, both
minority classes collapsed.** Tested against the true best recipe (300-D
audio, matching checkpoint B) as a clean single-variable change. Result:
60.42% accuracy / 59.70% weighted F1 / 37.99% macro F1 -- worse than B
(62.45%/61.50%/42.74%) on every metric, with **both** Fear and Disgust at
0.0 F1. This recipe already leans on focal loss, heavy class weights, and
auxiliary losses to protect minority classes; softening the main loss's
target distribution on top of that apparently blunts exactly the signal
those mechanisms need, rather than adding the mild, mostly-harmless
regularization label smoothing usually provides. Reverted (`LABEL_SMOOTHING`
back to 0.0); log kept as
`logs/train_final_run17_labelsmoothing0.1_REGRESSION_59.70wf1.log`.

**Pattern worth noting:** every attempt this session to modify the loss
function or attention mechanism directly on top of the best recipe --
speaker-relational bias (twice), sentiment loss (twice), label smoothing --
has regressed at least one minority class to 0.0 F1. The changes that
*did* work (RoBERTa text, speaker embeddings, ensembling) all added new
information or genuinely different models rather than reshaping how the
existing loss treats existing classes. This recipe's minority-class balance
looks more fragile to loss-level perturbation than to architectural
additions.

**Per-batch adaptive dropout (matching AMB-DSGDN's actual per-batch
granularity, not our earlier epoch-level approximation): a net loss, but no
collapse this time.** 61.65% accuracy / 60.62% weighted F1 / 43.39% macro
F1 -- worse than the best (62.45%/61.50%) on accuracy and weighted F1,
slightly better on macro F1, and every class stayed non-zero (Fear 0.239,
close to the best ever). Under this project's own selection rule (weighted
F1), still a net loss, so reverted. Faster-reacting dropout apparently
isn't the bottleneck here; log kept as
`logs/train_final_run18_perbatch_dropout_60.62wf1.log`.

**RoBERTa-large + graph attention fusion, tested together: a real attempt
at the two biggest architectural gaps, and a real regression.** Extracted
frozen RoBERTa-large (1024-D, matching AMB-DSGDN's actual text encoder,
`modules/text_features_roberta_large.py`) and built a genuine graph-based
relational mechanism (`models/graph_fusion.py`, using `torch_geometric`'s
GATConv over a per-dialogue graph with cross-modal, temporal, and
same-speaker edges) to replace the earlier scalar relational-bias hack.
Both were smoke-tested thoroughly (edge construction checked by hand,
gradients confirmed flowing, old-checkpoint backward compatibility
confirmed) before trusting them with training time. Tested together first,
given a real week of runway rather than the 2 days available when the bias
version was tried: 61.11% accuracy / 60.08% weighted F1 / 38.22% macro F1
-- worse than the best (62.45%/61.50%/42.74%) on every metric, and **Fear
and Disgust both collapsed to 0.0 F1 again** -- the exact same failure
signature as the scalar relational-bias attempts, despite a completely
different (and far more expressive) mechanism for encoding speaker
relationships. Reverted; log kept as
`logs/train_final_run19_robertalarge_graphfusion_REGRESSION_60.08wf1.log`.
Added a `USE_GRAPH_FUSION` toggle to `train_final.py` to isolate the two.

**Isolation result: RoBERTa-large alone is a genuinely mixed result, and
confirms the graph layer is the more likely culprit.** With
`USE_GRAPH_FUSION=False`: 63.41% accuracy / 60.85% weighted F1 / 39.17%
macro F1 (see bug note below -- these are the corrected numbers). Accuracy
is the **best of any single model this entire session** (+0.96 over the
previous best, 62.45%), weighted F1 is close (-0.65), and only **Disgust**
collapsed to 0.0 F1 -- Fear survived (0.114). That's a meaningfully
different (and much less severe) failure than the combined run, where
*both* Fear and Disgust hit exactly 0.0 -- evidence the graph fusion layer,
not the larger text encoder, is the primary driver of the double-collapse.
The distinctly different error pattern (predicts neutral far more often --
58.7% of test utterances vs. the usual ~48-50%) made this worth keeping as
a 6th ensemble candidate; log kept as
`logs/train_final_run20_robertalarge_alone_61.10wf1.log` (filename keeps
the original, pre-fix number for traceability to when it was logged). The
graph fusion layer itself remains disabled and not otherwise pursued
further this session -- two failed relational-modeling mechanisms (a
scalar bias, now a real graph) is a strong enough signal that the problem
is structural to this architecture's fused representation, not either
implementation.

**A real bug was caught here: `load_model()` ignored `use_graph_fusion`
entirely.** It never read or passed this flag when reconstructing a model
from a checkpoint, so every checkpoint loaded through it silently got the
model's own default (`True`) regardless of what it was actually trained
with -- corrupting evaluation for checkpoints A/B/C/D (predate graph fusion,
zero trained weights for it, evaluated with an active random graph layer
scrambling their forward pass) and F (explicitly trained with it off, same
problem). This surfaced because a "new best" ensemble result looked
suspicious enough to double-check. Fixed: `train_final.py` now saves
`use_graph_fusion` in checkpoint metadata, and `load_model()` reads it with
a default of `False` (not the model's own `True` default) for checkpoints
saved before this field existed, since every checkpoint currently in use
either predates the feature or was trained with it off. Verified B and F
both now correctly resolve to `use_graph_fusion=False`, and B's re-evaluated
solo score exactly reproduces its known-correct baseline (62.45%/61.50%/
42.74%), confirming the fix. The bug's actual impact was smaller than it
could have been -- the graph layer has a residual connection, so random
weights added noise rather than destroying the signal -- but it was real,
and every ensemble number below was recomputed after the fix rather than
trusting the first pass.

**With the fix in place: a new best result.** Testing the RoBERTa-large
checkpoint (F) as a 6th ensemble member across every combination with B:
**B+D+F is the new best, at 65.56% accuracy / 63.90% weighted F1 / 44.57%
macro F1** -- beating the previous best (B+C+D, 64.56%/63.60%/46.01%) on
accuracy and weighted F1 (the project's selection metric), though macro F1
is lower. **Gap to AMB-DSGDN (66.07%/66.18%) is now 0.51 / 2.28 points --
the closest this project has been all session**, down from ~5 points at
the start. Full sweep (16 combinations across A/B/C/D/F) in
`logs/ensemble_test_output.log`.

**GPU training pipeline stood up on Kaggle, validated end-to-end.** Set up
a full CLI-driven pipeline (Kaggle account, API token, a packaged 269MB
feature dataset, and a training kernel that clones this repo and runs
`training/train_final.py` unchanged -- it already auto-detects CUDA via
`config.DEVICE`, no code changes needed). Took three iterations to get the
kernel's data-loading right (see git history for the specific bugs -- wrong
assumptions about how Kaggle mounts an uploaded dataset), but the training
itself needed zero changes. Confirmed real speedup: **8.3 minutes total,
~23s/epoch on a Tesla T4, versus 45-180+ seconds/epoch on CPU** -- roughly
a 5-8x wall-clock improvement. Tested the resulting checkpoint (G, same
RoBERTa-large recipe as F) both solo (60.00%/59.98%/42.36%, genuinely
different from F despite identical code and seed -- F was trained on CPU and
G on the GPU, and the different floating-point numerics send the same
seeded run down a different trajectory; this is a CPU-vs-GPU difference,
not run-to-run randomness -- a later retrain on the same GPU reproduced G
bit-for-bit, see the re-verification note below) and as an
8th ensemble candidate: **it doesn't beat B+D+F in any combination**, same
pattern as the seed-43/44 variants -- confirms again that same-recipe
diversity (whether from a different seed or different hardware) isn't
enough for ensembling here, only genuinely different architectures/features
are. The pipeline itself remains available and now iterates 5-8x faster,
which matters most for trying something that needs many fast iterations
(e.g. redesigning the graph fusion layer with room to actually debug it, or
attempting real end-to-end fine-tuning) rather than for re-running the
existing recipe under different randomness.

**End-to-end text fine-tuning, tried for real: essentially ties the frozen
baseline, doesn't beat it.** Built a full fine-tuning pipeline
(`training/train_finetune.py`, `models/finetune_text_encoder.py`,
`training/dataset_finetune.py`) -- a genuinely trainable RoBERTa-base
(verified gradients reach the transformer layers before trusting it with
GPU time), same recipe as checkpoint B/F otherwise (auxiliary losses,
adaptive dropout, speaker embeddings, graph fusion and sentiment loss both
off), trained on Kaggle's GPU with a two-speed optimizer (2e-5 for the
encoder, 3e-4 for the rest). Validation looked genuinely promising --
0.5882 weighted F1 at epoch 7, the best validation score of the entire
session. The real test result: **62.57% accuracy / 61.50% weighted F1 /
39.93% macro F1** -- weighted F1 is *exactly* the same as the frozen
RoBERTa-base baseline (61.50%), accuracy is marginally better (+0.12), and
macro F1 is worse, because **Fear collapsed to 0.0 F1** -- the same
recurring failure pattern seen with the relational bias, sentiment loss,
label smoothing, and graph fusion attempts. This is a genuinely surprising,
disappointing result given the literature comparison's clear implication
that fine-tuning is the main thing separating this project from
higher-scoring 2024-2026 systems. Plausible reasons, none confirmed:
fine-tuning a 125M-parameter encoder on ~10K training utterances may need
more careful regularization or a shorter/warmed-up schedule than the
10-epoch, fixed-LR setup used here to avoid overfitting to majority
classes; or the project's minority-class protections (focal loss, class
weighting, auxiliary losses) may simply not compose well with *any*
sufficiently large change to the text pathway, fine-tuning included. Not
promoted to the main checkpoint; kept as `models/final_model_finetuned.pt`
and `logs/evaluate_finetune_output.log` for the record.

**Follow-up: freezing the bottom 8 of 12 layers fixes the Fear collapse and
gives a small, real win.** The full-fine-tuning result's own writeup named
overfitting the minority-class protections as a plausible cause -- tested
that hypothesis directly by freezing RoBERTa-base's embeddings and bottom 8
transformer layers, leaving only the top 4 (23.2% of the encoder, 28.9M of
124.6M params) trainable (`models/finetune_text_encoder.py`'s
`freeze_layers` param, `FREEZE_LAYERS = 8` in `train_finetune.py`).
Verified the freezing itself was real before spending GPU time: a local
smoke test confirmed frozen parameters are provably unchanged after
`optimizer.step()` while the top layers provably do change. Trained on
Kaggle's GPU (10.3 min, best epoch 9 of 10, val weighted F1 0.6065 -- better
than the full-fine-tune run's 0.5882). Real test result: **62.57% accuracy
/ 61.66% weighted F1 / 42.02% macro F1.** Same accuracy as the full-tuning
attempt, but weighted F1 is now *higher* than the frozen baseline (61.50%)
for the first time any fine-tuning variant has beaten it, and **Fear no
longer collapses** (F1 0.0 -> 0.126, recall 0.0 -> 0.14) -- direct evidence
for the overfitting hypothesis, though modest in size. Also re-tested
paired with checkpoint B: **B + layer-frozen fine-tune reaches 64.90%
accuracy / 63.50% weighted F1 / 44.16% macro F1**, again beating the
equivalent pairing with the full-fine-tune checkpoint (64.79%/63.36%/43.76%)
on every metric. This checkpoint (referred to as H below) supersedes the
full-fine-tuning one as this project's fine-tuned member going forward --
strictly better in every configuration actually measured. **Correction
(added later):** an earlier version of this paragraph said the frozen
DistilBERT (D) and RoBERTa-large (F) checkpoints behind the B+D+F+H result
were "no longer saved anywhere". That was wrong -- they were sitting inside
`MELD_best_ensemble_checkpoints.zip` (made on 2026-09-27) and were missed
because the search only looked for loose `.pt` files, never inside zip
archives. They are restored under `models/originals/` and the ensemble is
re-evaluated with them in the "Ensemble results under an honest selection
protocol" section below. (The original *H*, the full-fine-tune RoBERTa-base
checkpoint, really was deleted and is not recoverable as a file; see that
section for how it can be regenerated.)

**Follow-up 2: the same recipe on RoBERTa-large -- AMB-DSGDN's actual text
encoder -- produces the best single model this project has ever trained,
but Fear collapses again.** Everything up to this point had only fine-tuned
RoBERTa-base; RoBERTa-large had only ever been used frozen (checkpoint F).
Applied the same layer-freezing approach, scaled to RoBERTa-large's 24
layers: froze the bottom 16 (the same ~2/3 ratio as the RoBERTa-base run),
leaving the top 8 (28.7% of the encoder, 101.8M of 355.4M params)
trainable. Smoke-tested locally first (same frozen/trainable gradient
check as before, adapted for 24 layers). Trained on Kaggle's GPU (26.2 min,
best epoch 9 of 10, val weighted F1 0.6366 -- the best validation score of
any fine-tuning attempt this session, comfortably ahead of RoBERTa-base's
0.6065). First run's kernel errored *after* training finished, during the
output-copy step -- a script bug on this end (it still referenced the
pre-rename checkpoint filename), not a training failure; fixed the kernel
script to copy whatever checkpoint actually changed rather than a
hardcoded name, and reran the full 26-minute training since the finished
checkpoint had already been lost when the container tore down. Real test
result: **64.18% accuracy / 63.32% weighted F1 / 43.57% macro F1** -- the
best *solo* checkpoint this project has ever produced, beating every prior
single model including frozen RoBERTa-large (F, 63.41%/60.85%/39.17%) and
the RoBERTa-base fine-tune above on every metric. Disgust and Anger both
score unusually well for this project (F1 0.216 and 0.520). **But Fear
collapsed to 0.0 F1 again** -- the same failure the RoBERTa-base layer-
freezing had fixed. So layer-freezing's fix for Fear collapse doesn't
automatically transfer to a bigger encoder; whatever protects Fear at
RoBERTa-base scale is apparently overwhelmed again once the trainable
portion is large enough (101.8M trainable params here vs. 28.9M for the
RoBERTa-base version), even at a similar frozen-fraction ratio. Kept as
`models/final_model_finetuned_roberta_large.pt` (gitignored, like the other
fine-tuned checkpoints) and `logs/evaluate_finetune_roberta_large_output.log`.

**Two attempts to fix that Fear collapse both failed on validation.** With
`TRAIN_FREEZE_LAYERS` and `TRAIN_FEAR_WEIGHT_BOOST` / `TRAIN_DISGUST_WEIGHT_BOOST`
env overrides added to `train_finetune.py`, two single-variable variants of
the RoBERTa-large fine-tune were trained on Kaggle's GPU: (1) a heavier
freeze (bottom 20 of 24 layers instead of 16) and (2) Fear and Disgust
class weights x1.5 on top of the existing 1.15x, at the original freeze
depth. Judged on *validation* (the project's selection rule), neither beat
the existing fine-tune: best validation weighted F1 was 0.6207 for the
heavier freeze (macro F1 0.4601) and 0.6128 for the class-weight boost
(macro F1 0.4268), against 0.6366 / 0.4766 for the original. Neither
checkpoint was downloaded or scored on the test set -- a variant that loses
on validation has no claim on the test set. So the simple knobs do not fix
Fear here; the minority classes remain the open problem (see the ensemble
section for the current per-class numbers).

**But fine-tuning turned out to help anyway -- as an ensemble member, not
solo.** Tested the fine-tuned checkpoint (H) alongside B+D+F in every
combination (`evaluation/ensemble_test_with_finetune.py`, after first
verifying `FinetuneMELDDataset`'s test-set ordering exactly matches the
frozen-feature dataset's, 0 mismatches across all 280 dialogues, so H's
predictions can be combined positionally with the others). Even though H
alone never beat B+D+F alone, adding it produces a new best on every
metric: **B+D+F+H reaches 66.36% accuracy / 64.55% weighted F1 / 44.96%
macro F1**, and B+D+H (dropping F) reaches 65.98% / **64.56%** / **45.28%**
-- both strictly better than B+D+F's 65.56%/63.90%/44.57%. This fits the
same pattern as every other successful ensembling step this session:
identical recipes don't help (seed43, seed44, Kaggle-GPU checkpoint G all
failed to add value), but a genuinely different member does -- and a
fine-tuned encoder's errors are apparently different enough from a frozen
one's to be useful, even though its solo score is unremarkable. Full sweep
in `logs/ensemble_test_with_finetune_output.log`. The standing best result
is now **B+D+F+H by accuracy, or B+D+H by weighted F1/macro F1**.

**First re-verification attempt (retraining D and F): superseded, kept for
the record because the mistakes in it are instructive.** At the time, the
original D and F checkpoints behind the 66.36%/64.55% result were believed
lost (they were not -- see the correction above and the next section), so D
and F were retrained on Kaggle's GPU to pair them with H3 (the RoBERTa-large
fine-tune): D in 6.6 min (best val weighted F1 0.5840), F in 8.2 min
(0.5816), via new environment-variable overrides in `train_final.py`
(`TRAIN_TEXT_DIM`, `TRAIN_TEXT_PATH`, `TRAIN_OUTPUT_PATH`,
`TRAIN_USE_LEGACY_AUDIO`; unset, behavior is unchanged). Result:
**B+D'+F'+H3 reached 65.56% accuracy / 64.16% weighted F1 / 45.27% macro
F1**, below the original 66.36%/64.55%. Three facts, established later,
change how that should be read:

* **The retrained D was not the same recipe as the original D.** The
  original D was an older recipe (DistilBERT features, no speaker
  embeddings; solo 61.23%/60.43%, run8 in `logs/`); the retrain used the
  *current* speaker-aware recipe. They are different models, so no
  conclusion about "variance" can be drawn from comparing them.
* **The retrained F was the same recipe as the original F** (CPU run20):
  epoch 1 is nearly identical (train loss 0.9696 vs 0.9668, train accuracy
  0.481 vs 0.483) and the two then drift apart (val weighted F1 at epoch 3:
  0.530 vs 0.554), ending at 63.41% vs 60.00% test accuracy. Same seed,
  same code, different platform: the dropout masks come from different
  random streams on CPU and GPU and the floating-point numerics differ,
  and that is enough to send a 15+-epoch run somewhere several points
  away. **A single trained instance of this recipe therefore carries
  several points of run-to-run noise.** (A later four-seed study, below,
  shows the seed-42 GPU run was simply the *worst* of four: the other
  three GPU seeds score 62.7-63.9% accuracy, in line with the original
  F, so the GPU itself is not what lowered it.)
* **GPU retraining is deterministic for a fixed seed.** Retraining D and F
  two more times each (meant as a "multi-seed" check) gave checkpoints
  bit-identical to the first (all 105 weight tensors equal, zero
  difference), and the retrained F's solo score is *identical* to
  checkpoint G's from an earlier GPU run -- F', G and the repeats are one
  and the same run. An earlier draft of this note read F' and G as two
  independent GPU samples pointing to a systematic GPU effect; that was
  wrong. Different instances need a different seed, which `TRAIN_SEED` now
  provides.

The practical consequence for reading *any* number in this file: margins of
a point or two between single runs are inside run-to-run noise, which is
why the ensemble analysis below reports bootstrap confidence intervals.
Full sweep in `logs/ensemble_test_full_reverify_output.log`.

**Weighted ensembling: tried properly, and it overfits the validation set.**
Every ensemble result above averages member probabilities with equal
weight. Fit per-member weights instead
(`evaluation/weighted_ensemble_test.py`), two ways, both on the
**validation** set only: (1) softmax-parameterized weights minimizing
validation NLL via Nelder-Mead, (2) a direct grid search (0.1 steps)
maximizing validation weighted F1 -- the project's actual selection metric.
Both looked like clear wins on validation: equal weighting scores 0.6243
weighted F1 there, NLL-fitted weights reach 0.6390, and the grid search
reaches 0.6482 by weighting B and D to *exactly zero* and betting
everything on F+H. Applied to the test set, the story reverses: equal
weighting scores 65.56%/64.16%/45.27% (matching the re-verification result
above, as it should -- same members, same weights), NLL-fitted weights
score slightly worse (65.36%/64.13%/44.20%), and the grid-search weights
that looked *best* on validation score worse still (64.44%/63.59%/45.14%)
-- clearly beaten by plain equal weighting on every metric. This is
textbook overfitting: with only 1,108 validation utterances and a search
free to zero out entire members, the grid search found a combination that
fit validation noise rather than a genuinely better member-weighting.
**Conclusion: equal weighting stays.** This was evaluated honestly rather
than assumed, and the answer is a real no -- consistent with per-member
temperature scaling also not helping earlier
(`evaluation/calibrated_ensemble_test.py`). Full output in
`logs/weighted_ensemble_test_output.log`.

### Ensemble results under an honest selection protocol

Every ensemble number earlier in this file was produced the same way: score
many subsets on the test set and report the best one. That is a winner's-
curse bias -- the maximum of many noisy estimates is optimistic -- and it
is how the 66.36% headline arose. This section redoes the analysis with a
protocol that cannot flatter the result:

1. **Cache each member's predictions once**
   (`evaluation/cache_member_probs.py`; a fine-tuned member costs 10-20
   minutes of CPU encoding, so everything afterwards works on stored
   validation/test probabilities in `logs/prob_cache/`, which are small and
   committed so the numbers below can be reproduced without any
   checkpoint).
2. **Choose the ensemble on validation** (`evaluation/ensemble_from_cache.py`):
   every subset of the candidate members is scored there, the subset with
   the best validation weighted F1 is the one reported, and its test score
   is looked at once. The best-on-test subset is also printed but labelled
   optimistic. A selection-free alternative -- average *all* members -- is
   reported alongside.
3. **Report uncertainty** with a dialogue-level bootstrap (whole dialogues
   are resampled, since utterances inside one are not independent) and a
   paired bootstrap against single models.

Members (test accuracy / weighted F1 / macro F1; all trained on the training
split only):

| Member | What it is | Test |
|---|---|---|
| B | RoBERTa-base frozen features + speaker embeddings (`models/final_model.pt`) | 62.45 / 61.50 / 42.74 |
| D | DistilBERT frozen features, older non-speaker recipe | 61.23 / 60.43 / 41.44 |
| F | RoBERTa-large frozen features, graph fusion off | 63.41 / 60.85 / 39.17 |
| H2 | RoBERTa-base fine-tuned, bottom 8/12 layers frozen | 62.57 / 61.66 / 42.02 |
| H3 | RoBERTa-large fine-tuned, bottom 16/24 layers frozen | 64.18 / 63.32 / 43.57 |

The original D, F and B were recovered from `MELD_best_ensemble_checkpoints.zip`
(restored under `models/originals/`; B is bit-identical to `final_model.pt`).

Results over these five members (`logs/ensemble_from_cache_originals_pool.log`):

| Ensemble (equal-weight average) | Selection | Validation | Test |
|---|---|---|---|
| D + F + H3 | best of 31 subsets on validation | 66.70 / 64.74 / 48.28 | **66.48 / 64.74 / 44.59** |
| B + D + F + H2 + H3 | none (all members) | 66.34 / 64.19 / 46.84 | **66.63 / 64.83 / 44.73** |
| B + D + H2 + H3 | best of 31 subsets *on test* (optimistic, not a valid headline) | 65.52 / 63.83 / 48.05 | 67.09 / 65.63 / 45.81 |

Reproduce the table above from the committed probability caches (no
checkpoints or GPU needed, about a minute):

```bash
venv/Scripts/python.exe -m evaluation.ensemble_from_cache     --candidates B,D_orig,F_orig,H2_base,H3_large --reference B
```

For the validation-chosen ensemble the 95% dialogue-bootstrap interval is
accuracy [64.33%, 68.61%] and weighted F1 [62.44%, 67.07%]; AMB-DSGDN's
reported 66.07% / 66.18% falls inside both. Against the single model B the
improvement is +4.03 accuracy points (95% CI +2.59 to +5.52) and +3.24
weighted-F1 points (CI +1.75 to +4.78). The honest reading: **this project's
ensemble is statistically indistinguishable from AMB-DSGDN on both
metrics**, with a point estimate slightly above on accuracy and about 1.4
points below on weighted F1, and it is decisively better than any single
model here. The weakest classes are unchanged: Fear (F1 0.07, recall 4%)
and Disgust (F1 0.14) -- 50 and 68 test utterances respectively.

**Why Fear and Disgust fail (`evaluation/minority_error_analysis.py`,
`logs/minority_error_analysis_output.log`, on the validation-chosen
ensemble).** It is not a thresholding accident -- the model is genuinely
unsure about these classes:

* *Fear* (50 test utterances): the ensemble predicts Fear only 6 times in
  2,610 utterances (2 correct), and its probability for the true class on
  real Fear examples averages 0.10 and never exceeds 0.29. 48% of true Fear
  goes to neutral, 18% to anger, 10% each to sadness and surprise. Fear
  utterances are unusually *long* (mean 11.6 words vs 8.1 overall; 29 of the
  50 have 11+ words, of which 3.4% are recalled) and situational rather than
  lexical -- "there's no way Joey's gonna make it in time", "Monica kinda
  trusted me with something and she shouldn't have!" -- so what marks them as
  fear is the surrounding situation, not any local word, and agitated ones
  ("I'm freaking out!") sound like anger.
* *Disgust* (68 test utterances): predicted 31 times overall but only 7
  correct; true Disgust goes to neutral (35%) and anger (34%). Sarcastic or
  contemptuous remarks look like anger on the surface; interjections like
  "Ewww!" are rare.

This is the motivation for putting dialogue context inside the encoder
(next): the information that distinguishes these classes is mostly in the
neighbouring lines, which an isolated-utterance encoder never sees. A
one-parameter minority-class prior correction (divide probabilities by
class-prior^tau, tau fit on validation) was also tried on the chosen
ensemble: validation selects tau = 0 (no correction). As a pure operating
point it trades overall accuracy for minority recall -- at tau = 0.4, test
macro F1 rises from 44.6% to 46.4% while accuracy falls from 66.5% to 63.6%
-- which is worth knowing but is not an improvement on the project's
selection metric.

**Seed noise, measured.** Four seeds each of the current-recipe DistilBERT
(D) and RoBERTa-large (F) frozen-feature models were trained on Kaggle's GPU
(`TRAIN_SEED`; `evaluation/seed_stats.py`, `logs/seed_stats_output.log`):

| Family (4 seeds) | Test accuracy | Test weighted F1 | Equal-weight average of the 4 seeds |
|---|---|---|---|
| D, DistilBERT | 61.02 +/- 1.82 (58.39 to 62.26) | 60.03 +/- 1.36 (58.05 to 61.15) | 62.95 / 61.74 / 42.88 |
| F, RoBERTa-large | 62.35 +/- 1.65 (60.00 to 63.87) | 61.34 +/- 0.99 (59.98 to 62.32) | 64.10 / 62.80 / 43.92 |

The same recipe moves by about 2 accuracy points (one standard deviation)
from seed to seed, with a 4-point best-to-worst range -- so any single-run
difference smaller than that is not evidence of anything. Averaging the
seeds recovers +1.5 to +1.9 points over a typical single seed, so for a
frozen-feature family "train a few seeds and average" is worth it.
Whether it helps the *full* ensemble is a separate question, and the answer
is no: adding the seed-averaged D and F families to the member pool leaves
the validation-chosen ensemble unchanged (D + F + H3, test 66.48% / 64.74%),
and the twelve best subsets by validation all score 66.0-66.7% accuracy and
64.3-64.9% weighted F1 on test (`logs/ensemble_from_cache_seedavg_pool.log`).
The pool has plateaued around **66.5% / 64.8%**: more seeds of the same
families cannot move it, and a member that sees something different is
needed instead (the context-aware encoder below).

**Context-aware fine-tuning (the one remaining untried lever).** Every
fine-tuned encoder above embeds each utterance *in isolation*; all
cross-utterance reasoning is left to the BiLSTM/attention layers on top.
On MELD that is a real handicap -- many utterances are short and ambiguous
by themselves ("Yeah.", "What?", "Oh.") and only the neighbouring lines say
what the character feels. `ContextFinetuneTextEncoder`
(`models/finetune_text_encoder.py`) puts the neighbouring utterances inside
the transformer's input instead: for each target utterance it feeds the
preceding lines and the next reply, each tagged with a dialogue-relative
speaker letter (the same relative identities the fusion model already uses
-- never character names), and mean-pools only the target's tokens.
Zero-size windows fall back to the original encoder exactly, and old
checkpoints load unchanged. Before any GPU time was spent it was checked
for: the exact input format, window semantics (changing a neighbour inside
the window changes the target's embedding by 0.097; changing an utterance
outside it changes it by exactly 0), truncation that never clips the
target, gradients reaching only the unfrozen layers, and real training
steps. Runs launched on Kaggle's GPU: RoBERTa-large with 2 preceding lines
+ 1 following (same recipe otherwise as the 64.18% isolated fine-tune),
and RoBERTa-base with 3 + 1 at two seeds plus a no-context control at the
second seed, so a gain can be judged against seed noise rather than
against one run. *Results are added below when they finish; until then
nothing here claims the context window helps.*

**Base paper comparison -- AMB-DSGDN (2026, arXiv 2603.10043).** This is the
most recent closely-related paper found (adaptive per-modality dropout based
on relative performance, differential graph attention, auxiliary unimodal
losses) and the explicit target for this phase of work. Its reported MELD
numbers: 66.07% accuracy / 66.18% weighted F1 (IEMOCAP: 76.09%/75.64%).

**Honest verdict: statistically indistinguishable from AMB-DSGDN on both
metrics, not a clear win.** The best result obtained *without* selecting on
the test set is the equal-weight average of five members (B, D, F and the
two fine-tuned encoders H2/H3): **66.63% accuracy / 64.83% weighted F1**;
choosing the members on validation instead gives 66.48% / 64.74% (see
"Ensemble results under an honest selection protocol"). Against AMB-DSGDN's
66.07% / 66.18% that is about +0.4 points on accuracy and -1.4 on weighted
F1 -- but on 2,610 test utterances the 95% bootstrap interval is roughly
+/-2 points, and AMB-DSGDN's numbers fall inside it on both metrics. So the
defensible claim is *comparable performance*, reached with a much lighter
pipeline (frozen features plus a layer-frozen fine-tune, trained on free GPU
time), not that this project surpasses the paper. The project's earlier
66.36% headline came from picking the best of several ensembles on the test
set and shouldn't be quoted as a clean result. What *is* established
decisively is the size of the ensembling gain over a single model (+4.0
accuracy points, +3.2 weighted-F1 points over B, both with confidence
intervals clear of zero). Six single-model or pipeline changes adapted from
AMB-DSGDN's method helped (auxiliary unimodal losses, adaptive dropout, the
RoBERTa-base and then RoBERTa-large text upgrades, dialogue-relative
speaker embeddings, and end-to-end fine-tuning); the speaker-relational
attention bias (twice), sentiment loss (twice), per-batch dropout alone, a
full graph attention fusion layer, and fitted ensemble weights all did not.
Not every idea inspired by a stronger paper transfers cleanly, and
reporting the failures -- including a real bug caught mid-session, several
of this project's own earlier misreadings, and the winner's-curse bias in
the old headline -- is as much part of the record as the successes.

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
  finetune_text_encoder.py    Trainable text encoders: isolated-utterance and
                              context-aware (neighbouring lines in the transformer)
  graph_fusion.py             Graph-attention fusion layer (tried; did not help)
  baseline_model.py           Original bimodal baseline (reconstructed from checkpoint)
  final_model.pt               Checkpoint B -- the single model the demo serves
  baseline_model.pt            Trained baseline checkpoint
  legacy/                      Archived V1-V4 checkpoints + old model stub files
training/
  dataset.py                   MELDDataset: aligns text/audio/video/labels per dialogue
  train_final.py                Frozen-feature training script -> final_model.pt (env overrides:
                                TRAIN_TEXT_PATH/DIM, TRAIN_SEED, TRAIN_OUTPUT_PATH)
  train_finetune.py             End-to-end text fine-tuning (layer freezing, optional
                                dialogue context window, seeds; GPU recommended)
  dataset_finetune.py           Dataset variant yielding raw utterance text for fine-tuning
  legacy/                      Archived baseline/V2/V3/V4 training scripts
evaluation/
  evaluate_final.py             Overall + per-class metrics + confusion matrix
  evaluate_baseline.py          Same, for the baseline model
  modality_ablation.py          All 7 modality-combination results
  disagreement_analysis.py      Modality agreement/disagreement + adaptive-weight analysis
  cache_member_probs.py         Cache one member's val/test probabilities (logs/prob_cache/)
  ensemble_from_cache.py        Honest ensemble analysis: pick on validation, test once,
                                bootstrap CIs, paired comparison, optional prior correction
  seed_stats.py                 Seed-to-seed spread and seed-averaging for a model family
  evaluate_finetune.py          Test metrics for a fine-tuned checkpoint
  weighted_ensemble_test.py     Fitted ensemble weights (tried; overfit validation)
  ensemble_test.py              Original equal-weight subset sweep (selects on test; see README)
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
