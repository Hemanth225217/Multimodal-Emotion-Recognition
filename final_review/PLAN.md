# Final-review track

Started 2026-10-05. This is separate from the pre-final (guide) review.

**The full plan, with every lever, the tiers of targets, the gates and the two-person split, is in `MASTER_PLAN.md` (written 2026-10-07, planning only).**

## Ground rules

1. **The pre-final package is frozen.** Git tag `pre-final-review` (commit `c49da59`) marks the code, README, logs and
   caches that went to the pre-final review. The report, the 25-slide deck, the viva sheet, the demo and the zip in
   `Downloads` are not touched by this track. The original folder `Multimodal_Emotion_Recognition` stays on `main`.
2. **All new work lives on branch `final-review`** in this folder (`Multimodal_Emotion_Recognition_FINAL`, a git worktree),
   with notes and results under `final_review/`. New outputs get new file names. Never write `models/final_model.pt`
   or overwrite a committed cache: `training/train_final.py` writes there by default, so always set `TRAIN_OUTPUT_PATH`.
3. **Same honest protocol as before, fixed in advance:** every design choice is made on the validation set; the test set
   is scored once for each configuration listed here before it is run; every run is reported, failures included; any
   claimed improvement needs at least two seeds, a dialogue-level bootstrap interval and a paired comparison with the
   pre-final result; nothing (thresholds, weights, layers) is tuned on test. IEMOCAP stays out of scope.
4. **The target is not "beat every paper".** On 2026-10-05 the published MELD weighted-F1 results I could verify ran from
   about 66 to 73.1 (the pre-final README and viva sheet list them); our ensemble is 66.30. The aim is to close part of that
   gap with methods we can defend, to say clearly where we land, and to turn the audit findings into measurable repairs.

## The idea in plain words ("audit and repair")

The pre-final review showed, with numbers, that our fusion model is text-led and that audio and video are barely used:
the adaptive weights are nearly constant (text 0.545, SD 0.016), the video-only head predicts "neutral" for all 2,610
test utterances, the audio-only head never predicts fear or disgust, and the LOW disagreement level is almost entirely
"all three say neutral". The systems that score 71 to 73 differ most visibly in their audio and video encoders. So the
final-review contribution is to repair those weak parts and to measure whether the repair works, using the audit numbers
as the success criteria. A negative result is still a result and will be reported as one.

## Work packages (in order of value per effort)

| | Question | Success criterion (written before running) |
|---|---|---|
| W4 | Do the other speaker's lines help? Evaluation-only test on model B: full dialogue vs target alone vs target + same-speaker lines vs target + other-speaker lines. | A table with intervals; no threshold. |
| W4 | Selective prediction: does ensemble disagreement or confidence give a useful accuracy-coverage trade-off (from the cached probabilities, no training)? | Risk-coverage curve and AURC against a confidence-only baseline. |
| W1 | Do stronger audio embeddings (WavLM, HuBERT, wav2vec 2.0 all-layer probes) beat the openSMILE-style features? Probe every layer on validation, then train the fusion model with the chosen features. | Audio-only validation weighted F1 up by at least 3 points, and fused validation weighted F1 up by at least 0.5 over the control, at 3 seeds each. |
| W2 | Do face-based or CLIP-style frame embeddings make the video head informative? | Video-only head predicts at least 3 classes and beats the 48.12% always-neutral accuracy on validation. |
| W3 | Does a reliability-aware weight network (inputs: single-modality confidence) give utterance-adaptive weights without collapse? | Text-weight SD of at least 0.05 on test, no loss of validation weighted F1, and MEDIUM vs HIGH accuracy differing by at least 5 points among non-neutral predictions. |
| W5 | Does adding the new members improve the validation-chosen ensemble on test? | Paired bootstrap against the pre-final ensemble (67.89 / 66.30), reported whatever the sign. |
| W6 | Update the positioning table, report chapters, deck and viva sheet for the final review. | Done last, from verified numbers only. |

Controls matter: the current `train_final.py` uses per-batch adaptive dropout while model B used per-epoch dropout, so
every comparison uses a control trained with the same script and the same seeds, not model B itself.

## Where the heavy work runs

The laptop must stay free for the pre-final demo. Heavy extraction and training run on Kaggle's free T4 through kernels,
reading the public `zaber666/meld-dataset` (raw MELD videos) and the private `hemanths0411/meld-emotion-features`
(text features, CSVs). Results come back with `kaggle kernels output`. Light analyses run locally.

## Hazards

- `meld_dataset` and `meld_features` in this folder are **directory junctions** to the originals. Delete them with
  `rmdir` (or `Remove-Item` on the junction itself), never with a recursive delete of this folder, or the data is lost.
- Disk is tight (about 7 GB free): stream audio through ffmpeg, do not write temporary wav files.
- Do not run long CPU jobs while the demo is being presented.
