# Journal (newest last)

## 2026-10-05

- Pre-final state tagged `pre-final-review` (c49da59). Final-review work moved to branch `final-review`, worktree
  `C:\Users\Samsung Laptop\Multimodal_Emotion_Recognition_FINAL`; `meld_dataset` and `meld_features` are junctions to the
  originals (read-only use).
- Check: `evaluation/evaluate_final.py` run from this worktree reproduces model B (62.45% / 61.50% / 42.74%), so the
  junctions and the copied checkpoint work.
- Kaggle: the public dataset `zaber666/meld-dataset` contains the raw videos, so audio and video features can be computed
  on Kaggle without uploading anything. `hemanths0411/meld-emotion-features` (211 MB) holds the text features and CSVs.
- Found while reading `modules/audio_features_wav2vec2.py`: the earlier wav2vec 2.0 test used only the **last layer**,
  mean-pooled. For emotion, middle layers and WavLM-style models are normally better, so the earlier "no gain" result does
  not rule out stronger audio features. W1 starts with a layer-by-layer probe on validation.

## 2026-10-07

- **W1 audio probe finished** (kernel `meld-audio-probe`, T4, about 40 minutes; results in `final_review/results/audio_probe/`,
  summary printed by `final_review/summarize_probe.py`). Linear probe, train -> validation only, audio-only weighted F1:
  legacy openSMILE-style 300-D 34.9 (macro F1 24.3); wav2vec 2.0-base best layer 38.0 (its last layer 37.1, which is what the
  earlier "no gain" test used); WavLM-base-plus best layer 11: 41.2; HuBERT-large best layer 17: 43.0; **WavLM-large best
  layer 22: 44.8 (macro F1 33.7); mean of layers 22, 14, 21: 45.5 (macro F1 34.6)**. So the pre-registered W1 audio-only
  criterion (at least +3 points on validation) is met by a wide margin (about +10). The fused-model criterion is still open.
- **Bug found in my kernel:** the test split decoded only 1,595 of 2,610 clips. The public dataset has several copies of many
  clips and the kernel kept the first one found; for test that copy often could not be decoded. Train (9,988) and validation
  (1,108) match the local pipeline, so the probe results are unaffected. Fix: `kaggle/audio_extract_test/audio_extract.py`
  keeps every candidate path, tries the canonical folder first, falls back to the other copies, and logs why failures failed.

- **Kernel `meld-audio-test-features` ended with status ERROR and an empty log** (no failure message through the API). The script
  passes a syntax check and imports locally, so the likely cause is the title/id mismatch Kaggle warned about (the title slug
  is `meld-audio-test-features`, the metadata id was `meld-audio-test`). Not re-run: on 2026-10-07 you asked for planning only.
  Fix on restart: title it "MELD Audio Test" so the slug equals the id.
- Repository facts for the contribution statement: as of 2026-10-07, 67 commits, all authored by Hemanth S, 66 with a
  "Co-Authored-By: Claude" trailer. Pranitha N.S does not appear in the history (see `MASTER_PLAN.md` section 9).
- **`MASTER_PLAN.md` written** (planning only): tiers A to D of targets with rough odds, 15 levers with estimated gains and costs,
  five candidate contributions with a prior-art protocol, phases with gates, safeguards, logistics, risks, the two-person split.
  A Word copy is in `Downloads\Final_Review_Master_Plan.docx`.

## 2026-10-07 (afternoon)

- The user gave the deadline: **15 October** (8 days from today, a Wednesday). `MASTER_PLAN.md` section 0 is the 8-day schedule:
  experiments freeze Monday 12 October evening, test scored once on the 13th, documents finished on the 14th, the 15th is a buffer.
  Final selection rule fixed before any new run: greedy forward selection on validation (stop when the gain is below 0.1).
- Test-split audio kernel restarted with the title fixed to match its id (`hemanths0411/meld-audio-test`); this time Kaggle gave no
  title warning.

## 2026-10-08

- **Test-split audio kernel finished** (`hemanths0411/meld-audio-test`, T4). All 2,610 test clips decoded, 0 failed, 0 without
  video (2,056 clips taken from the second copy found, 554 from the first). WavLM-large and HuBERT-large features of shape
  (2610, 25, 1024) are saved with `index_test.json`, `labels_test.json` and `decode_report.json`. Together with the train and dev
  features in the `meld-audio-probe` output this completes the input for the L1 fusion run. The test files inside
  `meld-audio-probe` stay unusable (1,595 of 2,610 clips): when building pickles, give `make_audio_pickle.py` the
  `meld-audio-test` output folder first. The earlier kernel `meld-audio-test-features` (status ERROR) is superseded.
- **No experiment was run today.** Work was paused at the user's request (planning and a project handoff only). The L1 run
  (3 seeds each of: control with the old 300-D audio, WavLM-large layers 22/14/21, WavLM-large layer 22) therefore slips from
  Thursday 8 to Friday 9, and `MASTER_PLAN.md` section 0 needs re-baselining: Friday L1 and L5 with Gate 1 in the evening,
  Saturday the language-model run, Sunday Gate 3, freeze Monday 12 evening, final scoring Tuesday 13 (unchanged).
- **Uncommitted in this worktree:** the `audio_path` / `TRAIN_AUDIO_PATH` / `TRAIN_AUDIO_DIM` / `TRAIN_EPOCHS` plumbing in
  `modules/data_loader.py`, `training/dataset.py`, `training/dataset_finetune.py`, `training/train_final.py`,
  `training/train_finetune.py` and `evaluation/cache_member_probs.py`, plus the new `final_review/tools/make_audio_pickle.py`.
  Compile-checked only, never run on real features. First step when work resumes: a CPU smoke test with a synthetic audio
  pickle in the legacy layout, then commit and push, then write the fusion kernel.
