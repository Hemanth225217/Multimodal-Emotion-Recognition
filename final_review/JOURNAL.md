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
- **Evening, work resumed (new Claude session after a handoff).** Plumbing reviewed and committed (`e4495fa`, not yet pushed).
  Check added: `data_emotion.p` and the three MELD CSVs hold identical utterance sets for every dialogue (1,038 / 114 / 280),
  so the builder's CSV-ranked rows match the datasets' position-based alignment.
- **CPU smoke test passed.** Fake kernel-shaped outputs (3 layers x 1024-D, rows shuffled, one clip missing per split, plus a
  5-row decoy test file in the "probe" folder) went through the real `make_audio_pickle.py`: decoy ignored, all 13,705 rows in
  the right place, missing clips became zero rows. `train_final.py` with `TRAIN_AUDIO_DIM=1024 TRAIN_EPOCHS=1` trained one
  epoch in 1.1 min (Audio=1024D, val 1,108 utterances; the number itself is meaningless on random audio) and
  `cache_member_probs --audio-path` wrote val (1108, 7) and test (2610, 7). Smoke files deleted. A harmless stderr line
  "DLL load failed while importing _devicearray" (Windows Application Control on an optional numba module) appears at start.
- **Fusion kernel written:** `final_review/kaggle/fusion_audio/` (`hemanths0411/meld-fusion-audio`). Clones `final-review`,
  finds inputs by marker file (decode_report.json = test audio, probe_results.json = train/dev audio), refuses to run if
  the built test split has any missing clip, trains control / wavlm_mean / wavlm_l22 at seeds 42, 1, 2, caches val+test
  probabilities without printing test metrics, writes validation-only `summary.json`. `MODE = "smoke"` first (1 variant,
  1 seed, 1 epoch), then `"full"`. It needs the branch pushed before it can run.
- Laptop disk: C: had 2.7 GB free at the start of the evening and was falling (under 1 GB) from something outside this work.
- **Kaggle smoke run of `meld-fusion-audio` (version 1, MODE smoke) COMPLETE in 2.0 min.** Cloned commit 6648519; CUDA on;
  inputs found by marker (test audio = meld-audio-test, train/dev = meld-audio-probe); built WavLM-large (22/14/21) pickle has
  zero missing rows train/dev/test = 1/1/0, as required; Audio=1024D; one epoch took 0.4 min; val and test probabilities
  cached (1108, 7) / (2610, 7). The one-epoch validation number is a pipeline check only and is not a result.
  Estimated full run: 9 trainings of at most 25 epochs (~10 min each with early stopping) plus caching, about 1.5 to 2 GPU hours.
  Kernel switched to MODE = "full".
- **CPU analyses (while L1 runs on Kaggle): `final_review/analysis/headroom_selective.py`**, results in
  `final_review/results/analysis/`. The script first reproduces the pre-final headline on test (67.89 / 66.30 / 46.00).
  - **Headroom (L15), validation only, 9 families:** at least one family right on 86.3 % of utterances, all right on 33.8 %,
    none on 13.7 %; average of all 67.6 % vs best single 66.4 %. Fear: any family right on only 45 % (18 of 40), average-of-all
    recall 7.5 %; disgust 40.9 % / 13.6 %. So the oracle gap is large overall, but for fear and disgust even the oracle
    is weak: no combination rule can fix them; they need better inputs or training.
  - **Selective prediction, headline ensemble:** test accuracy rises from 67.9 % (answer all) to 75.1 % at 80 % coverage and
    85.2 % at 50 % (top-probability ranking); AURC 0.160 [0.141, 0.181] against 0.321 for a random ranking and 0.058 for
    the oracle. Validation behaves the same (69.5 -> 75.3 -> 83.6). Member agreement ranks utterances as well as the top
    probability, no better: AURC difference +0.0005 [-0.0016, +0.0024] on test. When all 5 families agree (1,395 test
    utterances) accuracy is 82.9 %; at 2 of 5, 33.5 %. Unlike model B's three-level disagreement indicator, ensemble
    agreement is informative.
- **Paper track added** (deadline 21 Oct); the plan is kept locally, outside git. Second dataset chosen: MC-EIU English
  (Hugging Face `YulangZhuo/MC-EIU`, CC BY-NC 4.0, gated with automatic approval, raw video in three archives of
  22.4 / 25.5 / 12.5 GB; dyadic scenes from Friends, The Big Bang Theory, Modern Family; same 7 emotions). IEMOCAP is the
  backup if its licence arrives in time. **The user dropped the LoRA language-model member (L8).**
- **Re-audit tool:** `final_review/analysis/modality_audit.py` (validation by default). Model B on validation reproduces its
  checkpoint's recorded val score exactly (59.12 / 58.27 / 43.63) and shows the same pattern as the pre-final test audit:
  text weight 0.545 (SD 0.017), largest on 100 % of utterances, mean text weight differs by only 0.015 across emotions;
  video-only predicts neutral for every utterance; audio-only predicts 5 classes (39.9 %); ablation: text only 58.84,
  text+audio 59.03, all 59.12; audio right while fused wrong on 63 of 639 non-neutral utterances; when text-only and
  audio-only disagree the fused prediction follows text 84.4 % of the time. Kaggle kernel `final_review/kaggle/audit_l1`
  (CPU only) will run the same audit on the 9 L1 checkpoints.
- **L1 full run COMPLETE** (`meld-fusion-audio` version 2, commit 6648519, 72 min on the T4). Validation only (test
  probabilities cached, not scored). Best-epoch validation, mean of seeds 42 / 1 / 2:
  control (300-D audio) acc 58.42, wF1 57.89 (SD 0.42), mF1 43.05;
  WavLM-large layer 22: acc 58.97, wF1 57.45 (SD 0.36), mF1 42.82 (wF1 -0.43 vs control);
  WavLM-large mean of 22/14/21: acc 57.94, wF1 56.68 (SD 0.21), mF1 42.45 (wF1 -1.21 vs control).
  Per run: control 58.09 / 57.41 / 58.16; l22 57.56 / 57.76 / 57.05; mean 56.53 / 56.92 / 56.58.
  **Gate 1 (fused val wF1 >= +0.5 over control): FAILED for both WavLM variants.** Better audio features, which raise
  audio-only probe wF1 by about 10 points, do not raise the fused model's validation score. Reported as a negative
  result. The CPU audit kernel `meld-audit-l1` was launched to see whether the fusion uses the new audio at all.
  Results and caches: `final_review/results/L1_fusion_audio/`.
- **L1 audit (validation, `meld-audit-l1` version 2, CPU), mean of 3 seeds** - control / WavLM l22 / WavLM mean:
  mean audio weight 0.212 / 0.108 / 0.146 (model B 0.238); text largest weight 99.3 / 97.9 / 97.6 %; accuracy added by
  audio (all minus text+video) +1.59 / +0.90 / +0.66 points; fused follows text when text-only and audio-only disagree
  84.6 / 93.9 / 94.2 %; audio-only right while fused wrong (non-neutral) 42 / 19.7 / 36. Audio-only accuracy inside the
  fused model barely moves (43.8 / 43.0 / 44.5). **Finding: with much stronger audio features the fusion gives audio less
  weight and follows text more.** The input repair does not reach the fusion; the bottleneck is the fusion's weighting,
  not the audio encoder. Results: `final_review/results/L1_audit/`.
- **MC-EIU English inspected** (`mc-eiu-inspect` version 6, run by the user from the editor so the HF_TOKEN secret is
  available; secrets are not passed to CLI-pushed runs): 45,009 utterances, 4,013 dialogues (Dia_No global), speakers 0/1,
  columns Sr_No, Subtitle, Script, Dia_No, Utt_No, video_name, Season, Episode, Begin/End_timestamp ("hh:mm:ss,ms"),
  emotion, intent, speaker. Shows: Modern Family 24,911, Friends 10,149, Big Bang Theory 9,949 utterances. Emotions:
  neutral 21,429, happy 12,622, anger 4,476, sad 2,650, surprise 1,732, fear 1,256, disgust 844. Clips are per utterance
  (`dia_<D>_utt_<U>.mp4`): zip 1 12,000 clips (22.4 GB), zip 2 18,000 (25.6 GB), rar about 15,000 (12.5 GB). **No split
  column, and the official split is not public** (authors' code uses 10-fold CV files that are not released; GitHub
  issue MC-EIU/MC-EIU#4, "No test partition", open and unanswered since Aug 2026). Decision (user): our own seeded,
  dialogue-level split with the paper's sizes 2,807 / 400 / 806 dialogues, released with the code; numbers are not
  comparable to the published 40-42 weighted F1.
- **MC-EIU extraction launched** by the user from the editor (`mc-eiu-extract` version 2, GPU T4 x2, HF_TOKEN attached).
  Kernel: self-test on 64 clips first, then the three archives, then text; MELD-identical features. Video path verified
  locally beforehand (reproduces stored MELD ResNet-18 vectors exactly on 3 clips); split logic verified (exact sizes,
  show proportions kept). Audio decoding could not be tested on the laptop (Windows Application Control blocks ffmpeg,
  exit code 0xC0E90002; not touched); the kernel's self-test covers it.
- **Context attribution (C3), first run: model B, validation, `analysis/context_attribution.py`** (paired dialogue
  bootstrap, 2,000 resamples; single model, masking is out-of-distribution for it):
  hiding the words of all context utterances: accuracy -5.08 [-8.16, -2.06], wF1 -5.69, 32 % of predictions flip;
  hiding the voice or the face of all context: no measurable effect (accuracy 0.00 [-0.99, 0.98] and +0.09 [-0.45, 0.62],
  flips 3.5 % / 1.3 %); same-speaker past, all modalities: accuracy +2.48 [0.60, 4.44] (hiding it HELPS this model);
  other-speaker past words: -1.03 [-3.07, 1.01] (not distinguishable from 0). Reading so far: the model uses its
  neighbours' words, not their voice or face. The same-speaker result must be replicated (seeds, other models, a model
  trained with context-modality dropout) before any claim.
