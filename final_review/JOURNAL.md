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
