# Final-review master plan

Written 2026-10-07. **Planning only: nothing in this plan is being run yet.** The pre-final package stays frozen (git tag `pre-final-review`). This plan replaces the short table in `PLAN.md`; the ground rules there still apply.

## 1. The goal, stated honestly

You asked for a plan to beat every published result on MELD and to find a contribution nobody has made. This is the most ambitious version of that I can defend. Two things come first.

- **"Beat every paper" cannot be guaranteed.** The highest MELD weighted F1 I could verify in a paper itself is 73.1 (MCN-CL). Our pre-final ensemble is 66.30, and any single result on this 2,610-utterance test set carries an interval of about ±2.2 points. Several published numbers also come without a stated number of runs or selection rule. We can plan to climb as high as honest methods allow, and we report where we land.
- **"Nobody has done it" cannot be proven.** What we can do is run a documented prior-art search for each contribution we claim, keep the queries and the closest papers in the repository, and say "to our knowledge".

| Tier | Target (weighted F1) | Published results it passes (as stated by the authors) | My rough odds |
|---|---|---|---|
| A | 68.0 or more | GraphSmile 66.71, MALN 66.9, CFN-ESA 67.2, TelME 67.37, Sync-TVA 67.40, M2FNet 67.85 | about 65% |
| B | 69.8 or more | also InstructERC 69.15, LaERC-S 69.27, MiSTER-E 69.5, xLSTM-based fusion 69.78 | about 25 to 30% |
| C | 72.0 or more | also AMuSE 71.32, DialogueLLM 71.90 (trained on three datasets), AM²-EmoJE 71.98 | about 4% |
| D | above 73.1 | also MCN-CL 73.1, which would beat every result I could verify | about 1% |

The odds are my own rough judgement from the size and overlap of the levers in section 3, not measurements. Treat them as orders of magnitude. A higher point estimate will be reported as "higher point estimate, not statistically distinguishable" unless the paired bootstrap interval excludes zero.

## 2. What the strongest published systems do (checked in the papers)

- **AM²-EmoJE (71.98):** SBERT text, PASST audio, face-and-body video through Bi-LSTMs. Single reported value.
- **AMuSE (71.32):** MPNet text, PASST audio, two Bi-LSTMs over speaker and dialogue context. Utterance level, standard test split.
- **MiSTER-E (69.5 ± 0.3):** speech and text only (no video, no speaker identity), mixture of experts, three random initialisations.
- **MCN-CL (73.1):** RoBERTa-base, openSMILE, ResNet-101 with pyramid squeeze attention. Number of runs and run selection not stated.
- **LLM route:** InstructERC (LLaMA-2 with LoRA) 69.15, LaERC-S 69.27. DialogueLLM 71.90: LLaMA-2-7B with LoRA (4.2 M trainable parameters), 5 hours on one 40 GB A100, trained on MELD, IEMOCAP and EmoryNLP with visual information turned into text, five seeds averaged.
- **SpeechCueLLM:** describing speech features in words improves an LLM (about +2 weighted F1 on IEMOCAP).
- **Audio only on MELD:** a published table lists WavLM-base at 35.2 weighted F1. Our own probe reached 44.8 on validation with WavLM-large layer 22.

What this tells us: the two big levers are a stronger text model (an LLM) and stronger audio. Several top numbers rely on extra training data or unstated protocols, so a MELD-only track must stay our primary comparison. A free T4 can run LoRA on a 7-billion-parameter model in 4-bit, slowly.

## 3. Catalogue of levers (every possibility I can see)

Gains are my estimates for our ensemble's weighted F1 (ranges, not promises). "P" is my probability that the lever helps at all. Cost is Kaggle T4 hours.

| Lever | What and why | Gain / P | Cost and risk |
|---|---|---|---|
| L1 WavLM audio in the fusion model | Probe: audio-only +10 points on validation. Use layers 22, 14, 21 of WavLM-large. | +0.3 to +1.2 / 0.7 | 1 to 2 h. Needs the test features re-extracted. |
| L2 Fine-tune WavLM-large top layers | Fine-tuning usually beats frozen probes. | +0.3 to +1.0 on top of L1 / 0.4 | 3 to 4 h. Overfitting on noisy labels. |
| L3 Dialogue-relative audio normalisation | Remove each speaker's average voice inside the dialogue before the model sees it. | +0.1 to +0.5 / 0.4 | Under 1 h. Uses only information inside the dialogue. |
| L4 Video: CLIP or face embeddings | The video head is constant "neutral" today. | 0 to +0.5 / 0.3 | 2 to 4 h. Finding the speaker's face is hard; most papers see small video gains. |
| L5 Frozen sentence or LLM embeddings as text input | Our frozen RoBERTa mean-pooled features are weak; AMuSE and AM²-EmoJE use sentence embedders. Add context prompts. | +0.2 to +1.0 / 0.6 | 2 to 3 h extraction, 2 h training. Cheap, many seeds. |
| L6 More seeds of the best families | Seed averaging gained 1.5 to 1.9 points before. | +0.2 to +0.6 / 0.8 | 6 to 10 h. Diminishing returns. |
| L7 Fine-tuning tricks | Weight averaging (EMA/SWA), adversarial training, R-Drop, layer-wise learning rates, wider context windows, speaker names. | +0.2 to +0.8 / 0.5 | 5 to 8 h. Each decided on validation. |
| L8 LLM with LoRA as an ensemble member | Ungated 7B model (Qwen2.5-7B-Instruct or Qwen3-8B), MELD only. The published LLM numbers are 69 to 72. | +0.5 to +2.0 / 0.6 | 8 to 14 h. Slowest lever. fp16 only on a T4. LLMs may have seen Friends scripts, which we disclose. |
| L9 Vocal cues in the LLM prompt | SpeechCueLLM idea, using our audio classifier's output and acoustic descriptors. | +0.2 to +0.8 over L8 / 0.4 | 2 to 3 h on top of L8. |
| L10 Extra public data (EmoryNLP, DailyDialog) | DialogueLLM did this. Reported as a separate track. | +0.3 to +1.5 / 0.4 | 4 to 8 h. Not like-for-like with MELD-only papers. |
| L11 Cross-validated stacking | Learn how to combine members, using out-of-fold predictions. | +0.2 to +0.8 / 0.3 | 8 to 12 h. Fitted weights lost to equal weights before. |
| L12 Emotion-shift or transition modelling | A transition prior or CRF over the dialogue, contrastive loss. | +0.2 to +0.8 / 0.3 | 3 to 6 h. |
| L13 Reliability-aware modality weights | Weights driven by each modality's own confidence. | 0 to +0.5 / 0.3 | 2 to 3 h. Mostly for interpretability. |
| L14 Class-prior adjustment | Chosen by cross-validation, not by the test set. | 0 to +0.4 / 0.3 | Under 1 h. Validation rejected it before. |
| L15 Headroom analysis (free) | How often is any member right? Decides whether L11 is worth running. | decides | CPU only. |

The gains overlap and shrink as they stack, so the totals in section 1 are well below the sum of this column.

## 4. Candidate contributions nobody should assume are new

Each needs a prior-art check before we claim it.

1. **N1: Which neighbours, which modality?** Test whether the other speaker's audio and video help beyond their words. A masking matrix on trained models (target alone, same-speaker past, other-speaker past, other-speaker future, each with text only or all modalities), plus training with context-modality dropout. This answers your original idea with data. Cost: evaluation only. Known neighbours: speaker-aware conversation models (ICON, DialogueRNN, DialogueGCN, MMGCN, AMB-DSGDN), which use speaker information but do not isolate role by modality.
2. **N2: Audit and repair of adaptive modality weights.** We showed the weights are nearly constant and the video head is constant. Repair with stronger unimodal encoders (L1, L4) and reliability-aware weights (L13), and measure weight spread and the informativeness of disagreement. Neighbours: AMB-DSGDN, Ada2I and several 2026 modality-balance papers; the audit is the new part.
3. **N3: Prediction sets for multimodal conversation emotion recognition, checked by modality disagreement.** Neighbour: conformal prediction for text-only ERC on MELD (Roohi et al., 2025). Ours would be multimodal, with coverage reported per disagreement level and per class.
4. **N4: Dialogue-relative audio normalisation (L3).** Neighbour: speaker normalisation in speech emotion recognition. New only as a measured step in the conversational pipeline.
5. **N5: Evidence in the prompt.** An LLM that sees calibrated single-modality predictions and conflict flags, and writes a short rationale. Neighbour: SpeechCueLLM. High risk, so it comes last.

**Prior-art protocol.** For each claim, run at least five queries on arXiv, the ACL Anthology and Semantic Scholar, and record the queries and the closest papers in `final_review/prior_art.md`. State the claim as "to our knowledge", with the closest work cited.

## 5. Schedule with decision gates

| Phase | Work | T4 hours | Gate at the end |
|---|---|---|---|
| 0 (done) | Separate branch, plan, audio probe (+10 audio-only), test-split bug found | 1 | none |
| 1: quick wins (2 to 3 working days) | Re-extract test audio (fix the failed job), L1 with 3 seeds against a control, L5 frozen embeddings, L15 headroom, N1 masking matrix, selective prediction | 8 to 10 | Keep audio only if fused validation weighted F1 rises by at least 0.5. Keep embeddings only if they beat the control by 1.0 or add at least 0.3 to the validation pool. |
| 2: strengthen (3 to 4 days) | L2, L3, L4, L6, L7, L13 | 12 to 16 | Decide whether the LLM budget is worth spending: go if the validation pool is still below the Tier A target. |
| 3: the big bet (4 to 6 days) | L8, L9, L10 as separate tracks | 12 to 20 | Keep an LLM member only if it raises the validation pool. |
| 4: freeze and write (2 to 3 days) | Frozen candidate list, one test run, paired bootstrap, updated comparison table, report chapters, deck, viva sheet, demo with the new model, zip | 1 to 2 | none |

Total: roughly 35 to 45 T4 hours, which is about two weekly quotas (30 hours each) and two to three weeks of wall-clock time. If the review date is close, the compressed version is Phase 1, the best part of Phase 2, and Phase 4, about 5 to 7 days.

## 6. Protocol and safeguards

- Every choice (layers, models, seeds to keep, windows, ensemble members) is made on validation. The test set is scored once, by a single script that reads a frozen candidate list committed before the run.
- At least 3 seeds for any claimed improvement; mean and spread reported; a control trained with the same script and seeds, never model B itself.
- A **MELD-only track** is the primary comparison. Anything using extra data (L10) is a clearly labelled second track.
- Disclose that large language models may have seen Friends dialogue during pre-training.
- Report every run, including failures, in `final_review/JOURNAL.md`.
- Stop rule: if Phase 1 shows no gain, stop chasing accuracy. Spend the remaining time on N1, N2, N3 and the documents.

## 7. Logistics

- **Compute:** Kaggle T4 (30 hours a week, sessions up to 12 hours), Colab as a backup. The laptop only runs light analyses and the demo.
- **Models:** open weights that need no sign-in (Qwen2.5-7B-Instruct, Qwen3-8B; confirm on each model page before use), so no access token has to be handled. I never type or read credentials.
- **Disk:** about 7 GB free on the laptop. Big outputs stay on Kaggle and are chained between kernels; only small result files are downloaded.
- **Checkpoint hygiene:** new file names, `TRAIN_OUTPUT_PATH` always set, model B never overwritten.
- **Known open issue:** the test-feature kernel (`meld-audio-test-features`) ended with an error and an empty log, with no failure message. The script passes a syntax check and imports locally. The likely cause is that the kernel title does not match its id (Kaggle warned about it). Fix on restart: title it "MELD Audio Test" so the slug equals the id, then check that a log appears.

## 8. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| LoRA run runs out of memory or time on a T4 | Smaller LLM (3 to 4 billion parameters), shorter context, checkpoint every epoch |
| fp16 training produces NaN (it did with DeBERTa) | Abort on a non-finite loss, upcast sensitive layers, lower the learning rate |
| Gains vanish on test | Report it. The audit, N1, N3 and the positioning table still stand |
| Too many choices inflate the winner's curse | Frozen candidate list, selection on validation only, the unselected average reported beside the headline |
| Kaggle quota or session limits | Chain kernels, save per-epoch outputs, use Colab for small jobs |
| Review date arrives early | Use the compressed schedule in section 5 |

## 9. Working as two people

**Proposed split (change it as you agree).** The aim is that each of you owns work you can truthfully describe and explain.

- **Hemanth:** audio track (L1 to L3), frozen embeddings (L5), seeds and the ensemble protocol (L6, section 6).
- **Pranitha:** prior-art checks and the comparison table (section 4), the analyses N1 and N3, the video track (L4), the demo upgrade, and the report and slide chapters for the final review.
- **Both:** one 15-minute walk-through each week where each explains the other's package, so either of you can answer any question.

**How the work is credited.** Every experiment gets an owner in the journal. If Pranitha wants her work visible in the repository, add her as a collaborator and have her commit from her own account, or commit with `--author` set to her name. Today the history shows only one author.

**What to tell the guide.** Say what each person actually did, in plain words, by area: literature and base-paper choice, data and features, model and training, experiments and evaluation, analysis and demo, report and slides. For each area name who led it and who helped.

**Be open about the AI assistant.** As of 7 October, all 67 commits in the repository are authored by Hemanth S, and 66 carry a "Co-Authored-By: Claude" line that GitHub shows. So the honest and safest line is to say that an AI coding assistant helped implement the project under your direction, and to check your course's policy on AI assistance. Then describe what you decided, ran, checked and can explain: which base paper, which experiments to run or drop (for example dropping graph fusion), what the results mean, and how to run the demo. Do not describe code as hand-written if it was not.

Template to fill in with the truth:

- *Hemanth S:* led ______; helped with ______; I can explain and demonstrate ______.
- *Pranitha N.S:* led ______; helped with ______; she can explain and demonstrate ______.
- *Both:* decisions we made together: ______. AI assistance: ______.

## 10. Ready to go (what happens when you say "go")

Already in place: separate worktree and branch, the frozen pre-final tag, this plan, the audio probe results (+10 audio-only on validation), the probe and extraction scripts, the evaluation scripts and committed caches, the working demo.

To write when we start: audio-path support in the loader, dataset and trainer; a training kernel that joins the Kaggle feature outputs and runs 3 seeds against a control; the frozen-embedding kernel; the LoRA kernel; the masking-matrix script (N1); the selective-prediction and headroom scripts; the frozen candidate file and the single test-scoring script.

First three actions: (1) restart the test-feature kernel with the title fix; (2) write and run the Phase 1 training kernel; (3) run the headroom analysis and the masking matrix locally on the CPU.

**What I need from you:** the date of the final review, who does what (section 9), whether extra public data is acceptable for a second track, and whether you want the LLM track, which is the most expensive.

## Sources

- AMB-DSGDN, arXiv 2603.10043. GraphSmile, arXiv 2407.21536. Sync-TVA, arXiv 2507.21395. MiSTER-E, arXiv 2602.23300. AMuSE, arXiv 2401.15164. AM²-EmoJE, arXiv 2402.10921. MCN-CL, arXiv 2511.10892.
- DialogueLLM, arXiv 2310.11374. InstructERC (github.com/LIN-SHANG/InstructERC). LaERC-S, arXiv 2403.07260. SpeechCueLLM, arXiv 2407.21315.
- Exposing Weaknesses in Emotion Recognition in Conversations, arXiv 2609.05806. Conformal prediction for ERC, Roohi et al., Natural Language Processing 2025.
