# VN-AV-DF Forensics (v003)

**A Vietnamese lip-sync deepfake benchmark: real speakers, six audio-visual cells, unseen generators and voices at test time.**

v003 is rebuilt from scratch. The earlier study (v001-v002, closed 14 October 2026) lives in the archived repository and its `v002` release.

## Benchmark design

| | Original audio | Spliced real audio | AI voice |
|---|---|---|---|
| **Real video** | ① real | ② sham (full + partial) | ③ AI voice: track A (whole sentence), B (one phrase, spliced); test adds C (inpainting) |
| **Lip-synced video** | ④ source | ⑤ donor | ⑥ both (lips driven by track B) |

- **Real data:** parts of 800 clips (20 men + 20 women × 20 clips, 4-7 s, 720p, audio–video offset corrected), split by speaker.
- **Generators:** Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking for train/validation/test; LatentSync 1.5 and Diff2Lip appear **only at test time**.
- **Voices:** VieNeu-TTS, F5-TTS (Vietnamese), OmniVoice, Seed-VC for train/validation/test; VoxCPM2, viXTTS and F5 inpainting appear **only at test time**.
- **Labels:** visual / audio / audio-visual mismatch intervals; `fake_intervals` is their union.
- **Per real clip:** each clip goes through half of the generators (2 of 4 in train/validation, 3 of 6 in test, balanced); 9 files in train/validation (6 with AI), 12 in test (9 with AI). About 7,800 files per part, 69% with AI. Rationale: [docs/v003_model_research.md](docs/v003_model_research.md) §10.

## Detectors

| Name | Role |
|---|---|
| `avh_align` (+ `avh_align_sup`) | **BL-1**, AVH-Align (CVPR 2025): audio-visual alignment learned from reals |
| `biolip` | **BL-2**, BioLip (2026): lip kinematics |
| `speechforensics`, `av_lipsync_plus`, `xlsr_aasist` | Reference baselines (training-free sync, supervised AV-HuBERT, audio-only) |
| `avcv` | **A1**: learned audio-lip cost volume over lags |
| `mtn` | **A2**: native-resolution mouth traces (NPR + temporal residuals, kinematics, processing-invariance augmentation) |
| `trihead` | **A3**: sync + trace + audio branches, three outputs, per-speaker reference clips |

Evaluation covers four tasks (ai / visual / audio / mismatch), per-cell, per-generator, per-voice and seen/unseen breakdowns, per-speaker AUC, localization inside partial segments, reference-normalized scores, and pre-registered gates with speaker-level bootstrap CIs. The test split is evaluated once and locked.

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Real data: plan → download → cut → identify → review → parts → export → archive |
| `generation/` | Steps 00-04: setup, plan, voice session, lip-sync sessions, finalize |
| `training/` | Steps 00-04: encoder setup, feature extraction, train, evaluate, report |
| `notebooks/` | `generate`, `prepare`, `train`, `demo` (Kaggle; W&B logging in `prepare` and `train`) |
| `src/vn_av_df/` | `generation/` (cells, plan, voice, generator adapters), `training/` (data, models, train, evaluate, report), `features/`, `demo/` (make a fake from a real video, analyze a video; Gradio), `tracking.py` (W&B) |
| `docs/` | `fake_plan.md` (what is generated, counts), `v003_model_research.md` (design rationale and sources, Vietnamese) |
| `environments/` | Isolated worker environments for generators, TTS and AV-HuBERT |
| `tests/` | Synthetic end-to-end and component tests |

How to run each step is described in **[PIPELINE.md](PIPELINE.md)** (written in Vietnamese).

## Status

- The code passes the synthetic end-to-end tests, which use mocked TTS, generators and ASR.
- The worker environments for the new generators and voices have **not been run on Kaggle yet**.
- No results yet.

## Ethics and usage limits

- Only the authors' own videos or videos used with permission. **Raw videos and fake samples of real people are not redistributed**; the data is for research only.
- Several generator and TTS checkpoints are restricted to non-commercial use.
- Detector scores are suspicion levels in 0-1 and are **not calibrated** probabilities.
