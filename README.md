# VN-AV-DF Forensics (v006)

**A Vietnamese lip-sync deepfake benchmark: real speakers, six audio-visual cells, unseen generators and voices at test time.**

| Version | Branch | What it is |
|---|---|---|
| v001-v002 | `v001`, `v002` | earlier study on one pilot part (closed 14 October 2026) |
| v003 | `v003` | the benchmark rebuilt from scratch; detectors M1, M2 and baselines on part 1 |
| v004 | `v004` | measured mismatch labels, the M3 lip–audio–head detector, all three parts (closed 7 November 2026): [docs/v004_report.md](docs/v004_report.md) |
| v005 | `v005` | **M3-R**, a detector trained on real re-compressed video (480p, 360p) with paired clean ↔ re-encoded consistency, and a resolution gate (closed 11 November 2026, seed 42): **[docs/v005_report.md](docs/v005_report.md)** |
| **v006** | `main` | the v005 roster retrained on seed 43 with extra validation points inside the first three epochs, so the learning curves start from chance level; report in [docs/v006_report.md](docs/v006_report.md) |

## Benchmark design

| | Original audio | Spliced real audio | AI voice |
|---|---|---|---|
| **Real video** | ① real | ② sham (full + partial) | ③ AI voice: track A (whole sentence), B (one phrase, spliced); test adds C (inpainting) |
| **Lip-synced video** | ④ source | ⑤ donor | ⑥ both (lips driven by track B) |

- **Real data:** parts of 600-800 clips (20 men + 20 women × 15-20 clips, 4-7 s, 720p, audio-video offset corrected), split by speaker.
- **Generators:** Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking for train/validation/test; LatentSync 1.5 and Diff2Lip appear **only at test time**.
- **Voices:** VieNeu-TTS, F5-TTS (Vietnamese), OmniVoice, Seed-VC for train/validation/test; VoxCPM2, viXTTS and F5 inpainting appear **only at test time**.
- **Labels:** visual / audio / audio-visual mismatch intervals; `fake_intervals` is their union.
- **Per real clip:** each clip goes through half of the generators (2 of 4 in train/validation, 3 of 6 in test, balanced); 9 files in train/validation (6 with AI), 12 in test (9 with AI). About 5,850-7,800 files per part, 69% with AI. Rationale: [docs/v003_generation.md](docs/v003_generation.md) §4.

## Detectors

Baselines are re-implemented from the authors' released code (differences listed in the model docstrings and in [docs/v003_model.md](docs/v003_model.md)). Names in **bold** are trained in v005 (core steps).

| Name | Role |
|---|---|
| `lipinc_v2` | **BL-V**, LIPINC-V2 (Datta et al. 2025): mouth-region vision temporal transformer (visual fake) |
| `ssl_aasist` | BL-A, XLS-R 300M fine-tuned with AASIST + RawBoost (Tak et al. 2022) (audio fake); did not learn in v003, optional in v004 |
| `xlsr_frozen` | **BL-A2**, frozen XLS-R 300M with a learned mix of 5 layer groups (v004) |
| `avh_align` | **BL-M**, AVH-Align (CVPR 2025): audio-visual alignment learned from reals only, initialized from the released AV-1M checkpoint |
| `avh_align_sup` | **BL-M/sup**, AVH-Align/sup: same network trained with labels |
| `speechforensics` | **BL-SF**, SpeechForensics (no training) |
| `bl_ens`, `bl_ens_av` | BL-Ens = max(BL-V, BL-A); **BL-Ens-AV** = max(BL-M/sup, BL-A2); z-normalized on validation reals (no training) |
| `trihead_v2` | M1 (v003): sync (lag cost volume) + mouth trace (NPR, head pose) + audio (learned XLS-R layer weights) branches, three outputs, per-speaker reference clips, pseudo-fake audio replacement, speaker gradient reversal; not in the v005 roster (code kept) |
| `avh_align_plus` | **M2**: AVH-Align scorer over a ±15-frame lag volume + lip–head motion + pseudo-fakes |
| `trihead_v4` | M3 (v004): relative / absolute sync streams, AVH-Align Φ channel, appearance, prosody ↔ head-motion branch, mismatch / offset / cell-type heads; not in the v005 roster (code kept) |
| `trihead_v5` | **M3-R** (v005): M3 without prosody, appearance and speaker loss; trained on clean / re-encoded pairs (480p and 360p) with a consistency loss, sham clips as hard negatives for the visual head, XLS-R layer and channel dropout; at inference a **resolution gate** skips the visual evidence below 360p (AI? from the audio head) |
| `trihead_v5_no_pair`, `_quality`, `_clean` | M3-R ablations: random compressed copies without pairing (**trained in v005**); with a quality-conditioned FiLM branch; trained on clean video only (both optional, not run yet) |

Evaluation covers four tasks (ai / visual / audio / mismatch), per-cell, per-generator, per-voice and seen/unseen breakdowns, per-speaker AUC, localization inside partial segments (visual and mismatch), reference-normalized scores, a delayed-audio stress test, operating points at 5% validation FPR, and pre-registered gates with speaker-level bootstrap CIs. Since v004 the mismatch label of lip-synced clips is **measured** (SyncNet at lag 0 against real clips) instead of set by construction, and each model is tested right after training (every test use is logged in `test-log.json`). Since v005 every model is also trained with **re-encoded copies** (`settings.VARIANTS`: 480p libx264 CRF 30 + AAC 64 kbps, 360p CRF 35 + AAC 32 kbps) and tested on them and on two levels never seen in training (360p VP9 + Opus, 288p), with false-alarm rates at the clean-validation threshold and gates R1-R8, R2b, R2c. A **processed-real** set (real, sham and voice-only clips passed through benign codec / rescaling chains like those inside the generator pipelines, labels kept; three of the seven chain families appear only at test time) checks whether a detector finds AI manipulation or just the processing trace of the fake pipeline.

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Real data workspace: step scripts (plan → download → cut → identify → review → parts → export → archive), `configs/`, `checkpoints/`, `data/` (ledger, raw, pool), `exports/` |
| `generation/` | Steps 00-04: setup, plan, voice session, lip-sync sessions, finalize |
| `training/` | Steps 00-04: encoder setup, feature extraction, train, evaluate, report |
| `notebooks/` | `generate`, `prepare`, `train`, `demo` (Kaggle; W&B logging in `prepare` and `train`) |
| `src/vn_av_df/` | `generation/` (cells, plan, voice, generator adapters), `training/` (data, models, train, evaluate, report), `features/`, `demo/` (make a fake from a real video, analyze a video; Gradio), `tracking.py` (W&B) |
| `src/vn_av_data/` | Real data pipeline code (ledger, cut, face / VAD / SyncNet / voice, review page, split, export), run by `data_pipeline/steps/` |
| `docs/` | Vietnamese design notes with rationale and sources: `v003_data.md` (real data pipeline, people and clip counts, audio-visual offset), `v003_generation.md` (six cells, generators and voices, counts per part), `v003_model.md` (methods, baselines, proposed models, evaluation); English `v004_report.md` and **`v005_report.md`** with their generated tables in `v004_results/`, `v005_results/` and figures in `figures/` |
| `environments/` | Isolated worker environments for generators, TTS and AV-HuBERT |
| `tests/` | Synthetic end-to-end and component tests; `tests/data_pipeline/` for the real data pipeline |

How to run each step is described in **[PIPELINE.md](PIPELINE.md)** (written in Vietnamese).

## Status (v005, closed 11 November 2026)

The numbers below are the closed v005 round (seed 42). v006 (14 November 2026) retrained the same roster on seed 43; see the replication note at the end of this section. Trained and tested once on Kaggle (3 parts, 31 unseen test speakers, 7,368 clips, seed 42); full report in [docs/v005_report.md](docs/v005_report.md).

| AI? AUC (test) | clean 720p | 480p | 360p |
|---|---|---|---|
| **M3-R** (`trihead_v5`) | **0.973** | **0.962** | **0.914** |
| M3-R without pairing | 0.972 | 0.959 | 0.911 |
| BL-Ens-AV | 0.935 | 0.913 | 0.848 |
| M2 | 0.930 | 0.907 | 0.851 |

- Pre-registered gates with speaker-bootstrap 95% CIs:
  - M3-R − BL-Ens-AV: +0.039 / +0.050 / +0.065 (clean / 480p / 360p);
  - M3-R − M2 at 480p: +0.055;
  - the consistency loss adds only +0.004 [+0.001; +0.006] at 480p.
- At the 5% clean-validation FPR threshold, M3-R flags 1.5% of clean and 2.1% of 480p real test clips, but 8.5% at 360p.
- M3-R does not treat misaligned real audio (sham, delayed audio) as AI.
- Unseen voices remain the weakest case: VoxCPM2 0.905, F5 inpainting 0.911.
- Not run in this round: R4 (clean-only training), R5 (quality branch), and the VP9 / processed-real evaluation variants.
- **Seed 43 replication (v006, [docs/v006_report.md](docs/v006_report.md)).**
  - M3-R AI? AUC is 0.971 / 0.960 / 0.907, and the gates R1, R2, R2b and R3 pass again.
  - Per-model AUCs differ by at most 0.009 between the two seeds.
  - The consistency-loss gain does not replicate: R6 is −0.001 [−0.004; +0.002].
  - The audio non-inferiority gate R8 misses by 0.00004: the CI lower bound is −0.01004.
  - Dense validation learning curves: [docs/figures/v006/c04b_dynamics.png](docs/figures/v006/c04b_dynamics.png).

## Ethics and usage limits

- Only the authors' own videos or videos used with permission. **Raw videos and fake samples of real people are not redistributed**; the data is for research only.
- Several generator and TTS checkpoints are restricted to non-commercial use.
- Detector scores are suspicion levels in 0-1 and are **not calibrated** probabilities.
