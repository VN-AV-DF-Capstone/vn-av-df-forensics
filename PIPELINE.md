# VN-AV-DF v003 - hướng dẫn chạy pipeline

```
data_pipeline (real, local) ─► part sạch vn-av-df-data-partN (800 clip, 20 nam + 20 nữ × 20 clip, split sẵn)
  └─► notebooks/generate.ipynb   phiên "voice" → các phiên "lipsync" → finalize      (generation/0x_*.py)
  └─► notebooks/prepare.ipynb    AV-HuBERT Large → crop miệng → XLS-R                (training/01_prepare.py)
  └─► notebooks/train.ipynb      train 6 baseline + 3 đề xuất → test một lần → report.md  (training/02-04)
```

Mỗi bước chạy được ở local bằng script trong `generation/` và `training/`, hoặc trên Kaggle bằng notebook tương ứng; cả hai cách gọi cùng các lệnh trong `src/vn_av_df/cli.py`. Cấu hình nằm ở `settings.py` (local) và các cell đầu của notebook (Kaggle). Pipeline dữ liệu real: [data_pipeline/README.md](data_pipeline/README.md).

| Bước | Script local | Code |
|---|---|---|
| Tải source / weight generator | `generation/00_setup.py [--generators ...]` | `generation/lipsync.py`, `wav2lip.py`, `musetalk.py` |
| Kế hoạch sinh | `generation/01_plan.py` | `generation/plan.py` |
| Phiên giọng (①②③ + track B) | `generation/02_voice.py` | `generation/voice/`, `generation/generate.py` |
| Phiên lip-sync (④⑤⑥) | `generation/03_lipsync.py --generators ...` | `generation/lipsync.py`, `generation/generate.py` |
| Gom → manifest có nhãn | `generation/04_finalize.py` | `generation/generate.py` |
| Tải encoder | `training/00_setup.py` | `assets.py` |
| Trích feature | `training/01_prepare.py [--encoders ...]` | `training/prepare.py`, `features/` |
| Train | `training/02_train.py [--models ...] [--seeds ...]` | `training/train.py`, `training/models.py` |
| Test một lần | `training/03_evaluate.py` | `training/evaluate.py` |
| Báo cáo | `training/04_report.py` | `training/report.py` |

Môi trường worker (generator, TTS, AV-HuBERT): [environments/README.md](environments/README.md).

## 6 cell và số sample mỗi clip real

| | ORIGINAL AUDIO | SPLICED REAL AUDIO | AI VOICE |
|---|---|---|---|
| **REAL VIDEO** | ① real | ② sham (full + partial) | ③ giọng AI: track A (cả câu), B (một cụm, ghép); test thêm C (sửa trực tiếp) |
| **LIP-SYNCED VIDEO** | ④ source | ⑤ donor | ⑥ both (vẽ môi theo track B) |

- Mỗi clip chạy **một nửa số generator** (2 / 4 ở train và validation, 3 / 6 ở test), mỗi generator một lần, cho ra full + partial. Gán theo vòng: mỗi generator chạy số lần bằng nhau và chia đều ④/⑤/⑥; trong một clip các generator ở ô khác nhau.
- Hệ giọng xoay vòng theo clip (1 hệ mỗi clip).
- Chi tiết và số liệu: [docs/fake_plan.md](docs/fake_plan.md); căn cứ: [docs/v003_model_research.md](docs/v003_model_research.md) §10.

| | Mỗi clip real |
|---|---|
| train / validation | 9 file (6 có AI): 1 real + 2 sham + 2 giọng + 2 generator × 2 |
| test | 12 file (9 có AI): 1 real + 2 sham + 3 giọng + 3 generator × 2 |

Bước finalize ghi thêm `shortcut_audit.json`: khoảng lặng đầu / cuối, tỉ lệ lặng, độ to, độ dài (một mình) có đoán được nhãn AI không. AUC ngoài [0,3; 0,7] là cờ đỏ, phải sửa cách sinh trước khi train.

## Generator và giọng

| | train / validation / test | chỉ test (chưa thấy khi train) |
|---|---|---|
| Generator | wav2lip_gan, musetalk_1_5, iplap, videoretalking | latentsync_1_5, diff2lip |
| Giọng | vieneu (VieNeu-TTS v3), f5_vi (F5-TTS tiếng Việt), omnivoice, seedvc (TTS → Seed-VC) | voxcpm2, vixtts; f5_inpaint (track C) |

Danh sách nằm ở `settings.py` (`TRAIN_GENERATORS`, `TEST_ONLY_GENERATORS`, `TRAIN_VOICES`, `TEST_ONLY_VOICES`).

- **Higgs Audio v3 đã được thay bằng viXTTS.** Higgs cần server SGLang-Omni (Python 3.12, benchmark trên H100), không chắc chạy được trên T4. viXTTS thuộc họ GPT + HiFi-GAN, khác họ với các hệ dùng khi train.
- **IP-LAP chỉ phát hành weight qua OneDrive.** Tải tay `landmarkgenerator_checkpoint.pth` và `renderer_checkpoint.pth`, upload thành Kaggle dataset, rồi đặt `IPLAP_CHECKPOINTS` trong notebook generate (local: `cfg["generation"]["iplap_checkpoints"]`).
- Môi trường worker của 4 generator mới và 6 hệ giọng **chưa chạy thử trên Kaggle**.

## Sinh trên Kaggle (notebook generate)

1. **Phiên giọng** (`PHASE = "voice"`, cần GPU). Chạy **trước**, vì cell ⑥ cần track B.
2. **Các phiên lip-sync** (`PHASE = "lipsync"`, `GENERATORS = [...]`), mỗi phiên vài generator. Phiên sau attach Output của phiên trước vào `PREVIOUS_OUTPUT`; job đã xong được bỏ qua.
3. Phiên cuối (đủ mọi job) tự finalize → `datasets/vn-av-df-data/<part>/manifest.jsonl`.

Mỗi job ghi `records/<key>.json`. Hết giờ (`TIME_BUDGET_HOURS`) hoặc lỗi thì chạy lại, job dở được xoá và làm lại. Lỗi riêng một clip (không thấy mặt, generator báo lỗi) thì bỏ job đó; bỏ 10 job liên tiếp thì dừng.

## Model

Chi tiết ở [docs/v003_model_research.md](docs/v003_model_research.md).

| Tên | Loại | Ý chính |
|---|---|---|
| speechforensics | baseline (NeurIPS 2024) | cos audio / visual AV-HuBERT, độ trễ cố định tốt nhất ±15, không train |
| avh_align / avh_align_sup | **BL-1** (CVPR 2025) | MLP căn tiếng-môi; InfoNCE chỉ trên real / bản có giám sát |
| biolip | **BL-2** (2026) | động học landmark môi, không phụ thuộc ngôn ngữ |
| av_lipsync_plus | baseline | AV-HuBERT + TCN, có giám sát (≈ B-AVH v002) |
| xlsr_aasist | baseline audio | XLS-R + AASIST-lite |
| avcv | **A1** | cost volume tiếng-môi theo độ trễ + CNN 2D → mismatch, visual |
| mtn | **A2** | crop miệng độ phân giải gốc, residual NPR + thời gian + động học môi, augment bất biến xử lý |
| trihead | **A3** | 3 nhánh (sync, trace, audio) → 3 đầu ra + tham chiếu người (FiLM) |

`settings.MODELS` là đội hình chính; ablation ở `settings.ABLATIONS`.

## Kết quả

Đánh giá chấm 4 tác vụ: ai / visual / audio / mismatch. Kèm theo:
- phân rã theo cell, generator, giọng, full / partial, đã thấy / chưa thấy khi train;
- AUC trong từng speaker;
- định vị trong đoạn partial (ô cao nhất có nằm trong đoạn, ô sát chỗ nối so với ô giữa);
- định vị chuẩn LAV-DF / AV-Deepfake1M: **AP theo IoU 0,5 / 0,75 / 0,95 và AR@1/5/10**, để so được với các bài khác;
- chế độ **ref-norm**: chuẩn hoá điểm theo 3 clip real tham chiếu của chính người đó (bỏ các clip đó và mọi file sinh từ chúng khỏi phép đo);
- **điểm vận hành**: ngưỡng FPR 1 % / 5 % trên real validation, TPR và báo nhầm ở test, precision khi fake chiếm 50 % / 10 % / 1 % ngoài thực tế; báo nhầm riêng cho real và sham. Lý do: tỉ lệ fake trong dữ liệu (~70 %) cao hơn nhiều ngoài đời, nên ngưỡng F1 tối đa sẽ báo nhầm quá nhiều (Layton et al., USENIX Security 2024). Demo dùng ngưỡng FPR 5 %.

`runs/<RUN_NAME>/report.md` gồm bảng mean ± std theo seed và các **cổng** đặt trước, mỗi cổng có ΔAUC kèm CI bootstrap theo speaker. Test chỉ chạy một lần và được khoá bằng `test-lock.json`.

## Log W&B (quá trình train)

Chỉ ghi cho **quá trình train**: notebook `prepare` và `train` (`USE_WANDB = True`, secret `WANDB_API_KEY`, project **private**). Pipeline dữ liệu real và phần sinh fake **không** ghi W&B; nhật ký của chúng nằm ở ledger (`data_pipeline`) và `generation.json` / `records/` (sinh fake). Không đẩy ảnh / video người thật. Code ở `src/vn_av_df/tracking.py`.

| Bước (job_type) | Ghi theo thời gian | Ghi lúc kết thúc / artifact |
|---|---|---|
| `prepare` | tiến độ, mẫu/phút theo encoder | tỉ lệ frame hợp lệ; **kiểm chéo độ lệch tiếng–hình bằng AV-HuBERT** trên clip real; `extraction.json` |
| `train` (mỗi model × seed) | mỗi epoch: loss tổng + từng đầu (frame / video), grad norm, lr, giây/epoch, VRAM, **AUC validation 4 tác vụ**, AUC validation (ai) theo ô / generator / giọng / kind | epoch tốt nhất, lý do dừng, giờ train, số tham số, có augment hay không; `best.pt` + `history.json` |
| `evaluate` (một lần) | — | bảng tổng (AUC, AP, EER, F1, FAR), phân rã, theo người, ref-norm, định vị, **ROC**, histogram điểm theo ô; `evaluation.json` + `test-lock.json` + mọi `metrics.json` + `predictions-test.npz` |
| `report` | — | bảng AUC mean ± std theo seed, **các cổng + CI 95%**; `report.md` + `gates.json` |

Mỗi run có config (cấu hình, seed, hash dữ liệu / test-lock) để tái lập. Run id lưu ở `wandb-<tên>.json` cạnh kết quả, chạy lại sau khi bị ngắt sẽ ghi tiếp đúng run.

## Demo (notebook `demo.ipynb`, Kaggle)

Chạy ở phiên tương tác có GPU, mở link `gradio.live` in ra ở cell cuối. Code ở `src/vn_av_df/demo/`.

| Tab | Làm gì |
|---|---|
| 1. Tạo fake từ video real | video ≤ 60 s → chọn ô ③ voice / ④ source / ⑤ donor (tải thêm tiếng thật khác) / ⑥ both, full hoặc partial (đoạn tự chọn hoặc nhập giây), generator, hệ giọng, câu mới cho cụm sửa → video fake + nhãn thật. Dùng lại đúng các hàm của pipeline sinh |
| 2. Phân tích video | chuẩn hoá → **đo và sửa độ lệch tiếng–hình bằng SyncNet** (báo riêng nếu lệch vượt ngưỡng BT.1359) → đặc trưng như lúc train → mỗi model chạy theo cửa sổ 7 s → điểm 4 tác vụ, kết luận theo ngưỡng validation (`metrics.json`), timeline có vẽ chồng nhãn thật. Thêm 1–3 video real của cùng người để dùng tham chiếu (TriHead) |

- Mặc định cài Wav2Lip + LatentSync 1.5 và F5-TTS tiếng Việt; muốn generator / giọng khác thì sửa `GENERATORS`, `VOICES` (mỗi môi trường cài thêm 5–15 phút).
- Chưa có checkpoint: `RUN_DIR = None` → chỉ chạy `speechforensics` (không cần train) để thử luồng.
- Video tối đa 60 s (giới hạn của bộ trích AV-HuBERT). Link `gradio.live` là công khai trong thời gian chạy: chỉ dùng video được phép, tắt khi xong.

## Thời gian / dung lượng (ước lượng)

- **Một part** (800 clip): 1.800 lần chạy generator và 800 job giọng; khoảng 7.800 file.
- **GPU cho sinh fake:** khoảng 55-100 giờ cho mỗi part (gần 2-4 tuần quota Kaggle). Chia phiên theo generator để chạy song song trên nhiều notebook, ghép bằng `PREVIOUS_OUTPUT`.
- **Trích feature và train:** vài giờ.

## Kiểm thử

```bash
python -m pytest            # tests/ (pipeline tổng hợp end-to-end) + data_pipeline/tests
python -m ruff check .
```

Test dùng TTS / generator / ASR giả lập, không thay cho việc chạy thử trên vài clip thật.
