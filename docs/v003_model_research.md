> Bản sao từ nhật ký nghiên cứu (`RESEARCH_WORKLOG/v003_model_research.md`, ngoài repo) ngày 18/10/2026. Sửa ở nhật ký rồi chép lại vào đây.

# v003 · Research kiến trúc model

Cập nhật 15/10/2026. Phạm vi: detector cho benchmark lip-sync tiếng Việt v003, gồm 6 cell và 3 câu hỏi cho mỗi sample:
- **visual fake?** (hình bị AI vẽ)
- **audio fake?** (giọng AI)
- **AV mismatch?** (tiếng và môi lệch nhau)

Quy ước:
- Số liệu lấy từ chính paper; khác điều kiện thì không so trực tiếp.
- "Chưa tìm thấy" nghĩa là chưa tìm thấy tính đến ngày trên, không khẳng định là chưa ai làm.
- Kết quả v001/v002 chỉ là **bằng chứng phụ**, vì test chỉ có 2-3 người.

Cách làm như v002: bài toán → lỗi X (lấy từ literature) → ngành khác giải X bằng Z → mang Z về, kèm đánh giá khả thi trên Kaggle 2×T4. Nguồn tìm: arXiv 2024-2026, CVPR/NeurIPS/ICME/ACM MM/EACL, Interspeech/ICASSP, tạp chí tiếng Trung (中国图象图形学报), luận văn tiếng Trung phồn thể (NCCU).

---

## 1. Toàn cảnh phương pháp

| Phương pháp | Năm · nơi | Ý chính | Train trên | Kết quả nổi bật (theo paper) | Code |
|---|---|---|---|---|---|
| **AVH-Align** | CVPR 2025 | AV-HuBERT (Large, fine-tune VSR) cho feature audio/visual riêng → MLP 1024→512→256→128→1 chấm độ khớp từng frame; InfoNCE trên ±15 frame; điểm video = log-sum-exp(−Φ_ii) | **Chỉ real** | Bền với lối tắt leading silence, tốt nhất AV-Deepfake1M ở nhóm real-only. Train FakeAVCeleb → test AV-1M chỉ 57,7 AUC | ✔ (bit-ml) |
| **SpeechForensics** | NeurIPS 2024 | AV-HuBERT Large **không fine-tune**; cosine audio/visual từng frame, lấy max theo độ lệch ±τ (τ≈15) | Chỉ real (không train detector) | FF++ 97,6, FakeAVCeleb 99,0 (Large) | ✔ |
| **BioLip** | arXiv 2026 | Chỉ landmark môi (64 điểm), y-coordinate, 4 bậc động học (vị trí, vận tốc, gia tốc, jerk) + thống kê + tỉ lệ vùng → CNN-1D 302K tham số | AVLips (W2L, TalkLip, SadTalker) | LipSyncTIMIT zero-shot TB **0,927** (W2L 0,97 · VideoReTalking 0,96 · Diff2Lip 0,93 · **IP-LAP 0,82**); CRF40 còn 0,76; 7 ngôn ngữ TB 0,83 | Chưa thấy |
| LIPINC / LIPINC-V2 | 2024 / 2025 | Bất nhất miệng theo thời gian; V2 dùng ViT + cross-attention | Lip-sync | LipSyncTIMIT TB 0,96 (CRF23) / 0,76 (CRF40) | ✔ |
| LoCC | ICME 2026 | So từng frame với "frame phản thực" dự đoán từ frame lân cận; teacher-student; định vị tới frame | LAV-DF, AV-1M | Theo paper hơn LIPINC trên LAV-DF / AV-1M / FakeAVCeleb / KoDF | ✔ |
| PIA | arXiv 2025 | Phoneme (WhisperX) + viseme (3D-CNN + EfficientNet) + độ mở môi + ArcFace theo thời gian | FakeAVCeleb, DeepSpeak | FakeAVCeleb AUC 99,8; DeepSpeak 98,1. **Pipeline phoneme chỉ tiếng Anh** | ✔ |
| LipFD | NeurIPS 2024 | Phổ âm + chuỗi frame miệng, hai đầu | AVLips | Trong miền >95%; **chéo bộ ~0,54** (X-AVDT đo lại) | ✔ |
| AV-Lip-Sync+ (≈ B-AVH của mình) | 2023 | AV-HuBERT + TCN đa tỉ lệ, có giám sát | Có fake | Mốc có giám sát | ✔ |
| X-AVDT | arXiv 2026 | Đảo ngược diffusion (Hallo) lấy cross-attention AV | MMDF | Chéo generator 95,3 AUROC, nhưng **~1 phút / clip 16 frame** | Trang dự án |
| KLASSify (hạng 3 AV-1M++ 2025) | 2025 | SSL audio + graph attention (kiểu AASIST) + đặc trưng hình thủ công | AV-1M++ | 92,8 AUC; định vị chỉ bằng audio IoU 0,35 | - |
| Pindrop (hạng 1 định vị AV-1M++) | 2025 | Ensemble mạng chuyên biệt + định vị kiểu ActionFormer | AV-1M++ | Điểm định vị 67,2 | - |
| XLS-R + AASIST (và XLSR-SLS, XLSR-Mamba) | 2022-2026 | SSL đa ngôn ngữ (128 ngôn ngữ, **có tiếng Việt**) + graph attention | ASVspoof | Mạnh trong miền, **yếu với hệ TTS chưa thấy** | ✔ |
| POI-Forensics | 2022 | Học danh tính AV từ real, so với video tham chiếu của chính người đó | Chỉ real | Bền với video chất lượng thấp | ✔ |
| AV-LMMDetect / Omni-Fake-R1 / ConLLM | 2026 | MLLM (Qwen2.5-Omni) fine-tune, có giải thích | FakeAVCeleb, MAVOS-DD | 85,1% acc open-set (base 32,3%) | Một phần |

## 2. Lỗi của detector hiện nay (bằng chứng chính từ literature)

| # | Lỗi | Bằng chứng | Phụ (v001/v002) |
|---|---|---|---|
| X1 | **Học thuộc dấu vết pixel của generator đã thấy** | BioLip: Xception 0,93 → 0,67 khi nén; LipFD chéo bộ 0,54; MesoInception ≈ ngẫu nhiên với generator lạ | P2: W2L 0,98 → MuseTalk 0,55 |
| X2 | **Lối tắt của dữ liệu** (leading silence, khác biệt nén / kích thước) | Smeu CVPR25: chỉ nhìn im lặng đầu clip đạt >98%; Fake-or-JPEG ECCVW24 | E6: nhánh artifact báo nhầm 79-100% clip chỉ bị xử lý |
| X3 | **Phụ thuộc ngôn ngữ** | BioLip: LipFD kém rõ trên tiếng Trung; PIA, TISAN chỉ chạy với phoneme tiếng Anh; AV-HuBERT pretrain bằng tiếng Anh | - |
| X4 | **Sụp khi nén / nhiễu** | BioLip: mọi phương pháp giảm 0,17-0,25 ở CRF40; SAVe 0,77 ở c40 | - |
| X5 | **Generator làm mượt theo thời gian** (IP-LAP, diffusion) | BioLip yếu nhất với IP-LAP (0,82 → 0,53 ở CRF40); X-AVDT: bộ dữ liệu cũ chủ yếu GAN | - |
| X6 | **Gộp thẳng điểm "lệch tiếng" vào điểm "AI"** | AVH-Align / SpeechForensics chỉ đo độ khớp, nên lệch tiếng thật (sham) cũng bị báo; BioLip không chạy được khi tiếng bị thay | v001: sync-only báo nhầm 88% sham |
| X7 | **Không định vị được, hoặc định vị bằng vết nối** | Survey partial-spoof 2025 (bám vùng overlap-add); UMMAFormer từ LAV-DF sang AV-1M rơi 98,8 → 51,6 AP | - |
| X8 | **Phát hiện giọng giả không tổng quát sang TTS lạ** | XLS-R+AASIST yếu với hệ chưa thấy; VietPrism: MiniMax EER 37%, VoxCPM2 35% | - |
| X9 | **Điểm nền lệch theo người / danh tính** | POI-Forensics là cả một hướng dựa trên tham chiếu người; PIA dùng ArcFace theo thời gian | v002: AUC trong từng người 0,96, gộp 0,78 |
| X10 | **Quá đắt cho quy mô lớn** | X-AVDT ~1 phút / clip; MLLM 7B không fine-tune được trên T4 | - |

## 3. Cầu nối liên ngành

| Ý tưởng (Z) | Ngành gốc | Giải lỗi | Trạng thái tìm kiếm |
|---|---|---|---|
| **Cost volume theo độ trễ + gộp 2D** (học tương quan rồi cho CNN 2D chạy trên trục thời gian × độ trễ, ước độ trễ bằng soft-argmin) | Stereo matching (GC-Net, PSMNet), optical flow (PWC-Net, RAFT) | X6, X7 | Chưa tìm thấy cho lip-sync. SyncNet / AVH-Align chấm 1 độ trễ; SpeechForensics lấy max toàn clip; TriCue (v002) dùng hồ sơ cos thô → TCN |
| **Residual upsampling (NPR) ở độ phân giải gốc của miệng** | Image forensics (Tan et al., CVPR 2024) | X1, X4 | Chưa tìm thấy cho vùng miệng lip-sync |
| **Residual "phản thực" theo thời gian** (frame − nội suy từ hai frame lân cận) | Nội suy frame / phát hiện bất thường video (dự đoán frame) | X5, X7 | LoCC (2026) đã làm dạng học teacher-student → v003 chỉ dùng như một luồng đặc trưng rẻ, **không nhận là ý mới** |
| **Động học landmark môi** (vận tốc, gia tốc, jerk) | Biomechanics / speech motor control | X3, X4 | BioLip đã làm → dùng làm **baseline** |
| **Học bất biến với xử lý lành tính** (augment đối xứng real/fake: nén lại, đổi cỡ, làm nét) | Domain generalization / anti-spoofing (RawBoost, codec augmentation) | X2, X4 | Phổ biến bên audio; v002 chưa làm (lỗi E6) |
| **Chuẩn hoá theo người bằng clip tham chiếu** | Speaker verification (enrollment, score normalization), POI-Forensics | X9 | Đã có cho face swap; **chưa tìm thấy cho lip-sync kèm 3 đầu ra** |
| **SSL đa ngôn ngữ cho audio** (XLS-R có tiếng Việt) | ASR / anti-spoofing | X3, X8 | Chuẩn; dùng làm nhánh audio + baseline |
| MLLM giải thích | Vision-language | Giải thích | Fine-tune 7B không vừa T4 → chỉ để sau, làm judge zero-shot |

Ghi chú riêng cho tiếng Việt:
- **Thanh điệu gần như không thấy trên môi.** Lỗi thanh điệu của TTS chỉ bắt được qua nhánh audio, không qua nhánh AV.
- **Encoder AV-HuBERT pretrain bằng tiếng Anh** (rủi ro X3). Cách giảm rủi ro:
  - các nhánh không phụ thuộc ngôn ngữ (trace, landmark);
  - XLS-R cho audio;
  - về sau có thể thay bằng XLAVS-R (đa ngôn ngữ).

## 4. Hai pipeline baseline được chọn

| | **BL-1 · AVH-Align** (CVPR 2025) | **BL-2 · BioLip** (2026) |
|---|---|---|
| Vì sao | Real-only, có code, tốt nhất AV-1M nhóm real-only, bền với lối tắt; đo "AV mismatch" | Tốt nhất zero-shot trên **đúng các generator của v003** (VideoReTalking, Diff2Lip, IP-LAP, W2L); không phụ thuộc ngôn ngữ; rẻ |
| Cài lại trong repo | Đúng kiến trúc paper trên cache AV-HuBERT; bản `/sup` có giám sát | Đúng kiến trúc paper (302K tham số); landmark lấy từ dlib 20 điểm môi trong cache AV-HuBERT, thay cho MediaPipe 64 điểm của paper - **khác paper, ghi rõ** |
| Điểm yếu đã biết | X1 khi train có fake; X3 | IP-LAP, nén mạnh; không bắt được giọng giả |

Baseline tham chiếu (rẻ, chạy kèm):
- **SpeechForensics:** không cần train, chạy trên cùng cache.
- **AV-Lip-Sync+ (B-AVH):** có giám sát.
- **XLS-R + AASIST-lite:** chỉ audio, dùng làm mốc cho đầu "audio fake?".

**Encoder AV-HuBERT:** v003 mặc định dùng **Large VoxCeleb2 + LRS3 fine-tune VSR** (đúng AVH-Align); Base giữ lại làm tuỳ chọn rẻ hơn.

## 5. Ba kiến trúc đề xuất

### A1 · AVCV: AV Cost-Volume Network

```
a, v (AV-HuBERT, 25 Hz) → chiếu học được Pa, Pv (Linear → LN, 256-d)
→ cost volume C[t, δ] = <Pa a_t, Pv v_{t+δ}> / √d,  δ ∈ [−15, 15]
→ CNN 2D trên (thời gian × độ trễ): 3 khối Conv2d 3×3, dilation (1, 2, 4) theo thời gian
→ soft-argmin theo δ: độ trễ ước lượng + độ nhọn đỉnh (entropy) + giá trị đỉnh
→ ghép [đặc trưng gộp theo δ ‖ a ‖ v chiếu] → TCN thời gian → 2 đầu:
   mismatch(t)  (lệch tiếng)   và   visual_fake(t) (miệng "khớp quá đều" của generator)
```

- **Giải:** X6 (tách "lệch" và "AI"), X7 (định vị theo ô 0,2 s, kèm độ trễ ước lượng).
- **Khác TriCue (v002):** tương quan được học (không phải cos thô); gộp 2D trên trục độ trễ (v002 làm 1D); có soft-argmin và độ nhọn đỉnh.
- **Kiểm chứng:** ablation bỏ CNN 2D (giữ 1D như TriCue), bỏ chiếu học được (về cos thô).

### A2 · MTN: Native-resolution Mouth-Trace Network

```
crop miệng xám 96×96 ở ĐỘ PHÂN GIẢI GỐC (hộp từ cache AV-HuBERT, không phóng lên 224)
→ augment bất biến xử lý (JPEG q 30-95, đổi cỡ 0,5-1, làm nét, gamma), áp cho cả real và fake
→ 3 luồng residual: NPR  x − up(down(x))
                    temporal  x_t − (x_{t−1} + x_{t+1}) / 2
                    ảnh gốc chuẩn hoá
→ CNN 3D nhỏ (3 khối Conv3d 3×3×3, 32-64 kênh, stride không gian 2) → pool không gian
→ ghép động học môi (BioLip: 4 bậc trên landmark môi) → TCN → visual_fake(t)
```

- **Giải:** X1 (dấu vết upsampling chung cho mọi generator vẽ ở độ phân giải thấp rồi phóng), X2/X4 (augment đối xứng), X5 (residual thời gian).
- **Rẻ:** cache crop khoảng 0,5 MB / sample, train end-to-end trên T4.
- **Kiểm chứng:** bỏ NPR / bỏ temporal / bỏ augment; so với DINOv2 224 (v002).

### A3 · TriHead-Ref: ba đầu ra + chuẩn hoá theo người

```
nhánh sync  = A1 (đặc trưng trước đầu ra)
nhánh trace = A2
nhánh audio = XLS-R-300M (trung bình lớp 6-12, 25 Hz) → AASIST-lite
             (graph attention theo thời gian × dải tần) → audio_fake(t)
hợp nhất muộn có cổng theo đầu ra:
  visual_fake ← trace + sync      audio_fake ← audio      mismatch ← sync (+ audio)
tham chiếu người: embedding trung bình của ≥ 3 clip real đã biết của cùng speaker → FiLM
  điều chỉnh 3 đầu (train: tham chiếu lấy từ clip real khác của cùng người trong cùng split)
```

- **Giải:** X9 (trong từng người v002 đạt 0,96, gộp chỉ 0,78), X8 (audio riêng, nhiều hệ TTS khi train), X6.
- **Protocol báo hai chế độ:** không tham chiếu / có tham chiếu (3 clip real của chính người đó được giữ riêng làm tham chiếu ở test, không tính vào điểm).

### Bảng ablation dự kiến

| Mã | Mô tả |
|---|---|
| A1 | AVCV đầy đủ · A1−2D · A1−learned (cos thô) |
| A2 | MTN đầy đủ · −NPR · −temporal · −augment · −kinematics |
| A3 | TriHead-Ref · −ref · −audio · −trace · −sync |
| BL | AVH-Align, AVH-Align/sup, BioLip, SpeechForensics, B-AVH, XLS-R+AASIST-lite |

## 6. Protocol và cổng

- Nhánh đánh giá: generator đã thấy / chưa thấy, giọng đã thấy / chưa thấy, người chưa thấy, định vị, lệch tiếng.
- Báo riêng: full / partial, độ dài đoạn giả, ô sát chỗ nối và ô giữa đoạn.
- Ba đầu ra chấm riêng: visual AUC, audio AUC, mismatch AUC, cộng điểm tổng "bị AI" = max(visual, audio).
- **Cổng:**
  - A1 hơn AVH-Align trên đầu mismatch;
  - A2 hơn BioLip trên generator chưa thấy và ở CRF cao;
  - A3 có tham chiếu hơn không tham chiếu ≥ 0,03 AUC gộp.

  Trượt cổng nào thì báo đúng như vậy; không đổi cổng sau khi xem test.

## 7. Nguồn

- AVH-Align: arXiv 2412.00175 (CVPR 2025)
- SpeechForensics: arXiv 2508.09913 (NeurIPS 2024)
- BioLip: arXiv 2604.16808
- LoCC: arXiv 2606.22772 (ICME 2026)
- PIA: arXiv 2510.14241
- LIPINC-V2: arXiv 2504.01470
- LipFD: arXiv 2401.15668
- X-AVDT: arXiv 2603.08483
- AV-Deepfake1M++: arXiv 2507.20579
- KLASSify: arXiv 2508.07337
- Pindrop: arXiv 2508.08141
- NPR: CVPR 2024, arXiv 2312.10461
- POI-Forensics: arXiv 2204.03083
- XLAVS-R: arXiv 2403.14402
- AV-LMMDetect: arXiv 2602.23393
- Omni-Fake: CVPR 2026
- Audio deepfake generalization: arXiv 2601.02914, 2604.26465
- Survey NCCU (thesis.lib.nccu.edu.tw)
- 中国图象图形学报 survey (doi 10.11834/jig.230088)

## 8. Triển khai (15/10/2026)

- Code: `vn-av-df-forensics/src/vn_av_df/generation/` (cells, plan, voice, lipsync, generate, wav2lip, musetalk) và `src/vn_av_df/training/` (prepare, data, models, train, evaluate, report), encoder XLS-R / crop miệng ở `features/encoders.py`; bước chạy `generation/00-04_*.py`, `training/00-04_*.py`; notebook `notebooks/{generate,prepare,train}.ipynb`; môi trường worker `environments/worker_envs.py`. Hướng dẫn: `PIPELINE.md`.
- Đã kiểm: test end-to-end trên dữ liệu tổng hợp (plan xoay vòng → giọng → lip-sync → finalize đúng hợp đồng nhãn → train 8 model 1 epoch → test → report), forward/backward của 19 cấu hình model, fbank khớp torchaudio.
- **Chưa kiểm** (cần GPU + Internet thật): cài môi trường và chạy thật 4 generator mới, 6 hệ giọng, PhoWhisper, AV-HuBERT Large, XLS-R.
- Thay đổi so với §4-§5 khi triển khai: VieNeu-TTS **v3** (thay v2, cùng tác giả); **viXTTS thay Higgs Audio v3** ở test (Higgs cần SGLang-Omni / H100); BioLip dùng 20 điểm môi dlib thay 64 điểm MediaPipe; AASIST-lite dùng self-attention thay HS-GAL đầy đủ; track C (sửa trực tiếp) chỉ lấy đoạn đã sửa ghép lại vào tiếng gốc (vocoder F5 tạo lại cả câu).

## 9. Độ lệch tiếng-hình sẵn có của video real (17/10/2026)

### 9.1. Vấn đề

v001 đã ghi "lệch A/V sẵn có của real chưa đo" (v001 §9) nhưng chưa xử lý. Generator lip-sync luôn vẽ môi khớp tiếng ở độ lệch 0 (theo "điểm 0" của SyncNet mà chúng dùng khi train). Real của một kênh lệch d frame thì có độ lệch d. Hệ quả:

1. **Lối tắt "khớp quá mức":** fake ④ source lệch 0, real cùng clip lệch d; với partial, đoạn vẽ lệch 0 nằm giữa hai đoạn lệch d. Detector có thể học "khớp hoàn hảo → fake", tức học đặc điểm của **kênh** chứ không phải của AI. Báo nhầm sẽ dao động theo người (đúng kiểu điểm nền theo người đã thấy ở v002, speaker 01; chưa kiểm chứng là do độ lệch).
2. **Nhãn mismatch nhiễu:** real lệch vài frame vẫn mang nhãn "không lệch".
3. **Ảnh hưởng không đều giữa model:** AVH-Align (BL-1) coi độ lệch 0 là cặp đúng khi train nên nhạy nhất; SpeechForensics / A1 AVCV dò độ lệch ±15 nên bền với độ lệch cố định nhưng có thể dùng chính "lệch 0, đỉnh nhọn" làm lối tắt; ref-norm (A3) khử phần lớn độ lệch cố định theo người.

### 9.2. Đo trên batch 1 (SyncNet v2, CPU)

- Bộ đo: SyncNet (Chung & Zisserman 2016, code `syncnet_python` MIT), crop mặt như `run_pipeline.py`, độ lệch −15…+15 frame.
- **Phép dịch biết trước:** dịch tiếng của 30 clip k ∈ {−6, −3, −1, +1, +3, +6} frame → đo lại đúng k ở **180/180** lần (độ phân giải 1 frame; chỉ kiểm phần chênh, không kiểm "điểm 0" tuyệt đối).
- **Theo video** (95 clip, 5 video):

| Video | Loại | Độ lệch cả video | Clip đồng ý ±1 frame | Độ nhọn đỉnh |
|---|---|---|---|---|
| JHnOdxWspmQ (podcast) | nhiều người | tiếng trễ 2 frame | 40/40 | 5,9 |
| 1hbZoEjumKE | 1 người | trễ 1 | 12/12 | 5,9 |
| IqeoeByYM4c | 1 người | trễ 1 | 14/14 | 5,2 |
| 50QybEYOOD4 | 1 người (lời dẫn đè cảnh) | — | 8/22 | 1,6 |
| ptzxTlDaw-A | 1 người (vlog, ghép cảnh) | — | 2/7 | 1,9 |

- Kết luận: video quay mặt người nói lệch **ổn định** 1–2 frame (tiếng trễ 40–80 ms, dưới ngưỡng mắt thường); video có lời dẫn đè hình cho độ lệch lung tung → bộ đo đồng thời **phát hiện mặt trên hình không phải người nói** ở mức video (lỗ hổng v001 §9 "chưa kiểm mặt trong clip đúng người nói").

### 9.3. Chọn bộ đo

| Model | LRS2, 5 frame | LRS2, 15 frame | Weight công khai |
|---|---|---|---|
| SyncNet (2016) | 75,8% | 96,1% | có |
| Perfect Match (2019) | 88,1 | 99,1 | ? |
| VocaLiST (2022) | 92,8 | 99,8 | không lấy được (theo PVSync) |
| ModEFormer (2023) | 94,5 | 99,8 | không thấy |
| IC-SyncNet (2024) | 96,5 | 99,9 | không thấy |
| StableSyncNet (LatentSync) | 94% HDTF (tác giả); 90,6% ở cửa sổ 16 frame theo PVSync | | có |
| PVSync (10/2026) | 95,1% (16 frame, bộ test riêng) | | chưa phát hành |

(Số theo IC-SyncNet Bảng 1 và PVSync; độ chính xác = đoán đúng ±1 frame.)

- SyncNet kém ở cửa sổ ngắn nhưng ngang các model mới khi gộp đủ dài; ta gộp 15–20 clip × 100–175 frame mỗi video → **SyncNet đủ dùng**, và khớp thực nghiệm 9.2.
- Không dùng StableSyncNet làm bộ đo chính: nó thuộc LatentSync (generator chỉ ở test); sửa real theo "điểm 0" của nó có thể vô tình làm real giống fake LatentSync hơn.
- AV-HuBERT (SpeechForensics) làm ý kiến thứ hai khi trích feature trên Kaggle (fairseq không chạy dễ trên Windows), không chặn pipeline.

### 9.4. Quyết định

1. **Đo trong bước cắt** (`03_cut`): sau mỗi video, SyncNet trên tối đa 20 clip rải đều; độ lệch video = đỉnh đường cong trung bình. Trạng thái: `ok` (đỉnh ≥ 3,0 và ≥ 80% clip đồng ý ±1 frame, ≥ 3 clip), `few` (đạt nhưng < 3 clip: vẫn sửa, gắn cờ xem kỹ), `unreliable` (còn lại: mọi clip tự loại, cờ `sync_unreliable`, người duyệt chọn lại được). Ngưỡng 3,0 đặt giữa nhóm tin được (5,2–5,9) và nhóm hỏng (1,6–1,9) của batch 1.
2. **Sửa khi export:** dịch tiếng của mọi clip theo độ lệch của video về 0 (copy hình, encode lại tiếng, đệm lặng ≤ vài frame ở mép); manifest ghi `audio_lag_frames`, `av_sync_status`. Lệch lớn nhưng ổn định vẫn sửa, không bỏ. Không chọn clip theo điểm sync (giữ nguyên tắc v001).
3. **Augment khi train: dịch tiếng so với hình k ∈ [−1, +2] frame** (dương = tiếng trễ), đều cho real và fake, nhãn giữ nguyên. Căn cứ:
   - **Ngưỡng nhận biết của người (ITU-R BT.1359, thử nghiệm chủ quan trên clip người đọc tin):** bắt đầu nhận ra khi tiếng **sớm ~45 ms** hoặc **trễ ~125 ms**; chấp nhận được tới khoảng +90 / −185 ms. Không đối xứng: tiếng trễ được chịu đựng gấp ~3 lần. ATSC còn đề xuất chặt hơn (sớm ≤ 15 ms, trễ ≤ 45 ms) cho phát sóng. Khoảng −1…+2 frame (sớm 40 ms … trễ 80 ms) nằm trong vùng chưa nhận ra của BT.1359.
   - **"Điểm 0" giữa các công cụ lệch ~1 frame:** ModEFormer thấy LRS2 lệch ~1 frame so với LRS3/VoxCeleb2 dù đều đã sửa bằng SyncNet; Wav2Lip train trên LRS2, LatentSync theo StableSyncNet → fake của các generator có thể lệch nhau ±1 frame.
   - **Sai số đo:** độ phân giải 1 frame (9.2) → real sau khi sửa còn sai ±0,5–1 frame.
   - **Augment lớn có hại:** DFD-Lab augment lệch tới ±200 ms làm giảm hiệu năng trên dữ liệu ngoài ở cả 3 model; PVSync coi clip lệch > 3 frame so với mode của bộ dữ liệu là dữ liệu bẩn.
   - Thí nghiệm chọn biên độ (không / ±1 / −1…+2 / ±3, đo AUC + đường báo nhầm khi real bị dịch k = −3…+3) để sau.
4. **Không nghiên cứu lệch thời gian thuần** (cùng tiếng, dịch đi) như một loại fake trong v003. Ô ② sham vẫn là thay tiếng bằng đoạn nói khác.
5. **Demo (sau):** ước lượng độ lệch toàn video (SyncNet hoặc đầu độ lệch của A1), dịch về 0 rồi mới chấm "AI?"; nếu lệch vượt vùng chấp nhận thì báo riêng "lệch thời gian ≈ X ms".

### 9.5. Nguồn

- ITU-R BT.1359 (tóm tắt): [Fora Soft](https://www.forasoft.com/learn/audio-for-video/articles-audio/lip-sync-itu-r-bt-1359-tolerance-windows), [TV Technology](https://www.tvtechnology.com/opinions/av-synchronization-how-bad-is-bad), [nghiên cứu chủ quan trên video thể thao, arXiv 2212.01686](https://ar5iv.labs.arxiv.org/html/2212.01686). Chưa đọc bản gốc ITU; con số chấp nhận được có nguồn ghi +90/−185, nguồn khác ghi +90/−190.
- SyncNet: Chung & Zisserman, "Out of Time", ACCV 2016 Workshops; code [joonson/syncnet_python](https://github.com/joonson/syncnet_python).
- Perfect Match: arXiv 1809.08001 · VocaLiST: arXiv 2204.02090 · ModEFormer: arXiv 2303.11551 · IC-SyncNet: arXiv 2409.00971 · PVSync: arXiv 2610.09223 · LatentSync / StableSyncNet: arXiv 2412.09262.
- DFD-Lab: arXiv 2609.23830 · DeepFake Doctor: arXiv 2506.05851.

## 10. Thiết kế số lượng dữ liệu: số người, clip real / người, fake / clip real (18/10/2026)

### 10.1. Câu hỏi

Bao nhiêu clip real mỗi người, mỗi clip real sinh bao nhiêu fake, tỉ lệ fake : real bao nhiêu là hợp lý, dựa trên nghiên cứu đã công bố (không dựa cảm tính).

### 10.2. Bằng chứng (đã đọc bản gốc)

| Nghiên cứu | Kết quả | Ý nghĩa |
|---|---|---|
| Scaling Laws for Deepfake Detection (arXiv 2510.16320; 5,8 triệu ảnh real / 51 nguồn, 8,8 triệu fake / 102 phương pháp) | 1 − AUC giảm theo luỹ thừa khi tăng **số nguồn real** (α = 0,476) và **số phương pháp giả** (α = 0,375), chưa bão hoà. Giảm số ảnh real 10 lần chỉ làm AUC 0,886 → 0,879. Khuyên không "thu thêm ảnh từ cùng nguồn" | Đa dạng (người, nguồn, phương pháp) > số lượng lặp |
| Implicit Identity Leakage (CVPR 2023) | Detector học thông tin danh tính, nguyên nhân chính của kém tổng quát | Ít người → dễ học danh tính |
| Bansal et al. 2017 (nhận dạng mặt) | Cùng tổng số ảnh: nhiều người tốt hơn nhiều ảnh cho ít người (mạng cỡ thường) | Cùng hướng |
| GenD (WACV 2026, arXiv 2508.06248) | Fake cùng nguồn với real (paired) hơn không paired +4,7 điểm AUC trung bình trên 14 bộ, cùng tổng số video; unpaired học vẹt nhanh hơn | Fake phải sinh từ chính clip real |
| Layton et al. (USENIX Security 2024; thí nghiệm trên giọng nói) | Train 90 % fake → trên tập toàn real báo nhầm nhiều hơn báo đúng; về 25 % fake giảm mạnh báo nhầm nhưng cũng giảm bắt fake. "Model phải train với phân bố lớp giống tập đánh giá". Khuyên: tăng media thật, báo nhiều chỉ số (ROC, PR, FPR/TPR, điểm thô), báo theo nhiều base-rate (×10 / ÷10) | Tỉ lệ + ngưỡng phải khớp tỉ lệ fake ngoài đời |
| Smeu et al. (CVPR 2025) | FakeAVCeleb, AV-Deepfake1M: fake mở đầu bằng khoảng lặng; bộ phân loại chỉ nhìn khoảng lặng > 98 % | Cần kiểm lối tắt cấu trúc |
| Referee (arXiv 2510.27475), ID-Reveal (ICCV 2021), POI-Forensics | Phương pháp dùng tham chiếu người cần ít real: Referee 1 clip 1–3 s đủ (AUC 99,71 %), ID-Reveal 6 s | 3 clip tham chiếu ≈ 16 s là đủ |
| GBDF (2022) | Lệch giới → kém ở nữ; cân giới công bằng hơn, không giảm độ chính xác | Giữ cân giới (20 nam / 20 nữ mỗi part) |
| Deepfake-Eval-2024 (arXiv 2503.02857) | Deepfake ngoài đời 2024: detector mã nguồn mở giảm ~50 % AUC (video) so với benchmark học thuật | Benchmark cần nhiễu thực tế |

### 10.3. Các bộ dữ liệu (số từ paper gốc)

| Bộ | Người | Real | Real / người | Fake / real |
|---|---|---|---|---|
| FakeAVCeleb | 500 | 500 | 1 | 39 |
| Celeb-DF | 59 | 590 | 10 | ~9,6 |
| LAV-DF | 153 | 36.431 | ~238 | ~2,7 |
| KoDF | 403 | 62.166 | ~154 | ~2,8 |
| DFDC (bản phát hành; quay 3.426 người, 14,4 video / người) | 960 (train 486) | ~19 nghìn ở train (tự tính) | ~245 (tự tính) | ~5 |
| AV-Deepfake1M | 2.068 | 286.721 | ~139 | **3,00** ở mọi split → suy ra mỗi real sinh đúng 3 fake (1 mỗi loại tiếng / hình / cả hai); mỗi fake nhiều chỗ sửa (⌈độ dài / 10 s⌉); phương pháp rút cho từng fake |
| AV-Deepfake1M++ (ACM MM 2025) | 7.109 | 627.936 | ~88 | 2,27 (train 2,70; testA 1,88; testB 1,03); 3 nguồn real, 36 loại nhiễu, test khác người / nguồn / phương pháp; **train / validation chia ngẫu nhiên theo mẫu** |

### 10.4. Quyết định

| | Trước | Sau |
|---|---|---|
| Người / part | 50 (25 nam + 25 nữ), split 15 / 4 / 6 mỗi giới | **40 (20 nam + 20 nữ)**, split **12 / 3 / 5** mỗi giới (60 / 15 / 25 %); raw giữ nguyên |
| Clip real / người | 10 | **20** (đã duyệt), từ 2–3 video khác nhau |
| Chỉ tiêu cắt nguồn 1 người | 30–50 | **40–60** (chỉ tiêu tối đa, không bắt buộc); thời lượng giữ ≥ 20 phút |
| Generator / clip real | mọi generator (4 / 6) | **một nửa: 2 (train, validation), 3 (test)**, rút cân bằng (mỗi generator chạy bằng nhau, chia đều 3 ô, bộ generator chung clip đổi theo vòng); phần tiếng giữ nguyên (1 hệ giọng mỗi clip) |
| File / clip real | 13 / 18 | **9 (6 AI) / 12 (9 AI)** |
| Một part | 500 clip, 7.100 file | **800 clip, ~7.800 file** (5.400 AI / 2.400 không AI = 69 % / 31 %; train 2 : 1) |
| Lần chạy lip-sync / part | 2.240 | **1.800** (train + validation 1.200, test 600) + 800 job giọng; GPU ~55–100 giờ |
| Ngưỡng | F1 tối đa trên validation | + ngưỡng **FPR 1 % / 5 %** trên real validation, báo precision ở tỉ lệ fake **50 % / 10 % / 1 %**, FPR riêng cho real và sham; demo dùng ngưỡng FPR 5 % |
| Kiểm lối tắt | — | `shortcut_audit.json` khi finalize: AUC của khoảng lặng đầu / cuối, tỉ lệ lặng, độ to, độ dài (một mình) với nhãn AI; ngoài [0,3; 0,7] → cờ đỏ |
| ref-norm | bỏ clip tham chiếu | bỏ clip tham chiếu **và mọi file sinh từ chúng** |

Lý do chính:
- **Người là nút thắt**, không phải clip. Thêm clip từ người đã có gần như không tốn công thu thập. 20 clip vẫn xa vùng bão hoà của các bộ lớn (88–139 real / người), nên còn lợi: real cho AVH-Align gấp đôi, 3 clip tham chiếu không chiếm 30 % số clip, FPR đo trên 200 real / part (10 người test × 20).
- **Một nửa generator mỗi clip**: cùng GPU lip-sync với "10 clip × mọi generator" nhưng gấp đôi clip real làm nền, mỗi clip real lặp ít hơn (6–9 file AI thay vì 10–15), gần cách AV-Deepfake1M (~3 fake / real, phương pháp rút cho từng fake). Mất so sánh "mọi generator trên cùng clip"; so sánh tổng hợp theo generator / ô vẫn cân.
- **Không cố định số clip khác nhau theo người** (20–60): người nhiều clip lấn át (identity leakage), AUC gộp lệch, cùng GPU làm được thêm part (thêm người).
- **Không dùng nhóm "real_only"** (clip chỉ làm real): phương án một nửa generator cho tỉ lệ tốt hơn mà mọi real đều có fake cặp (GenD).

### 10.5. So với chuẩn (AV-Deepfake1M++)

Hơn: chia theo người ở cả train / validation / test; ② sham (lệch không AI); 6 ô, 4 tác vụ; sửa độ lệch tiếng–hình của real; encode giống hệt real và fake; tiếng Việt.
Kém: quy mô (200 người = 5 part, ~39 nghìn file vs 7.109 người, 2 triệu video); một loại nguồn (YouTube); **chưa có bản test nhiễu thực tế** (nén lại, giảm độ phân giải, rớt frame…), việc kế tiếp nên làm (chỉ cần ffmpeg trên file đã sinh, không tốn GPU sinh).

### 10.6. Nguồn

- Scaling Laws for Deepfake Detection: arXiv 2510.16320 · Implicit Identity Leakage: CVPR 2023 · Bansal et al.: arXiv 1705.07426 · GenD: arXiv 2508.06248 · Layton et al.: USENIX Security 2024 · Smeu et al.: arXiv 2412.00175 · Referee: arXiv 2510.27475 · ID-Reveal: ICCV 2021 · POI-Forensics: arXiv 2204.03083 · GBDF: arXiv 2207.10246 · Deepfake-Eval-2024: arXiv 2503.02857
- Bộ dữ liệu: FakeAVCeleb arXiv 2108.05080 · Celeb-DF arXiv 1909.12962 · LAV-DF arXiv 2204.06228 · KoDF arXiv 2103.10094 · DFDC arXiv 2006.07397 · AV-Deepfake1M arXiv 2311.15308 · AV-Deepfake1M++ arXiv 2507.20579


**Đoạn giả (18/10/2026):** giữ **1 đoạn** mỗi bản partial (LAV-DF: 62,72 % video 1 đoạn, 10,55 % 2 đoạn, đoạn trung bình 0,65 s; AV-Deepfake1M: ⌈độ dài / 10 s⌉ đoạn → clip < 10 s có 1 đoạn). Không có nghiên cứu đối chứng cho thấy train nhiều đoạn giúp định vị; 2 đoạn để làm thí nghiệm phụ sau. Đổi: độ dài đoạn **0,3–3 s** (gồm sửa 1 từ); thêm chỉ số định vị **AP@IoU 0,5 / 0,75 / 0,95 và AR@1/5/10**.
