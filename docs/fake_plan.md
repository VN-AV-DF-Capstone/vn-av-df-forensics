# VN-AV-DF v003: kế hoạch sinh fake cho một part

Một part có **800 clip real** (40 người × 20 clip) và sinh ra khoảng **7.800 file**: 9 file cho mỗi clip train / validation, 12 file cho mỗi clip test. Căn cứ chọn các con số: `docs/v003_model_research.md` §10.

## Sáu ô: mỗi file thuộc ô nào

|  | Tiếng gốc | Tiếng thật của clip khác | Giọng AI |
|---|---|---|---|
| **Hình thật (môi gốc)** | ① real | ② sham | ③ voice |
| **Môi do generator vẽ lại** | ④ source | ⑤ donor | ⑥ both |

| Ô | Cách tạo | Có AI? | Phần bị giả | Tiếng khớp môi? |
|---|---|---|---|---|
| ① **real** | Clip gốc, giữ nguyên | Không | — | Khớp |
| ② **sham** | Giữ hình, thay tiếng bằng tiếng thật của **chính người đó** ở một clip khác | Không | — | **Không khớp** |
| ③ **voice** | Giữ hình, thay tiếng bằng **giọng AI** nhân bản giọng người đó | Có | Tiếng | Không khớp |
| ④ **source** | Generator vẽ lại môi theo **tiếng gốc** của clip. Tiếng không đổi, chỉ môi bị giả | Có | Môi | Khớp |
| ⑤ **donor** | Generator vẽ lại môi theo **tiếng thật của một clip khác** của cùng người, rồi dùng tiếng đó làm tiếng của video. Người đó "nói" một câu họ chưa từng nói ở cảnh này | Có | Môi | Khớp |
| ⑥ **both** | Sửa **một cụm** trong câu thành giọng AI, rồi generator vẽ môi theo tiếng đã sửa. Giả cả môi lẫn tiếng | Có | Môi + một cụm tiếng | Khớp |

- **full:** giả cả clip. **partial:** chỉ giả **một** đoạn 0,3–3 s (đủ ngắn để chỉ là một từ); phần còn lại là gốc.
- Ô ③ có 3 kiểu: **A** đọc lại cả câu bằng giọng AI, **B** sửa một cụm rồi ghép vào tiếng gốc, **C** sửa trực tiếp trên sóng âm (chỉ có ở test).
- "Clip khác" ở ② và ⑤ luôn là clip của **cùng người, cùng split**.

## Bảng chia một part

|  | **train** | **validation** | **test** | **Cả part** |
|---|---|---|---|---|
| Người (nam + nữ) | 24 (12+12) | 6 (3+3) | 10 (5+5) | 40 (20+20) |
| Tỉ lệ theo người | 60% | 15% | 25% | 100% |
| Clip real (20 clip/người) | 480 | 120 | 200 | 800 |
| ① real | 1 | 1 | 1 | |
| ② sham (full + partial) | 2 | 2 | 2 | |
| ③ voice | 2 (A, B) | 2 (A, B) | 3 (A, B, C) | |
| ④⑤⑥ lip-sync: generator mỗi clip × 2 (full + partial) | **2** × 2 = 4 | **2** × 2 = 4 | **3** × 2 = 6 | |
| **File mỗi clip real** | **9** | **9** | **12** | |
| ↳ có AI / không AI | 6 / 3 | 6 / 3 | 9 / 3 | |
| **Tổng file** | **4.320** | **1.080** | **2.400** | **7.800** |
| ↳ có AI / không AI | 2.880 / 1.440 | 720 / 360 | 1.800 / 600 | 5.400 / 2.400 (69% / 31%) |
| Generator (rút từ) | Wav2Lip, MuseTalk 1.5, IP-LAP, VideoReTalking | như train | như train + **LatentSync 1.5, Diff2Lip** | |
| Hệ giọng | VieNeu, F5-TTS VN, OmniVoice, Seed-VC | như train | như train + **VoxCPM2, viXTTS, F5 inpainting** | |

Phần in đậm ở cột test là người, generator và giọng **chưa thấy khi train**, dùng để đo khả năng tổng quát.

## Cách gán generator vào ④⑤⑥

- Mỗi clip chỉ chạy **một nửa số generator**: 2 trong 4 (train / validation), 3 trong 6 (test). Phần tiếng không đổi: mỗi clip dùng 1 hệ giọng.
- Gán theo **vòng** trên cả split: mỗi vòng, mỗi generator chạy đúng 1 clip, nên **số lần chạy mỗi generator bằng nhau** (train 240 lần / generator, test 100 lần / generator mỗi part).
- Ô của mỗi generator xoay đều, nên **mỗi generator chia đều ④ / ⑤ / ⑥**. Trong một clip, các generator luôn ở **ô khác nhau**; bộ generator chung clip được rút ngẫu nhiên theo seed và đổi theo vòng.
- Ví dụ 2 clip train liên tiếp: clip 1 → Wav2Lip ④ + IP-LAP ⑤; clip 2 → MuseTalk ⑥ + VideoReTalking ④.
- Một part chạy **1.800 lần lip-sync** (train + validation 1.200, test 600) và 800 job giọng.

## Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `datasets/vn-av-df-data/<part>/clips/` | toàn bộ file mp4 (real + fake), encode cùng một cấu hình |
| `datasets/vn-av-df-data/<part>/manifest.jsonl` | mỗi dòng một file: split, người, ô, full/partial, generator, giọng, nhãn AI, các đoạn bị giả |
| `datasets/vn-av-df-data/<part>/shortcut_audit.json` | kiểm lối tắt: khoảng lặng, độ to, độ dài có tự đoán được nhãn không (cờ đỏ nếu AUC ngoài 0,3–0,7) |

## Bạn sẽ nhận được gì

Mỗi part được gửi dưới dạng **một file zip** `vn-av-df-data-partN.zip`, hoặc một **Kaggle dataset private** cùng tên. Đây là video người thật: **không chia sẻ lại, không để public**.

| File trong part | Nội dung |
|---|---|
| `clips/<clip_id>.mp4` | 800 clip real, 4–7 s, 25 fps, cạnh ngắn ≤ 720, H.264 + AAC 48 kHz; **tiếng đã được dịch cho khớp môi** (độ lệch đo bằng SyncNet ở bước dữ liệu) |
| `manifest.jsonl` | mỗi dòng một clip: `clip_id`, `speaker_id`, `gender`, `split`, `duration_s`, nguồn (`url`, `channel`, đoạn giây trong video gốc), độ lệch đã sửa (`audio_lag_frames`, `av_sync_status`), `flags` |
| `speakers.csv` | 40 người: giới, split, kênh, số video |
| `split-lock.json` | khoá phân chia train / validation / test theo người (**không đổi split**) |
| `dataset_info.json`, `quality_report.json` | thống kê part, hash manifest, kết quả kiểm định dạng |

Kèm theo: link repo code (notebook `notebooks/generate.ipynb`), và weight IP-LAP (chỉ phát hành qua OneDrive) nếu bạn chạy IP-LAP.

**Bạn dùng part này để:** gắn vào notebook `generate` (`CLEAN_PART`), chạy phiên giọng trước rồi các phiên lip-sync. Phiên cuối tự finalize ra bộ dữ liệu như mục "Đầu ra" ở trên.

## Vì sao chọn các con số

| Lựa chọn | Lý do | Căn cứ |
|---|---|---|
| **Nhiều người** là ưu tiên (40 người / part, thêm part = thêm người) | Lỗi giảm rõ khi tăng **số nguồn real khác nhau**. Thêm ảnh từ cùng nguồn gần như không giúp: giảm số ảnh real đi 10 lần, AUC chỉ từ 0,886 xuống 0,879. Ít người thì detector dễ học "đây là ai" thay vì "có bị giả không" | [1], [2], [3] |
| **Cân 20 nam / 20 nữ** | Bộ dữ liệu lệch giới làm detector kém hơn ở nhóm ít; cân giới thì công bằng hơn mà không giảm độ chính xác | [4] |
| **Chia split theo người** (12 / 3 / 5 mỗi giới) | Không người nào xuất hiện ở hai split, nên test đo đúng khả năng với **người chưa thấy**. Bộ lớn nhất hiện nay (AV-Deepfake1M++) chỉ làm vậy ở test, còn train / validation chia ngẫu nhiên theo mẫu | [5] |
| **20 clip / người**, lấy từ 2–3 video khác nhau | Tìm người mới là phần khó nhất; thêm clip từ người đã có gần như không tốn công thu thập. 20 vẫn thấp hơn nhiều so với các bộ lớn (88–139 real / người), nên còn lợi: gấp đôi real cho model học từ real, đủ clip tham chiếu, đo báo nhầm trên 200 real / part | [5], [6] |
| **Mọi fake sinh từ chính clip real** (paired) | Fake cùng nguồn với real tốt hơn fake khác nguồn **+4,7 điểm AUC** trung bình trên 14 bộ, với cùng số video | [7] |
| **Mỗi clip chỉ chạy một nửa số generator** (2 / 4, test 3 / 6), giọng giữ nguyên | Cùng chi phí GPU với "10 clip × mọi generator", nhưng gấp đôi số clip real làm nền và mỗi clip real lặp ít hơn. Gần cách AV-Deepfake1M (đúng 3 fake mỗi real, phương pháp rút cho từng fake) | [6] |
| **Có AI / không AI ≈ 69 % / 31 %** | Tỉ lệ cùng vùng với các bộ lớn (AV-Deepfake1M 75 % fake, AV-Deepfake1M++ 69 %). Fake quá nhiều làm detector báo nhầm nhiều trên video thật; vì vậy báo cáo thêm ngưỡng theo tỉ lệ báo nhầm cố định và precision ở nhiều tỉ lệ fake ngoài đời | [6], [8] |
| **Generator và giọng chỉ có ở test** (LatentSync, Diff2Lip, VoxCPM2, viXTTS, F5 inpainting) | Đo khả năng phát hiện phương pháp **chưa từng thấy**; số phương pháp giả cũng là trục làm detector tổng quát tốt hơn | [1] |
| **② sham** (tiếng thật ghép lệch, không AI) | Tách hai câu hỏi "có AI không" và "tiếng có khớp môi không"; nếu thiếu, detector chỉ học "lệch = giả" | thiết kế v001–v002 |
| **Partial: đúng 1 đoạn, 0,3–3 s** | Clip ngắn (4–7 s) thì các bộ khác cũng chỉ 1 đoạn (LAV-DF: 62,72 % video 1 đoạn, đoạn trung bình 0,65 s; AV-Deepfake1M: 1 đoạn / 10 s). Đoạn ngắn tới 0,3 s để có cả kiểu sửa một từ | [9], [6] |
| **Sửa độ lệch tiếng–hình của real, encode giống hệt real và fake** | Video YouTube hay lệch 1–2 frame. Generator luôn khớp ở độ lệch 0, nên nếu không sửa, detector học "khớp hoàn hảo = giả". Encode khác nhau thì detector học khác biệt nén | §9 nghiên cứu |
| **Kiểm lối tắt khoảng lặng / độ to / độ dài** | Hai bộ phổ biến (FakeAVCeleb, AV-Deepfake1M) có lối tắt: fake mở đầu bằng khoảng lặng, một bộ phân loại chỉ nhìn khoảng lặng đạt > 98 % | [10] |

Chi tiết và số liệu đầy đủ: `docs/v003_model_research.md` §9–§10.

## Nghiên cứu đã tham khảo

1. Scaling Laws for Deepfake Detection (2025). [arXiv 2510.16320](https://arxiv.org/abs/2510.16320)
2. Dong et al. Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization. CVPR 2023. [link](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.html)
3. Bansal et al. The Do's and Don'ts for CNN-based Face Verification. ICCV Workshops 2017. [arXiv 1705.07426](https://arxiv.org/abs/1705.07426)
4. Nadimpalli & Rattani. GBDF: Gender Balanced DeepFake Dataset. ICPR Workshops 2022. [arXiv 2207.10246](https://arxiv.org/abs/2207.10246)
5. Cai et al. AV-Deepfake1M++. ACM Multimedia 2025. [arXiv 2507.20579](https://arxiv.org/abs/2507.20579)
6. Cai et al. AV-Deepfake1M. ACM Multimedia 2024. [arXiv 2311.15308](https://arxiv.org/abs/2311.15308)
7. Yermakov et al. Deepfake Detection that Generalizes Across Benchmarks (GenD). WACV 2026. [arXiv 2508.06248](https://arxiv.org/abs/2508.06248)
8. Layton et al. Measuring Structural Limitations of Deepfake Media Datasets. USENIX Security 2024. [pdf](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf)
9. Cai et al. LAV-DF (Do You Really Mean That?). DICTA 2022. [arXiv 2204.06228](https://arxiv.org/abs/2204.06228)
10. Smeu et al. Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets. CVPR 2025. [arXiv 2412.00175](https://arxiv.org/abs/2412.00175)

Các bộ dùng để so số lượng: FakeAVCeleb ([arXiv 2108.05080](https://arxiv.org/abs/2108.05080)), Celeb-DF ([arXiv 1909.12962](https://arxiv.org/abs/1909.12962)), KoDF ([arXiv 2103.10094](https://arxiv.org/abs/2103.10094)), DFDC ([arXiv 2006.07397](https://arxiv.org/abs/2006.07397)).
