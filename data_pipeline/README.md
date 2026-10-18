# Pipeline dữ liệu real v003

Chạy ở **máy local**: từ danh sách URL YouTube ra **part** real (20 nam + 20 nữ × 20 clip = 800 clip, chia sẵn train / validation / test theo người). Part là đầu vào của bước sinh fake (`generation/`, `notebooks/generate.ipynb`). Căn cứ các con số: [docs/v003_model_research.md](../docs/v003_model_research.md) §9–§10.

Chỉ dùng video của chính mình hoặc đã được phép. Raw, clip và part là dữ liệu người thật: **không chia sẻ công khai**.

## Cài đặt

```powershell
conda activate vn-av-df
python 00_install.py                 # từ thư mục gốc repo: cài data_pipeline + package chính
python data_pipeline/steps/00_setup.py   # tải YuNet, Silero VAD, SFace, WeSpeaker, SyncNet v2
```

Cần Node.js trong PATH (yt-dlp dùng để tải YouTube).

## Các bước

| Bước | Lệnh | Làm gì |
|---|---|---|
| 01 | `01_plan.py [--cookies-from-browser firefox]` | Đọc `data/inputs/urls.csv` (url, speakers = 1 / n, note). Nguồn 1 người: video trải theo ngày đăng đến khi ≥ 20 phút (3–10 video). Nguồn nhiều người: 15 video mới nhất. Loại video dọc, < 25 fps, < 720p, lồng tiếng, trùng. Chia **batch raw ≤ 20 giờ và ≤ 16 GB** |
| 02 | `02_download.py [N]` | Tải batch (720p). Lỗi 403 tạm thời thì tự thử lại; YouTube đòi xác minh bot thì dừng, chạy lại với `--cookies-from-browser firefox` |
| 03 | `03_cut.py [N]` | Cắt clip ứng viên 4–7 s. Nguồn 1 người: 40–60 clip rải đều (chỉ tiêu tối đa). Nguồn nhiều người: cắt hết, kiểm 5 điểm cùng một mặt. Mỗi frame: đúng 1 mặt ≥ 48 px (khung 640), hụt ≤ 3 frame (≤ 2 liền nhau), không chuyển cảnh, không jump cut. **Đo độ lệch tiếng–hình bằng SyncNet** (≤ 20 clip / video); video không đo được → clip tự loại, cờ `sync_unreliable` |
| 04 | `04_identify.py [N]` | Gán speaker bằng mặt (SFace) + giọng (WeSpeaker), khớp với người đã có (trong kênh, rồi toàn bộ), gợi ý gộp. `--calibrate` sau vài chục người để chỉnh ngưỡng |
| 05 | `05_review.py` | Trang duyệt http://127.0.0.1:8001: gộp / tách người, gán giới (M / F), **chọn 20 clip** (S chọn, X loại, R đổi người, N xong, E loại người). Clip không chọn để nguyên là được. Tắt bằng Ctrl+C |
| 06 | `06_parts.py` | Gom người đã duyệt đủ 20 clip thành part 20 nam + 20 nữ; split theo người 12 / 3 / 5 mỗi giới; người chung video đi cùng nhau. Part tạo xong bị khoá |
| 07 | `07_export.py N` | Xuất part ra `exports/vn-av-df-data/vn-av-df-data-partN/` + zip. **Dịch tiếng theo độ lệch đo ở bước 03** (về 0); manifest có `audio_lag_frames`, `av_sync_status` |
| 08 | `08_archive.py raw N` / `part N` `[--uploaded]` | Đóng zip raw; đánh dấu zip raw / part đã lên Drive |
| 09 | `09_clean.py raw N` / `part N` / `pool` | Xoá bản ở máy, chỉ khi đã `--uploaded`. `pool` xoá clip không chọn của người đã duyệt xong / bị loại |
| 10 | `10_status.py` | Toàn cảnh: batch, clip, người, part, dung lượng |

`[N]` = số batch (bỏ trống: batch đầu còn việc). Mọi bước dừng được bằng Ctrl+C, chạy lại thì làm tiếp.

## Thứ tự

```
mỗi batch:  01 plan → 02 download → 03 cut → 04 identify → 05 review → 08 archive raw N (up Drive, --uploaded) → 09 clean raw N
đủ người:   06 parts → 07 export N → up zip lên Kaggle (private) + Drive → 08 archive part N --uploaded → 09 clean part N
```

## Thư mục `data/` (một thư mục cho mọi batch và part)

| Đường dẫn | Nội dung | Giữ / xoá |
|---|---|---|
| `inputs/urls.csv` | danh sách URL | giữ |
| `ledger/ledger.sqlite` | **sổ cái**: URL, video, lý do loại, batch, clip, độ lệch, speaker, quyết định duyệt, part, nơi lưu trên Drive. Kèm `backups/` (30 bản gần nhất) và `csv/` (xem bằng Excel) | **giữ, up Drive mỗi lần archive** |
| `raw/batch_XXX/` | video gốc + `videos.csv` | xoá sau khi lên Drive (`09_clean raw`) |
| `pool/clips/`, `pool/faces/` | clip ứng viên, ảnh dải mặt | clip xoá theo part / pool; ảnh mặt giữ |
| `registry/` | ảnh mặt đại diện mỗi người | giữ |
| `archive/` | zip chờ up Drive | xoá cùng raw / part |

Cấu hình: [configs/data.yaml](configs/data.yaml). Đổi luật cắt sau khi đã có clip sẽ bị chặn (luật cắt phải thống nhất); thử luật mới trên workspace riêng: `$env:VNAV_DATA="data_trial"`.

## Kiểm thử

```powershell
python -m pytest data_pipeline/tests
```
