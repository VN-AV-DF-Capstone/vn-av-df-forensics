"""Chuẩn hoá video đầu vào của demo và sửa độ lệch tiếng-hình (giống bước dữ liệu).

- `normalize`: 25 fps CFR, cạnh ngắn ≤ 720, tối đa `max_seconds` giây, H.264 + AAC 48 kHz.
- `align`: đo độ lệch cả video bằng SyncNet trên các cửa sổ 7 s (như bước cắt), dịch tiếng về 0
  khi đo được; trả kèm cảnh báo khi lệch vượt ngưỡng người xem nhận ra (ITU-R BT.1359: tiếng
  sớm ~45 ms / trễ ~125 ms).
"""

from pathlib import Path

from vn_av_df.common.runtime import run
from vn_av_df.data.media import ffmpeg, probe

WINDOW = 175  # 7 s ở 25 fps
EARLY_MS, LATE_MS = 45, 125


def normalize(source, output, max_seconds=60):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = "scale=w='if(gte(iw,ih),-2,min(trunc(iw/2)*2,720))':h='if(gte(iw,ih),min(trunc(ih/2)*2,720),-2)'"
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-i", str(source), "-t", str(max_seconds)),
            *("-vf", f"fps=25,{scale}", "-fps_mode", "cfr"),
            *("-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"),
            *("-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1"),
            *("-movflags", "+faststart", str(output)),
        ]
    )
    probe(output)  # thiếu hình hoặc tiếng thì báo lỗi ngay
    return output


def sync_rules():
    """Ngưỡng đo độ lệch lấy từ cấu hình data pipeline (cut.sync)."""
    from vn_av_data.common.runtime import load_config

    config = Path(__file__).resolve().parents[3] / "data_pipeline/configs/data.yaml"
    cut = load_config(config)["cut"]
    return cut["sync"], cut


class Aligner:
    def __init__(self, meter=None):
        self.rules, cut = sync_rules()
        if meter is None:
            from vn_av_data.features.syncnet import SyncMeter

            base = Path(__file__).resolve().parents[3] / "data_pipeline"
            meter = SyncMeter(base / cut["syncnet_model"], base / cut["face_model"])
        self.meter = meter

    def measure(self, video):
        from vn_av_data.data.media import decode
        from vn_av_data.features.syncnet import summarize

        media = decode(video, max_duration=61, max_side=640, sample_rate=16000)
        frames, pcm = media["frames"], media["pcm"]
        curves = []
        for start in range(0, max(1, len(frames) - WINDOW // 2), WINDOW):
            end = min(len(frames), start + WINDOW)
            if end - start < 50:
                continue
            curve = self.meter.curve(frames[start:end], pcm[start * 640 : end * 640])
            if curve is not None:
                curves.append(curve)
        return summarize(curves, {**self.rules, "min_clips": 1})

    def align(self, video, output):
        """→ (đường dẫn video đã sửa, thông tin độ lệch)."""
        from vn_av_data.data.export import place_clip

        sync = self.measure(video)
        lag = sync.get("audio_lag_frames", 0) if sync["status"] in ("ok", "few") else 0
        ms = lag * 40
        sync["audio_lag_ms"] = ms
        sync["noticeable"] = bool(ms > LATE_MS or -ms > EARLY_MS)
        sync["corrected"] = bool(lag)
        place_clip(video, output, lag, probe(video)["duration_s"])
        return Path(output), sync
