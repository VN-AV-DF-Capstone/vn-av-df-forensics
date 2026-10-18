"""Trích feature cho dataset v003 đã finalize: avhubert → mouth (cần hộp miệng) → xlsr.

Cache: <cfg["cache"]>/<encoder>/<sample_id>.npz (+ .json). Chạy lại thì bỏ qua cache còn khớp
chữ ký; dừng giữa chừng không làm hỏng cache (ghi file tạm rồi đổi tên).
"""

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vn_av_df import tracking
from vn_av_df.common.runtime import write_json
from vn_av_df.dataset import media_path, training_dataset

ORDER = ("avhubert", "mouth", "xlsr")


def make(cfg, name, device):
    spec = {**cfg["encoders"][name], **({"device": device} if device else {})}
    if name == "avhubert":
        from vn_av_df.features.avhubert import AVHubertVideoEncoder

        return AVHubertVideoEncoder(spec)
    if name == "xlsr":
        from vn_av_df.features.encoders import XLSREncoder

        return XLSREncoder(spec)
    if name == "mouth":
        from vn_av_df.features.encoders import MouthEncoder

        return MouthEncoder(spec)
    raise ValueError(f"Unknown encoder {name}")


def prepare(cfg, encoders=ORDER, factory=None):
    rows, selection = training_dataset(cfg, verify_media=False)
    cache = Path(cfg["cache"])
    reports = {}
    chosen = [e for e in ORDER if e in encoders]
    run = tracking.open_run(
        cfg,
        "prepare",
        f"{cache.name}-prepare-{'+'.join(chosen)}",
        group=cache.name,
        config={
            "encoders": {e: cfg["encoders"][e] for e in chosen},
            "selection": selection,
            "samples": len(rows),
        },
        folder=cache,
    )
    for name in [e for e in ORDER if e in encoders]:
        devices = list(cfg.get("prepare_devices") or [None])
        copies = int(cfg["encoders"][name].get("workers_per_device", 1))
        pool_devices = [d for d in devices for _ in range(copies)]
        available = queue.Queue()
        for device in pool_devices:
            available.put((factory or make)(cfg, name, device))
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("source_clip_id") or row["sample_id"]].append(row)
        began, guard, done = time.perf_counter(), threading.Lock(), [0]

        def extract(group):
            encoder = available.get()
            try:
                for row in group:
                    request = {"video": str(media_path(cfg["dataset"], row))}
                    if name == "mouth":
                        request["box_cache"] = str(cache / "avhubert" / f"{row['sample_id']}.npz")
                    encoder.extract(request, folder / f"{row['sample_id']}.npz")
                    with guard:
                        done[0] += 1
                        spent = time.perf_counter() - began
                        if done[0] % 25 == 0 or done[0] == len(rows):
                            run.log(
                                {
                                    f"{name}/done": done[0],
                                    f"{name}/samples_per_min": done[0] / max(spent, 1e-6) * 60,
                                }
                            )
                            print(
                                f"{name}: {done[0]}/{len(rows)} | {spent / 60:.1f} phút, còn "
                                f"~{spent / done[0] * (len(rows) - done[0]) / 60:.1f} phút",
                                flush=True,
                            )
            finally:
                available.put(encoder)

        try:
            with ThreadPoolExecutor(max_workers=len(pool_devices)) as pool:
                for future in [pool.submit(extract, g) for g in groups.values()]:
                    future.result()
        finally:
            while not available.empty():
                getattr(available.get(), "close", lambda: None)()
        elapsed = time.perf_counter() - began
        write_json(
            folder / "extraction.json",
            {
                "samples": len(rows),
                "selection": selection,
                "elapsed_s": elapsed,
                "devices": pool_devices,
            },
        )
        reports[name] = {"samples": len(rows), "elapsed_s": round(elapsed, 1)}
        run.summary({f"{name}/elapsed_h": elapsed / 3600, f"{name}/samples": len(rows)})
        log_cache(run, name, folder, rows)
        run.artifact(f"{cache.name}-{name}-extraction", "extraction", [folder / "extraction.json"])
    run.finish()
    return reports


def log_cache(run, name, folder, rows):
    """Chất lượng cache: tỉ lệ frame hợp lệ; với AV-HuBERT thêm kiểm chéo độ lệch tiếng-hình
    trên clip real (đã sửa về 0 ở data pipeline bằng SyncNet; ở đây đo lại bằng đặc trưng
    AV-HuBERT theo cách của SpeechForensics)."""
    if not run.active:
        return
    valid, lags, peaks = [], [], []
    for row in rows:
        path = folder / f"{row['sample_id']}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            if name == "avhubert":
                ok = z["audio_valid"] & z["visual_valid"]
                valid.append(float(ok.mean()))
                if row.get("cell") == "real":
                    lag, peak = tracking.lag_profile(z["audio"], z["visual"], ok)
                    if lag is not None:
                        lags.append(lag)
                        peaks.append(peak)
            elif name == "mouth":
                valid.append(float(z["valid"].mean()))
    run.hist(f"{name}/valid_frame_ratio", valid)
    if valid:
        run.summary({f"{name}/mean_valid_ratio": float(np.mean(valid))})
    if lags:
        lags = np.asarray(lags)
        run.hist("sync_check/avhubert_lag_real_frames", lags)
        run.hist("sync_check/avhubert_peak_real", peaks)
        run.summary(
            {
                "sync_check/real_clips": len(lags),
                "sync_check/share_within_1_frame": float(np.mean(np.abs(lags) <= 1)),
                "sync_check/median_lag": float(np.median(lags)),
            }
        )
