"""Explicit, resumable downloads. Record exact revisions and hashes for reproducibility."""

import os
import shutil
import time
import urllib.request
from pathlib import Path

from vn_av_data.common.runtime import read_json, sha, write_json

# (mục cấu hình, khoá đường dẫn) → URL. License: YuNet / SFace Apache-2.0 (OpenCV Zoo),
# Silero VAD MIT, WeSpeaker ResNet34-LM CC-BY-4.0 (Hugging Face Wespeaker), SyncNet v2 (VGG Oxford,
# code syncnet_python MIT; chỉ dùng cho nghiên cứu).
OPENCV_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
MEDIA = [
    ("cut", "face_model", f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    (
        "cut",
        "vad_model",
        "https://raw.githubusercontent.com/snakers4/silero-vad/v6.0/src/silero_vad/data/silero_vad.onnx",
    ),
    (
        "cut",
        "sface_model",
        f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    (
        "cut",
        "syncnet_model",
        "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    ),
    (
        "identify",
        "voice_model",
        "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/"
        "voxceleb_resnet34_LM.onnx",
    ),
]


def fetch_file(url, destination):
    destination = Path(destination)
    record = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists():
        if record.exists() and read_json(record)["sha256"] == sha(destination):
            return read_json(record)
        raise ValueError(f"Unverified existing asset: {destination}; use a new asset path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        # Một số máy chủ (Hugging Face) đôi khi cắt kết nối giữa chừng: thử lại vài lần.
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
                    shutil.copyfileobj(response, out)
                break
            except OSError:
                if attempt == 4:
                    raise
                print(f"  tải lại {destination.name} (lần {attempt + 1})", flush=True)
                time.sleep(3 * attempt)
        if partial.stat().st_size < 1024:
            raise ValueError(f"Invalid model download: {url}")
        os.replace(partial, destination)
        result = {"url": url, "sha256": sha(destination), "bytes": destination.stat().st_size}
        write_json(record, result)
        return result
    finally:
        partial.unlink(missing_ok=True)


def setup_assets(cfg):
    return {key: fetch_file(url, cfg[section][key]) for section, key, url in MEDIA}
