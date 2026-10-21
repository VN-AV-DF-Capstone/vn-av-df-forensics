"""Phần chung của worker giọng (chạy trong môi trường riêng của từng hệ TTS / VC).

`<worker>.py --serve <project root>`: nạp model một lần, mỗi dòng stdin một yêu cầu JSON
(xem vn_av_df/worker.py). Model tải về thư mục VNAV_MODELS (đặt bởi phía gọi).
"""

import os
import sys
from pathlib import Path


def models_dir(name):
    folder = Path(os.environ.get("VNAV_MODELS", "weights/voice")) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_wav(path, wav, rate):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(wav, dtype=np.float32).reshape(-1)
    try:
        import soundfile

        soundfile.write(str(path), data, int(rate))
    except ImportError:
        from scipy.io import wavfile

        wavfile.write(str(path), int(rate), (np.clip(data, -1, 1) * 32767).astype("int16"))
    return {"sample_rate": int(rate), "samples": int(data.size)}


def main(load, handle):
    if len(sys.argv) < 3 or sys.argv[1] != "--serve":
        raise SystemExit("Usage: <worker> --serve <project root>")
    sys.path.insert(0, str(Path(sys.argv[2]) / "src"))
    from vn_av_df.worker import serve

    state = load()

    def run(request):
        try:
            return handle(state, request)
        finally:
            # Hai worker cùng một GPU (vd. VoxCPM2 + F5 sửa giọng ở test): worker rảnh trả cache.
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    serve(run)
