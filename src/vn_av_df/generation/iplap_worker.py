"""Chạy inference_single.py của IP-LAP (cwd = repo IP-LAP, python của env gen_iplap).

Upstream giữ nguyên thuật toán; chỉ đổi I/O: upstream ghi sketch|kết quả cạnh nhau rồi mux ffmpeg,
ở đây chỉ giữ nửa kết quả (FFV1, đúng kích thước gốc) và bỏ bước mux. Upstream ghi kết quả đầu
3 lần (frame 0, 1 = frame 2), số frame vẫn khớp. Upstream tách đường dẫn theo '/' và '.', nên
input được chép vào thư mục tương đối không có dấu chấm. Phải chạy được trên Python 3.7.

    python iplap_worker.py --video face.mp4 --audio audio.wav --output result.avi
"""

import argparse
import os
import random
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # face_alignment mới torch.compile FAN; không có triton (Windows) thì lỗi khi chạy.
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import cv2
    import numpy as np
    import torch
    import torchvision

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    work, temp = Path(f"iplap{os.getpid()}"), Path(f"tempfile_of_iplap{os.getpid()}")
    work.mkdir()
    shutil.copyfile(args.video, work / f"face{Path(args.video).suffix}")
    shutil.copyfile(args.audio, work / "audio.wav")
    sys.argv = [
        "inference_single.py",
        *("--input", f"{work.name}/face{Path(args.video).suffix}"),
        *("--audio", f"{work.name}/audio.wav", "--output_dir", work.name),
    ]
    writer = cv2.VideoWriter

    class ResultHalf:
        """Upstream ghi [sketch | kết quả]; chỉ giữ nửa phải, lossless."""

        def __init__(self, path, fourcc, fps, size):
            self.width = size[0] // 2
            self.stream = writer(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (self.width, size[1]))

        def write(self, frame):
            self.stream.write(np.ascontiguousarray(frame[:, self.width :]))

        def release(self):
            self.stream.release()

    cv2.VideoWriter = ResultHalf
    call = subprocess.call
    # Bỏ lệnh mux cuối (đọc result.avi); các lệnh ffmpeg chuẩn hoá đầu vào vẫn chạy.
    subprocess.call = lambda command, **kw: 0 if "result.avi" in command else call(command, **kw)
    # VGG19 chỉ dùng cho loss khi train và được nạp từ checkpoint renderer: khỏi tải ImageNet;
    # torchvision mới đã bỏ tham số pretrained.
    vgg19 = torchvision.models.vgg19
    torchvision.models.vgg19 = lambda pretrained=False, **kw: vgg19(**kw)
    sys.path.insert(0, os.getcwd())
    try:
        runpy.run_path("inference_single.py", run_name="__main__")
        shutil.copyfile(temp / "result.avi", args.output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
