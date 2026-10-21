"""Isolated compatibility bridge to the official Wav2Lip inference code.

Keeps model/detection/inference upstream; adjusts librosa API, device and lossless
intermediate output. Inputs in cwd have fixed names, avoiding upstream shell paths.
--serve: nạp model và bộ dò mặt một lần, mỗi yêu cầu {"cwd": thư mục có face.avi/audio.wav}.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
# Thư mục script (src/vn_av_df/generation) không được che package upstream của Wav2Lip; bỏ khỏi path.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import zipfile
from types import SimpleNamespace


def load_generator(path, model_factory, device):
    """Accept official TorchScript or state_dict weights without unsafe pickle fallback."""
    import torch

    scripted = False
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            scripted = any(name.endswith("/constants.pkl") for name in archive.namelist())
    if scripted:
        model = torch.jit.load(str(path), map_location=device)
        kind = "torchscript"
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                "Expected an official TorchScript archive or checkpoint with state_dict"
            )
        model = model_factory()
        model.load_state_dict(
            {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()},
            strict=True,
        )
        model = model.to(device)
        kind = "state_dict"
    model.eval()
    with torch.inference_mode():
        for batch in (1, 8):
            audio = torch.zeros(batch, 1, 80, 16, device=device)
            face = torch.zeros(batch, 6, 96, 96, device=device)
            output = model(audio, face)
            if output.shape != (batch, 3, 96, 96) or not torch.isfinite(output).all():
                raise ValueError("Generator forward check failed: expected finite [B,3,96,96]")
    return model, kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ffmpeg", required=True)
    # Upstream mặc định 16; lô 1 chậm hơn nhiều mà không đổi kết quả dò mặt.
    parser.add_argument("--face-det-batch", type=int, default=16)
    args = parser.parse_args()
    import librosa
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select cpu or a compatible generator environment")
    sys.path.insert(0, str(args.upstream.resolve()))
    sys.argv = [
        str(args.upstream / "inference.py"),
        "--checkpoint_path",
        str(args.checkpoint.resolve()),
        "--face",
        "face.avi",
        "--audio",
        "audio.wav",
        "--outfile",
        "generated.mkv",
        "--face_det_batch_size",
        str(args.face_det_batch),
        "--wav2lip_batch_size",
        "8",
    ]
    spec = importlib.util.spec_from_file_location(
        "official_wav2lip_inference", args.upstream / "inference.py"
    )
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    inference.device = args.device
    hp = inference.audio.hp
    inference.audio._mel_basis = librosa.filters.mel(
        sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "librosa", "opencv-python", "scipy", "numba", "tqdm")
    }
    print(json.dumps(dict(versions=versions, device=args.device)), flush=True)
    print("Checking generator checkpoint and forward before face detection...", flush=True)
    model, checkpoint_format = load_generator(args.checkpoint, inference.Wav2Lip, args.device)
    print(
        json.dumps(
            dict(
                checkpoint_format=checkpoint_format,
                forward_check="passed",
                batches=[1, 8],
                device=args.device,
            )
        ),
        flush=True,
    )
    if args.check:
        return
    inference.load_model = lambda path: model
    # Use lossless intermediate media for both classes; avoid a fake-only DIVX pass.
    fourcc = inference.cv2.VideoWriter_fourcc
    inference.cv2.VideoWriter_fourcc = lambda *code: fourcc(
        *("FFV1" if "".join(code) == "DIVX" else code)
    )

    def mux_lossless(command, **unused):
        # Upstream has one final mux call because our input is already a WAV.
        if not Path("temp/result.avi").is_file():
            raise RuntimeError("Wav2Lip did not produce its video intermediate")
        subprocess.run(
            [
                args.ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-i",
                "temp/result.avi",
                "-i",
                "audio.wav",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                "generated.mkv",
            ],
            check=True,
        )
        return 0

    inference.subprocess = SimpleNamespace(call=mux_lossless)
    # Upstream tạo lại S3FD (nạp weight lên GPU) mỗi lần dò mặt; giữ một bản cho mọi video.
    detectors = {}
    factory = inference.face_detection.FaceAlignment

    def cached_detector(*options, **named):
        key = repr((options, sorted(named.items())))
        if key not in detectors:
            detectors[key] = factory(*options, **named)
        return detectors[key]

    inference.face_detection.FaceAlignment = cached_detector

    home = os.getcwd()

    def run_one(request=None):
        # Tên file cố định trong cwd (face.avi, audio.wav, generated.mkv) như bản một lần.
        if request:
            os.chdir(request["cwd"])
        try:
            Path("temp").mkdir(exist_ok=False)
            inference.main()
            if not Path("generated.mkv").is_file():
                raise RuntimeError("Wav2Lip did not produce generated.mkv")
        finally:
            # Windows không xoá được thư mục đang là cwd của một process (WinError 32).
            os.chdir(home)

    if args.serve:
        from vn_av_df.worker import serve

        serve(run_one)
    else:
        run_one()


if __name__ == "__main__":
    main()
