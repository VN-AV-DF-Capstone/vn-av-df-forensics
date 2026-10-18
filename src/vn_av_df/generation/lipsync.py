"""Adapter generator lip-sync của v003 (ngoài Wav2Lip / MuseTalk đã có).

Mỗi lần gọi chạy CLI chính thức của repo trong môi trường riêng (python của env đặt ở
cfg["envs"][env]); đầu vào chuẩn hoá: face.mp4 (đúng số frame, 25 fps, kích thước gốc,
gần lossless) + audio.wav 16 kHz mono. Đầu ra được đưa về đúng số frame / kích thước rồi ghi
FFV1 + PCM (như adapter cũ) để pipeline chỉ lấy hình.

| generator        | họ                         | env          | weight                                   |
|------------------|----------------------------|--------------|------------------------------------------|
| videoretalking   | nhiều tầng, có làm đẹp mặt | gen_vrt      | Google Drive (tự tải bằng gdown)         |
| iplap            | landmark → render          | gen_iplap    | OneDrive: TẢI TAY (cfg iplap_checkpoints) |
| latentsync_1_5   | latent diffusion 256 px    | gen_latentsync | Hugging Face ByteDance/LatentSync-1.5 |
| diff2lip         | diffusion trên pixel       | gen_diff2lip | Google Drive (tự tải bằng gdown)         |
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

from vn_av_df.common.runtime import read_json, run, sha, write_json
from vn_av_df.data.media import ffmpeg

REPOS = {
    "videoretalking": "https://github.com/OpenTalker/video-retalking.git",
    "iplap": "https://github.com/Weizhi-Zhong/IP_LAP.git",
    "latentsync_1_5": "https://github.com/bytedance/LatentSync.git",
    "diff2lip": "https://github.com/soumik-kanad/diff2lip.git",
}
ENVS = {
    "videoretalking": "gen_vrt",
    "iplap": "gen_iplap",
    "latentsync_1_5": "gen_latentsync",
    "diff2lip": "gen_diff2lip",
}
DRIVE_FOLDERS = {
    "videoretalking": "18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0",
    "diff2lip": "1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5",
}


def repo_dir(cfg, name):
    return Path(cfg["root"]) / "external" / name


def python_of(cfg, name):
    path = (cfg.get("envs") or {}).get(ENVS[name])
    if not path:
        raise FileNotFoundError(
            f"No python for {ENVS[name]}; run environments/worker_envs.py {ENVS[name]}"
        )
    return path


def setup(cfg, name):
    """Clone repo (ghi commit), tải weight. IP-LAP cần weight tải tay (OneDrive)."""
    from vn_av_df.assets import pinned_repository

    repo = repo_dir(cfg, name)
    pinned_repository(REPOS[name], repo)
    if name in DRIVE_FOLDERS:
        target = repo / "checkpoints"
        if not target.exists() or not any(target.iterdir()):
            import gdown

            gdown.download_folder(id=DRIVE_FOLDERS[name], output=str(target), quiet=False)
    elif name == "latentsync_1_5":
        from huggingface_hub import snapshot_download

        snapshot_download(
            "ByteDance/LatentSync-1.5",
            local_dir=str(repo / "checkpoints"),
            allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
        )
    elif name == "iplap":
        source = Path(cfg["generation"].get("iplap_checkpoints") or "")
        target = repo / "test/checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        for file in ("landmarkgenerator_checkpoint.pth", "renderer_checkpoint.pth"):
            if not (target / file).is_file():
                if not (source / file).is_file():
                    raise FileNotFoundError(
                        f"IP-LAP weight {file} missing. Tải từ OneDrive trong README của IP-LAP, "
                        "upload thành Kaggle dataset rồi đặt cfg['generation']['iplap_checkpoints']"
                    )
                shutil.copy2(source / file, target / file)
    return provenance(cfg, name)


def weights(cfg, name):
    repo = repo_dir(cfg, name)
    if name == "latentsync_1_5":
        return [repo / "checkpoints/latentsync_unet.pt", repo / "checkpoints/whisper/tiny.pt"]
    if name == "iplap":
        return [
            repo / "test/checkpoints/landmarkgenerator_checkpoint.pth",
            repo / "test/checkpoints/renderer_checkpoint.pth",
        ]
    found = sorted(p for p in (repo / "checkpoints").rglob("*") if p.is_file())
    return found


def provenance(cfg, name):
    repo = repo_dir(cfg, name)
    files = weights(cfg, name)
    if not files or any(not p.is_file() for p in files):
        raise FileNotFoundError(f"{name} weights missing; run the generator setup")
    lock = repo / "vn_av_revision.json"
    return {
        "generator": name,
        "version": read_json(lock)["commit"] if lock.is_file() else "unknown",
        "checkpoint_sha256": sha(files[0]),
        "adapter_sha256": sha(__file__),
    }


def prepare_inputs(video, audio_source, folder, frames, width, height):
    face, audio = folder / "face.mp4", folder / "audio.wav"
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-i", video, "-an"),
            *("-vf", f"fps=25,scale={width}:{height}", "-frames:v", frames),
            *("-c:v", "libx264", "-crf", "8", "-preset", "veryfast", "-pix_fmt", "yuv420p", face),
        ]
    )
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-i", audio_source, "-vn"),
            *("-af", f"atrim=duration={frames / 25},apad=whole_dur={frames / 25}"),
            *("-ar", "16000", "-ac", "1", audio),
        ]
    )
    return face, audio


def normalize_output(produced, output, frames, width, height):
    """Đúng số frame (đệm frame cuối nếu thiếu) và kích thước gốc; ghi FFV1 lossless."""
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-i", produced),
            *("-vf", f"fps=25,scale={width}:{height},tpad=stop_mode=clone:stop={frames}"),
            *("-frames:v", frames, "-an", "-c:v", "ffv1", output),
        ]
    )


def command(cfg, name, repo, face, audio, out_dir):
    py = python_of(cfg, name)
    if name == "videoretalking":
        out = out_dir / "result.mp4"
        return [py, "inference.py", "--face", face, "--audio", audio, "--outfile", out], out
    if name == "latentsync_1_5":
        out = out_dir / "result.mp4"
        config = cfg["generation"].get("latentsync_config", "configs/unet/stage2.yaml")
        return [
            py,
            *("-m", "scripts.inference", "--unet_config_path", config),
            *("--inference_ckpt_path", "checkpoints/latentsync_unet.pt"),
            *("--inference_steps", "20", "--guidance_scale", "1.5", "--enable_deepcache"),
            *("--video_path", face, "--audio_path", audio, "--video_out_path", out),
        ], out
    if name == "iplap":
        # Tên output: <tên input bỏ 4 ký tự cuối>result_N_25_Nl_15.mp4 trong output_dir.
        out = out_dir / "faceresult_N_25_Nl_15.mp4"
        return [
            py,
            *("inference_single.py", "--input", face, "--audio", audio),
            *("--output_dir", out_dir),
        ], out
    if name == "diff2lip":
        script = (repo / "scripts/inference_single_video.sh").read_text(encoding="utf-8")
        model = sorted((repo / "checkpoints").rglob("*.pt"))[0]
        out = out_dir / "result.mp4"
        values = {"video_path": face, "audio_path": audio, "out_path": out, "model_path": model}
        for key, value in values.items():
            script = re.sub(rf"(?m)^{key}=.*$", lambda _, k=key, v=value: f'{k}="{v}"', script)
        runner = out_dir / "run_diff2lip.sh"
        runner.write_text(script, encoding="utf-8")
        return ["bash", runner], out
    raise ValueError(f"Unknown generator {name}")


def make_synthesize(name):
    def synthesize(cfg, video, audio_video, output, frames, width, height, gpu=None):
        repo = repo_dir(cfg, name)
        with tempfile.TemporaryDirectory(dir=Path(output).parent) as tmp:
            folder = Path(tmp)
            face, audio = prepare_inputs(video, audio_video, folder, frames, width, height)
            args, produced = command(
                cfg, name, repo, face.resolve(), audio.resolve(), folder.resolve()
            )
            env = os.environ.copy()
            env["PATH"] = str(Path(python_of(cfg, name)).parent) + os.pathsep + env.get("PATH", "")
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = folder / "generator.log"
            try:
                run(args, cwd=repo, timeout=3600, log=log, env=env)
            except RuntimeError as exc:
                text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
                if re.search(r"face not detected|no face|face detect", text, re.I):
                    raise RuntimeError(f"Worker failed: Face not detected ({name})") from exc
                raise RuntimeError(f"Worker failed: {name} error: {text[-1500:]}") from exc
            if not produced.is_file():
                raise RuntimeError(f"Worker failed: {name} produced no video")
            normalize_output(produced, output, frames, width, height)

    return synthesize


class Adapter:
    """Cùng giao diện với vn_av_df.wav2lip / musetalk: setup, provenance, synthesize."""

    def __init__(self, name):
        if name not in REPOS:
            raise ValueError(f"Unknown generator {name}")
        self.name = name
        self.synthesize = make_synthesize(name)

    def setup(self, cfg):
        return setup(cfg, self.name)

    def provenance(self, cfg):
        return provenance(cfg, self.name)


def receipt(cfg, name, info):
    write_json(repo_dir(cfg, name) / "vn_av_generator.json", info)
