"""Môi trường worker của v003 trên Kaggle / Linux (micromamba, ngoài /kaggle/working).

    python environments/worker_envs.py gen_latentsync --env-root /kaggle/temp/vn-av-df-workers
In ra đường dẫn python; gán vào cfg["envs"][<tên>] (settings.ENVS ở local; notebook làm tự động).

Mỗi công thức ghi lại phiên bản theo README / requirements chính thức của upstream. CHƯA CHẠY THỬ
trên Kaggle: nếu pip lỗi, sửa công thức tại đây (log ở <env-root>/.kaggle-tools/<tên>/setup.log).
Generator cần source đã clone trước (generation/00_setup.py) vì cài requirements của repo.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from bootstrap_worker import execute, micromamba, remove_package_cache, write_json

CU111 = "https://download.pytorch.org/whl/torch_stable.html"
CU121 = "https://download.pytorch.org/whl/cu121"
# transformers 5.x (kéo theo omnivoice, voxcpm, coqui-tts) tắt PyTorch < 2.6 → lỗi khi import model.
CU126 = "https://download.pytorch.org/whl/cu126"
TORCH_28 = ["torch==2.8.0", "torchaudio==2.8.0", "--index-url", CU126]

RECIPES = {
    # VideoReTalking README: Python 3.8, torch 1.9.0+cu111, requirements.txt.
    "gen_vrt": {
        "python": "3.8",
        "conda": ["ffmpeg", "dlib"],
        "pip": [
            ["torch==1.9.0+cu111", "torchvision==0.10.0+cu111", "-f", CU111],
            # requirements.txt upstream trừ dlib==19.24.0: pip build dlib từ source lỗi trên
            # Kaggle; dùng dlib của conda ở trên.
            [
                *("basicsr==1.4.2", "kornia==0.5.1", "face-alignment==1.3.4", "ninja==1.10.2.3"),
                *("einops==0.4.1", "facexlib==0.2.5", "librosa==0.9.2", "gradio>=3.7.0"),
                "numpy==1.23.4",
            ],
        ],
        "probe": "import torch, cv2, face_alignment, dlib",
    },
    # IP-LAP README: Python 3.7.13, torch 1.10.0, torchvision 0.11.0, requirements.txt. PyPI đã gỡ
    # mediapipe 0.8.11 (cũ nhất còn lại 0.10.5, Python >= 3.8) nên dùng Python 3.8 và pin lại:
    # mediapipe 0.10.9 (bản cuối không kéo jax), scipy 1.10.1 (1.2.3 không có wheel 3.8),
    # numba 0.56.4 (bản mới hơn đòi numpy >= 1.22), face-alignment 1.3.4 (1.3.5 đã đổi
    # LandmarksType._2D mà code IP-LAP dùng thành TWO_D). lws chỉ có source, setup.py import numpy.
    "gen_iplap": {
        "python": "3.8",
        "conda": ["ffmpeg"],
        "pip": [
            ["torch==1.10.0+cu111", "torchvision==0.11.0+cu111", "-f", CU111],
            ["numpy==1.21.6", "wheel"],
            ["--no-build-isolation", "lws==1.2.7"],
            [
                *("typing_extensions==4.3.0", "opencv-contrib-python==4.6.0.66"),
                *("opencv-python==4.6.0.66", "numpy==1.21.6", "tqdm==4.64.1", "mediapipe==0.10.9"),
                *("face-alignment==1.3.4", "librosa==0.9.2", "numba==0.56.4", "scipy==1.10.1"),
                *("lws==1.2.7", "tensorboardX==2.5.1", "piq==0.7.0"),
            ],
        ],
        "probe": "import torch, cv2, lws, librosa, mediapipe, face_alignment; face_alignment.LandmarksType._2D",
    },
    # LatentSync: Python 3.10 + requirements.txt (đã pin torch / diffusers của repo).
    "gen_latentsync": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [["-r", "external/latentsync_1_5/requirements.txt"]],
        "probe": "import torch, diffusers",
    },
    # Diff2Lip README: Python 3.9, ffmpeg 5.0.1, requirements.txt; guided-diffusion cần mpi4py.
    # requirements.txt không cài thẳng được: "-e ./guided-diffusion" tính theo thư mục đang đứng,
    # av 9.2.0 không còn wheel (12.0.0 vẫn hợp write_video của torchvision 0.13), torchaudio không
    # pin kéo torch 2.x (Diff2Lip không dùng). numpy < 2 cho torch 1.12; librosa 0.9.2 cần
    # pkg_resources (setuptools < 81), six (librosa gọi lúc chạy, không kéo theo). mpi4py + mpich lấy
    # từ conda thay cho mpi4py-mpich.
    "gen_diff2lip": {
        "python": "3.9",
        "conda": ["ffmpeg=5.0.1", "mpi4py", "mpich"],
        "pip": [
            [
                "torch==1.12.1",
                "torchvision==0.13.1",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu113",
            ],
            [
                *("librosa==0.9.2", "opencv-python==4.5.5.64", "opencv-contrib-python==4.6.0.66"),
                *("tensorboard==2.11.0", "tqdm==4.64.1", "av==12.0.0", "numpy==1.23.5"),
                *("setuptools<81", "six", "-e", "external/diff2lip/guided-diffusion"),
            ],
        ],
        "probe": "import torch, mpi4py, six, librosa, cv2, av, guided_diffusion",
    },
    # VieNeu v3 (GPU): torch CUDA + transformers==4.57.6 (theo trang PyPI).
    "tts_vieneu": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [["torch", "torchaudio", "--index-url", CU121], ["vieneu", "transformers==4.57.6"]],
        "probe": "from vieneu import Vieneu",
    },
    "tts_f5": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            ["torch==2.4.1", "torchaudio==2.4.1", "--index-url", CU121],
            ["f5-tts", "huggingface_hub"],
        ],
        "probe": "from f5_tts.infer.utils_infer import load_model",
    },
    "tts_omnivoice": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            TORCH_28,  # README OmniVoice: torch 2.8.0
            ["git+https://github.com/k2-fsa/OmniVoice.git", "soundfile"],
        ],
        "probe": "from omnivoice import OmniVoice",
    },
    # VoxCPM: Python >= 3.10, torch >= 2.5, CUDA >= 12.
    "tts_voxcpm": {
        "python": "3.11",
        "conda": ["ffmpeg"],
        "pip": [TORCH_28, ["voxcpm", "soundfile"]],
        "probe": "from voxcpm import VoxCPM",
    },
    "tts_xtts": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            TORCH_28,
            ["coqui-tts", "huggingface_hub"],
        ],
        "probe": "from TTS.tts.models.xtts import Xtts",
    },
    "vc_seedvc": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "clone": "https://github.com/Plachtaa/seed-vc.git",
        "pip": [
            # requirements.txt upstream mở đầu bằng 3 dòng torch nightly (--index-url áp cho cả
            # file) rồi mới pin torch==2.4.0: pip không tìm được 2.4.0 trên index nightly. Cài
            # đúng các pin của file đó, torch lấy từ index ổn định.
            ["torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0", "--index-url", CU121],
            [
                *("accelerate", "scipy==1.13.1", "librosa==0.10.2", "huggingface-hub>=0.28.1"),
                *("munch==4.0.0", "einops==0.8.0", "descript-audio-codec==1.0.0"),
                *("gradio==5.23.0", "pydub==0.25.1", "resemblyzer", "jiwer==3.0.3"),
                *("transformers==4.46.3", "FreeSimpleGUI==5.1.1", "soundfile==0.12.1"),
                *("sounddevice==0.5.0", "modelscope==1.18.1", "funasr==1.1.5", "numpy==1.26.4"),
                *("hydra-core==1.3.2", "pyyaml", "python-dotenv"),
            ],
        ],
        "probe": "import torch, torchaudio, librosa, munch, transformers.modeling_utils",
    },
}


def setup(name, root, env_root):
    recipe = RECIPES[name]
    root, env_root = Path(root).resolve(), Path(env_root).resolve()
    prefix = env_root / f".venv-{name}"
    python = prefix / "bin/python"
    folder = env_root / ".kaggle-tools" / name
    receipt, log = folder / "environment.json", folder / "setup.log"
    signature = {"recipe": recipe, "script": Path(__file__).read_text(encoding="utf-8")}
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous.get("status") == "ready" and previous.get("recipe") == recipe:
        return python
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "CUDA_HOME", "CUDA_PATH")
    }
    env.update(
        PYTHONNOUSERSITE="1",
        PIP_NO_CACHE_DIR="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        MPLBACKEND="Agg",
    )
    write_json(receipt, {"status": "installing", "recipe": recipe, "python": str(python)})
    if recipe.get("clone"):
        target = root / "external" / Path(recipe["clone"]).stem
        if not target.exists():
            execute(["git", "clone", "--depth", "1", recipe["clone"], target], root, log, env)
    mamba = micromamba(env_root / ".kaggle-tools/bin")
    spec = folder / "env.yml"
    spec.write_text(
        "channels: [conda-forge]\ndependencies:\n"
        + "".join(f"  - {p}\n" for p in [f"python={recipe['python']}", "pip", *recipe["conda"]]),
        encoding="utf-8",
    )
    command = "install" if python.exists() else "create"
    execute(
        [
            mamba,
            "--no-rc",
            "-r",
            env_root / ".kaggle-tools/mamba",
            command,
            "-y",
            "-p",
            prefix,
            "-f",
            spec,
        ],
        root,
        log,
        env,
    )
    for args in recipe["pip"]:
        execute([python, "-m", "pip", "install", "--no-input", *args], root, log, env)
    remove_package_cache(env_root / ".kaggle-tools/mamba")
    execute([python, "-c", recipe["probe"]], root, log, env)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (folder / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    write_json(
        receipt,
        {
            "status": "ready",
            "recipe": recipe,
            "python": str(python),
            "signature_chars": len(json.dumps(signature)),
        },
    )
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=sorted(RECIPES))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--env-root", default=None)
    args = parser.parse_args()
    for item in args.names:
        print(f"{item}={setup(item, args.root, args.env_root or args.root)}", flush=True)
