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

RECIPES = {
    # VideoReTalking README: Python 3.8, torch 1.9.0+cu111, requirements.txt.
    "gen_vrt": {
        "python": "3.8",
        "conda": ["ffmpeg", "dlib"],
        "pip": [
            ["torch==1.9.0+cu111", "torchvision==0.10.0+cu111", "-f", CU111],
            ["-r", "external/videoretalking/requirements.txt"],
        ],
        "probe": "import torch, cv2, face_alignment",
    },
    # IP-LAP README: Python 3.7.13, torch 1.10.0, torchvision 0.11.0, requirements.txt.
    "gen_iplap": {
        "python": "3.7",
        "conda": ["ffmpeg"],
        "pip": [
            ["torch==1.10.0+cu111", "torchvision==0.11.0+cu111", "-f", CU111],
            ["-r", "external/iplap/requirements.txt"],
        ],
        "probe": "import torch, cv2",
    },
    # LatentSync: Python 3.10 + requirements.txt (đã pin torch / diffusers của repo).
    "gen_latentsync": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [["-r", "external/latentsync_1_5/requirements.txt"]],
        "probe": "import torch, diffusers",
    },
    # Diff2Lip README: Python 3.9, ffmpeg 5.0.1, requirements.txt; guided-diffusion cần mpi4py.
    "gen_diff2lip": {
        "python": "3.9",
        "conda": ["ffmpeg=5.0.1", "mpi4py"],
        "pip": [
            [
                "torch==1.12.1",
                "torchvision==0.13.1",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu113",
            ],
            ["-r", "external/diff2lip/requirements.txt"],
        ],
        "probe": "import torch, mpi4py",
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
            ["torch", "torchaudio", "--index-url", CU121],
            ["git+https://github.com/k2-fsa/OmniVoice.git", "soundfile"],
        ],
        "probe": "from omnivoice import OmniVoice",
    },
    # VoxCPM: Python >= 3.10, torch >= 2.5, CUDA >= 12.
    "tts_voxcpm": {
        "python": "3.11",
        "conda": ["ffmpeg"],
        "pip": [["torch>=2.5", "torchaudio", "--index-url", CU121], ["voxcpm", "soundfile"]],
        "probe": "from voxcpm import VoxCPM",
    },
    "tts_xtts": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "pip": [
            ["torch==2.4.1", "torchaudio==2.4.1", "--index-url", CU121],
            ["coqui-tts", "huggingface_hub"],
        ],
        "probe": "from TTS.tts.models.xtts import Xtts",
    },
    "vc_seedvc": {
        "python": "3.10",
        "conda": ["ffmpeg"],
        "clone": "https://github.com/Plachtaa/seed-vc.git",
        "pip": [
            ["torch", "torchaudio", "--index-url", CU121],
            ["-r", "external/seed-vc/requirements.txt"],
        ],
        "probe": "import torch",
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
