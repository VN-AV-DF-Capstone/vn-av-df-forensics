"""Tạo worker riêng trên Kaggle/Linux; không sửa Python của notebook hoặc máy local.

Chạy sau action tải source/weights. Chỉ dùng stdlib để bootstrap không phụ thuộc
torch/fairseq/MMLab của kernel. Receipt chỉ chuyển ready sau pip check và import probe.
--env-root đặt môi trường (~10+ GB) ngoài checkout, ví dụ /kaggle/temp, để không tính vào
giới hạn Output 20 GB của /kaggle/working; cache pip/micromamba không giữ lại sau khi cài.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

MAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
MMCV_INDEX = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"


def digest(path):
    """Hash file cấu hình/source để không tái dùng môi trường sai phiên bản."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Ghi receipt nguyên tử, giữ trạng thái installing khi cài đặt bị gián đoạn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute(command, cwd, log, env=None):
    """Stream log ra notebook, đồng thời giữ file để tra lỗi pip/conda."""
    command = [str(arg) for arg in command]
    print("Run:", " ".join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nRun: " + " ".join(command) + "\n")
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                stream.write(line)
                print(line, end="", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"Worker setup failed ({code}); inspect {log}")


def extract_micromamba(archive, destination):
    """Chỉ đọc đúng binary regular file; không extract đường dẫn/link từ archive."""
    with tarfile.open(archive, "r:bz2") as package:
        member = package.getmember("bin/micromamba")
        if not member.isfile():
            raise ValueError("Micromamba archive entry must be a regular file")
        with package.extractfile(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    destination.chmod(0o755)


def micromamba(folder):
    """Tải từ endpoint chính thức một lần và lưu URL/hash; không cần conda có sẵn."""
    binary, receipt = folder / "micromamba", folder / "micromamba.json"
    folder.mkdir(parents=True, exist_ok=True)
    if binary.is_file():
        if not receipt.is_file() or json.loads(receipt.read_text())["sha256"] != digest(binary):
            raise ValueError("Micromamba binary differs from its receipt")
        return binary
    archive = folder / "micromamba.tar.bz2.partial"
    staged = folder / "micromamba.partial"
    try:
        with urllib.request.urlopen(MAMBA_URL, timeout=120) as response, archive.open("wb") as out:
            resolved_url = response.url
            shutil.copyfileobj(response, out)
        extract_micromamba(archive, staged)
        write_json(
            receipt,
            {
                "url": MAMBA_URL,
                "resolved_url": resolved_url,
                "archive_sha256": digest(archive),
                "sha256": digest(staged),
            },
        )
        staged.replace(binary)
    finally:
        archive.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
    return binary


def installation_commands(worker, python, root, constraints):
    """Lệnh cài theo worker; mọi pip đều gọi bằng interpreter đích tuyệt đối."""
    pip = [python, "-m", "pip", "install", "--no-input", "--constraint", constraints]
    commands = [
        pip
        + ["torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", TORCH_INDEX]
    ]
    if worker == "avhubert":
        # -v: pip giấu output build; cần lỗi compiler thật trong setup.log.
        commands.append(
            pip + ["--no-build-isolation", "-v", "-e", root / "external/av_hubert/fairseq"]
        )
    else:
        commands.extend(
            [
                pip + ["-r", root / "external/MuseTalk/requirements.txt"],
                pip + ["mmengine==0.10.7", "mmcv==2.0.1", "--only-binary=mmcv", "-f", MMCV_INDEX],
                pip + ["mmdet==3.1.0", "mmpose==1.1.0"],
            ]
        )
    return commands


def probe_code(worker, root, output, require_cuda):
    """Kiểm import/API thư viện và CUDA; không coi đây là inference checkpoint thật."""
    imports = (
        "sys.path[:0] = [str(root/'external/av_hubert/avhubert'), str(root/'external/av_hubert/fairseq')]\n"
        "import dlib, fairseq, hubert, hubert_pretraining, skimage, python_speech_features\n"
        if worker == "avhubert"
        else "sys.path.insert(0,str(root/'external/MuseTalk'))\n"
        "from mmcv.ops import get_compiling_cuda_version\n"
        "from mmpose.apis import init_model\n"
        "from musetalk.utils.utils import load_all_model\n"
        "from musetalk.utils.audio_processor import AudioProcessor\n"
        "from musetalk.utils.face_parsing import FaceParsing\n"
    )
    return (
        "import json, sys\nfrom pathlib import Path\nimport torch, numpy as np, av, cv2\n"
        "import imageio_ffmpeg\n"
        f"root=Path({str(root)!r})\n"
        "assert sys.version_info[:2] == (3,10), 'Worker requires Python 3.10'\n"
        "assert torch.__version__.split('+')[0] == '2.0.1', 'Unexpected worker torch'\n"
        "torch.from_numpy(np.zeros(2,dtype=np.float32)).numpy()\n"
        + imports
        + (
            "assert torch.cuda.is_available(), 'Enable a Kaggle GPU before using this worker'\n"
            "torch.ones(1,device='cuda').sum().item()\n"
            if require_cuda
            else ""
        )
        + "result={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,"
        "'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,"
        "'ffmpeg':imageio_ffmpeg.get_ffmpeg_exe(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\n"
        + f"Path({str(output)!r}).write_text(json.dumps(result,indent=2),encoding='utf-8')\n"
    )


def remove_package_cache(mamba_root):
    """Xoá gói conda đã tải/giải nén sau khi cài xong; env đã có bản riêng (hardlink/copy)."""
    cache = mamba_root / "pkgs"
    if not cache.is_dir():
        return 0
    size = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file() and not p.is_symlink())
    shutil.rmtree(cache, ignore_errors=True)
    return size


def setup_worker(root, worker, require_cuda=True, env_root=None):
    """Tạo/resume môi trường do project quản lý, chỉ sau khi source upstream đã tải.

    Source/weights đọc từ root; môi trường, micromamba, receipt và log nằm ở env_root
    (mặc định root) để receipt luôn đi cùng môi trường nó mô tả.
    """
    if worker not in {"avhubert", "musetalk"}:
        raise ValueError("Unknown worker")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError(
            "Automatic bootstrap targets Kaggle/Linux x86_64; see manual setup for local"
        )
    root = Path(root).resolve()
    env_root = Path(env_root).resolve() if env_root else root
    specification = root / "environments" / f"{worker}.yml"
    source = root / "external" / ("av_hubert" if worker == "avhubert" else "MuseTalk")
    required = [specification, source / "vn_av_revision.json"]
    required += (
        [source / "fairseq/setup.py", source / "avhubert/hubert.py"]
        if worker == "avhubert"
        else [source / "requirements.txt", source / "musetalk/utils/utils.py"]
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Run the notebook asset setup cell first; missing {path}")
    signature = {str(p.relative_to(root)): digest(p) for p in required}
    signature["bootstrap"] = digest(__file__)
    prefix = env_root / f".venv-{worker}"
    python = prefix / "bin/python"
    tools = env_root / ".kaggle-tools"
    folder = tools / worker
    receipt, log = folder / "environment.json", folder / "setup.log"
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous["signature"] != signature:
        raise ValueError("Worker specification/source changed; use a fresh Kaggle working checkout")
    if prefix.exists() and not previous:
        raise ValueError(f"Refuse to change an unmanaged environment: {prefix}")
    env = os.environ.copy()
    # Không cho pip/conda hoặc PYTHONPATH của kernel đổi nơi cài/nơi import của worker.
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "VIRTUAL_ENV"):
        env.pop(name, None)
    # Kaggle đặt CUDA_HOME (CUDA 12.x). fairseq thấy biến này thì build thêm extension .cu, torch
    # cu118 từ chối vì lệch CUDA. Worker không cần extension CUDA tự build (torch có runtime riêng;
    # mmcv dùng wheel dựng sẵn) nên bỏ biến khi cài.
    for name in ("CUDA_HOME", "CUDA_PATH"):
        env.pop(name, None)
    env.update(
        # Kernel Jupyter đặt MPLBACKEND=module://matplotlib_inline...; worker không có gói đó,
        # import mmpose → matplotlib sẽ lỗi ở bước probe. Worker không vẽ: dùng Agg.
        MPLBACKEND="Agg",
        PYTHONNOUSERSITE="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        # Wheel torch/MMLab đã nằm trong env; cache pip chỉ chiếm thêm vài GB.
        PIP_NO_CACHE_DIR="1",
        PIP_CONSTRAINT=str(folder / "constraints.txt"),
    )
    if not previous or previous["status"] != "ready":
        write_json(receipt, {"status": "installing", "signature": signature, "python": str(python)})
        constraints = folder / "constraints.txt"
        constraints.write_text(
            "torch==2.0.1\ntorchvision==0.15.2\ntorchaudio==2.0.2\n"
            "numpy==1.23.5\nscipy==1.10.1\nopencv-python==4.9.0.80\n"
            "setuptools<70\n",
            encoding="utf-8",
        )
        mamba = micromamba(tools / "bin")
        # Chạy lại env update nếu lần đầu dừng giữa phần conda/pip của YAML.
        command = "install" if python.exists() else "create"
        execute(
            [
                mamba,
                "--no-rc",
                "-r",
                tools / "mamba",
                command,
                "-y",
                "-p",
                prefix,
                "-f",
                specification,
            ],
            root,
            log,
            env,
        )
        for command in installation_commands(worker, python, root, constraints):
            execute(command, root, log, env)
        freed = remove_package_cache(tools / "mamba")
        print(f"Removed micromamba package cache: {freed / 1e9:.2f} GB", flush=True)
    write_json(receipt, {"status": "checking", "signature": signature, "python": str(python)})
    execute([python, "-m", "pip", "check"], root, log, env)
    probe = folder / "probe.json"
    execute([python, "-c", probe_code(worker, root, probe, require_cuda)], source, log, env)
    # Freeze là artifact tái lập, không chỉ một thông báo 'đã cài'.
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    (folder / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    write_json(
        receipt,
        {
            "status": "ready",
            "signature": signature,
            "python": str(python),
            "probe": json.loads(probe.read_text()),
            "pip_freeze_sha256": digest(folder / "pip-freeze.txt"),
        },
    )
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and verify an isolated Kaggle/Linux worker."
    )
    parser.add_argument("worker", choices=("avhubert", "musetalk"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--env-root",
        type=Path,
        default=None,
        help="Nơi đặt môi trường/log (mặc định --root); Kaggle: thư mục ngoài /kaggle/working",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    print(
        setup_worker(
            args.root, args.worker, require_cuda=not args.allow_cpu, env_root=args.env_root
        )
    )
