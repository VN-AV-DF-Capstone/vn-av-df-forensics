"""Kiểm bootstrap không tải package thật: isolation, resume, failure và archive safety."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_worker", Path(__file__).resolve().parents[1] / "environments/bootstrap_worker.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def installer(tmp_path, monkeypatch):
    """Giả lập process con, giữ filesystem/receipt thật để kiểm khi bị ngắt."""
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap, "micromamba", lambda path: path / "micromamba")
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **k: "numpy==1.23.5\n")
    for worker, source in (("avhubert", "av_hubert"), ("musetalk", "MuseTalk")):
        for relative in (
            f"environments/{worker}.yml",
            f"external/{source}/vn_av_revision.json",
            f"external/{source}/requirements.txt",
            f"external/{source}/fairseq/setup.py",
            f"external/{source}/avhubert/hubert.py",
            f"external/{source}/musetalk/utils/utils.py",
        ):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture", encoding="utf-8")
    commands = []

    def execute(command, cwd, log, env=None):
        command = [str(x) for x in command]
        commands.append((command, env))
        if "-p" in command:
            python = Path(command[command.index("-p") + 1]) / "bin/python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fixture")
        if "-c" in command:
            prefix = Path(command[0]).parents[1]  # <env_root>/.venv-<worker>/bin/python
            worker = prefix.name.removeprefix(".venv-")
            path = prefix.parent / ".kaggle-tools" / worker / "probe.json"
            path.write_text(json.dumps({"cuda_available": True}))

    monkeypatch.setattr(bootstrap, "execute", execute)
    return tmp_path, commands, execute


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_worker_isolated_and_ready_resume_rechecks_without_reinstall(
    installer, monkeypatch, worker
):
    root, commands, _ = installer
    monkeypatch.setenv("PYTHONPATH", "notebook-only-packages")
    monkeypatch.setenv("PIP_TARGET", "do-not-write-here")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    python = bootstrap.setup_worker(root, worker)
    assert python == root / f".venv-{worker}/bin/python"
    assert all("PYTHONPATH" not in env and "PIP_TARGET" not in env for _, env in commands)
    # CUDA_HOME của Kaggle (12.x) làm fairseq build extension .cu lệch torch cu118.
    assert all("CUDA_HOME" not in env for _, env in commands)
    for command, _ in commands:
        if "pip" in command:
            assert command[0] == str(python)
    installed = "\n".join(" ".join(c) for c, _ in commands)
    assert ("mmpose==1.1.0" in installed) == (worker == "musetalk")
    assert ("fairseq" in installed) == (worker == "avhubert")
    receipt = root / ".kaggle-tools" / worker / "environment.json"
    assert json.loads(receipt.read_text())["status"] == "ready"
    commands.clear()
    bootstrap.setup_worker(root, worker)
    assert len(commands) == 2  # pip check và import/CUDA probe vẫn phải chạy lại.
    assert all("install" not in command for command, _ in commands)
    (root / "environments" / f"{worker}.yml").write_text("changed")
    with pytest.raises(ValueError, match="specification/source changed"):
        bootstrap.setup_worker(root, worker)


def test_env_root_keeps_environment_outside_checkout_and_drops_caches(installer):
    root, commands, _ = installer
    workers = root.parent / "kaggle-temp"
    cache = workers / ".kaggle-tools/mamba/pkgs/torch-2.0.1/lib.so"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x" * 1000)
    python = bootstrap.setup_worker(root, "musetalk", env_root=workers)
    assert python == workers / ".venv-musetalk/bin/python"
    receipt = workers / ".kaggle-tools/musetalk/environment.json"
    assert json.loads(receipt.read_text())["status"] == "ready"
    # Checkout (Output Kaggle) không chứa env/receipt; cache gói conda bị xoá, pip không cache.
    assert not (root / ".venv-musetalk").exists() and not (root / ".kaggle-tools").exists()
    assert not cache.parent.parent.exists()
    assert all(env["PIP_NO_CACHE_DIR"] == "1" for _, env in commands)
    assert any(str(workers / ".kaggle-tools/mamba") in command for command, _ in commands)


def test_failed_install_resumes_but_failed_probe_is_not_ready(installer, monkeypatch):
    root, commands, real_execute = installer

    def fail(command, *args, **kwargs):
        if "--index-url" in command:
            raise RuntimeError("interrupted package download")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap.setup_worker(root, "avhubert")
    receipt = root / ".kaggle-tools/avhubert/environment.json"
    assert json.loads(receipt.read_text())["status"] == "installing"
    monkeypatch.setattr(bootstrap, "execute", real_execute)
    commands.clear()
    bootstrap.setup_worker(root, "avhubert")
    assert "install" in commands[0][0]  # Môi trường đã tạo, tiếp tục dependency còn dở.

    def fail_probe(command, *args, **kwargs):
        if "-c" in command:
            raise RuntimeError("CUDA unavailable")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail_probe)
    with pytest.raises(RuntimeError, match="CUDA"):
        bootstrap.setup_worker(root, "avhubert")
    assert json.loads(receipt.read_text())["status"] != "ready"


def test_missing_source_unmanaged_environment_and_non_linux_fail_early(installer, monkeypatch):
    root, commands, _ = installer
    source = root / "external/av_hubert/fairseq/setup.py"
    source.unlink()
    with pytest.raises(FileNotFoundError, match="asset setup"):
        bootstrap.setup_worker(root, "avhubert")
    (root / ".venv-musetalk").mkdir()
    with pytest.raises(ValueError, match="unmanaged"):
        bootstrap.setup_worker(root, "musetalk")
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Kaggle/Linux"):
        bootstrap.setup_worker(root, "musetalk")
    assert not commands


def test_archive_extraction_never_writes_other_members(tmp_path):
    archive, target = tmp_path / "package.tar.bz2", tmp_path / "micromamba"
    with tarfile.open(archive, "w:bz2") as package:
        for name in ("../escaped", "bin/micromamba"):
            member = tarfile.TarInfo(name)
            member.size = 7
            package.addfile(member, io.BytesIO(b"fixture"))
    bootstrap.extract_micromamba(archive, target)
    assert target.read_bytes() == b"fixture"
    assert not (tmp_path.parent / "escaped").exists()
    with tarfile.open(archive, "w:bz2") as package:
        member = tarfile.TarInfo("bin/micromamba")
        member.type, member.linkname = tarfile.SYMTYPE, "../elsewhere"
        package.addfile(member)
    with pytest.raises(ValueError, match="regular file"):
        bootstrap.extract_micromamba(archive, target)


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_generated_probe_is_valid_python(worker, tmp_path):
    compile(bootstrap.probe_code(worker, tmp_path, tmp_path / "probe.json", True), "probe", "exec")


@pytest.mark.parametrize("notebook", ["generate", "prepare", "train", "demo"])
def test_notebook_cells_compile_and_use_settings(notebook):
    path = Path(__file__).resolve().parents[1] / "notebooks" / f"{notebook}.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    for source in code:
        compile(source, str(path), "exec")
    text = "\n".join(code)
    assert "cfg = settings.config()" in text
    if notebook == "demo":
        assert "build_app" in text and "share=True" in text
    else:
        assert "prune_output" in text  # Output chỉ giữ kết quả.
        # W&B chỉ ở quá trình train (prepare + train), không ở bước sinh fake.
        assert ("WANDB_API_KEY" in text) == (notebook in ("prepare", "train"))
