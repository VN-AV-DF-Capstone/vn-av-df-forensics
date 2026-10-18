"""Explicit, resumable downloads. Record exact revisions and hashes for reproducibility."""

import os
import shutil
import urllib.request
from pathlib import Path

from vn_av_df.common.runtime import read_json, run, sha, write_json

# Revision pin sẵn; repo không có ở đây thì lấy revision hiện tại lúc tải lần đầu và ghi lại.
HF_REVISIONS = {}


def hf_snapshot(name, target, patterns):
    """Tải đúng revision đã pin; ghi revision trước để phiên bị ngắt resume cùng snapshot."""
    from huggingface_hub import snapshot_download

    target = Path(target)
    revision_file = target / "vn_av_revision.json"
    if revision_file.exists():
        revision = read_json(revision_file)["revision"]
    else:
        from huggingface_hub import HfApi

        revision = HF_REVISIONS.get(name) or HfApi().model_info(name).sha
    write_json(revision_file, {"repo_id": name, "revision": revision})
    print(f"Downloading {name}@{revision}", flush=True)
    try:
        snapshot_download(
            name, revision=revision, local_dir=target, max_workers=2, allow_patterns=patterns
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download {name} failed. Rerun training/00_setup.py "
            f"on a working Internet connection; partial files retained. {exc}"
        ) from exc
    return {"repo_id": name, "revision": revision}


def pinned_repository(url, target, recursive=False):
    """Lần đầu ghi commit, lần sau yêu cầu đúng commit; không pull ngầm."""
    target = Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", *(["--recurse-submodules"] if recursive else []), url, target])
    commit = run(["git", "rev-parse", "HEAD"], cwd=target).strip()
    lock = target / "vn_av_revision.json"
    if lock.exists() and read_json(lock)["commit"] != commit:
        raise ValueError("Upstream commit changed; use new source/cache directory")
    write_json(lock, dict(repository=url, commit=commit))
    return commit


def pinned_snapshot(repo_id, target, patterns):
    """Resolve revision một lần trước download để resume cùng tài sản."""
    from huggingface_hub import HfApi, snapshot_download

    target = Path(target)
    lock = target / "vn_av_revision.json"
    revision = read_json(lock)["revision"] if lock.exists() else HfApi().model_info(repo_id).sha
    write_json(lock, dict(repo_id=repo_id, revision=revision))
    snapshot_download(
        repo_id, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=2
    )


def setup_avhubert(encoder):
    """Chuẩn bị AV-HuBERT (Base mặc định, hoặc checkpoint_url) và landmark; fairseq chỉ ở worker."""
    import bz2

    pinned_repository(
        "https://github.com/facebookresearch/av_hubert.git", encoder["repo"], recursive=True
    )
    result = {
        "checkpoint": fetch_file(
            encoder.get(
                "checkpoint_url",
                "https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter4.pt",
            ),
            encoder["checkpoint"],
        )
    }
    landmark = Path(encoder["landmarks"])
    compressed = landmark.with_suffix(".dat.bz2")
    fetch_file("https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", compressed)
    if not landmark.exists():
        from vn_av_df.common.runtime import atomic_bytes

        atomic_bytes(landmark, bz2.decompress(compressed.read_bytes()))
    result["mean_face"] = fetch_file(encoder["mean_face_url"], encoder["mean_face"])
    return result


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
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if partial.stat().st_size < 1024:
            raise ValueError(f"Invalid model download: {url}")
        os.replace(partial, destination)
        result = {"url": url, "sha256": sha(destination), "bytes": destination.stat().st_size}
        write_json(record, result)
        return result
    finally:
        partial.unlink(missing_ok=True)
