"""Seed-VC (Plachtaa/seed-vc): đổi giọng câu nói nguồn sang âm sắc của người tham chiếu.

Gọi inference.py của repo (đường dẫn: biến VNAV_SEEDVC_REPO) cho từng yêu cầu.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import main


def load():
    repo = Path(os.environ.get("VNAV_SEEDVC_REPO", "external/seed-vc")).resolve()
    if not (repo / "inference.py").is_file():
        raise FileNotFoundError(f"Seed-VC repo missing: {repo}")
    return repo


def handle(repo, request):
    if request["mode"] != "vc":
        raise ValueError(f"Unsupported mode {request['mode']}")
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(
            [
                sys.executable,
                "inference.py",
                *("--source", request["source"], "--target", request["target"]),
                *("--output", folder, "--diffusion-steps", "25"),
                *("--length-adjust", "1.0", "--inference-cfg-rate", "0.7"),
            ],
            cwd=repo,
            check=True,
        )
        produced = sorted(Path(folder).glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if not produced:
            raise RuntimeError("Seed-VC produced no wav")
        out = Path(request["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced[-1], out)
    return {"out": str(out)}


if __name__ == "__main__":
    main(load, handle)
