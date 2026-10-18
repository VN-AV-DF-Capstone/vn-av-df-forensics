"""Chạy trong môi trường Anaconda chính đang active để cài thư viện.

PROFILE = "training" tương đương: python -m pip install -e ./data_pipeline -e ".[training]"
Script này không tải checkpoint và không cài môi trường worker (environments/).
Tải encoder: python training/00_setup.py
"""

import subprocess
import sys
from pathlib import Path

# "data": thu thập/review; "generation" và "training": các bước sinh / train ở local.
# Trên Kaggle, notebook tự cài.
PROFILE = "training"

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if PROFILE not in {"data", "generation", "training"}:
        raise ValueError("PROFILE must be data, generation or training")
    args = [sys.executable, "-m", "pip", "install", "-e", str(root / "data_pipeline")]
    args += ["-e", str(root) if PROFILE == "data" else f"{root}[{PROFILE}]"]
    subprocess.run(args, check=True)
