"""Đường dẫn chung của các bước real v003 (chạy từ bất kỳ thư mục nào).

DATA: workspace (ledger, raw, pool, registry, archive). Thử luật cắt mới trên workspace riêng:
    set VNAV_DATA=data_trial   (PowerShell: $env:VNAV_DATA="data_trial")
Part export ra exports/<DATASET_NAME>/<DATASET_NAME>-partN (generation đọc từ đó).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from data_settings import DATASET_NAME  # noqa: E402, F401

DATA = os.environ.get("VNAV_DATA", "data")
EXPORTS = "exports"
