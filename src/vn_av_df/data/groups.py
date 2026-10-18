"""Đọc/ghi manifest; nhóm liên thông người/nguồn dùng chung với bước chia split (data_pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

from vn_av_data.data.split import SPLITS, connected_groups, identity_list, identity_nodes

from vn_av_df.common.runtime import atomic_bytes

__all__ = [
    "SPLITS",
    "connected_groups",
    "identity_list",
    "identity_nodes",
    "read_manifest",
    "write_manifest",
]


def read_manifest(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_manifest(path, rows):
    atomic_bytes(
        path,
        (
            "\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in rows) + "\n"
        ).encode(),
    )
