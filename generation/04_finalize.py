"""Gom record → manifest.jsonl (mặc định giữ mọi sample; review.csv để loại mẫu lỗi)."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["finalize", *sys.argv[1:]]))
