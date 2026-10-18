"""Kế hoạch sinh cho part sạch đang chọn (PART trong data_settings.py)."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["plan", *sys.argv[1:]]))
