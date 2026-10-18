"""Phiên giọng: ① real, ② sham, ③ giọng AI (track A/B/C) và track B cho ⑥. Chạy trước 03."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["voice", *sys.argv[1:]]))
