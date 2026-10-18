"""Phiên lip-sync (④⑤⑥): python generation/03_lipsync.py --generators wav2lip_gan musetalk_1_5"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["lipsync", *sys.argv[1:]]))
