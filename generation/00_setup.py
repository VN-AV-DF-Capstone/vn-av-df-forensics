"""Tải source / weight các generator trong settings.py.

    python generation/00_setup.py                              mọi generator
    python generation/00_setup.py --generators iplap diff2lip  chỉ vài generator

Môi trường worker (Python riêng của từng generator / TTS): environments/worker_envs.py.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["setup", *sys.argv[1:]]))
