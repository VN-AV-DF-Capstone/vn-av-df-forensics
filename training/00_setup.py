"""Tải encoder: AV-HuBERT Large (source + checkpoint + landmark) và XLS-R-300M."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import bootstrap

bootstrap()
from vn_av_df.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["setup", "--generators", "--encoders", *sys.argv[1:]]))
