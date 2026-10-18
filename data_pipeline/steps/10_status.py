"""Xem toàn cảnh: batch, clip, speaker, part, dung lượng còn trống."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["status"])
