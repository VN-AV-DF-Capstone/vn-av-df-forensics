"""Gom speaker đã duyệt đủ thành part (20 nam + 20 nữ × 20 clip), chia split trong part."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["parts"])
