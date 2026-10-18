"""Duyệt theo speaker ở http://127.0.0.1:8001 : chọn 20 clip + gán giới cho từng speaker."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["review"])
