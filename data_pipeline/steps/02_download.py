"""Tải một batch: `02_download.py` (batch đầu còn việc) hoặc `02_download.py 3`.

Ctrl+C rồi chạy lại để tải tiếp; video lỗi mạng được thử lại.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["download"], batch_step=True)
