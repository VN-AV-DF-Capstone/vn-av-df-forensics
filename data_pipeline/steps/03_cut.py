"""Cắt clip ứng viên của một batch vào pool: `03_cut.py` hoặc `03_cut.py 3`.

Đo bộ phát hiện jump cut trên raw của batch: `03_cut.py --jumpcut-eval 3`.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--jumpcut-eval":
        batch = sys.argv[2]
        sys.argv = sys.argv[:1] + sys.argv[3:]
        run(["jumpcut-eval", "--batch", batch])
    run(["cut"], batch_step=True)
