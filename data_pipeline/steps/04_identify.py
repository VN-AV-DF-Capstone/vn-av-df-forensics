"""Gán speaker cho clip mới cắt của một batch: `04_identify.py` hoặc `04_identify.py 3`.

Hiệu chỉnh ngưỡng sau khi đã duyệt vài chục speaker: `04_identify.py --calibrate`.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    if sys.argv[1:2] == ["--calibrate"]:
        sys.argv = sys.argv[:1]
        run(["calibrate"])
    run(["identify"], batch_step=True)
