"""Đóng gói và đánh dấu đã lên Drive (xoá bản ở máy: 09_clean.py).

    08_archive.py raw 3              ZIP raw batch 3 (sau khi đã cut + identify xong)
    08_archive.py raw 3 --uploaded   đánh dấu: zip raw batch 3 đã lên Drive
    08_archive.py part 1             nhắc file zip của part 1 (07_export đã tạo)
    08_archive.py part 1 --uploaded  đánh dấu: zip part 1 đã lên Drive
Nhớ up cả thư mục data/ledger/ lên Drive mỗi lần.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["archive"])
