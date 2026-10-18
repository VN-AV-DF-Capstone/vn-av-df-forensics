"""Xoá bản ở máy; chỉ chạy được với thứ đã đánh dấu lên Drive (08_archive.py ... --uploaded).

    09_clean.py raw 3     xoá video raw batch 3 + zip (clip đã cắt trong pool vẫn giữ)
    09_clean.py part 1    xoá thư mục export part 1 + zip + mọi clip trong pool của 40 người đó
    09_clean.py pool      xoá clip thừa: không được chọn, của speaker đã duyệt xong hoặc bị loại
                          (kể cả clip tự loại do sync); chỉ clip mà raw của batch đã lên Drive
Ledger vẫn giữ mọi thông tin của thứ đã xoá.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["clean"])
