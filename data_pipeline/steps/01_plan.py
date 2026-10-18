"""Đọc data/inputs/urls.csv (url, speakers = 1 | n, note) → chọn video → chia batch raw.

Lần đầu chạy sẽ tạo file mẫu urls.csv. Thêm URL rồi chạy lại: chỉ xử lý dòng mới / dòng lỗi.
Danh sách từng batch: data/raw/batch_XXX/videos.csv. Thêm --cookies-from-browser firefox nếu
YouTube chặn.
"""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["plan"])
