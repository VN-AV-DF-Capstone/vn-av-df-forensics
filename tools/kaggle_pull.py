"""Tải Output notebook Kaggle (version mới nhất), bỏ qua file đã có mà không gọi mạng.

    python tools/kaggle_pull.py <owner/slug> <thư mục đích> [regex lọc] [số luồng]

Ví dụ (từ thư mục vn-av-df-forensics, Output nằm dưới vn-av-df-forensics/ nên đích là ..):
    python tools/kaggle_pull.py linhxm/vn-av-df-train .. "/runs/"

Khác `kaggle kernels output`: file đã có (đọc được trọn vẹn) bỏ qua ngay, mỗi file tự thử lại khi
mạng lỗi, tải song song, ghi .part rồi mới đổi tên. Chạy lại bao nhiêu lần cũng được.
Kaggle API chỉ trả Output của version mới nhất. Cần Kaggle CLI bản mới (Python ≥3.11) và token.
"""

import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def complete(path):
    """File cũ (có thể tải dở) chỉ được giữ khi đọc được trọn vẹn."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as z:  # Bị cắt cụt thì mất central directory → lỗi.
                z.namelist()
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def list_files(api, owner, slug):
    files, token = [], None
    while True:
        for attempt in range(10):
            try:
                with api.build_kaggle_client() as kaggle:
                    request = ApiListKernelSessionOutputRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    request.page_size = 200
                    if token:
                        request.page_token = token
                    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
                break
            except Exception as e:
                print(f"Liệt kê lỗi ({e}), thử lại {attempt + 1}/10", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit("Không liệt kê được danh sách file")
        files += [(f.file_name, f.url) for f in response.files or []]
        print(f"Đã liệt kê {len(files)} file", flush=True)
        token = response.next_page_token
        if not token:
            return files


def fetch(session, url, target):
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(8):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
            part.replace(target)
            return None
        except Exception as e:
            error = e
            time.sleep(3 * (attempt + 1))
    return f"{target}: {error}"


def main():
    kernel, dest = sys.argv[1], Path(sys.argv[2])
    pattern = re.compile(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    owner, slug = kernel.split("/")[:2]
    api = KaggleApi()
    api.authenticate()
    files = [
        (name, url)
        for name, url in list_files(api, owner, slug)
        if not pattern or pattern.search(name)
    ]
    todo = [(name, url) for name, url in files if not complete(dest / name)]
    print(
        f"Tổng {len(files)} file, đã có {len(files) - len(todo)}, cần tải {len(todo)}", flush=True
    )
    failed, start = [], time.time()
    session = requests.Session()
    with ThreadPoolExecutor(threads) as pool:
        jobs = [pool.submit(fetch, session, url, dest / name) for name, url in todo]
        for i, job in enumerate(as_completed(jobs), 1):
            if job.result():
                failed.append(job.result())
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} | {(time.time() - start) / 60:.1f} phút", flush=True)
    for line in failed:
        print("LỖI", line)
    print("Xong." if not failed else f"{len(failed)} file lỗi: chạy lại lệnh để tải tiếp.")


if __name__ == "__main__":
    main()
