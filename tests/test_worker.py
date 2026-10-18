import sys
from pathlib import Path

import pytest

from vn_av_df import worker

# Worker giả: in log kiểu tqdm (dở dòng) rồi trả kết quả; "fail" ném lỗi, "crash" chết hẳn.
SCRIPT = """
import os, sys
sys.path.insert(0, {src!r})
from vn_av_df.worker import serve
loaded = {{"model": "nạp một lần"}}

def handle(request):
    print("\\r 50%|#####     |", end="", flush=True)
    if request["x"] == "fail":
        raise ValueError("bad input")
    if request["x"] == "crash":
        os._exit(3)
    return {{"echo": request["x"], "pid": os.getpid(), "backend": os.environ.get("MPLBACKEND"), **loaded}}

serve(handle)
"""


@pytest.fixture
def command(tmp_path):
    path = tmp_path / "fake_worker.py"
    src = str(Path(__file__).resolve().parents[1] / "src")
    path.write_text(SCRIPT.format(src=src), encoding="utf-8")
    return [sys.executable, str(path)]


def test_worker_reuses_process_reports_failures_and_restarts(command, monkeypatch):
    # Kernel Jupyter đặt backend inline; worker ở môi trường riêng phải nhận Agg.
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    try:
        first = worker.get("fake", lambda: worker.Worker(command, startup_timeout=60))
        a = first.call({"x": 1}, timeout=60)
        b = first.call({"x": 2}, timeout=60)
        # Cùng một process cho mọi yêu cầu: model chỉ nạp một lần.
        assert a["echo"] == 1 and b["echo"] == 2 and a["pid"] == b["pid"]
        assert a["model"] == "nạp một lần" and a["backend"] == "Agg"
        with pytest.raises(RuntimeError, match="bad input"):
            first.call({"x": "fail"}, timeout=60)
        assert first.alive and first.call({"x": 3}, timeout=60)["pid"] == a["pid"]
        with pytest.raises(RuntimeError, match="exited"):
            first.call({"x": "crash"}, timeout=60)
        # Worker chết: lần lấy sau mở process mới.
        second = worker.get("fake", lambda: worker.Worker(command, startup_timeout=60))
        assert second is not first and second.call({"x": 4}, timeout=60)["pid"] != a["pid"]
    finally:
        worker.close_all()
