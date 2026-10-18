"""Worker sinh chạy liên tục: nạp model một lần, nhận nhiều yêu cầu qua stdin/stdout.

Mỗi generator vẫn chạy bằng Python của môi trường riêng (process riêng), chỉ không khởi
động lại process và nạp lại model cho từng cặp. Dòng giao thức có tiền tố riêng nên log
upstream (print/tqdm) không lẫn vào kết quả; log đó được giữ lại để báo khi lỗi.
Worker chết giữa chừng thì lần gọi sau tự mở worker mới (cặp đang làm bị báo lỗi).
"""

import collections
import json
import os
import queue
import subprocess
import sys
import threading
import traceback

READY, DONE, FAIL = "@@VNAV_READY", "@@VNAV_DONE", "@@VNAV_FAIL"


def serve(handle):
    """Vòng lặp phía worker: mỗi dòng stdin một yêu cầu JSON, trả DONE/FAIL trên dòng riêng.

    Chỉ dùng stdlib để chạy được trong môi trường worker; xuống dòng trước tiền tố vì
    tqdm của upstream có thể để dở dòng (\\r) trên stdout/stderr đã gộp.
    """
    print("\n" + READY, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            result = handle(json.loads(line)) or {}
            print("\n" + DONE + " " + json.dumps(result), flush=True)
        except Exception as exc:  # noqa: BLE001 -- báo lỗi của một cặp, worker vẫn sống
            error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            print("\n" + FAIL + " " + json.dumps(error), flush=True)


class Worker:
    """Phía gọi: mở process worker, chờ READY, gửi từng yêu cầu và chờ kết quả có timeout."""

    def __init__(self, command, cwd=None, env=None, startup_timeout=1800):
        self.command = [str(arg) for arg in command]
        self.tail = collections.deque(maxlen=80)
        self.lines = queue.Queue()
        # MPLBACKEND inline của kernel Jupyter làm import matplotlib trong worker lỗi; dùng Agg.
        env = {**(os.environ if env is None else env), "MPLBACKEND": "Agg"}
        self.process = subprocess.Popen(
            self.command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()
        self._wait(READY, startup_timeout)

    def _read(self):
        for line in self.process.stdout:
            text = line.strip("\r\n")
            if text.startswith("@@VNAV_"):
                self.lines.put(text)
            elif text.strip():
                self.tail.append(text.split("\r")[-1])
        self.lines.put(None)

    def _wait(self, expected, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"Worker timeout after {timeout}s: {self._log()}") from None
        if line is None:
            code = self.process.wait()
            raise RuntimeError(f"Worker exited ({code}): {self.command[:3]}\n{self._log()}")
        kind, _, payload = line.partition(" ")
        if kind == FAIL:
            error = json.loads(payload)
            raise RuntimeError(f"Worker failed: {error['error']}\n{error['traceback'][-3000:]}")
        if kind != expected:
            raise RuntimeError(f"Unexpected worker reply {kind}")
        return json.loads(payload) if payload else {}

    def _log(self):
        return "\n".join(self.tail)

    @property
    def alive(self):
        return self.process.poll() is None

    def call(self, request, timeout=3600):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._wait(DONE, timeout)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()


_workers, _locks = {}, {}
_lock = threading.Lock()


def get(key, factory):
    """Một worker cho mỗi key (generator, GPU, cấu hình); mở lại nếu worker cũ đã chết.

    Khoá riêng từng key: hai GPU nạp model song song, không chờ nhau.
    """
    with _lock:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        worker = _workers.get(key)
        if worker is None or not worker.alive:
            worker = factory()
            with _lock:
                _workers[key] = worker
        return worker


def close_all():
    with _lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.close()
