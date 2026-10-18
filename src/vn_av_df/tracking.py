"""Log W&B cho quá trình train v003: trích feature, train, test, báo cáo (bằng chứng cho báo cáo).

Bật bằng cfg["wandb"] (notebook prepare / train đặt; project để private):
    {"project": "vn-av-df-v003", "entity": None, "mode": "online"}
Không đặt cfg["wandb"] thì mọi hàm ở đây không làm gì: local và test không cần cài wandb.
Không đẩy ảnh / video người thật; chỉ số liệu, bảng, biểu đồ và file kết quả.

Mỗi run ghi config + hash dữ liệu/code để tái lập; id run lưu ở `<thư mục>/wandb-<tên>.json` để
phiên sau mở lại đúng run (resume) khi chạy tiếp sau khi bị ngắt.
"""

import math
import os
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import read_json, write_json

PROJECT = "vn-av-df-v003"


class Run:
    """Một run W&B; `active` False thì mọi lệnh bỏ qua."""

    def __init__(self, wandb=None, run=None):
        self.wandb, self.run = wandb, run

    @property
    def active(self):
        return self.run is not None

    def log(self, data, step=None):
        if self.active and data:
            clean = {k: v for k, v in data.items() if v is not None and not _nan(v)}
            self.run.log(clean, step=step) if step is not None else self.run.log(clean)

    def summary(self, data):
        if self.active:
            for k, v in data.items():
                if v is not None and not _nan(v):
                    self.run.summary[k] = v

    def table(self, key, columns, rows):
        if self.active and rows:
            self.run.log(
                {key: self.wandb.Table(columns=list(columns), data=[list(r) for r in rows])}
            )

    def hist(self, key, values):
        values = [float(v) for v in values if v is not None and not _nan(v)]
        if self.active and values:
            self.run.log({key: self.wandb.Histogram(values)})

    def line(self, key, xs, ys, x="x", y="y", title=None):
        if self.active and len(xs):
            table = self.wandb.Table(
                columns=[x, y], data=[[float(a), float(b)] for a, b in zip(xs, ys)]
            )
            self.run.log({key: self.wandb.plot.line(table, x, y, title=title or key)})

    def bar(self, key, counts, label="group", value="count", title=None):
        if self.active and counts:
            table = self.wandb.Table(
                columns=[label, value], data=[[str(k), v] for k, v in counts.items()]
            )
            self.run.log({key: self.wandb.plot.bar(table, label, value, title=title or key)})

    def artifact(self, name, kind, paths, metadata=None):
        if not self.active:
            return
        art = self.wandb.Artifact(name, type=kind, metadata=metadata or {})
        added = False
        for path in map(Path, paths):
            if path.is_file():
                art.add_file(str(path))
                added = True
            elif path.is_dir():
                art.add_dir(str(path), name=path.name)
                added = True
        if added:
            self.run.log_artifact(art)

    def finish(self):
        if self.active:
            self.run.finish()
            self.run = None


def _nan(value):
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def open_run(cfg, job_type, name, group=None, config=None, folder=None, tags=()):
    """Mở run; `folder` có thì lưu id để phiên sau tiếp tục đúng run."""
    options = cfg.get("wandb")
    if not options:
        return Run()
    import wandb

    record = Path(folder) / f"wandb-{name}.json" if folder else None
    previous = read_json(record) if record and record.exists() else {}
    run = wandb.init(
        project=options.get("project", PROJECT),
        entity=options.get("entity"),
        mode=options.get("mode", "online"),
        dir=options.get("dir") or os.environ.get("WANDB_DIR"),
        job_type=job_type,
        group=group,
        name=name,
        id=previous.get("id"),
        resume="allow",
        config=config or {},
        tags=list(tags),
        reinit=True,
    )
    if record:
        write_json(record, {"id": run.id, "name": name, "job_type": job_type, "group": group})
    return Run(wandb, run)


# ---------------------------------------------------------------- nội dung dùng chung
def lag_profile(a, v, valid, max_lag=15):
    """Độ lệch tốt nhất giữa đặc trưng tiếng và hình (cos trung bình theo độ lệch, như
    SpeechForensics); dương = tiếng trễ. Trả (lag, độ nhọn = max − median)."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    n = len(a)
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, lag), min(n, n + lag)
        if hi - lo < 10:
            scores.append(np.nan)
            continue
        keep = valid[lo:hi] & valid[lo - lag : hi - lag]
        sims = (a[lo:hi] * v[lo - lag : hi - lag]).sum(1)
        scores.append(float(sims[keep].mean()) if keep.any() else np.nan)
    scores = np.asarray(scores)
    if np.all(np.isnan(scores)):
        return None, None
    best = int(np.nanargmax(scores))
    return best - max_lag, float(np.nanmax(scores) - np.nanmedian(scores))


def curves(run, key, series, x="fpr", y="tpr", title=None):
    """Nhiều đường trên một biểu đồ: series = {tên: (xs, ys)}."""
    if not run.active or not series:
        return
    data = [[name, float(a), float(b)] for name, (xs, ys) in series.items() for a, b in zip(xs, ys)]
    table = run.wandb.Table(columns=["series", x, y], data=data)
    run.run.log({key: run.wandb.plot.line(table, x, y, stroke="series", title=title or key)})
