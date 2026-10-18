"""Demo: phân tích một video bằng các model v003 đã train (checkpoint trong `runs/<RUN>/`).

1. Chuẩn hoá (25 fps, ≤ 720p, ≤ 60 s) và sửa độ lệch tiếng-hình bằng SyncNet (như bước dữ liệu).
2. Trích đặc trưng như lúc train (AV-HuBERT, crop miệng, XLS-R), nạp bằng chính bộ nạp của train.
3. Chạy model trên cửa sổ 7 s chồng nửa (model được train trên clip 4-7 s); điểm frame lấy trung
   bình các cửa sổ phủ frame đó, điểm video = cửa sổ cao nhất.
4. Ngưỡng mỗi tác vụ lấy từ `metrics.json` nếu đã chạy test: ngưỡng FPR 5 % trên real validation
   (ngoài đời fake hiếm; ngưỡng F1 tối đa chỉ dùng khi thiếu điểm vận hành).
Video tham chiếu (tuỳ chọn, real của cùng người): dùng cho model có tham chiếu người (TriHead).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from vn_av_df.common.runtime import read_json
from vn_av_df.training.data import BenchData, collate, to_device
from vn_av_df.training.evaluate import TASKS, frame_head, task_scores
from vn_av_df.training.models import pool_scores
from vn_av_df.training.prepare import make
from vn_av_df.training.train import load_model

WINDOW, STRIDE, REF_FRAMES = 175, 88, 50


def windows(length):
    if length <= WINDOW:
        return [(0, length)]
    starts = list(range(0, length - WINDOW + 1, STRIDE))
    if starts[-1] + WINDOW < length:
        starts.append(length - WINDOW)
    return [(s, s + WINDOW) for s in starts]


def blank_targets(item, length):
    for task in TASKS:
        item[f"y_{task}"] = np.zeros(length, np.float32)
    item["video"] = dict.fromkeys(TASKS, 0.0)
    return item


class Detector:
    def __init__(self, cfg, run_dir, models=None, device=None, aligner=None, encoders=None):
        self.cfg = dict(cfg)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_dir = Path(run_dir)
        folders = sorted(p for p in run_dir.iterdir() if (p / "best.pt").is_file())
        if models:
            folders = [p for p in folders if p.name.rsplit("_seed", 1)[0] in models]
        if not folders:
            raise FileNotFoundError(f"Không có checkpoint nào trong {run_dir}")
        first = {}
        for folder in folders:  # mỗi model một seed (seed nhỏ nhất)
            first.setdefault(folder.name.rsplit("_seed", 1)[0], folder)
        self.models = {}
        for name, folder in first.items():
            metrics = folder / "metrics.json"
            thresholds = (
                {
                    t: (m.get("operating_points", {}).get("fpr_0.05") or {}).get(
                        "threshold", m.get("threshold")
                    )
                    for t, m in read_json(metrics)["tasks"].items()
                }
                if metrics.is_file()
                else {}
            )
            self.models[name] = (load_model(folder / "best.pt", self.device), thresholds)
        self.needs = set().union(*(m.needs for m, _ in self.models.values()))
        self.aligner = aligner
        self.encoders = encoders or {}

    def encoder(self, name):
        if name not in self.encoders:
            self.encoders[name] = make(self.cfg, name, self.device)
        return self.encoders[name]

    def features(self, video, cache, sample_id):
        """Trích đặc trưng vào `cache/<encoder>/<sample_id>.npz` (cùng bố cục lúc train)."""
        needs = {"avhubert"} if self.needs & {"av", "mouth"} else set()
        if "mouth" in self.needs:
            needs.add("mouth")
        if "ssl" in self.needs:
            needs.add("xlsr")
        for name in ("avhubert", "mouth", "xlsr"):
            if name not in needs:
                continue
            (cache / name).mkdir(parents=True, exist_ok=True)
            request = {"video": str(video)}
            if name == "mouth":
                request["box_cache"] = str(cache / "avhubert" / f"{sample_id}.npz")
            self.encoder(name).extract(request, cache / name / f"{sample_id}.npz")

    def prepare(self, video, work, name):
        """Chuẩn hoá + sửa độ lệch → (video đã xử lý, thông tin độ lệch)."""
        from vn_av_df.demo.media import Aligner, normalize

        normal = normalize(video, work / f"{name}_normal.mp4")
        if self.aligner is None:
            self.aligner = Aligner()
        return self.aligner.align(normal, work / f"{name}_aligned.mp4")

    @torch.no_grad()
    def score(self, model, item, length, refs):
        frames = {}
        counts = np.zeros(length)
        best = {}
        for a, b in windows(length):
            part = blank_targets({k: v[a:b] for k, v in item.items()}, b - a)
            part["index"] = 0
            if model.references:
                part["refs"] = refs[: model.references]
            batch = to_device(collate([part]), self.device)
            heads = model(batch)
            mask = batch["mask"]
            values = {
                h: v[0].float().cpu().numpy() for h, v in heads.items() if not h.startswith("_")
            }
            pooled = {
                h: float(pool_scores(v[:1], mask[:1], model.pool)[0])
                for h, v in heads.items()
                if not h.startswith("_")
            }
            prediction = {"frames": values, "video": pooled}
            for task, value in task_scores(model, prediction).items():
                best[task] = max(best.get(task, -np.inf), value)
            for task in TASKS:
                curve = frame_head(model, prediction, task)
                if curve is not None:
                    frames.setdefault(task, np.zeros(length))[a:b] += curve[: b - a]
            counts[a:b] += 1
        return best, {t: v / np.maximum(counts, 1) for t, v in frames.items()}

    def analyze(self, video, references=(), work=None):
        work = Path(work or tempfile.mkdtemp(prefix="vnav-demo-"))
        cache = work / "cache"
        aligned, sync = self.prepare(video, work, "input")
        self.features(aligned, cache, "input")
        ref_ids = []
        for i, ref in enumerate(references):
            ref_aligned, _ = self.prepare(ref, work, f"ref{i}")
            self.features(ref_aligned, cache, f"ref{i}")
            ref_ids.append(f"ref{i}")
        results, length = {}, 0
        for name, (model, thresholds) in self.models.items():
            # Mỗi model nạp đúng đặc trưng nó cần (mask hợp lệ giống lúc train).
            data = BenchData({**self.cfg, "cache": str(cache)}, [], model.needs)
            item, size = data.load({"sample_id": "input"})
            refs = [data.load({"sample_id": r}, None, REF_FRAMES)[0] for r in ref_ids]
            length = max(length, size)
            video_scores, frame_scores = self.score(model, item, size, refs)
            results[name] = {
                "video": video_scores,
                "frames": frame_scores,
                "thresholds": thresholds,
                "decision": {
                    t: (v >= thresholds[t]) if thresholds.get(t) is not None else None
                    for t, v in video_scores.items()
                },
            }
        return {
            "video": str(aligned),
            "duration_s": length / 25,
            "sync": sync,
            "references": len(ref_ids),
            "models": results,
        }


def training_free_run(folder, dim=1024):
    """Chưa có checkpoint: thư mục run chỉ có speechforensics (không cần train) để thử luồng."""
    from vn_av_df.training.models import build
    from vn_av_df.training.train import save

    target = Path(folder) / "speechforensics_seed0"
    target.mkdir(parents=True, exist_ok=True)
    model = build("speechforensics", dim=dim)
    config = {"name": "speechforensics", "dim": dim, "hidden": 128, "dropout": 0.1}
    save(target / "best.pt", {"config": config, "state": model.state_dict(), "epoch": 0})
    return Path(folder)
