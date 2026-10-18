"""Tiện ích test: workspace tạm, cấu hình, video tổng hợp, mặt / embedding giả lập."""

import copy
from pathlib import Path

import numpy as np
import pytest

from vn_av_data.common.runtime import load_config, run, sha
from vn_av_data.data.media import ffmpeg
from vn_av_data.ledger import Ledger
from vn_av_data.workspace import Workspace

CONFIG = Path(__file__).resolve().parents[1] / "configs/data.yaml"


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config(CONFIG))


@pytest.fixture
def ws(tmp_path):
    return Workspace.at(tmp_path / "data", tmp_path / "exports", "vn-test")


@pytest.fixture
def ledger(ws):
    led = Ledger(ws.ledger)
    yield led
    led.close()


def make_video(path, seconds, size="1280x720", fps=25, delay=0.0, freq=440):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-f", "lavfi"),
            *("-i", f"testsrc2=s={size}:r={fps}:d={seconds}"),
            *(["-itsoffset", str(delay)] if delay else []),
            *("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={seconds}"),
            *("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
            *("-c:a", "aac", path),
        ]
    )
    return path


def unit(seed, dim=128):
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeTracker:
    """Một mặt hợp lệ ở mọi frame; `jumps` = các frame (đếm từ lúc reset) mà mặt nhảy chỗ."""

    def __init__(self, jumps=()):
        self.previous = None
        self.jumps = set(jumps)
        self.count = 0

    def inspect(self, image):
        if self.previous is None:
            self.count = 0
        self.count += 1
        shift = 60.0 if any(self.count > j for j in self.jumps) else 0.0
        x = 200.0 + shift + 0.2 * self.count
        lm = np.array(
            [[x + 30, 100], [x + 70, 100], [x + 50, 125], [x + 35, 150], [x + 65, 150]],
            np.float32,
        )
        self.previous = np.array([x, 70, 100, 110])
        face = np.r_[[x, 70, 100, 110], lm.reshape(-1), [0.99]].astype(np.float32)
        return {"valid": True, "box": [x, 70, 100, 110], "landmarks": lm, "face": face}


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def align(self, image, face):
        return np.full((112, 112, 3), 128, np.uint8)

    def embed(self, aligned):
        return self.vector


def add_input(ledger, url, speakers="1", kind="channel"):
    ledger.insert("inputs", {"url": url, "kind": kind, "speakers": speakers, "status": "planned"})
    return ledger.scalar("SELECT input_id FROM inputs WHERE url=?", (url,))


def add_raw_video(ws, ledger, video_id, input_id, batch, seconds=20, channel="ch1", rank=0):
    path = make_video(ws.raw(batch) / "videos" / f"{video_id}.mp4", seconds)
    if not ledger.scalar("SELECT COUNT(*) FROM batches WHERE batch=?", (batch,)):
        ledger.insert("batches", {"batch": batch, "created_at": "now"})
    ledger.insert(
        "videos",
        {
            "video_id": video_id,
            "input_id": input_id,
            "rank": rank,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": video_id,
            "channel": channel,
            "channel_id": channel,
            "duration_s": seconds,
            "status": "accepted",
            "batch": batch,
            "dl_status": "done",
            "file": path.relative_to(ws.root).as_posix(),
            "sha256": sha(path),
            "probe": {"fps": 25.0, "width": 1280, "height": 720},
            "raw_location": "local",
        },
    )
    return path
