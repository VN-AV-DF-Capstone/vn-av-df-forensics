"""Demo v003: tạo fake từ một video real (mọi ô) và phân tích theo cửa sổ (TTS / ASR / generator /
encoder / SyncNet giả lập, CPU)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_pipeline import FakeTranscriber, fake_call, fake_synthesize, make_clip, options

from vn_av_df.demo.detect import Detector, windows
from vn_av_df.demo.fake import generate_one
from vn_av_df.generation.voice.engines import Engine


def demo_cfg(tmp):
    return {"generation": options(), "device": "cpu", "root": str(tmp)}


@pytest.mark.parametrize(
    "cell,kind",
    [
        ("voice", "full"),
        ("voice", "partial"),
        ("source", "partial"),
        ("donor", "full"),
        ("both", "full"),
        ("both", "partial"),
    ],
)
def test_generate_one_every_cell(tmp_path, cell, kind):
    from vn_av_df.data.media import decode

    cfg = demo_cfg(tmp_path)
    real = make_clip(tmp_path / "real.mp4", seconds=6, freq=300)
    donor = make_clip(tmp_path / "donor.mp4", seconds=8, freq=500)
    path, label = generate_one(
        cfg,
        real,
        cell,
        kind=kind,
        generator="g_a",
        voice="f5_vi",
        donor=donor if cell == "donor" else None,
        output=tmp_path / "out" / "fake.mp4",
        engine=Engine(cfg, call=fake_call),
        transcriber=FakeTranscriber(),
        synthesize=fake_synthesize,
    )
    media = decode(path, max_side=320, sample_rate=48000)
    assert len(media["frames"]) == 150 and label["label"] == 1
    assert path.with_suffix(".json").is_file()
    if kind == "partial":
        assert label["fake_intervals"] and label["fake_intervals"] != [[0.0, 6.0]]
    if cell == "voice":
        assert label["audio_fake_intervals"] and not label["visual_fake_intervals"]
        assert label["av_mismatch_intervals"] == label["audio_fake_intervals"]
    if cell == "both":
        assert label["visual_fake_intervals"] and label["audio_fake_intervals"]
        assert label["edit"]["new"]
    if cell == "donor":
        assert label["visual_fake_intervals"] == [[0.0, 6.0]] and not label["av_mismatch_intervals"]


def test_user_span_and_text(tmp_path):
    cfg = demo_cfg(tmp_path)
    real = make_clip(tmp_path / "real.mp4", seconds=6, freq=300)
    _, label = generate_one(
        cfg,
        real,
        "voice",
        kind="partial",
        span=(2.0, 3.0),
        new_text="ba nghìn",
        output=tmp_path / "o.mp4",
        engine=Engine(cfg, call=fake_call),
        transcriber=FakeTranscriber(),
    )
    assert label["audio_fake_intervals"] == [[2.0, 3.0]] and label["edit"]["new"] == "ba nghìn"
    with pytest.raises(ValueError, match="không hợp lệ"):
        generate_one(
            cfg,
            real,
            "source",
            kind="partial",
            span=(0.1, 0.4),
            output=tmp_path / "p.mp4",
            synthesize=fake_synthesize,
        )


def test_windows_cover_every_frame():
    for length in (60, 175, 176, 400, 1500):
        spans = windows(length)
        covered = np.zeros(length, int)
        for a, b in spans:
            covered[a:b] += 1
            assert b - a <= 175
        assert covered.min() >= 1 and spans[-1][1] == length


class FakeAligner:
    def align(self, video, output):
        import shutil

        shutil.copy2(video, output)
        return Path(output), {"status": "ok", "audio_lag_frames": 0, "audio_lag_ms": 0, "conf": 5.0}


def fake_features(dim=32):
    """features(): ghi cache giả cùng bố cục với lúc train (độ dài = số frame của video)."""

    def features(self, video, cache, sample_id):
        from test_pipeline import write_fake_caches

        from vn_av_df.data.media import probe

        for name in ("avhubert", "xlsr", "mouth"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        row = {"sample_id": sample_id, "duration_s": probe(video)["duration_s"]}
        write_fake_caches({"cache": str(cache)}, [row], dim)

    return features


def test_detector_scores_long_video_in_windows(tmp_path, monkeypatch):
    from vn_av_df.common.runtime import write_json
    from vn_av_df.training.models import build
    from vn_av_df.training.train import save

    runs = tmp_path / "runs"
    for name in ("avcv", "mtn", "trihead", "speechforensics"):
        model = build(name, dim=32, hidden=16)
        folder = runs / f"{name}_seed42"
        folder.mkdir(parents=True)
        save(
            folder / "best.pt",
            {
                "config": {"name": name, "dim": 32, "hidden": 16, "dropout": 0.1},
                "state": model.state_dict(),
                "epoch": 1,
            },
        )
    write_json(runs / "avcv_seed42/metrics.json", {"tasks": {"mismatch": {"threshold": 0.0}}})
    monkeypatch.setattr(Detector, "features", fake_features())
    video = make_clip(tmp_path / "long.mp4", seconds=12)
    ref = make_clip(tmp_path / "ref.mp4", seconds=4, freq=700)
    detector = Detector(
        {"training": {"audio_lag_aug": [0, 0]}}, runs, device="cpu", aligner=FakeAligner()
    )
    with torch.no_grad():
        result = detector.analyze(video, [ref], tmp_path / "work")
    assert result["references"] == 1 and set(result["models"]) == {
        "avcv",
        "mtn",
        "trihead",
        "speechforensics",
    }
    avcv = result["models"]["avcv"]
    assert set(avcv["video"]) >= {"mismatch", "visual"}
    assert len(avcv["frames"]["mismatch"]) == 300  # 12 s × 25 fps, ghép từ các cửa sổ 7 s
    assert avcv["decision"]["mismatch"] in (True, False) and avcv["decision"]["visual"] is None
    from vn_av_df.demo.app import summary_rows, timeline_figure

    assert len(summary_rows(result)) >= 4
    figure = timeline_figure(result, {"fake_intervals": [[2.0, 4.0]]})
    assert figure.axes
