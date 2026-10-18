from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder, FakeTracker, add_input, add_raw_video, make_video, unit

from vn_av_data.data import cut as cut_module
from vn_av_data.data.cut import (
    Inspector,
    cut_batch,
    cut_media,
    identity_frames,
    media_origin,
    plan_clips,
    single_target,
    split_quota,
    spread_order,
)
from vn_av_data.data.media import decode, probe
from vn_av_data.data.vad import audio_blocks, speech_regions


@pytest.fixture
def cut_cfg(cfg, tmp_path):
    for key in ("face_model", "vad_model", "sface_model", "syncnet_model"):
        path = tmp_path / f"{key}.onnx"
        path.write_bytes(b"model")
        cfg["cut"][key] = str(path)
    return cfg


def test_cut_media_normalizes_to_720p_25fps_and_keeps_av_offset(tmp_path):
    source = make_video(tmp_path / "big.mp4", 3, size="1920x1080", fps=50, delay=0.8)
    origin, _ = media_origin(source)
    out = tmp_path / "clip.mp4"
    cut_media(source, out, 0.4, 2.4, origin)
    info = probe(out)
    assert (info["width"], info["height"], info["fps"]) == (1280, 720, 25.0)
    media = decode(out, sample_rate=48000)
    power = np.abs(media["pcm"])
    assert power[: int(0.3 * 48000)].max() < 1e-3  # tiếng bắt đầu ở 0,8 s nguồn = 0,4 s clip
    assert power[int(0.5 * 48000) : int(0.6 * 48000)].max() > 0.05
    with pytest.raises(FileExistsError):
        cut_media(source, out, 0.4, 2.4, origin)


def test_vad_blocks_keep_timestamp_gaps(tmp_path):
    source = make_video(tmp_path / "d.mp4", 3, size="320x240", delay=0.8)
    origin, _ = media_origin(source)
    pcm = np.concatenate(list(audio_blocks(source, origin)))
    assert np.abs(pcm[: int(0.7 * 16000)]).max() < 1e-3
    assert speech_regions([0] * 20 + [1] * 40 + [0] * 20, 2.56)[0][0] == pytest.approx(0.49)


def test_windows_quota_and_spread():
    cfg = {"min_seconds": 4, "max_seconds": 7, "overlap_seconds": 0}
    windows = plan_clips([[0, 30]], [12], 30, cfg)
    assert all(4 - 1e-6 <= b - a <= 7 + 1e-6 for a, b in windows)
    assert not any(a < 12 < b for a, b in windows)
    rules = {"min_clips": 30, "max_clips": 50, "low_speech_minutes": 5, "full_speech_minutes": 20}
    assert single_target(60, rules) == 30
    assert single_target(12.5 * 60, rules) == 40
    assert single_target(3600, rules) == 50
    assert split_quota(10, [1, 1, 2]) == [3, 2, 5] or sum(split_quota(10, [1, 1, 2])) == 10
    queues = spread_order([(i, i + 5) for i in range(0, 100, 5)], 4, 100)
    assert len(queues) == 4 and all(queues)
    assert queues[0][0] == (10, 15)  # gần tâm khúc đầu (12,5 s) nhất
    points = identity_frames(150, seed=3)
    assert points[0] == 0 and points[-1] == 149 and 74 in points and len(points) == 5
    assert points == identity_frames(150, seed=3)


def test_inspector_rejects_jump_cut_and_identity_change(tmp_path, cut_cfg):
    source = make_video(tmp_path / "s.mp4", 8)
    origin, _ = media_origin(source)
    cfg = cut_cfg["cut"]
    ok = Inspector(cfg, FakeTracker(), FakeEmbedder(unit(1))).inspect(source, origin, 0, 6, 1, True)
    assert ok["ok"] and ok["consistency"] == pytest.approx(1.0) and len(ok["aligned"]) == 5
    jump = Inspector(cfg, FakeTracker(jumps=[70]), FakeEmbedder(unit(1)))
    result = jump.inspect(source, origin, 0, 6, 1, True)
    assert not result["ok"] and result["reason"] == "jump_cut"

    class Switching(FakeEmbedder):
        calls = 0

        def embed(self, aligned):
            Switching.calls += 1
            return unit(1) if Switching.calls < 2 else unit(2)

    changed = Inspector(cfg, FakeTracker(), Switching(None)).inspect(source, origin, 0, 6, 1, True)
    assert not changed["ok"] and changed["reason"] == "identity_change"
    # Nguồn một người không chặn theo 5 điểm.
    Switching.calls = 0
    assert Inspector(cfg, FakeTracker(), Switching(None)).inspect(source, origin, 0, 6, 1, False)[
        "ok"
    ]


class GappyTracker(FakeTracker):
    """Như FakeTracker nhưng hụt mặt ở các frame `missing` (đếm từ 1 sau lần reset)."""

    def __init__(self, missing):
        super().__init__()
        self.missing = set(missing)

    def inspect(self, image):
        result = super().inspect(image)
        if self.count in self.missing:
            return {"valid": False, "reason": "no_face"}
        return result


@pytest.mark.parametrize(
    "missing,ok",
    [
        ((), True),
        ((40,), True),  # hụt 1 frame
        ((40, 41, 90), True),  # 2 frame liền + 1 frame lẻ = 3
        ((40, 41, 42), False),  # 3 frame liền nhau
        ((20, 60, 90, 120), False),  # 4 frame
    ],
)
def test_inspector_tolerates_short_face_gaps(tmp_path, cut_cfg, missing, ok):
    source = make_video(tmp_path / "s.mp4", 8)
    origin, _ = media_origin(source)
    inspector = Inspector(cut_cfg["cut"], GappyTracker(missing), FakeEmbedder(unit(1)))
    result = inspector.inspect(source, origin, 0, 6, 1, True)
    assert result["ok"] is ok
    if ok:
        assert result["gaps"] == len(missing)
    else:
        assert result["reason"] == "face_frame"


def test_fill_gaps_interpolates_between_good_frames():
    from vn_av_data.data.cut import fill_gaps

    items = [None, np.full((5, 2), 1.0), None, None, np.full((5, 2), 4.0), None]
    out = fill_gaps(items, (5, 2))
    assert out[:, 0, 0].tolist() == [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]


def full_speech(path, origin, duration):
    return [[0.0, duration]]


class FakeMeter:
    """Đường cong khoảng cách giả: video trong `bad` cho độ lệch lung tung, còn lại lệch `lag`."""

    def __init__(self, lag=2, bad=()):
        self.lag, self.bad, self.calls = lag, set(bad), 0

    def measure(self, path):
        from vn_av_data.features.syncnet import VSHIFT

        self.calls += 1
        lag = (self.calls * 7) % 11 - 5 if Path(path).parent.name in self.bad else self.lag
        curve = np.full(2 * VSHIFT + 1, 14.0)
        curve[VSHIFT + lag] = 8.0
        return curve


def test_sync_summary_and_measure_marks_unreliable_video(ws):
    from vn_av_data.data.cut import measure_sync
    from vn_av_data.features.syncnet import VSHIFT, read_curve, summarize

    rules = {"clips_per_video": 20, "min_clips": 3, "min_conf": 3.0, "min_agree": 0.8}
    curve = np.full(2 * VSHIFT + 1, 14.0)
    curve[VSHIFT + 2] = 8.0
    assert read_curve(curve) == (2, 6.0)
    assert summarize([curve] * 5, rules)["status"] == "ok"
    assert summarize([curve] * 2, rules)["status"] == "few"  # ít clip: sửa nhưng nhắc xem
    assert summarize([], rules)["status"] == "no_clips"
    clips = [
        {"start_s": float(i), "file": f"pool/clips/v1/c{i}.mp4", "flags": "", "decision": "pending"}
        for i in range(30)
    ]
    meter = FakeMeter(lag=2)
    result = measure_sync(meter, ws, clips, rules)
    assert result["audio_lag_frames"] == 2 and result["status"] == "ok" and meter.calls == 20
    assert all(c["decision"] == "pending" for c in clips)
    result = measure_sync(FakeMeter(bad={"v1"}), ws, clips, rules)
    assert result["status"] == "unreliable"
    assert all(c["decision"] == "reject" and "sync_unreliable" in c["flags"] for c in clips)


def test_cut_batch_single_source_spreads_and_resumes(ws, ledger, cut_cfg, monkeypatch):
    one = add_input(ledger, "https://www.youtube.com/@one/videos", "1")
    add_raw_video(ws, ledger, "aaaaaaaaaaa", one, 1, seconds=30, rank=0)
    add_raw_video(ws, ledger, "bbbbbbbbbbb", one, 1, seconds=30, rank=1)
    many = add_input(ledger, "https://www.youtube.com/watch?v=ccccccccccc", "n", "video")
    add_raw_video(ws, ledger, "ccccccccccc", many, 1, seconds=20, channel="ch2")
    cut_cfg["cut"]["single"].update(min_clips=4, max_clips=4)
    inspector = Inspector(cut_cfg["cut"], FakeTracker(), FakeEmbedder(unit(1)))
    meter = FakeMeter(lag=2, bad={"ccccccccccc"})
    original = cut_module.Cutter.cut_video
    calls = []

    def interrupt_second(self, video, speakers, quota):
        calls.append(video["video_id"])
        if video["video_id"] == "bbbbbbbbbbb" and calls.count("bbbbbbbbbbb") == 1:
            (ws.clips / "bbbbbbbbbbb").mkdir(parents=True, exist_ok=True)
            (ws.clips / "bbbbbbbbbbb/partial.mp4").write_bytes(b"x")
            raise KeyboardInterrupt
        return original(self, video, speakers, quota)

    monkeypatch.setattr(cut_module.Cutter, "cut_video", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        cut_batch(
            ws, ledger, 1, cut_cfg, inspector=inspector, speech_detector=full_speech, meter=meter
        )
    assert ledger.scalar("SELECT cut_status FROM videos WHERE video_id='aaaaaaaaaaa'") == "done"
    assert not (ws.clips / "bbbbbbbbbbb").exists()
    result = cut_batch(
        ws, ledger, 1, cut_cfg, inspector=inspector, speech_detector=full_speech, meter=meter
    )
    assert result["skipped"] == 1 and not result.get("failed")
    single = ledger.rows("SELECT * FROM clips WHERE input_id=? ORDER BY video_id, start_s", (one,))
    assert len(single) == 4  # chỉ tiêu 4 clip chia đều 2 video
    assert {c["video_id"] for c in single} == {"aaaaaaaaaaa", "bbbbbbbbbbb"}
    starts = [c["start_s"] for c in single if c["video_id"] == "aaaaaaaaaaa"]
    assert len(starts) == 2 and abs(starts[1] - starts[0]) >= 7  # rải ra, không liền nhau
    multi = ledger.rows("SELECT * FROM clips WHERE input_id=?", (many,))
    assert len(multi) == 3  # nguồn nhiều người: cắt hết (20 s → 7 + 7 + 6)
    for clip in single + multi:
        assert (ws.root / clip["file"]).is_file()
        assert (ws.faces / clip["video_id"] / f"{clip['clip_id']}.jpg").is_file()
        assert clip["face"] is not None
        assert 4 <= clip["duration_s"] <= 7
    # Độ lệch đo theo video: nguồn một người tin được → giữ; video "lời dẫn đè hình" → tự loại.
    info = ledger.row("SELECT cut_info FROM videos WHERE video_id='aaaaaaaaaaa'")["cut_info"]
    assert info["sync"]["audio_lag_frames"] == 2 and info["sync"]["status"] == "few"  # 2 clip
    assert all(c["decision"] == "pending" and "sync_few" in c["flags"] for c in single)
    assert all(c["decision"] == "reject" and "sync_unreliable" in c["flags"] for c in multi)
    # Đổi luật cắt sau khi đã có clip: bị chặn.
    cut_cfg["cut"]["max_seconds"] = 8.0
    with pytest.raises(ValueError, match="changed"):
        cut_batch(
            ws, ledger, 1, cut_cfg, inspector=inspector, speech_detector=full_speech, meter=meter
        )


@pytest.mark.parametrize("lag", [5, -5])
def test_export_shifts_audio_and_keeps_video(tmp_path, lag):
    from vn_av_data.data.export import place_clip

    source = make_video(tmp_path / "s.mp4", 3, delay=1.0)  # tiếng bắt đầu ở giây 1,0
    target = tmp_path / "t.mp4"
    place_clip(source, target, lag, probe(source)["duration_s"])

    def onset(path):
        media = decode(path, max_duration=10, max_side=64)
        return np.argmax(np.abs(media["pcm"]) > 0.05) / 16000, len(media["frames"])

    (before, frames), (after, frames_after) = onset(source), onset(target)
    assert frames_after == frames
    assert after == pytest.approx(before - lag * 0.04, abs=0.01)
