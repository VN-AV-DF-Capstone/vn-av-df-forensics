import pytest

from vn_av_data.data import plan as plan_module
from vn_av_data.data.plan import choose_single, plan_sources, read_inputs, spread
from vn_av_data.data.youtube import check, classify, describe, estimate_mb


def info(
    vid,
    date="20260101",
    duration=600,
    fps=25,
    height=720,
    dubbed=False,
    lang="vi",
    size=None,
    vertical=False,
):
    width = height * 16 // 9
    video = {
        "format_id": "136",
        "vcodec": "avc1",
        "acodec": "none",
        "width": height if vertical else width,
        "height": width if vertical else height,
        "fps": fps,
        "vbr": 800,
        "filesize": size,
    }
    audio = {"format_id": "140", "vcodec": "none", "acodec": "mp4a", "abr": 128, "language": lang}
    formats = [video, audio]
    if dubbed:
        audio = {**audio, "language": "en", "format_note": "English (dubbed)"}
        formats.append(audio)
    return {
        "title": vid,
        "channel": "Kênh",
        "channel_id": "UC1",
        "upload_date": date,
        "duration": duration,
        "formats": formats,
        "requested_formats": [video, audio],
    }


def ids(n, prefix="v"):
    return [f"{prefix}{i:010d}"[:11] for i in range(n)]


def test_classify_urls():
    assert classify("https://youtu.be/abcdefghijk") == (
        "video",
        "https://www.youtube.com/watch?v=abcdefghijk",
    )
    assert classify("https://www.youtube.com/@ten-kenh/featured")[1].endswith("/@ten-kenh/videos")
    assert classify("https://www.youtube.com/channel/UCabc/videos")[0] == "channel"
    kind, url = classify("https://www.youtube.com/playlist?list=PLx")
    assert kind == "playlist" and url.endswith("list=PLx")
    with pytest.raises(ValueError):
        classify("https://example.com/watch?v=abcdefghijk")


def test_read_inputs_creates_template_and_parses_speakers(tmp_path):
    path = tmp_path / "urls.csv"
    with pytest.raises(FileNotFoundError):
        read_inputs(path)
    path.write_text(
        "url,speakers,note\n# ghi chú\nhttps://youtu.be/abcdefghijk,1,\nhttps://youtu.be/bbcdefghijk,N,x\n",
        encoding="utf-8",
    )
    rows = read_inputs(path)
    assert [r["speakers"] for r in rows] == ["1", "n"]
    path.write_text("url,speakers\nhttps://youtu.be/abcdefghijk,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_inputs(path)


def test_check_rejects_quality_and_dubbed_tracks():
    assert check(describe(info("a"))) is None
    assert "fps" in check(describe(info("a", fps=24)))
    assert "short side" in check(describe(info("a", height=480)))
    assert "vertical" in check(describe(info("a", vertical=True)))
    assert "dubbed" in check(describe(info("a", dubbed=True)))
    assert "language" in check(describe(info("a", lang="en")))
    assert estimate_mb(describe(info("a", size=200e6))) == pytest.approx(
        200 + 128 * 600 / 8000, abs=0.1
    )


def test_spread_avoids_very_long_videos_and_spans_dates():
    rows = [
        {"video_id": str(i), "upload_date": f"2026{i + 1:02d}01", "duration_s": d, "rank": i}
        for i, d in enumerate([600, 700, 9000, 800, 900, 1000])
    ]
    chosen = spread(rows, 3, 5400)
    assert [c["video_id"] for c in chosen] == ["0", "3", "5"]


def test_choose_single_by_total_duration():
    rules = {"target_minutes": 20, "min_videos": 3, "max_videos": 10, "prefer_max_s": 5400}

    def rows(durations):
        return [
            {
                "video_id": str(i),
                "upload_date": f"2026{i % 12 + 1:02d}01",
                "duration_s": d,
                "rank": i,
            }
            for i, d in enumerate(durations)
        ]

    assert len(choose_single(rows([1800] * 8), rules)) == 3  # video dài: vẫn ≥ 3 buổi quay
    assert len(choose_single(rows([180] * 12), rules)) == 7  # 3 phút/video: 7 video ≥ 20 phút
    assert len(choose_single(rows([60] * 15), rules)) == 10  # video ngắn: dừng ở 10
    assert len(choose_single(rows([900, 600]), rules)) == 2  # kênh chỉ có 2 video


def setup_inputs(ws, text):
    ws.inputs.parent.mkdir(parents=True, exist_ok=True)
    ws.inputs.write_text("url,speakers\n" + text, encoding="utf-8")


def test_plan_limits_batches_and_duplicates(ws, ledger, cfg):
    single = ids(10, "s")
    multi = ids(20, "m")
    playlist = ids(30, "p")
    dates = {vid: f"2026{(i % 12) + 1:02d}{(i % 27) + 1:02d}" for i, vid in enumerate(playlist)}
    entries = {
        "https://www.youtube.com/@one/videos": [{"video_id": v} for v in single],
        "https://www.youtube.com/@many/videos": [{"video_id": v} for v in multi],
        "https://www.youtube.com/playlist?list=PL1": [{"video_id": v} for v in playlist]
        + [{"video_id": single[0]}],
    }
    setup_inputs(
        ws,
        "https://www.youtube.com/@one,1\nhttps://www.youtube.com/@many,n\n"
        "https://www.youtube.com/playlist?list=PL1,n\n",
    )

    def metadata(url):
        vid = url.rsplit("=", 1)[1]
        if vid == multi[1]:
            return info(vid, fps=24)  # bị loại, phải lấy video kế tiếp
        return info(vid, date=dates.get(vid, "20260105"), duration=3600)

    cfg["plan"]["batch_max_hours"] = 20
    summary = plan_sources(
        ws,
        ledger,
        cfg,
        metadata_loader=metadata,
        flat_loader=lambda url, limit: entries[url][:limit],
    )
    assert not summary["errors"]
    accepted = ledger.rows("SELECT * FROM videos WHERE status='accepted'")
    by_input = {}
    for v in accepted:
        by_input.setdefault(v["input_id"], []).append(v)
    assert len(by_input[1]) == 3  # một người: 3 video × 1 h ≥ 20 phút
    assert len(by_input[2]) == 15 and multi[1] not in {v["video_id"] for v in by_input[2]}
    newest = sorted(playlist, key=lambda v: dates[v], reverse=True)[:15]
    assert {v["video_id"] for v in by_input[3]} == set(newest)
    assert ledger.scalar("SELECT COUNT(*) FROM duplicates") == 1
    # 1 giờ / video: nguồn 1 (3 h) + nguồn 2 (15 h) = 18 h ≤ 20 → chung batch 1; nguồn 3 (15 h) mới.
    hours = {b["batch"]: b["hours"] for b in summary["batches"]}
    assert hours == {1: 18.0, 2: 15.0}
    assert (ws.raw(1) / "videos.csv").is_file()
    # Plan lại: dòng đã plan không đổi; thêm dòng mới được nối vào batch 2 (chưa tải).
    setup_inputs(
        ws,
        "https://www.youtube.com/@one,1\nhttps://www.youtube.com/@many,n\n"
        "https://www.youtube.com/playlist?list=PL1,n\nhttps://youtu.be/zzzzzzzzzzz,1\n",
    )
    summary = plan_sources(
        ws, ledger, cfg, metadata_loader=metadata, flat_loader=lambda url, limit: entries[url]
    )
    assert summary["inputs_processed"] == 1
    assert ledger.scalar("SELECT batch FROM videos WHERE video_id='zzzzzzzzzzz'") == 2


def test_plan_freezes_started_batch_and_retries_errors(ws, ledger, cfg):
    setup_inputs(ws, "https://youtu.be/aaaaaaaaaaa,1\n")
    plan_sources(ws, ledger, cfg, metadata_loader=lambda u: info("a"), flat_loader=None)
    ledger.update("videos", "video_id", "aaaaaaaaaaa", {"dl_status": "done"})
    setup_inputs(ws, "https://youtu.be/aaaaaaaaaaa,1\nhttps://youtu.be/bbbbbbbbbbb,1\n")
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("network timeout")
        return info("b")

    summary = plan_sources(ws, ledger, cfg, metadata_loader=flaky, flat_loader=None)
    assert summary["errors"]
    summary = plan_sources(ws, ledger, cfg, metadata_loader=flaky, flat_loader=None)
    assert not summary["errors"]
    assert ledger.scalar("SELECT batch FROM videos WHERE video_id='bbbbbbbbbbb'") == 2
    with pytest.raises(ValueError):
        setup_inputs(ws, "https://youtu.be/aaaaaaaaaaa,n\n")
        plan_module.register_inputs(ledger, read_inputs(ws.inputs))


def test_download_retries_transient_403_then_gives_up(ws, ledger, cfg, tmp_path):
    from conftest import make_video

    from vn_av_data.data.acquisition import download_batch

    setup_inputs(ws, "https://youtu.be/aaaaaaaaaaa,1\nhttps://youtu.be/bbbbbbbbbbb,1\n")
    plan_sources(ws, ledger, cfg, metadata_loader=lambda u: info("a"), flat_loader=None)
    source = make_video(tmp_path / "src.mp4", 2)
    calls, waits = [], []

    def downloader(url, folder, vid):
        calls.append(vid)
        if vid == "aaaaaaaaaaa" and calls.count(vid) < 3:
            raise RuntimeError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        if vid == "bbbbbbbbbbb":
            raise RuntimeError("ERROR: HTTP Error 403: Forbidden")
        target = folder / f"{vid}.mp4"
        target.write_bytes(source.read_bytes())
        return target

    with pytest.raises(RuntimeError, match="1 downloads failed"):
        download_batch(ws, ledger, 1, downloader=downloader, wait=waits.append)
    status = {r["video_id"]: r["dl_status"] for r in ledger.rows("SELECT * FROM videos")}
    assert status == {"aaaaaaaaaaa": "done", "bbbbbbbbbbb": "failed"}
    assert calls.count("aaaaaaaaaaa") == 3  # 2 lần 403 rồi tải được
    assert calls.count("bbbbbbbbbbb") == 4  # 1 lần + 3 lần thử lại rồi mới ghi failed
    assert waits == [5, 20, 5, 20, 60]


def test_download_stops_batch_on_bot_check(ws, ledger, cfg):
    from vn_av_data.data.acquisition import download_batch

    setup_inputs(ws, "https://youtu.be/aaaaaaaaaaa,1\nhttps://youtu.be/bbbbbbbbbbb,1\n")
    plan_sources(ws, ledger, cfg, metadata_loader=lambda u: info("a"), flat_loader=None)
    calls = []

    def downloader(url, folder, vid):
        calls.append(vid)
        raise RuntimeError("ERROR: [youtube] x: Sign in to confirm you’re not a bot.")

    with pytest.raises(RuntimeError, match="1 downloads failed"):
        download_batch(ws, ledger, 1, downloader=downloader, wait=lambda s: None)
    assert calls == ["aaaaaaaaaaa"]  # không thử lại, không gọi video sau
