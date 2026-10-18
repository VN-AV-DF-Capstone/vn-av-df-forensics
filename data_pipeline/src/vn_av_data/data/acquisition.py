"""Bước 02 download: tải các video đã chọn của một batch (720p, HTTPS), ghi kết quả vào ledger.

Tên file cố định theo video_id: bị ngắt (Ctrl+C, mất mạng) thì chạy lại sẽ tải tiếp từ file
`.part` của yt-dlp; video đã tải và khớp hash thì bỏ qua.
"""

import re
import time

from vn_av_data.common.runtime import sha
from vn_av_data.data.media import MAX_SHORT_SIDE, MIN_FPS, ffmpeg, probe, quality_issue

# Luật chọn định dạng dùng chung cho plan (xem trước, không tải) và download (tải thật).
# Chỉ luồng HTTPS: luồng HLS tải tiếp có thể mất header mp4.
DOWNLOAD_FORMAT = (
    f"bestvideo[ext=mp4][protocol=https][fps>={MIN_FPS - 0.5}]"
    "+bestaudio[ext=m4a][protocol=https]"
    f"/best[ext=mp4][protocol=https][fps>={MIN_FPS - 0.5}]"
)
# res = cạnh ngắn: video 1080p/4K lấy bản 720 có sẵn của YouTube (video dọc cũng đúng).
FORMAT_SORT = [f"res:{MAX_SHORT_SIDE}"]


def youtube_options(cookies_from_browser=None, force_ipv4=False):
    """Tuỳ chọn mạng chung của yt-dlp; cần Node.js để giải mã link YouTube."""
    import shutil

    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js must be on PATH for the YouTube downloader")
    options = {"socket_timeout": 30, "js_runtimes": {"node": {"path": node}}}
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if force_ipv4:
        options["source_address"] = "0.0.0.0"
    return options


def media_summary(meta):
    return {
        k: meta.get(k)
        for k in (
            "duration_s",
            "width",
            "height",
            "fps",
            "video_codec",
            "video_bitrate_kbps",
            "total_bitrate_kbps",
        )
    }


# Lỗi tạm thời phía YouTube: link tải cấp cho lần đó bị từ chối (403) hoặc mạng chập chờn.
# Gọi lại yt-dlp = xin link mới; thường lần sau tải được.
TRANSIENT = re.compile(
    r"HTTP Error (403|429|5\d\d)|timed out|Connection (reset|aborted)|IncompleteRead"
)
RETRY_WAITS = (5, 20, 60)
BOT_CHECK = re.compile(r"confirm you.re not a bot")
BOT_HINT = (
    "\nYouTube yêu cầu xác minh không phải bot → dừng batch. Chạy lại với phiên đăng nhập:\n"
    "    python 02_download.py --cookies-from-browser firefox\n"
    "(Firefox đã đăng nhập YouTube; Chrome/Edge trên Windows thường không đọc được cookie.)\n"
    "Hoặc chờ vài giờ rồi chạy lại."
)


def clean(exc):
    return re.sub(r"\x1b\[[0-9;]*m", "", str(exc))


def drop_parts(folder, vid):
    """Xoá luồng tải dở (.f*): ghép hỏng hoặc link cũ thì tải lại từ đầu."""
    for part in folder.glob(vid + ".f*"):
        part.unlink(missing_ok=True)


def fetch(downloader, video, folder, wait):
    """Tải một video; lỗi tạm thời (403, 429, 5xx, timeout) thì chờ rồi xin link mới, tối đa 3 lần."""
    for attempt, delay in enumerate((*RETRY_WAITS, None), 1):
        try:
            return downloader(video["url"], folder, video["video_id"])
        except Exception as exc:  # noqa: BLE001 -- chỉ thử lại lỗi tạm thời
            if delay is None or not TRANSIENT.search(clean(exc)):
                raise
            print(f"  lần {attempt}: {clean(exc)[:120]} → thử lại sau {delay} s", flush=True)
            drop_parts(folder, video["video_id"])
            wait(delay)


def download_batch(
    ws,
    ledger,
    batch,
    cookies_from_browser=None,
    force_ipv4=False,
    downloader=None,
    wait=time.sleep,
):
    """`downloader(url, folder, video_id)` thay cho yt-dlp trong test; `wait` thay time.sleep."""
    videos = ledger.rows(
        "SELECT * FROM videos WHERE batch=? AND status='accepted' ORDER BY input_id, rank",
        (batch,),
    )
    if not videos:
        raise ValueError(f"Batch {batch} has no accepted videos; run plan first")
    if ledger.scalar("SELECT location FROM batches WHERE batch=?", (batch,)) not in ("local", None):
        raise ValueError(f"Batch {batch} raw is already archived; nothing to download")
    folder = ws.raw(batch) / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        network = youtube_options(cookies_from_browser, force_ipv4)

        def downloader(url, target, vid):
            import yt_dlp

            options = {
                **network,
                "format": DOWNLOAD_FORMAT,
                "format_sort": FORMAT_SORT,
                "outtmpl": str(target / (vid + ".%(ext)s")),
                "noplaylist": True,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg(),
                "continuedl": True,
                "overwrites": False,
                "quiet": True,
                "no_warnings": True,
                "noprogress": False,
            }
            with yt_dlp.YoutubeDL(options) as client:
                client.extract_info(url, download=True)
            return target / (vid + ".mp4")

    done = failed = rejected = 0
    for number, v in enumerate(videos, 1):
        head = f"[{number}/{len(videos)}] {v['video_id']}"
        path = folder / (v["video_id"] + ".mp4")
        if v["dl_status"] == "done" and path.is_file() and sha(path) == v["sha256"]:
            done += 1
            continue
        if v["dl_status"] == "rejected":
            rejected += 1
            continue
        print(f"{head} {v['title'][:60]} ({(v['duration_s'] or 0) / 60:.1f} phút)", flush=True)
        try:
            path = fetch(downloader, v, folder, wait)
            meta = probe(path)
            issue = quality_issue(meta)
            values = {
                "file": path.relative_to(ws.root).as_posix(),
                "sha256": sha(path),
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "probe": media_summary(meta),
                "dl_status": "rejected" if issue else "done",
                "dl_error": issue or "",
                "raw_location": "local",
            }
            ledger.update("videos", "video_id", v["video_id"], values)
            if issue:
                print(f"  loại sau khi tải: {issue}", flush=True)
                rejected += 1
            else:
                done += 1
        except KeyboardInterrupt:
            print("\nDừng. Chạy lại cùng batch để tải tiếp từ chỗ dừng.", flush=True)
            raise
        except Exception as exc:  # noqa: BLE001 -- ghi lỗi, tải tiếp video khác
            message = clean(exc)
            drop_parts(folder, v["video_id"])
            ledger.update(
                "videos", "video_id", v["video_id"], {"dl_status": "failed", "dl_error": message}
            )
            print(f"  lỗi: {message[:300]}", flush=True)
            failed += 1
            if BOT_CHECK.search(message):
                # IP bị nghi là bot: video sau cũng lỗi, gọi tiếp chỉ làm bị chặn lâu hơn.
                print(BOT_HINT, flush=True)
                break
    ledger.log("download", f"batch {batch}: done {done}, failed {failed}, rejected {rejected}")
    summary = {"batch": batch, "downloaded": done, "failed": failed, "rejected": rejected}
    size = ledger.scalar(
        "SELECT SUM(size_mb) FROM videos WHERE batch=? AND dl_status='done'", (batch,)
    )
    summary["size_gb"] = round((size or 0) / 1000, 2)
    if failed:
        raise RuntimeError(f"{failed} downloads failed in batch {batch}; run again to retry")
    return summary
