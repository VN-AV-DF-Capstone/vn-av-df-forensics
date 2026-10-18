# Adapted from the user-authorized VN-AV-DF-Capstone collect workflow.
"""YouTube: phân loại URL (video / playlist / kênh), bung danh sách, hỏi metadata, luật loại.

Luật loại chỉ dùng metadata đã lưu trong ledger: đổi ngưỡng rồi plan lại không cần hỏi mạng.
"""

import copy
import re
from urllib.parse import parse_qs, urlparse

from vn_av_data.data.media import (
    MAX_SHORT_SIDE,
    MAX_SOURCE_HOURS,
    MIN_FPS,
    MIN_SHORT_SIDE,
    MIN_SOURCE_SECONDS,
)

HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
# Lỗi YouTube cho biết video không xem được (loại hẳn); lỗi khác coi là lỗi mạng (hỏi lại).
UNAVAILABLE = re.compile(
    r"private|unavailable|removed|terminated|members|age|copyright|not available|deleted",
    re.IGNORECASE,
)
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


def video_id(value):
    value = value.strip()
    if VIDEO_ID.fullmatch(value):
        return value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Expected a YouTube URL or video ID: {value}")
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.strip("/")
    elif host in HOSTS:
        parts = parsed.path.strip("/").split("/")
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
            candidate = parts[1]
    else:
        raise ValueError(f"Expected a YouTube URL: {value}")
    if not VIDEO_ID.fullmatch(candidate):
        raise ValueError(f"No valid video ID: {value}")
    return candidate


def classify(url):
    """('video', url chuẩn) | ('playlist', url) | ('channel', url tab /videos)."""
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in HOSTS | {"youtu.be", "www.youtu.be"}:
        raise ValueError(f"Not a YouTube URL: {url}")
    parts = [p for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)
    if parts[:1] == ["playlist"] and query.get("list"):
        return "playlist", f"https://www.youtube.com/playlist?list={query['list'][0]}"
    if host.endswith("youtu.be") or parts[:1] in (["watch"], ["shorts"], ["live"], ["embed"]):
        return "video", "https://www.youtube.com/watch?v=" + video_id(url)
    if parts and (parts[0].startswith("@") or parts[0] in ("channel", "c", "user")):
        base = parts[:1] if parts[0].startswith("@") else parts[:2]
        if len(base) < (1 if parts[0].startswith("@") else 2):
            raise ValueError(f"Incomplete channel URL: {url}")
        # Tab /videos: video thường, mới nhất trước (không gồm shorts / live).
        return "channel", "https://www.youtube.com/" + "/".join(base) + "/videos"
    raise ValueError(f"Unsupported YouTube URL (video, playlist or channel): {url}")


def flat_entries(url, options, limit):
    """Bung playlist/kênh không hỏi từng video; giữ đúng thứ tự YouTube trả về."""
    import yt_dlp

    quiet = {
        **options,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(quiet) as client:
        info = client.extract_info(url, download=False)
    entries = []
    for item in (info or {}).get("entries") or []:
        if item and item.get("id") and VIDEO_ID.fullmatch(item["id"]):
            entries.append({"video_id": item["id"], "title": item.get("title") or ""})
    return entries


def youtube_metadata(url, options):
    """Một lần hỏi YouTube: metadata thô + định dạng bước tải sẽ chọn (chọn offline, không tải)."""
    import yt_dlp

    from vn_av_data.data.acquisition import DOWNLOAD_FORMAT, FORMAT_SORT

    quiet = {**options, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(quiet) as client:
        raw = client.extract_info(url, download=False, process=False)
    selector = {**quiet, "format": DOWNLOAD_FORMAT, "format_sort": FORMAT_SORT}
    try:
        with yt_dlp.YoutubeDL(selector) as client:
            chosen = client.process_ie_result(copy.deepcopy(raw), download=False)
        raw["requested_formats"] = chosen.get("requested_formats") or [chosen]
    except yt_dlp.utils.YoutubeDLError:  # "Requested format is not available"
        raw["requested_formats"] = []
    return raw


def number(value, digits=1):
    return None if value in (None, "") else round(float(value), digits)


def describe(info):
    """Metadata lưu vào ledger (cột `meta`) từ kết quả yt-dlp (thô + requested_formats)."""
    formats = info.get("formats") or []
    video = [f for f in formats if f.get("vcodec") not in (None, "none")]
    audio = [
        f for f in formats if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
    ]
    largest = max(video, key=lambda f: min(f.get("width") or 0, f.get("height") or 0), default={})
    chosen = info.get("requested_formats") or []
    v = next((f for f in chosen if f.get("vcodec") not in (None, "none")), {})
    a = next((f for f in chosen if f.get("acodec") not in (None, "none")), {})
    languages = {f.get("language") for f in audio}
    note = (a.get("format_note") or "").lower()

    def size(f):
        value = f.get("filesize") or f.get("filesize_approx")
        return value / 1e6 if value else None

    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "channel_id": info.get("channel_id") or "",
        "upload_date": info.get("upload_date") or "",
        "duration_s": number(info.get("duration")),
        "live_status": info.get("live_status") or "",
        "availability": info.get("availability") or "",
        "max_width": largest.get("width"),
        "max_height": largest.get("height"),
        "native_fps": number(max((f.get("fps") or 0 for f in video), default=0), 3),
        "vertical": bool((largest.get("height") or 0) > (largest.get("width") or 0)),
        "v_format_id": v.get("format_id") or "",
        "v_codec": v.get("vcodec") or "",
        "v_width": v.get("width"),
        "v_height": v.get("height"),
        "v_fps": number(v.get("fps"), 3),
        "v_bitrate_kbps": number(v.get("vbr") or v.get("tbr")),
        "v_filesize_mb": number(size(v)),
        "dynamic_range": v.get("dynamic_range") or "",
        "a_format_id": a.get("format_id") or "",
        "a_codec": a.get("acodec") or "",
        "a_bitrate_kbps": number(a.get("abr") or a.get("tbr")),
        "a_filesize_mb": number(size(a)),
        "a_sample_rate": a.get("asr"),
        "a_channels": a.get("audio_channels"),
        "a_language": a.get("language") or "",
        # Nhiều track = có track lồng tiếng (vd. lồng tiếng AI của YouTube); kiểm track đã chọn.
        "audio_track_count": len(languages) if audio else None,
        "audio_is_original": len(languages) <= 1 or "original" in note,
    }


def check(meta, require_vietnamese=True):
    """Lý do loại theo metadata, hoặc None nếu đạt."""
    if meta.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        return f"live stream ({meta['live_status']})"
    if meta.get("availability") not in ("", None, "public", "unlisted"):
        return f"availability {meta['availability']}"
    duration = float(meta.get("duration_s") or 0)
    if not MIN_SOURCE_SECONDS <= duration <= MAX_SOURCE_HOURS * 3600:
        return f"duration {duration:.0f}s outside {MIN_SOURCE_SECONDS}s-{MAX_SOURCE_HOURS}h"
    # Dung sai 0,5: 25 fps danh định đôi khi ghi 24,99; 23,976 fps vẫn bị loại.
    if float(meta.get("native_fps") or 0) < MIN_FPS - 0.5:
        return f"fps {float(meta.get('native_fps') or 0):.2f} < {MIN_FPS}"
    if not meta.get("v_format_id") or not meta.get("a_format_id"):
        return f"no HTTPS mp4 format with ≥{MIN_FPS} fps and m4a audio"
    if int(meta["v_height"]) > int(meta["v_width"]):
        return f"vertical video {meta['v_width']}x{meta['v_height']} (v003 chỉ dùng video ngang)"
    short = min(int(meta["v_width"]), int(meta["v_height"]))
    if short < MIN_SHORT_SIDE:
        return f"short side {short} px < {MIN_SHORT_SIDE}"
    if short > MAX_SHORT_SIDE:
        return f"no ≤{MAX_SHORT_SIDE} px format (short side {short})"
    if not meta.get("audio_is_original", True):
        return "selected audio track is not the original (dubbed track)"
    language = (meta.get("a_language") or "").lower()
    if require_vietnamese and language and not language.startswith("vi"):
        return f"audio language {language}"
    return None


def estimate_mb(meta, fallback_gb_per_hour=0.5):
    """Dung lượng tải ước tính: filesize YouTube báo, không có thì suy từ bitrate."""
    duration = float(meta.get("duration_s") or 0)
    video = meta.get("v_filesize_mb")
    if video is None and meta.get("v_bitrate_kbps"):
        video = meta["v_bitrate_kbps"] * duration / 8 / 1000
    audio = meta.get("a_filesize_mb")
    if audio is None and meta.get("a_bitrate_kbps"):
        audio = meta["a_bitrate_kbps"] * duration / 8 / 1000
    if video is None:
        return round(fallback_gb_per_hour * 1000 * duration / 3600, 1)
    return round(video + (audio or 0), 1)
