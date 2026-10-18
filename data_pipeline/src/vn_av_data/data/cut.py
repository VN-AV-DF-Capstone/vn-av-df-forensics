"""Bước 03 cut: cắt clip ứng viên 4-7 s từ raw của một batch vào pool chung.

Nguồn một người (`speakers=1`): VAD cả video, rồi chỉ kiểm/cắt vài cửa sổ rải đều theo thời gian
đến khi đủ 40-60 clip cho người đó (tuỳ lượng tiếng nói) → rẻ, không cần quét hết video.
Nguồn nhiều người (`speakers=n`): quét mặt 4 fps + chuyển cảnh toàn video, cắt mọi cửa sổ đạt.

Mỗi cửa sổ phải qua (đọc mọi frame trên lưới 25 Hz của cửa sổ):
1. đúng một mặt hợp lệ, liền mạch ở mọi frame; 2. không chuyển cảnh; 3. không có jump cut chắc
chắn (jumpcut.py; nghi ngờ thì gắn cờ); 4. nguồn nhiều người: 5 điểm (đầu, ngẫu nhiên đầu-giữa,
giữa, ngẫu nhiên giữa-cuối, cuối) phải cùng một khuôn mặt (SFace).
5 mặt đã căn được lưu thành ảnh dải `pool/faces/<video>/<clip>.jpg` (xem khi duyệt) và trung bình
embedding của chúng là template của clip cho bước identify.

Cắt xong một video: đo độ lệch tiếng-hình bằng SyncNet trên tối đa `sync.clips_per_video` clip rải
đều → `cut_info["sync"]` (audio_lag_frames, conf, agree). Bước export dịch tiếng theo độ lệch đó.
Video không đo được (clip không đồng ý, đỉnh bẹt: thường là lời dẫn đè hình / ghép cảnh) →
mọi clip tự loại, gắn cờ `sync_unreliable`; người duyệt vẫn chọn lại được.

Đơn vị lưu là một video: clip của video ghi xong mới commit vào ledger. Ngắt giữa chừng thì
chạy lại sẽ xoá phần dở của video đó và cắt lại; video đã xong được bỏ qua.
"""

import math
import shutil
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from vn_av_data.common.runtime import fingerprint, run, sha
from vn_av_data.data import jumpcut
from vn_av_data.data.media import CLIP_FPS, MAX_SHORT_SIDE, ffmpeg, probe
from vn_av_data.ledger import now

STRIP_SIDE = 96


def media_origin(path):
    import av

    info = probe(path)
    with av.open(str(path)) as container:
        starts = [info[k] for k in ("video_start_s", "audio_start_s") if info[k] is not None]
        origin = (
            float(container.start_time / av.time_base)
            if container.start_time is not None
            else min(starts, default=0.0)
        )
    return origin, info


def plan_clips(speech, scenes, duration, cfg):
    """Cửa sổ min-max giây nằm trong đoạn có tiếng nói, không vắt qua chuyển cảnh."""
    minimum, maximum, overlap = cfg["min_seconds"], cfg["max_seconds"], cfg["overlap_seconds"]
    if not (0 < minimum <= maximum <= 60 and 0 <= overlap < minimum):
        raise ValueError("Need 0 <= overlap < min <= max <= 60 seconds")
    boundaries = sorted({0.0, duration, *(s for s in scenes if 0 < s < duration)})
    result = []
    for start, end in speech:
        for left, right in pairwise(boundaries):
            a, b = max(start, left), min(end, right)
            a, b = math.ceil(a * 25 - 1e-6) / 25, math.floor(b * 25 + 1e-6) / 25
            while b - a >= minimum - 1e-6:
                stop = min(b, a + maximum)
                # Đuôi ngắn: kéo cửa sổ cuối về sau thay vì bỏ phần đuôi.
                if 0 < b - stop < minimum - overlap:
                    stop = b
                    a = max(a, stop - maximum)
                result.append((round(a, 6), round(stop, 6)))
                if stop == b:
                    break
                a = stop - overlap
    return sorted(set(result))


def scan_visual(path, cfg, origin, tracker):
    """Quét mặt 4 fps + chuyển cảnh toàn video (chỉ cho nguồn nhiều người)."""
    import av
    import cv2

    samples, scenes, previous_hist, next_sample = [], [], None, 0.0
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame without timestamp")
            t = float(frame.pts * frame.time_base) - origin
            image = frame.to_ndarray(format="bgr24")
            hist = histogram(image)
            if (
                previous_hist is not None
                and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
                < cfg["scene_correlation"]
            ):
                scenes.append(max(0, t))
                tracker.previous = None
            previous_hist = hist
            if t + 1e-6 < next_sample:
                continue
            info = tracker.inspect(scaled(image, cfg["scan_max_side"]))
            samples.append({"time_s": max(0, t), "valid": info["valid"]})
            next_sample = t + 1 / cfg["sample_fps"]
    return samples, scenes


def histogram(image):
    import cv2

    tiny = cv2.resize(image, (96, 54))
    hist = cv2.calcHist([tiny], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist, hist)
    return hist


def scaled(image, max_side):
    import cv2

    scale = min(1, max_side / max(image.shape[:2]))
    if scale >= 1:
        return image
    return cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def face_crop(image, box, side=32):
    import cv2

    x, y, w, h = (int(round(v)) for v in box)
    crop = image[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0:
        return np.zeros((side, side), np.uint8)
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (side, side))


def identity_frames(count, seed):
    """5 chỉ số frame: đầu, ngẫu nhiên (đầu, giữa), giữa, ngẫu nhiên (giữa, cuối), cuối."""
    rng = np.random.default_rng(seed)
    last = count - 1
    return sorted(
        {
            0,
            int(rng.integers(max(1, last // 10), max(2, last // 2))),
            last // 2,
            int(rng.integers(last // 2 + 1, max(last // 2 + 2, last - last // 10))),
            last,
        }
    )


def fill_gaps(items, shape):
    """Frame hụt (None): nội suy tuyến tính giữa hai frame tốt kề bên; ở hai đầu lấy frame gần nhất."""
    good = [i for i, x in enumerate(items) if x is not None]
    if not good:
        return np.zeros((len(items), *shape), np.float32)
    out = []
    for i, x in enumerate(items):
        if x is not None:
            out.append(np.asarray(x, np.float32))
            continue
        before = max((g for g in good if g < i), default=None)
        after = min((g for g in good if g > i), default=None)
        if before is None or after is None:
            out.append(np.asarray(items[after if before is None else before], np.float32))
        else:
            w = (i - before) / (after - before)
            a, b = np.asarray(items[before], np.float32), np.asarray(items[after], np.float32)
            out.append((1 - w) * a + w * b)
    return np.stack(out)


def window_pcm(path, origin, start, end):
    """Tiếng mono 16 kHz đúng khoảng [start, end) trên cùng gốc thời gian với hình."""
    import av

    total = int(round((end - start) * 16000))
    pcm = np.zeros(total, np.float32)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        container.seek(int(max(0.0, origin + start - 1) / stream.time_base), stream=stream)
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

        def frames():
            for frame in container.decode(stream):
                yield from resampler.resample(frame)
                if frame.pts is not None and float(frame.pts * frame.time_base) - origin > end:
                    return

        for f in frames():
            if f.pts is None:
                continue
            first = int(round((float(f.pts * f.time_base) - origin - start) * 16000))
            samples = f.to_ndarray().reshape(-1)
            lo, hi = max(0, first), min(total, first + len(samples))
            if hi > lo:
                pcm[lo:hi] = samples[lo - first : hi - first]
    return pcm


class Inspector:
    """Kiểm một cửa sổ ở mọi frame (lưới 25 Hz). Dùng chung cho cắt và đo jump cut."""

    def __init__(self, cfg, tracker, embedder):
        self.cfg, self.tracker, self.embedder = cfg, tracker, embedder

    def features(self, path, origin, start, end, points=(), stop_on_fail=True):
        import av
        import cv2

        expected = int(round((end - start) * CLIP_FPS))
        wanted = set(points)
        out = {"landmarks": [], "crops": [], "aligned": [], "embeddings": [], "reason": None}
        self.tracker.previous = None
        max_bad, max_run = self.cfg["max_bad_frames"], self.cfg["max_bad_run"]
        bad = run_length = 0
        previous_hist, index, grid = None, 0, start
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(max(0.0, origin + start - 1) / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Video frame without timestamp")
                t = float(frame.pts * frame.time_base) - origin
                if t < grid - 0.02:
                    continue
                if index >= expected or t >= end - 1e-6:
                    break
                grid += 1 / CLIP_FPS
                image = frame.to_ndarray(format="bgr24")
                hist = histogram(image)
                if (
                    previous_hist is not None
                    and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
                    < self.cfg["scene_correlation"]
                ):
                    out["reason"] = out["reason"] or "scene_change"
                    if stop_on_fail:
                        return out
                previous_hist = hist
                small = scaled(image, self.cfg["scan_max_side"])
                info = self.tracker.inspect(small)
                if not info["valid"]:
                    # Hụt mặt thoáng qua (nghiêng đầu, mờ chuyển động) được bỏ qua trong giới hạn;
                    # landmark / ảnh mặt của frame đó nội suy từ hai frame tốt kề bên.
                    bad, run_length = bad + 1, run_length + 1
                    if bad > max_bad or run_length > max_run:
                        out["reason"] = out["reason"] or "face_frame"
                        if stop_on_fail:
                            return out
                    out["landmarks"].append(None)
                    out["crops"].append(None)
                else:
                    run_length = 0
                    out["landmarks"].append(info["landmarks"])
                    out["crops"].append(face_crop(small, info["box"]))
                    if index in wanted:
                        aligned = self.embedder.align(small, info["face"])
                        out["aligned"].append(aligned)
                        out["embeddings"].append(self.embedder.embed(aligned))
                index += 1
        if index < expected * 0.95:
            out["reason"] = out["reason"] or "coverage"
        out["gaps"] = bad
        out["landmarks"] = fill_gaps(out["landmarks"], (5, 2))
        out["crops"] = fill_gaps(out["crops"], (32, 32))
        return out

    def inspect(self, path, origin, start, end, seed, gate_identity):
        expected = int(round((end - start) * CLIP_FPS))
        points = identity_frames(expected, seed)
        feats = self.features(path, origin, start, end, points)
        if feats["reason"]:
            return {"ok": False, "reason": feats["reason"]}
        pcm = window_pcm(path, origin, start, end)
        jump = jumpcut.detect(feats["landmarks"], feats["crops"], pcm, self.cfg["jumpcut"])
        if jump["confident"]:
            return {"ok": False, "reason": "jump_cut", "jump": jump}
        embeddings = np.asarray(feats["embeddings"], np.float32)
        if len(embeddings) < 3:
            return {"ok": False, "reason": "identity_points"}
        mean = embeddings.mean(0)
        mean /= max(1e-8, float(np.linalg.norm(mean)))
        consistency = float((embeddings @ mean).min())
        if gate_identity and consistency < self.cfg["identity_consistency"]:
            return {"ok": False, "reason": "identity_change", "consistency": consistency}
        return {
            "ok": True,
            "face": mean,
            "consistency": consistency,
            "aligned": feats["aligned"],
            "suspect": jump["suspect"],
            "gaps": feats["gaps"],
        }


def cut_media(source, output, start, end, origin):
    """Encode clip: 25 fps CFR, cạnh ngắn ≤ 720, H.264 CRF 18 + AAC 48 kHz; hình và tiếng
    trừ CÙNG một mốc thời gian (giữ nguyên độ lệch A/V gốc)."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    absolute, stop = origin + start, origin + end
    filters = (
        f"[0:v:0]trim=start={absolute:.6f}:end={stop:.6f},"
        f"setpts=PTS-({absolute:.6f})/TB,"
        f"fps={CLIP_FPS},"
        f"scale=w='if(gte(iw,ih),-2,min(trunc(iw/2)*2,{MAX_SHORT_SIDE}))'"
        f":h='if(gte(iw,ih),min(trunc(ih/2)*2,{MAX_SHORT_SIDE}),-2)'[v];"
        f"[0:a:0]atrim=start={absolute:.6f}:end={stop:.6f},"
        f"asetpts=PTS-({absolute:.6f})/TB[a]"
    )
    try:
        run(
            [
                ffmpeg(),
                *("-nostdin", "-v", "error", "-y", "-copyts", "-ss", str(start)),
                *("-i", str(source), "-filter_complex", filters, "-map", "[v]", "-map", "[a]"),
                *("-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"),
                *("-fps_mode", "cfr", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"),
                *("-avoid_negative_ts", "disabled", "-movflags", "+faststart", partial),
            ]
        )
        info = probe(partial)
        if not info["duration_s"] or abs(info["duration_s"] - (end - start)) > 0.2:
            raise ValueError("Encoded clip duration disagrees with requested source interval")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def write_strip(aligned, target):
    import cv2

    tiles = [cv2.resize(a, (STRIP_SIDE, STRIP_SIDE)) for a in aligned]
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Cannot encode face strip")
    partial = target.with_suffix(".partial.jpg")
    partial.write_bytes(data.tobytes())
    partial.replace(target)


def single_target(speech_s, rules):
    """30 clip khi ≤ low phút tiếng nói, tăng tuyến tính tới 50 clip ở full phút."""
    minutes = speech_s / 60
    lo, hi = rules["low_speech_minutes"], rules["full_speech_minutes"]
    share = min(1.0, max(0.0, (minutes - lo) / max(1e-6, hi - lo)))
    return int(round(rules["min_clips"] + share * (rules["max_clips"] - rules["min_clips"])))


def split_quota(total, weights):
    """Chia `total` theo trọng số, phần dư theo phần thập phân lớn nhất."""
    weights = np.asarray(weights, float)
    if total <= 0 or weights.sum() <= 0:
        return [0] * len(weights)
    raw = total * weights / weights.sum()
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base.tolist()


def spread_order(windows, quota, duration):
    """Thứ tự thử cửa sổ để clip rải đều: chia trục thời gian thành `quota` khúc, mỗi vòng lấy
    cửa sổ gần tâm khúc nhất còn lại của từng khúc chưa có clip."""
    if not windows or quota <= 0:
        return []
    width = duration / quota
    queues = [[] for _ in range(quota)]
    for w in windows:
        mid = (w[0] + w[1]) / 2
        queues[min(quota - 1, int(mid // width))].append(w)
    for k, q in enumerate(queues):
        center = (k + 0.5) * width
        q.sort(key=lambda w: abs((w[0] + w[1]) / 2 - center))
    return queues


class Cutter:
    def __init__(self, ws, ledger, cfg, inspector):
        self.ws, self.ledger, self.cfg, self.inspector = ws, ledger, cfg, inspector

    def detect_speech(self, path, origin, duration):
        from vn_av_data.data.vad import detect_speech

        return detect_speech(
            path, self.cfg["vad_model"], origin, duration, self.cfg["merge_silence_seconds"]
        )

    def accept(self, video, path, origin, start, end, result):
        clip_id = "clip_" + fingerprint([video["video_id"], start, end])[:24]
        target = self.ws.clips / video["video_id"] / f"{clip_id}.mp4"
        target.unlink(missing_ok=True)
        cut_media(path, target, start, end, origin)
        write_strip(result["aligned"], self.ws.faces / video["video_id"] / f"{clip_id}.jpg")
        return {
            "clip_id": clip_id,
            "video_id": video["video_id"],
            "input_id": video["input_id"],
            "batch": video["batch"],
            "start_s": start,
            "end_s": end,
            "duration_s": round(end - start, 3),
            "file": target.relative_to(self.ws.root).as_posix(),
            "sha256": sha(target),
            "size_mb": round(target.stat().st_size / 1e6, 3),
            "face": result["face"],
            "face_consistency": round(result["consistency"], 4),
            "flags": ",".join(
                flag
                for flag, on in (
                    ("jump_suspect", result["suspect"]),
                    ("face_gap", result.get("gaps")),
                )
                if on
            ),
            "jump": ",".join(str(t) for t in result["suspect"]),
            "created_at": now(),
        }

    def cut_video(self, video, speakers, quota):
        cfg = self.cfg
        path = self.ws.root / video["file"]
        if sha(path) != video["sha256"]:
            raise ValueError(f"Raw video changed: {path}")
        origin, info = media_origin(path)
        duration = info["duration_s"]
        if not duration or duration > cfg["max_source_hours"] * 3600:
            raise ValueError("Source duration missing or exceeds configured limit")
        cut_info = video.get("cut_info") or {}
        speech = cut_info.get("speech")
        if speech is None:
            speech = self.detect_speech(path, origin, duration)
        reasons, clips, inspected = Counter(), [], 0
        single = speakers == "1"
        if single:
            windows = plan_clips(speech, [], duration, {**cfg, "overlap_seconds": 0.0})
            queues = spread_order(windows, quota, duration)
            budget = int(math.ceil(quota * cfg["single"]["attempts_factor"])) + 5
            filled = set()
            while len(clips) < quota and inspected < budget and any(queues):
                # Mỗi vòng: khúc chưa có clip trước; hết khúc trống thì lấy khúc còn cửa sổ.
                live = [k for k, q in enumerate(queues) if q]
                for k in [k for k in live if k not in filled] or live:
                    if len(clips) >= quota or inspected >= budget:
                        break
                    start, end = queues[k].pop(0)
                    inspected += 1
                    result = self.inspector.inspect(
                        path,
                        origin,
                        start,
                        end,
                        int(fingerprint([video["video_id"], start])[:12], 16),
                        False,
                    )
                    if result["ok"]:
                        clips.append(self.accept(video, path, origin, start, end, result))
                        filled.add(k)
                    else:
                        reasons[result["reason"]] += 1
        else:
            samples, scenes = scan_visual(path, cfg, origin, self.inspector.tracker)
            for start, end in plan_clips(speech, scenes, duration, cfg):
                inside = [s for s in samples if start <= s["time_s"] < end]
                coverage = len(inside) / max(1, (end - start) * cfg["sample_fps"])
                ratio = sum(s["valid"] for s in inside) / max(1, len(inside))
                if coverage < 0.8 or ratio < cfg["min_face_ratio"]:
                    reasons["face_scan"] += 1
                    continue
                inspected += 1
                result = self.inspector.inspect(
                    path,
                    origin,
                    start,
                    end,
                    int(fingerprint([video["video_id"], start])[:12], 16),
                    True,
                )
                if result["ok"]:
                    clips.append(self.accept(video, path, origin, start, end, result))
                else:
                    reasons[result["reason"]] += 1
        speech_s = sum(b - a for a, b in speech)
        return clips, {
            "speech": speech,
            "speech_s": round(speech_s, 1),
            "duration_s": duration,
            "quota": quota if single else None,
            "inspected": inspected,
            "accepted": len(clips),
            "rejected": dict(reasons),
        }


def cut_signature(cfg):
    here = Path(__file__).resolve()
    return fingerprint(
        {
            "config": {k: v for k, v in cfg.items() if k not in ("face_model", "vad_model")},
            "code": [
                sha(here),
                sha(here.with_name("jumpcut.py")),
                sha(here.with_name("vad.py")),
                sha(here.parents[1] / "features/face.py"),
                sha(here.parents[1] / "features/syncnet.py"),
            ],
            "models": [
                sha(cfg[k]) for k in ("face_model", "vad_model", "sface_model", "syncnet_model")
            ],
        }
    )


def measure_sync(meter, ws, clips, rules):
    """Độ lệch của cả video từ tối đa `clips_per_video` clip rải đều theo thời gian."""
    from vn_av_data.features.syncnet import summarize

    ordered = sorted(clips, key=lambda c: c["start_s"])
    count = min(rules["clips_per_video"], len(ordered))
    picks = sorted({round(i * (len(ordered) - 1) / max(1, count - 1)) for i in range(count)})
    curves = [meter.measure(ws.root / ordered[i]["file"]) for i in picks]
    result = summarize([c for c in curves if c is not None], rules)
    flag = {"unreliable": "sync_unreliable", "no_clips": "sync_unreliable", "few": "sync_few"}
    if result["status"] in flag:
        for clip in clips:
            if flag[result["status"]] == "sync_unreliable":
                clip["decision"] = "reject"
            clip["flags"] = ",".join(filter(None, [clip["flags"], flag[result["status"]]]))
    return result


def cut_tools(cfg):
    """Bộ kiểm cửa sổ + bộ đo SyncNet."""
    from vn_av_data.features.face import FaceEmbedder, FaceTracker
    from vn_av_data.features.syncnet import SyncMeter

    tracker = FaceTracker(
        cfg["face_model"], min_size=cfg["min_face_size"], max_gap=cfg["max_bad_run"]
    )
    inspector = Inspector(cfg, tracker, FaceEmbedder(cfg["sface_model"]))
    return inspector, SyncMeter(cfg["syncnet_model"], cfg["face_model"])


def cut_one(cutter, meter, video, speakers, quota):
    """Cắt một video + đo độ lệch; chỉ ghi file vào pool, không đụng ledger."""
    began = time.monotonic()
    clips, info = cutter.cut_video(video, speakers, quota)
    if clips:
        info["sync"] = measure_sync(meter, cutter.ws, clips, cutter.cfg["sync"])
    info["elapsed_s"] = round(time.monotonic() - began, 1)
    return clips, info


def cut_batch(ws, ledger, batch, cfg, inspector=None, speech_detector=None, meter=None):
    """Cắt lần lượt từng video đã tải của batch; test truyền inspector / meter giả."""
    cfg = dict(cfg["cut"])
    pending = ledger.scalar(
        "SELECT COUNT(*) FROM videos WHERE batch=? AND status='accepted' "
        "AND dl_status NOT IN ('done','rejected')",
        (batch,),
    )
    if pending:
        raise ValueError(f"Batch {batch}: {pending} videos not downloaded yet; run download first")
    signature = cut_signature(cfg)
    old = ledger.get_meta("cut_signature")
    if old and old != signature and ledger.scalar("SELECT COUNT(*) FROM clips"):
        raise ValueError(
            "Cut rules/code/models changed after clips were cut. Restore the old rules, or use a "
            "new workspace (VNAV_DATA) for a trial run."
        )
    ledger.set_meta("cut_signature", signature)
    if inspector is None or meter is None:
        tools = cut_tools(cfg)
        inspector, meter = inspector or tools[0], meter or tools[1]
    cutter = Cutter(ws, ledger, cfg, inspector)
    if speech_detector is not None:
        cutter.detect_speech = speech_detector
    videos = ledger.rows(
        "SELECT v.*, i.speakers FROM videos v JOIN inputs i USING(input_id) "
        "WHERE v.batch=? AND v.dl_status='done' ORDER BY v.input_id, v.rank",
        (batch,),
    )
    by_input = {}
    for v in videos:
        by_input.setdefault(v["input_id"], []).append(v)
    totals, jobs = Counter(), []
    for input_id, group in by_input.items():
        speakers = group[0]["speakers"]
        quotas = {v["video_id"]: None for v in group}
        if speakers == "1":
            quotas = single_quotas(cutter, ledger, group, cfg["single"])
        for v in group:
            if v["cut_status"] == "done":
                totals["skipped"] += 1
                continue
            fresh = ledger.row("SELECT * FROM videos WHERE video_id=?", (v["video_id"],))
            jobs.append((input_id, fresh, speakers, quotas[v["video_id"]]))

    def clean(video_id):  # phần dở của lần chạy bị ngắt / video lỗi
        shutil.rmtree(ws.clips / video_id, ignore_errors=True)
        shutil.rmtree(ws.faces / video_id, ignore_errors=True)

    def finish(input_id, video, outcome):
        minutes = (video["duration_s"] or 0) / 60
        head = f"[input {input_id}] {video['video_id']} ({minutes:.1f} phút)"
        if isinstance(outcome, Exception):
            clean(video["video_id"])
            ledger.update(
                "videos",
                "video_id",
                video["video_id"],
                {"cut_status": "failed", "cut_info": {"error": str(outcome)}},
            )
            print(f"{head}\n  lỗi: {outcome}", flush=True)
            totals["failed"] += 1
            return
        clips, info = outcome
        with ledger.tx() as conn:
            for clip in clips:
                ledger.insert("clips", clip, conn=conn)
            ledger.update(
                "videos",
                "video_id",
                video["video_id"],
                {"cut_status": "done", "cut_info": info},
                conn,
            )
        totals["videos"] += 1
        totals["clips"] += len(clips)
        sync = info.get("sync") or {}
        print(
            f"{head}\n  {info['accepted']} clip / {info['inspected']} cửa sổ kiểm, loại "
            f"{info['rejected']} | lệch {sync.get('audio_lag_frames', '-')} frame "
            f"({sync.get('status', '-')}) | {info['elapsed_s']:.0f}s",
            flush=True,
        )

    for _, video, _, _ in jobs:
        clean(video["video_id"])
    if jobs:
        print(f"{len(jobs)} video cần cắt", flush=True)
    try:
        for input_id, video, speakers, quota in jobs:
            try:
                outcome = cut_one(cutter, meter, video, speakers, quota)
            except Exception as exc:  # noqa: BLE001 -- video hỏng không chặn cả batch
                outcome = exc
            finish(input_id, video, outcome)
    except KeyboardInterrupt:
        for _, video, _, _ in jobs:
            status = ledger.scalar(
                "SELECT cut_status FROM videos WHERE video_id=?", (video["video_id"],)
            )
            if status != "done":
                clean(video["video_id"])
        print("\nDừng. Chạy lại cùng batch để cắt tiếp (video dở cắt lại từ đầu).")
        raise
    ledger.log("cut", f"batch {batch}: {dict(totals)}")
    if totals["failed"]:
        raise RuntimeError(f"{totals['failed']} videos failed; run cut again to retry them")
    return {"batch": batch, **totals}


def single_quotas(cutter, ledger, group, rules):
    """Chỉ tiêu clip cho từng video của một người: VAD mọi video (lưu lại) → tổng 40-60 clip
    chia theo lượng tiếng nói; video đã cắt giữ nguyên số clip đã có."""
    speech = {}
    for v in group:
        info = v.get("cut_info") or {}
        if "speech" not in info:
            path = cutter.ws.root / v["file"]
            origin, media = media_origin(path)
            info = {**info, "speech": cutter.detect_speech(path, origin, media["duration_s"])}
            ledger.update("videos", "video_id", v["video_id"], {"cut_info": info})
            v["cut_info"] = info
        speech[v["video_id"]] = sum(b - a for a, b in info["speech"])
    target = single_target(sum(speech.values()), rules)
    done = {
        v["video_id"]: (v.get("cut_info") or {}).get("accepted", 0)
        for v in group
        if v["cut_status"] == "done"
    }
    left = [v["video_id"] for v in group if v["video_id"] not in done]
    shares = split_quota(max(0, target - sum(done.values())), [speech[i] for i in left])
    return {**{i: None for i in done}, **dict(zip(left, shares))}


def jumpcut_eval(ws, ledger, cfg, batch, windows=60, seed=0, inspector=None):
    """Đo bộ phát hiện jump cut trên raw thật của một batch (raw phải còn ở máy).

    Lấy đoạn liền mạch dài ~10,5 s quanh các clip đã cắt (đã qua kiểm mặt), trích đặc trưng mọi
    frame, rồi tạo jump cut giả bằng cách xoá một đoạn 0,2-3 s (xem jumpcut.evaluate).
    """
    from vn_av_data.features.face import FaceEmbedder, FaceTracker

    cfg = dict(cfg["cut"])
    if inspector is None:
        tracker = FaceTracker(
            cfg["face_model"], min_size=cfg["min_face_size"], max_gap=cfg["max_bad_run"]
        )
        inspector = Inspector(cfg, tracker, FaceEmbedder(cfg["sface_model"]))
    clips = ledger.rows(
        "SELECT c.video_id, c.start_s, v.file, v.duration_s FROM clips c JOIN videos v "
        "USING(video_id) WHERE c.batch=? AND v.raw_location='local' ORDER BY c.clip_id",
        (batch,),
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(clips)
    span = cfg["max_seconds"] + 3.5
    sequences, tried = [], 0
    for c in clips:
        if len(sequences) >= windows:
            break
        start = max(0.0, c["start_s"] - 1.5)
        if start + span > (c["duration_s"] or 0):
            continue
        tried += 1
        path = ws.root / c["file"]
        origin, _ = media_origin(path)
        feats = inspector.features(path, origin, start, start + span)
        if feats["reason"]:
            continue
        feats["pcm"] = window_pcm(path, origin, start, start + span)
        sequences.append(feats)
        print(f"  chuỗi {len(sequences)}/{windows}", flush=True)
    report = jumpcut.evaluate(
        sequences,
        cfg["jumpcut"],
        samples=max(200, 4 * len(sequences)),
        seed=seed,
        min_s=cfg["min_seconds"],
        max_s=cfg["max_seconds"],
    )
    return {"batch": batch, "sequences": len(sequences), "tried": tried, **report}
