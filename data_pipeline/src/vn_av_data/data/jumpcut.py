"""Phát hiện jump cut: video bị cắt bỏ một đoạn nhưng người vẫn ở gần đúng chỗ cũ.

Tín hiệu theo frame (lưới 25 Hz), so với nền cục bộ ±`context_s` quanh frame:
- landmark: độ dịch trung bình 5 điểm YuNet giữa hai frame, chia khoảng cách hai mắt.
  Jump cut = MỘT gai đơn lẻ (frame trước và sau đều yên). Di chuyển chậm hay quay đầu nhanh
  trải đều qua nhiều frame nên không tạo gai đơn lẻ.
- appearance: sai khác ảnh mặt xám 32×32 giữa hai frame liên tiếp (z-score theo MAD).
- audio: khoảng cách phổ log-mel 80 ms trước / sau điểm cắt (z-score theo MAD); chỉ là tín hiệu
  phụ vì lời nói tự nhiên cũng đổi phổ nhanh.

Mức kết luận:
- confident: gai landmark + (appearance hoặc audio) → loại cửa sổ.
- suspect: chỉ một bằng chứng → giữ clip, gắn cờ để người duyệt xem.
`evaluate()` đo recall / false positive bằng jump cut giả: xoá một đoạn 0,2-3 s khỏi chuỗi frame
thật của chính nguồn đó (đúng bản chất của jump cut).
"""

import numpy as np

DEFAULTS = {
    "context_s": 1.0,
    "landmark_ratio": 6.0,  # gai >= ratio × trung vị chuyển động nền
    "landmark_min": 0.06,  # và >= 6% khoảng cách hai mắt
    "calm_ratio": 3.0,  # 2 frame mỗi bên gai phải <= calm_ratio × nền
    "appearance_z": 6.0,
    "audio_z": 5.0,
}
FPS = 25
SAMPLES_PER_FRAME = 640  # 16 kHz


def landmark_motion(landmarks):
    """(T, 5, 2) → (T,) độ dịch chuẩn hoá; frame 0 = 0."""
    lm = np.asarray(landmarks, np.float32)
    iod = np.linalg.norm(lm[:, 0] - lm[:, 1], axis=1)
    iod = np.maximum(iod, 1.0)
    step = np.linalg.norm(lm[1:] - lm[:-1], axis=2).mean(1)
    scale = (iod[1:] + iod[:-1]) / 2
    return np.r_[0.0, step / scale].astype(np.float32)


def appearance_change(crops):
    """(T, 32, 32) ảnh xám 0..255 → (T,) sai khác tuyệt đối trung bình với frame trước."""
    c = np.asarray(crops, np.float32) / 255.0
    return np.r_[0.0, np.abs(c[1:] - c[:-1]).mean((1, 2))].astype(np.float32)


def audio_change(pcm, frames, half_frames=2):
    """Khoảng cách log-phổ trung bình `half_frames` frame trước / sau mỗi ranh giới frame."""
    pcm = np.asarray(pcm, np.float32)
    need = frames * SAMPLES_PER_FRAME
    pcm = np.pad(pcm[:need], (0, max(0, need - len(pcm))))
    blocks = pcm.reshape(frames, SAMPLES_PER_FRAME) * np.hanning(SAMPLES_PER_FRAME)
    spec = np.abs(np.fft.rfft(blocks, axis=1)) ** 2
    # 32 dải tần đều theo log trên 0-8 kHz là đủ cho thay đổi tiếng nền / phòng.
    edges = np.unique(np.geomspace(1, spec.shape[1] - 1, 33).astype(int))
    bands = np.stack([spec[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    logs = np.log(bands + 1e-8)
    out = np.zeros(frames, np.float32)
    for t in range(half_frames, frames - half_frames + 1):
        before = logs[t - half_frames : t].mean(0)
        after = logs[t : t + half_frames].mean(0)
        out[t] = float(np.sqrt(np.mean((after - before) ** 2)))
    return out


def robust_z(values, t, lo, hi):
    """z-score của values[t] so với values[lo:hi] bỏ t-1..t+1 (trung vị + MAD)."""
    ctx = np.r_[values[lo : max(lo, t - 1)], values[min(hi, t + 2) : hi]]
    if len(ctx) < 5:
        return 0.0
    med = float(np.median(ctx))
    mad = float(np.median(np.abs(ctx - med))) * 1.4826
    return (float(values[t]) - med) / max(mad, 1e-4)


def detect(landmarks, crops, pcm=None, cfg=None):
    """Trả {"confident": [giây], "suspect": [giây]} tính từ đầu chuỗi."""
    cfg = {**DEFAULTS, **(cfg or {})}
    motion = landmark_motion(landmarks)
    appearance = appearance_change(crops)
    frames = len(motion)
    audio = audio_change(pcm, frames) if pcm is not None and len(pcm) else None
    ctx = max(3, round(cfg["context_s"] * FPS))
    floor = cfg["landmark_min"] / cfg["landmark_ratio"]
    found = {"confident": [], "suspect": []}
    last = -10
    for t in range(1, frames):
        lo, hi = max(1, t - ctx), min(frames, t + ctx + 1)
        neighbours = np.r_[motion[max(1, t - 2) : t], motion[t + 1 : t + 3]]
        background = np.r_[motion[lo : max(lo, t - 2)], motion[min(hi, t + 3) : hi]]
        base = max(float(np.median(background)) if len(background) else 0.0, floor)
        spike = motion[t] >= max(cfg["landmark_min"], cfg["landmark_ratio"] * base) and (
            not len(neighbours) or neighbours.max() <= cfg["calm_ratio"] * base
        )
        app = robust_z(appearance, t, lo, hi) >= cfg["appearance_z"]
        aud = False
        if audio is not None:
            aud = max(robust_z(audio, u, lo, hi) for u in range(max(1, t - 1), min(frames, t + 2)))
            aud = aud >= cfg["audio_z"]
        level = None
        if spike and (app or aud):
            level = "confident"
        elif spike or (app and aud):
            level = "suspect"
        if level and t - last > 3:
            found[level].append(round(t / FPS, 2))
            last = t
    return found


def evaluate(sequences, cfg=None, samples=200, seed=0, min_s=4.0, max_s=7.0):
    """Đo trên chuỗi đặc trưng thật (dict landmarks/crops/pcm ở 25 Hz, dài >= max_s + 3 s).

    Âm tính: một đoạn liền mạch dài min_s-max_s. Dương tính: cùng độ dài nhưng xoá một đoạn
    0,2-3 s ở vị trí ngẫu nhiên (cách hai đầu >= 1 s).
    """
    rng = np.random.default_rng(seed)
    usable = [s for s in sequences if len(s["landmarks"]) >= (max_s + 3.2) * FPS]
    if not usable:
        raise ValueError("Need feature sequences of at least max_s + 3.2 seconds")
    stats = {"negatives": 0, "false_confident": 0, "false_suspect": 0}
    stats.update({"positives": 0, "hit_confident": 0, "hit_any": 0})
    by_gap = {}

    def take(seq, idx):
        pcm = seq.get("pcm")
        audio = None
        if pcm is not None:
            audio = np.concatenate(
                [pcm[i * SAMPLES_PER_FRAME : (i + 1) * SAMPLES_PER_FRAME] for i in idx]
            )
        return seq["landmarks"][idx], seq["crops"][idx], audio

    for i in range(samples):
        seq = usable[i % len(usable)]
        total = len(seq["landmarks"])
        length = int(rng.uniform(min_s, max_s) * FPS)
        if i % 2 == 0:
            start = int(rng.integers(0, total - length))
            found = detect(*take(seq, np.arange(start, start + length)), cfg)
            stats["negatives"] += 1
            stats["false_confident"] += bool(found["confident"])
            stats["false_suspect"] += bool(found["suspect"] or found["confident"])
            continue
        gap = int(rng.uniform(0.2, 3.0) * FPS)
        start = int(rng.integers(0, total - length - gap))
        cut = int(rng.integers(FPS, length - FPS))
        idx = np.r_[
            np.arange(start, start + cut), np.arange(start + cut + gap, start + length + gap)
        ]
        found = detect(*take(seq, idx), cfg)
        hit_time = cut / FPS
        near = [t for t in found["confident"] if abs(t - hit_time) <= 0.12]
        any_near = near + [t for t in found["suspect"] if abs(t - hit_time) <= 0.12]
        stats["positives"] += 1
        stats["hit_confident"] += bool(near)
        stats["hit_any"] += bool(any_near)
        bucket = "<0.5s" if gap < 0.5 * FPS else "0.5-1.5s" if gap < 1.5 * FPS else ">=1.5s"
        entry = by_gap.setdefault(bucket, [0, 0])
        entry[0] += 1
        entry[1] += bool(any_near)
    return {
        "recall_confident": stats["hit_confident"] / max(1, stats["positives"]),
        "recall_any": stats["hit_any"] / max(1, stats["positives"]),
        "false_positive_confident": stats["false_confident"] / max(1, stats["negatives"]),
        "false_positive_any": stats["false_suspect"] / max(1, stats["negatives"]),
        "recall_any_by_gap": {k: round(v[1] / v[0], 3) for k, v in sorted(by_gap.items())},
        **stats,
    }
