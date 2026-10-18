"""Xử lý tiếng dùng chung cho ② sham và ③ / ⑥ giọng AI (cùng một cách ghép, để vết ghép
không trở thành dấu hiệu riêng của giọng AI).

Mọi PCM là float32 mono 48 kHz, cùng lưới thời gian với video (1920 mẫu / frame 25 Hz).
"""

import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np

RATE = 48000


def resample(pcm, rate, target=RATE):
    from scipy.signal import resample_poly

    pcm = np.asarray(pcm, np.float32)
    if rate == target:
        return pcm
    g = gcd(int(rate), int(target))
    return resample_poly(pcm, target // g, int(rate) // g).astype(np.float32)


def read_wav(path, target=RATE):
    from scipy.io import wavfile

    rate, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return resample(data.astype(np.float32), rate, target)


def write_wav(path, pcm, rate=RATE):
    from scipy.io import wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.wav")
    wavfile.write(partial, rate, np.clip(np.asarray(pcm) * 32767, -32768, 32767).astype(np.int16))
    partial.replace(path)


def rms(pcm):
    pcm = np.asarray(pcm, np.float64)
    return float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0.0


def trim_silence(pcm, threshold_db=-40.0, frame=480):
    """Bỏ lặng đầu/cuối của câu TTS (theo năng lượng 10 ms) trước khi co giãn vào chỗ trống."""
    pcm = np.asarray(pcm, np.float32)
    if len(pcm) < frame:
        return pcm
    blocks = pcm[: len(pcm) // frame * frame].reshape(-1, frame)
    level = 20 * np.log10(np.sqrt((blocks**2).mean(1)) + 1e-9)
    loud = np.flatnonzero(level > level.max() + threshold_db)
    if not len(loud):
        return pcm
    return pcm[loud[0] * frame : (loud[-1] + 1) * frame]


def time_stretch(pcm, target_samples, ffmpeg):
    """Đổi độ dài không đổi cao độ bằng bộ lọc atempo của ffmpeg; trả (pcm, tỉ lệ)."""
    pcm = np.asarray(pcm, np.float32)
    if not len(pcm) or target_samples <= 0:
        raise ValueError("Empty audio cannot be stretched")
    ratio = len(pcm) / target_samples  # >1: phải nói nhanh hơn
    if abs(ratio - 1) < 0.005:
        out = pcm
    else:
        chain, left = [], ratio
        while left > 2.0:
            chain.append(2.0)
            left /= 2.0
        while left < 0.5:
            chain.append(0.5)
            left /= 0.5
        chain.append(left)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            write_wav(src, pcm)
            subprocess.run(
                [
                    str(ffmpeg),
                    *("-nostdin", "-v", "error", "-y", "-i", str(src)),
                    *("-af", ",".join(f"atempo={c:.6f}" for c in chain), str(dst)),
                ],
                check=True,
                capture_output=True,
            )
            out = read_wav(dst)
    out = out[:target_samples]
    if len(out) < target_samples:
        out = np.pad(out, (0, target_samples - len(out)))
    return out.astype(np.float32), float(ratio)


def noise_floor(pcm, seconds=0.2):
    """Đoạn yên nhất (theo RMS 200 ms) của tiếng gốc, dùng làm tiếng nền phủ lên đoạn chèn."""
    size = int(seconds * RATE)
    pcm = np.asarray(pcm, np.float32)
    if len(pcm) < 2 * size:
        return np.zeros(size, np.float32)
    starts = range(0, len(pcm) - size, size // 2)
    best = min(starts, key=lambda s: rms(pcm[s : s + size]))
    return pcm[best : best + size].copy()


def splice(base, insert, start, end, crossfade_s=0.015, match_loudness=True, noise=None):
    """Thay base[start:end] (mẫu) bằng insert (đã đúng độ dài), crossfade hai đầu.

    match_loudness: đưa RMS đoạn chèn về RMS đoạn bị thay. noise: tiếng nền lặp lại để phủ
    đoạn chèn (TTS sạch hơn hẳn tiếng thu thật).
    """
    base = np.asarray(base, np.float32).copy()
    insert = np.asarray(insert, np.float32).copy()
    if len(insert) != end - start or not 0 <= start < end <= len(base):
        raise ValueError("Insert must exactly fill a valid span")
    if match_loudness:
        target, current = rms(base[start:end]), rms(insert)
        if current > 1e-6 and target > 1e-6:
            insert *= target / current
    if noise is not None and len(noise):
        reps = int(np.ceil(len(insert) / len(noise)))
        insert += np.tile(noise, reps)[: len(insert)]
    fade = max(1, min(int(crossfade_s * RATE), (end - start) // 4))
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    insert[:fade] = insert[:fade] * ramp + base[start : start + fade] * (1 - ramp)
    insert[-fade:] = insert[-fade:] * ramp[::-1] + base[end - fade : end] * (1 - ramp[::-1])
    base[start:end] = insert
    return np.clip(base, -1, 1)


def fit_whole(pcm, samples, ffmpeg, max_ratio=0.15):
    """Đưa cả câu TTS về đúng độ dài clip: co giãn trong ±max_ratio, phần dư đệm lặng / cắt."""
    pcm = trim_silence(pcm)
    ratio = len(pcm) / samples
    target = int(round(len(pcm) / min(max(ratio, 1 - max_ratio), 1 + max_ratio)))
    out, _ = time_stretch(pcm, target, ffmpeg)
    if len(out) >= samples:
        return out[:samples], ratio
    pad = samples - len(out)
    return np.pad(out, (pad // 2, pad - pad // 2)), ratio
