"""Speaker embedding từ tiếng của clip: WeSpeaker ResNet34-LM (ONNX, VoxCeleb, CC-BY-4.0).

Đặc trưng đầu vào là log-mel fbank 80 chiều theo Kaldi (cửa sổ Hamming 25 ms, bước 10 ms,
pre-emphasis 0,97, bỏ DC, rồi trừ trung bình theo thời gian) - đúng cách WeSpeaker trích khi
train. Viết bằng numpy để data pipeline không cần torch; test so khớp với torchaudio.
"""

import numpy as np

from vn_av_data.common.runtime import require_file

SAMPLE_RATE = 16000
FRAME, SHIFT, NFFT, MELS = 400, 160, 512, 80
EPS = np.finfo(np.float32).eps


def mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, np.float64) / 700.0)


def mel_banks(bins=MELS, nfft=NFFT, rate=SAMPLE_RATE, low=20.0, high=None):
    high = rate / 2 if not high else high
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (bins + 1)
    freqs = mel(rate / nfft * np.arange(nfft // 2))
    left = lo + np.arange(bins)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (freqs[None] - left) / (center - left)
    down = (right - freqs[None]) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))  # cột Nyquist = 0 như Kaldi


BANKS = mel_banks()
WINDOW = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))


def fbank(pcm):
    """pcm float [-1, 1] 16 kHz → (T, 80) log-mel (chưa trừ trung bình)."""
    x = np.asarray(pcm, np.float64) * 32768.0
    if len(x) < FRAME:
        raise ValueError("Audio shorter than one 25 ms frame")
    count = 1 + (len(x) - FRAME) // SHIFT
    index = np.arange(FRAME)[None] + SHIFT * np.arange(count)[:, None]
    frames = x[index]
    frames = frames - frames.mean(1, keepdims=True)
    frames = np.concatenate(
        [frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1
    )
    frames = frames * WINDOW
    power = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2
    return np.log(np.maximum(power @ BANKS.T, EPS)).astype(np.float32)


class VoiceEmbedder:
    def __init__(self, path, threads=2):
        import onnxruntime as ort

        require_file(path, "WeSpeaker model; run 00_setup.py")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def embed(self, pcm):
        feats = fbank(pcm)
        feats = feats - feats.mean(0, keepdims=True)
        (vector,) = self.session.run(None, {"feats": feats[None]})
        vector = vector.reshape(-1).astype(np.float32)
        return vector / max(1e-8, float(np.linalg.norm(vector)))


def clip_pcm(path):
    """Tiếng mono 16 kHz của cả clip."""
    import av

    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
