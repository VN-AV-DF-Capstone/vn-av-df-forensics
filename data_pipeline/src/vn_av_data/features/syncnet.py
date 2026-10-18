"""Đo độ lệch tiếng-hình của clip real bằng SyncNet (Chung & Zisserman, "Out of Time", 2016).

Weight `syncnet_v2.model` và kiến trúc lấy từ syncnet_python (MIT, (c) 2016-present Joon Son
Chung); chạy bằng torch trên CPU (~3 s/clip). MFCC viết lại bằng numpy theo đúng
python_speech_features 0.6 (MIT) mà SyncNet dùng khi train: cửa sổ chữ nhật 25 ms, bước 10 ms,
pre-emphasis 0,97, 26 mel, 13 hệ số, lifter 22, c0 = log năng lượng.

Mỗi clip: hộp mặt YuNet mỗi frame (làm mượt median 13 frame như run_pipeline.py) → crop 224
(mở rộng 40%) → đặc trưng 5 frame hình / 0,2 s MFCC → khoảng cách theo độ lệch -15..+15 frame.
Quy ước: `audio_lag` > 0 = tiếng trễ hơn hình (đo được bằng cách dịch tiếng biết trước).
Độ lệch của một video = đỉnh của đường cong trung bình mọi clip đo; tin được khi đỉnh đủ nhọn
và đa số clip đồng ý (±1 frame). Lý do và ngưỡng: docs/v003_model_research.md §9.
"""

import numpy as np

VSHIFT = 15
CROP_SCALE = 0.4
SAMPLES_PER_FRAME = 640  # 16 kHz / 25 fps


# ---------------------------------------------------------------- MFCC (python_speech_features)
def _hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def _filterbanks(nfilt=26, nfft=512, rate=16000):
    points = np.linspace(_hz2mel(0), _hz2mel(rate / 2), nfilt + 2)
    bins = np.floor((nfft + 1) * _mel2hz(points) / rate)
    bank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            bank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            bank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return bank


FILTERS = _filterbanks()


def mfcc(signal, rate=16000, frame=400, step=160, nfft=512, numcep=13, lifter=22):
    """[frames, 13]; giống python_speech_features.mfcc với tham số mặc định."""
    from scipy.fftpack import dct

    signal = np.asarray(signal, np.float64)
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    count = 1 if len(signal) <= frame else 1 + int(np.ceil((len(signal) - frame) / step))
    padded = np.concatenate([signal, np.zeros((count - 1) * step + frame - len(signal))])
    index = np.arange(frame)[None] + (np.arange(count) * step)[:, None]
    power = np.square(np.abs(np.fft.rfft(padded[index], nfft))) / nfft
    energy = power.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    feat = power @ FILTERS.T
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = dct(np.log(feat), type=2, axis=1, norm="ortho")[:, :numcep]
    feat *= 1 + (lifter / 2.0) * np.sin(np.pi * np.arange(numcep) / lifter)
    feat[:, 0] = np.log(energy)
    return feat


# ---------------------------------------------------------------- model
def torch_model():
    """Kiến trúc S của syncnet_python (tên lớp giữ nguyên để nạp state_dict gốc)."""
    import torch.nn as nn

    def conv2(i, o, k, s=(1, 1), p=(1, 1)):
        return [nn.Conv2d(i, o, k, stride=s, padding=p), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    def conv3(i, o, k, s=(1, 1, 1), p=0):
        return [nn.Conv3d(i, o, k, stride=s, padding=p), nn.BatchNorm3d(o), nn.ReLU(inplace=True)]

    def fc():
        return nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024)
        )

    class S(nn.Module):
        def __init__(self):
            super().__init__()
            self.netcnnaud = nn.Sequential(
                *conv2(1, 64, (3, 3)),
                nn.MaxPool2d((1, 1), (1, 1)),
                *conv2(64, 192, (3, 3)),
                nn.MaxPool2d((3, 3), (1, 2)),
                *conv2(192, 384, (3, 3)),
                *conv2(384, 256, (3, 3)),
                *conv2(256, 256, (3, 3)),
                nn.MaxPool2d((3, 3), (2, 2)),
                *conv2(256, 512, (5, 4), p=(0, 0)),
            )
            self.netfcaud, self.netfclip = fc(), fc()
            self.netcnnlip = nn.Sequential(
                *conv3(3, 96, (5, 7, 7), (1, 2, 2)),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2)),
                *conv3(96, 256, (1, 5, 5), (1, 2, 2), (0, 1, 1)),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
                *conv3(256, 256, (1, 3, 3), p=(0, 1, 1)),
                *conv3(256, 256, (1, 3, 3), p=(0, 1, 1)),
                *conv3(256, 256, (1, 3, 3), p=(0, 1, 1)),
                nn.MaxPool3d((1, 3, 3), (1, 2, 2)),
                *conv3(256, 512, (1, 6, 6)),
            )

    return S()


# ---------------------------------------------------------------- đo
def face_track(frames, detector):
    """(cx, cy, nửa cạnh) mỗi frame; frame hụt lấy frame gần nhất; None nếu hụt > 20%."""
    from scipy.signal import medfilt

    boxes = []
    for image in frames:
        h, w = image.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(image)
        if faces is None or not len(faces):
            boxes.append(None)
            continue
        f = max(faces, key=lambda f: f[2] * f[3])
        boxes.append((f[0] + f[2] / 2, f[1] + f[3] / 2, max(f[2], f[3]) / 2))
    good = [i for i, b in enumerate(boxes) if b is not None]
    if len(good) < 0.8 * len(boxes):
        return None
    filled = [
        boxes[min(good, key=lambda g: abs(g - i))] if b is None else b for i, b in enumerate(boxes)
    ]
    arr = np.asarray(filled, np.float64)
    kernel = min(13, len(arr) if len(arr) % 2 else len(arr) - 1)
    return np.stack([medfilt(arr[:, j], kernel_size=kernel) for j in range(3)], 1)


def face_crops(frames, track):
    """Crop như run_pipeline.py: hộp vuông mở rộng 40% (lệch xuống dưới), đệm xám 110."""
    import cv2

    out = []
    for image, (mx, my, bs) in zip(frames, track):
        pad = int(bs * (1 + 2 * CROP_SCALE))
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), constant_values=110)
        my, mx = my + pad, mx + pad
        face = padded[
            int(my - bs) : int(my + bs * (1 + 2 * CROP_SCALE)),
            int(mx - bs * (1 + CROP_SCALE)) : int(mx + bs * (1 + CROP_SCALE)),
        ]
        out.append(cv2.resize(face, (224, 224)))
    return np.stack(out)


def distance_curve(video_feats, audio_feats):
    """Khoảng cách trung bình theo độ lệch; index i ↔ syncnet offset VSHIFT - i."""
    n = min(len(video_feats), len(audio_feats))
    v, a = video_feats[:n], audio_feats[:n]
    padded = np.concatenate([np.zeros((VSHIFT, a.shape[1]), a.dtype), a, np.zeros_like(a[:VSHIFT])])
    win = 2 * VSHIFT + 1
    windows = np.stack([padded[i : i + win] for i in range(n)])  # [n, win, d]
    diff = windows - v[:, None] + 1e-6  # như torch.pairwise_distance (eps = 1e-6)
    return np.sqrt((diff**2).sum(-1)).mean(0)


def read_curve(curve):
    """→ (audio_lag_frames, confidence = median - min)."""
    index = int(np.argmin(curve))
    return index - VSHIFT, float(np.median(curve) - curve[index])


def summarize(curves, rules):
    """Độ lệch của cả video từ đường cong từng clip."""
    if not curves:
        return {"status": "no_clips", "clips": 0}
    lags = [read_curve(c)[0] for c in curves]
    lag, conf = read_curve(np.mean(curves, 0))
    agree = float(np.mean([abs(x - lag) <= 1 for x in lags]))
    if conf < rules["min_conf"] or agree < rules["min_agree"]:
        status = "unreliable"
    else:  # đỉnh rõ, các clip đồng ý; ít clip thì vẫn sửa nhưng nhắc người duyệt xem
        status = "ok" if len(curves) >= rules["min_clips"] else "few"
    return {
        "status": status,
        "audio_lag_frames": lag,
        "conf": round(conf, 3),
        "agree": round(agree, 3),
        "clips": len(curves),
        "clip_lags": lags,
    }


class SyncMeter:
    def __init__(self, weights, face_model, threads=4):
        import cv2
        import torch

        torch.set_num_threads(threads)
        self.model = torch_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.eval()
        self.detector = cv2.FaceDetectorYN.create(str(face_model), "", (320, 320), 0.6)

    def run(self, cnn, fc, window, count, batch=32):
        """Đặc trưng cho `count` cửa sổ; `window(i)` dựng cửa sổ thứ i (dựng theo batch cho đỡ RAM)."""
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, count, batch):
                x = np.stack([window(j) for j in range(i, min(count, i + batch))])
                out.append(fc(cnn(torch.from_numpy(x)).flatten(1)).numpy())
        return np.concatenate(out)

    def curve(self, frames, pcm):
        """frames [T, H, W, 3] BGR 25 fps, pcm 16 kHz cùng mốc → đường cong, hoặc None."""
        track = face_track(frames, self.detector)
        if track is None or len(frames) < 2 * VSHIFT:
            return None
        faces = face_crops(frames, track).astype(np.float32).transpose(3, 0, 1, 2)  # [3,T,H,W]
        count, m = len(frames) - 5, self.model
        cc = mfcc(np.asarray(pcm, np.float64) * 32768.0).T.astype(np.float32)  # [13, 100 Hz]
        cc = np.pad(cc, ((0, 0), (0, max(0, count * 4 + 20 - cc.shape[1]))))
        video = self.run(m.netcnnlip, m.netfclip, lambda i: faces[:, i : i + 5], count)
        audio = self.run(m.netcnnaud, m.netfcaud, lambda i: cc[None, :, i * 4 : i * 4 + 20], count)
        return distance_curve(video, audio)

    def measure(self, path):
        from vn_av_data.data.media import decode

        media = decode(path, max_duration=10, max_side=640, sample_rate=16000)
        return self.curve(media["frames"], media["pcm"])
