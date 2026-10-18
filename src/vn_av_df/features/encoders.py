"""Encoder của v003 (ngoài AV-HuBERT đã có): XLS-R cho audio, crop miệng độ phân giải gốc.

Mọi cache là npz + json cùng tên sample, lưới 25 Hz (`times_s`), kèm `feature_signature`
(asset + code + tuỳ chọn) và `source_fingerprint` (hash video) để không dùng nhầm cache cũ.

- xlsr: facebook/wav2vec2-xls-r-300m (128 ngôn ngữ, có tiếng Việt), trung bình hidden state các
  lớp [layer_low, layer_high], 50 Hz → 25 Hz (trung bình từng cặp) → `ssl` [T, 1024] float16.
- mouth: hộp miệng + landmark dlib từ cache AV-HuBERT (khung ≤ 640 px) quy về khung gốc 720p,
  crop xám `crop` × `crop` (mặc định 96, gần kích thước thật của miệng ở 720p, không phóng lên
  224) → `mouth` [T, crop, crop] uint8; `lips` [T, 20, 2] landmark môi (48-67) chuẩn hoá theo
  tâm miệng và bề rộng hai mép (đầu vào kiểu BioLip).
"""

from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import fingerprint, read_json, require_file, save_npz, sha, write_json


def source_fingerprint(video):
    return fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})


def fresh(output, signature, source):
    meta_path = Path(output).with_suffix(".json")
    if Path(output).is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        return (
            meta.get("feature_signature") == signature
            and meta.get("source_fingerprint") == source
            and meta.get("feature_sha256") == sha(output)
        )
    return False


def audio_16k(video):
    import av

    chunks = []
    with av.open(str(video)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class XLSREncoder:
    kind = "xlsr"

    def __init__(self, cfg, model=None):
        self.cfg = dict(cfg)
        self.low, self.high = cfg.get("layer_low", 6), cfg.get("layer_high", 12)
        self.device = cfg.get("device", "cpu")
        if model is None:
            import torch
            from transformers import Wav2Vec2Model

            folder = Path(cfg["model"])
            weights = folder / "model.safetensors"
            if not weights.is_file():
                weights = folder / "pytorch_model.bin"
            assets = {
                "config": sha(require_file(folder / "config.json", "XLS-R config")),
                "weights": sha(require_file(weights, "XLS-R weights")),
            }
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            model = Wav2Vec2Model.from_pretrained(str(folder), torch_dtype=dtype)
        else:
            assets = {"model": "test-fixture"}
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.signature = fingerprint(
            {
                "assets": assets,
                "code": sha(__file__),
                "layers": [self.low, self.high],
                "format": "xlsr-25hz-v1",
            }
        )

    def extract(self, request, output):
        import torch

        video = Path(request["video"])
        source = source_fingerprint(video)
        if fresh(output, self.signature, source):
            return read_json(Path(output).with_suffix(".json"))
        pcm = audio_16k(video)
        frames = int(round(len(pcm) / 640))
        x = torch.from_numpy((pcm - pcm.mean()) / (pcm.std() + 1e-7))[None]
        x = x.to(self.device, dtype=next(self.model.parameters()).dtype)
        with torch.no_grad():
            hidden = self.model(x, output_hidden_states=True).hidden_states
        layers = torch.stack(hidden[self.low : self.high + 1]).mean(0)[0].float().cpu().numpy()
        if len(layers) % 2:
            layers = np.vstack([layers, layers[-1:]])
        ssl = layers.reshape(-1, 2, layers.shape[1]).mean(1)
        ssl = np.pad(ssl, ((0, max(0, frames - len(ssl))), (0, 0)), mode="edge")[:frames]
        save_npz(
            output,
            ssl=ssl.astype(np.float16),
            times_s=np.arange(frames) / 25,
            valid=np.ones(frames, bool),
        )
        meta = {
            "format": "xlsr-25hz-v1",
            "step_s": 0.04,
            "duration_s": frames / 25,
            "feature_signature": self.signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta


def native_frames(video, count):
    """Frame xám ở độ phân giải gốc trên lưới 25 Hz (đúng count frame)."""
    import av
    import cv2

    frames, grid, origin = [], 0.0, None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            t = float(frame.pts * frame.time_base)
            origin = t if origin is None else origin
            if t - origin < grid - 0.02:
                continue
            frames.append(cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY))
            grid += 0.04
            if len(frames) == count:
                break
    while frames and len(frames) < count:
        frames.append(frames[-1])
    return frames


def crop_square(gray, cx, cy, side, size):
    import cv2

    scale = size / side
    matrix = np.array([[scale, 0, size / 2 - scale * cx], [0, scale, size / 2 - scale * cy]])
    return cv2.warpAffine(
        gray,
        matrix,
        (size, size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def normalized_lips(landmarks):
    """[T,68,2] → [T,20,2]: môi 48-67, trừ tâm miệng, chia bề rộng hai mép (48-54)."""
    lips = landmarks[:, 48:68].astype(np.float32)
    center = lips.mean(1, keepdims=True)
    width = np.linalg.norm(lips[:, 6] - lips[:, 0], axis=1)[:, None, None]
    return ((lips - center) / np.maximum(width, 1e-3)).astype(np.float16)


class MouthEncoder:
    kind = "mouth"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.size = int(cfg.get("crop", 96))
        self.signature = fingerprint(
            {
                "code": sha(__file__),
                "crop": self.size,
                "box_max_side": cfg.get("box_max_side", 640),
                "format": "mouth-native-v1",
            }
        )

    def extract(self, request, output):
        video = Path(request["video"])
        source = source_fingerprint(video)
        box_path = Path(request["box_cache"])
        box_meta = read_json(box_path.with_suffix(".json"))
        signature = fingerprint([self.signature, box_meta["feature_sha256"]])
        if fresh(output, signature, source):
            return read_json(Path(output).with_suffix(".json"))
        with np.load(box_path, allow_pickle=False) as z:
            boxes = z["mouth_boxes"].astype(np.float32)
            landmarks = z["landmarks"].astype(np.float32)
            valid = (z["visual_valid"] & (boxes[:, 2] > 0)).copy()
        frames = native_frames(video, len(boxes))
        if not frames:
            raise ValueError(f"No frames decoded: {video}")
        height, width = frames[0].shape
        scale = max(height, width) / min(max(height, width), self.cfg.get("box_max_side", 640))
        mouth = np.zeros((len(boxes), self.size, self.size), np.uint8)
        for i, (cx, cy, side) in enumerate(boxes):
            if valid[i]:
                mouth[i] = crop_square(frames[i], cx * scale, cy * scale, side * scale, self.size)
        save_npz(
            output,
            mouth=mouth,
            lips=normalized_lips(landmarks),
            valid=valid,
            times_s=np.arange(len(boxes)) / 25,
            native_side=np.float32(boxes[:, 2].mean() * scale),
        )
        meta = {
            "format": "mouth-native-v1",
            "step_s": 0.04,
            "duration_s": len(boxes) / 25,
            "feature_signature": signature,
            "feature_sha256": sha(output),
            "source_fingerprint": source,
            "box_sha256": box_meta["feature_sha256"],
        }
        write_json(Path(output).with_suffix(".json"), meta)
        return meta
