"""Nạp cache v003 thành tensor theo sample (lưới 25 Hz) và gom batch có đệm + mask.

Nhãn theo frame: visual / audio / mismatch = tỉ lệ frame nằm trong khoảng tương ứng; nhãn video:
label (bị AI), synthetic_visual, synthetic_audio, có mismatch.
Augment độ lệch tiếng-hình (chỉ khi train): dịch đặc trưng tiếng (AV-HuBERT audio, XLS-R) so với
hình k frame, k ngẫu nhiên đều trong cfg["training"]["audio_lag_aug"] (mặc định −1…+2; dương =
tiếng trễ), như nhau cho real và fake, nhãn giữ nguyên; frame mất tiếng ở mép bị mask. Real đã
được sửa độ lệch về 0 ở data pipeline nhưng còn sai số ±1 frame và mỗi generator có "điểm 0" riêng;
khoảng −1…+2 nằm trong vùng người xem không nhận ra lệch (ITU-R BT.1359: tiếng sớm ~45 ms, trễ
~125 ms). Căn cứ: docs/v003_model_research.md §9.
Tham chiếu người (enrollment): R clip real của cùng speaker trong cùng split. Khi train lấy ngẫu
nhiên (bỏ clip gốc của chính sample); khi đánh giá cố định R clip real đầu tiên theo sample_id
và các clip đó không được tính điểm ở chế độ "ref".
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vn_av_df.dataset import temporal_targets

KEYS = {
    "visual": "visual_fake_intervals",
    "audio": "audio_fake_intervals",
    "mismatch": "av_mismatch_intervals",
}


def frame_targets(row, frames):
    times = np.arange(frames) / 25
    out = {name: temporal_targets(row, times, 0.04, key) for name, key in KEYS.items()}
    out["ai"] = temporal_targets(row, times, 0.04, "fake_intervals")
    return {k: np.clip(v, 0, 1).astype(np.float32) for k, v in out.items()}


def video_targets(row):
    return {
        "ai": float(row["label"]),
        "visual": float(bool(row.get("synthetic_visual"))),
        "audio": float(bool(row.get("synthetic_audio"))),
        "mismatch": float(bool(row.get("av_mismatch_intervals"))),
    }


AUDIO_KEYS = ("a", "ssl")


def lag_audio(item, lag):
    """Tiếng trễ `lag` frame so với hình (lag < 0: sớm); đệm 0 ở mép và mask frame đó."""
    if not lag:
        return item
    keys = [k for k in AUDIO_KEYS if k in item]
    if not keys:
        return item
    n = len(item[keys[0]])
    for key in keys:
        x = item[key]
        pad = np.zeros((abs(lag), *x.shape[1:]), x.dtype)
        item[key] = np.concatenate([pad, x[:-lag]]) if lag > 0 else np.concatenate([x[-lag:], pad])
    keep = np.ones(n, bool)
    if lag > 0:
        keep[:lag] = False
    else:
        keep[lag:] = False
    item["lag_valid"] = keep
    return item


def augment_mouth(mouth, rng):
    """Xử lý lành tính áp như nhau cho real và fake (cùng tham số cho cả clip)."""
    import cv2

    out = mouth.copy()
    size = out.shape[1]
    quality = rng.randint(30, 95) if rng.random() < 0.7 else None
    factor = rng.uniform(0.5, 1.0) if rng.random() < 0.5 else None
    gamma = rng.uniform(0.7, 1.4) if rng.random() < 0.5 else None
    sharpen = rng.random() < 0.2
    blur = rng.uniform(0.3, 1.0) if not sharpen and rng.random() < 0.3 else None
    table = None if gamma is None else (((np.arange(256) / 255.0) ** gamma) * 255).astype(np.uint8)
    for i in range(len(out)):
        x = out[i]
        if factor:
            small = max(8, int(size * factor))
            x = cv2.resize(
                cv2.resize(x, (small, small), interpolation=cv2.INTER_AREA),
                (size, size),
                interpolation=cv2.INTER_LINEAR,
            )
        if blur:
            x = cv2.GaussianBlur(x, (0, 0), blur)
        if sharpen:
            x = cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x, (0, 0), 1.0), -0.5, 0)
        if table is not None:
            x = cv2.LUT(x, table)
        if quality:
            ok, data = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, quality])
            x = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if ok else x
        out[i] = x
    return out


def enrollment(rows, count):
    """Mỗi (split, speaker): `count` clip real đầu tiên theo sample_id."""
    reals = defaultdict(list)
    for r in rows:
        if r.get("cell") == "real" and r.get("speaker_id"):
            reals[(r["split"], r["speaker_id"])].append(r)
    return {k: sorted(v, key=lambda r: r["sample_id"])[:count] for k, v in reals.items()}


class BenchData(torch.utils.data.Dataset):
    """needs ⊆ {"av", "ssl", "mouth"}; references = số clip tham chiếu mỗi sample (0 = tắt)."""

    def __init__(
        self,
        cfg,
        rows,
        needs,
        augment=False,
        lag_augment=False,
        references=0,
        ref_frames=50,
        fixed_references=False,
        seed=0,
        crop_frames=None,
    ):
        self.cache = Path(cfg["cache"])
        self.rows, self.needs, self.augment = list(rows), set(needs), augment
        self.lag_augment = lag_augment
        self.references, self.ref_frames, self.seed = references, ref_frames, seed
        self.crop_frames = crop_frames  # train: cắt ngẫu nhiên một đoạn để tiết kiệm bộ nhớ
        self.fixed = enrollment(self.rows, references) if fixed_references else None
        pools = defaultdict(list)
        for r in self.rows:
            if r.get("cell") == "real" and r.get("speaker_id"):
                pools[(r["split"], r["speaker_id"])].append(r)
        self.pools = pools
        self.epoch = 0
        self.lag_range = tuple((cfg.get("training") or {}).get("audio_lag_aug") or (0, 0))

    def __len__(self):
        return len(self.rows)

    def load(self, row, rng=None, frames=None):
        sid = row["sample_id"]
        item, lengths = {}, []
        if "av" in self.needs:
            with np.load(self.cache / "avhubert" / f"{sid}.npz", allow_pickle=False) as z:
                item["a"], item["v"] = z["audio"].astype(np.float32), z["visual"].astype(np.float32)
                item["av_valid"] = (z["audio_valid"] & z["visual_valid"]).copy()
            lengths.append(len(item["a"]))
        if "ssl" in self.needs:
            with np.load(self.cache / "xlsr" / f"{sid}.npz", allow_pickle=False) as z:
                item["ssl"] = z["ssl"].astype(np.float32)
            lengths.append(len(item["ssl"]))
        if "mouth" in self.needs:
            with np.load(self.cache / "mouth" / f"{sid}.npz", allow_pickle=False) as z:
                item["mouth"], item["lips"] = z["mouth"].copy(), z["lips"].astype(np.float32)
                item["mouth_valid"] = z["valid"].copy()
            if self.augment and rng is not None:
                item["mouth"] = augment_mouth(item["mouth"], rng)
            lengths.append(len(item["mouth"]))
        if self.lag_augment and rng is not None and self.lag_range != (0, 0):
            item = lag_audio(item, rng.randint(*self.lag_range))
        length = min(lengths + ([frames] if frames else []))
        item = {k: v[:length] for k, v in item.items()}
        valid = np.ones(length, bool)
        for key in ("av_valid", "mouth_valid", "lag_valid"):
            if key in item:
                valid &= item[key]
        item["valid"] = valid
        return item, length

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{row['sample_id']}")
        item, length = self.load(row, rng)
        targets = frame_targets(row, length)
        item.update({f"y_{k}": v for k, v in targets.items()})
        if self.crop_frames and length > self.crop_frames:
            start = rng.randrange(length - self.crop_frames + 1)
            item = {k: v[start : start + self.crop_frames] for k, v in item.items()}
        item["index"] = index
        item["video"] = video_targets(row)
        if self.references:
            key = (row["split"], row.get("speaker_id"))
            if self.fixed is not None:
                refs = self.fixed.get(key, [])
            else:
                pool = [
                    r
                    for r in self.pools.get(key, [])
                    if r["source_clip_id"] != row["source_clip_id"]
                ]
                refs = rng.sample(pool, min(self.references, len(pool)))
            item["refs"] = [self.load(r, None, self.ref_frames)[0] for r in refs[: self.references]]
        return item


def collate(batch):
    """Đệm theo T dài nhất; mask = valid & trong độ dài thật."""
    size = max(len(b["valid"]) for b in batch)

    def stack(key, dtype=torch.float32):
        arrays = [b[key] for b in batch]
        shape = (len(arrays), size, *arrays[0].shape[1:])
        out = np.zeros(shape, arrays[0].dtype)
        for i, x in enumerate(arrays):
            out[i, : len(x)] = x
        return torch.as_tensor(out, dtype=dtype)

    out = {"mask": stack("valid", torch.bool), "index": torch.tensor([b["index"] for b in batch])}
    for key in ("a", "v", "ssl", "lips"):
        if key in batch[0]:
            out[key] = stack(key)
    if "mouth" in batch[0]:
        out["mouth"] = stack("mouth", torch.uint8)
    for key in ("y_ai", "y_visual", "y_audio", "y_mismatch"):
        out[key] = stack(key)
    out["video"] = {k: torch.tensor([b["video"][k] for b in batch]) for k in batch[0]["video"]}
    if "refs" in batch[0]:
        count = max(len(b["refs"]) for b in batch)
        out["ref_count"] = torch.tensor([len(b["refs"]) for b in batch])
        if count:
            flat = [r for b in batch for r in b["refs"]] or []
            ref_size = max(len(r["valid"]) for r in flat)
            refs = {}
            for key in ("a", "v", "ssl", "mouth", "lips", "valid"):
                if key not in flat[0]:
                    continue
                sample = flat[0][key]
                dtype = sample.dtype
                tensor = np.zeros((len(batch), count, ref_size, *sample.shape[1:]), dtype)
                for i, b in enumerate(batch):
                    for j, r in enumerate(b["refs"]):
                        tensor[i, j, : len(r[key])] = r[key]
                refs[key] = torch.as_tensor(tensor)
            out["refs"] = refs
    return out


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch.to(device) if torch.is_tensor(batch) else batch
