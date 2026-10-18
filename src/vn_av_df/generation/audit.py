"""Kiểm lối tắt của bộ dữ liệu đã sinh: đặc trưng "ngây thơ" một mình có đoán được nhãn không?

Smeu et al. (CVPR 2025) cho thấy FakeAVCeleb và AV-Deepfake1M có lối tắt: fake mở đầu bằng
khoảng lặng, một bộ phân loại chỉ nhìn khoảng lặng đạt > 98 %. Ở đây đo cho từng file: khoảng lặng
đầu / cuối, tỉ lệ khung lặng, độ to trung bình, độ dài; AUC của TỪNG đặc trưng (không train) với
nhãn AI, cả bộ và theo từng ô so với ① real. |AUC − 0,5| > 0,2 (AUC ngoài [0,3; 0,7]) → cờ đỏ: sửa
cách sinh trước khi train. Kết quả: `<dataset>/shortcut_audit.json`.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

FRAME = 320  # 20 ms ở 16 kHz
FEATURES = ("lead_silence_s", "trail_silence_s", "silence_ratio", "loudness_db", "duration_s")
FLAG = 0.2


def audio_features(pcm):
    pcm = np.asarray(pcm, np.float64)
    frames = pcm[: len(pcm) // FRAME * FRAME].reshape(-1, FRAME)
    if not len(frames):
        return dict.fromkeys(FEATURES, 0.0)
    db = 20 * np.log10(np.sqrt((frames**2).mean(1)) + 1e-9)
    loud = db > db.max() - 30  # lặng = thấp hơn khung to nhất 30 dB
    first = int(np.argmax(loud))
    last = len(loud) - 1 - int(np.argmax(loud[::-1]))
    return {
        "lead_silence_s": first * FRAME / 16000,
        "trail_silence_s": (len(loud) - 1 - last) * FRAME / 16000,
        "silence_ratio": float(1 - loud.mean()),
        "loudness_db": float(db[loud].mean()),
        "duration_s": len(pcm) / 16000,
    }


def auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def audit(out, rows, reader=None):
    """rows = manifest đã finalize; reader(path) → pcm 16 kHz (mặc định giải mã bằng PyAV)."""
    if reader is None:
        from vn_av_df.features.encoders import audio_16k as reader
    feats = {r["sample_id"]: audio_features(reader(Path(out) / r["video"])) for r in rows}
    labels = [r["label"] for r in rows]
    report = {"samples": len(rows), "flag_if_auc_outside": [0.5 - FLAG, 0.5 + FLAG], "overall": {}}
    for name in FEATURES:
        report["overall"][name] = auc(labels, [feats[r["sample_id"]][name] for r in rows])
    reals = [r for r in rows if r.get("cell") == "real"]
    by_cell = defaultdict(list)
    for r in rows:
        if r.get("cell") != "real":
            by_cell[f"{r['cell']}/{r.get('kind')}"].append(r)
    report["by_cell_vs_real"] = {}
    for group, members in sorted(by_cell.items()):
        values = {}
        for name in FEATURES:
            values[name] = auc(
                [1] * len(members) + [0] * len(reals),
                [feats[r["sample_id"]][name] for r in members + reals],
            )
        report["by_cell_vs_real"][group] = values
    flags = [
        {"group": group, "feature": name, "auc": value}
        for group, values in [("overall", report["overall"]), *report["by_cell_vs_real"].items()]
        for name, value in values.items()
        if value is not None and abs(value - 0.5) > FLAG
    ]
    report["flags"] = flags
    report["means"] = {
        cell: {
            name: float(
                np.mean([feats[r["sample_id"]][name] for r in rows if r.get("cell") == cell])
            )
            for name in FEATURES
        }
        for cell in sorted({r.get("cell") for r in rows})
    }
    return report
