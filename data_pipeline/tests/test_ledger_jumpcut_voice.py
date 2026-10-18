import numpy as np
import pytest

from vn_av_data.data import jumpcut
from vn_av_data.features.voice import BANKS, fbank, mel_banks
from vn_av_data.ledger import Ledger


def test_ledger_transaction_rolls_back_and_backups(ws, ledger):
    ledger.insert("inputs", {"url": "u1", "kind": "video", "speakers": "1"})
    with pytest.raises(KeyboardInterrupt):
        with ledger.tx() as conn:
            ledger.insert("inputs", {"url": "u2", "kind": "video", "speakers": "1"}, conn=conn)
            with ledger.tx():  # lồng nhau thuộc giao dịch ngoài
                ledger.insert("inputs", {"url": "u3", "kind": "video", "speakers": "1"})
            raise KeyboardInterrupt
    assert ledger.scalar("SELECT COUNT(*) FROM inputs") == 1
    vector = np.arange(4, dtype=np.float32)
    ledger.insert(
        "speakers", {"speaker_id": "spk_0001", "face_sum": vector, "photos": [{"video": "v"}]}
    )
    row = ledger.row("SELECT * FROM speakers")
    assert np.allclose(row["face_sum"], vector) and row["photos"] == [{"video": "v"}]
    backup = ledger.backup("test")
    assert backup.is_file()
    assert (ws.ledger.parent / "csv/speakers.csv").is_file()
    copy = Ledger(backup)
    assert copy.scalar("SELECT COUNT(*) FROM speakers") == 1
    copy.close()


def track(frames, jump_at=None, drift=0.2, turn=None):
    """Landmark tổng hợp: trôi chậm; jump_at = dịch 60 px trong 1 frame; turn = quay đầu 8 frame."""
    base = np.array([[30, 100], [70, 100], [50, 125], [35, 150], [65, 150]], np.float32)
    rng = np.random.default_rng(0)
    out, crops, x = [], [], 200.0
    for t in range(frames):
        x += drift
        if jump_at is not None and t == jump_at:
            x += 60
        if turn is not None and turn <= t < turn + 8:
            x += 7  # di chuyển nhanh nhưng liên tục qua 8 frame
        out.append(base + [x, 0] + rng.normal(0, 0.15, base.shape))
        level = 100 + (60 if jump_at is not None and t >= jump_at else 0)
        crops.append(np.full((32, 32), level, np.float32) + rng.normal(0, 1, (32, 32)))
    return np.asarray(out, np.float32), np.asarray(crops, np.float32)


def test_jumpcut_detects_single_frame_jump_but_not_smooth_motion():
    lm, crops = track(150, jump_at=80)
    found = jumpcut.detect(lm, crops)
    assert found["confident"] == [pytest.approx(80 / 25, abs=0.05)]
    lm, crops = track(150, drift=1.5)
    assert jumpcut.detect(lm, crops) == {"confident": [], "suspect": []}
    lm, crops = track(150, turn=60)
    assert jumpcut.detect(lm, crops)["confident"] == []


def test_jumpcut_evaluate_on_sequences():
    lm, crops = track(400)
    # Mặt / ảnh thay đổi theo thời gian để đoạn bị xoá tạo ra bước nhảy.
    lm = lm + np.linspace(0, 400, 400)[:, None, None].astype(np.float32) * [1, 0]
    crops = crops + np.linspace(0, 200, 400)[:, None, None].astype(np.float32)
    report = jumpcut.evaluate([{"landmarks": lm, "crops": crops}], samples=40, seed=1)
    assert report["positives"] == 20 and report["negatives"] == 20
    assert report["recall_any"] > 0.5
    assert report["false_positive_confident"] == 0


def test_fbank_shape_and_mel_banks():
    assert BANKS.shape == (80, 257)
    assert np.allclose(BANKS[:, -1], 0)
    assert mel_banks(40).shape == (40, 257)
    pcm = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    feats = fbank(pcm)
    assert feats.shape == (98, 80)
    assert np.isfinite(feats).all()
    with pytest.raises(ValueError):
        fbank(pcm[:100])
