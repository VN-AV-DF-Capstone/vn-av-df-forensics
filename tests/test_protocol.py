import numpy as np
import pytest

from vn_av_df.dataset import temporal_targets, validate_rows


def row(sid="a", split="train", label=0):
    return {
        "sample_id": sid,
        "split": split,
        "label": label,
        "duration_s": 4.0,
        "fake_intervals": [] if label == 0 else [[1, 2]],
        "review_status": "keep",
        "source_id": sid,
        "speaker_id": sid,
        "sha256": sid[0] * 64,
        "generator": "none" if label == 0 else "wav2lip_gan",
        "generator_version": "fixture",
    }


def test_transitive_donor_leakage():
    rows = [row("a"), row("b", "test", 1)]
    rows[1]["parent_ids"] = ["a"]
    with pytest.raises(ValueError, match="leakage"):
        validate_rows(rows)


def test_modality_union_and_reviewed_repost_leakage():
    r = row(label=1)
    r.update(
        synthetic_audio=False,
        synthetic_visual=True,
        audio_fake_intervals=[],
        visual_fake_intervals=[[1, 2]],
    )
    validate_rows([r])
    r["visual_fake_intervals"] = [[1, 3]]
    with pytest.raises(ValueError, match="Union"):
        validate_rows([r])
    copies = [row("a"), row("b", "test")]
    for item in copies:
        item["duplicate_group_id"] = "reviewed-repost"
    with pytest.raises(ValueError, match="leakage"):
        validate_rows(copies)


def test_unknown_intervals():
    r = row(label=1)
    r["fake_intervals"] = None
    validate_rows([r])
    assert np.all(temporal_targets(r, np.arange(10) * 0.2, 0.2) == -1)
    r = row()
    r["fake_intervals"] = None
    with pytest.raises(ValueError, match="Real"):
        validate_rows([r])


def test_short_fake_fraction():
    r = row(label=1)
    r["fake_intervals"] = [[0.05, 0.1]]
    y = temporal_targets(r, np.arange(4) * 0.2, 0.2)
    assert y[0] == pytest.approx(0.25)
