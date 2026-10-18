import json

import pytest
from conftest import CONFIG, add_input, add_raw_video

from vn_av_data.cli import main, pick_batch
from vn_av_data.data import archive


def test_raw_archive_requires_finished_batch_then_tracks_location(ws, ledger):
    one = add_input(ledger, "u", "1")
    add_raw_video(ws, ledger, "aaaaaaaaaaa", one, 1, seconds=3)
    with pytest.raises(ValueError, match="not downloaded/cut/identified"):
        archive.pack_raw(ws, ledger, 1)
    ledger.update("videos", "video_id", "aaaaaaaaaaa", {"cut_status": "done", "id_status": "done"})
    packed = archive.pack_raw(ws, ledger, 1)
    assert (ws.archive / packed["zip"]).is_file()
    assert (ws.raw(1) / "SHA256SUMS.txt").read_text().split()[1] == "videos/aaaaaaaaaaa.mp4"
    with pytest.raises(ValueError, match="--uploaded"):
        archive.delete_local(ws, ledger, "raw", 1)
    archive.mark_uploaded(ws, ledger, "raw", 1)
    archive.delete_local(ws, ledger, "raw", 1)
    assert not ws.raw(1).exists() and not (ws.archive / packed["zip"]).exists()
    row = ledger.row("SELECT * FROM videos WHERE video_id='aaaaaaaaaaa'")
    assert row["raw_location"] == "drive:" + packed["zip"] and row["sha256"]


def test_cli_status_backup_and_batch_pick(ws, ledger, capsys):
    one = add_input(ledger, "u", "1")
    add_raw_video(ws, ledger, "aaaaaaaaaaa", one, 2, seconds=3)
    assert pick_batch(ledger, "cut") == 2
    with pytest.raises(ValueError):
        pick_batch(ledger, "identify")
    args = ["--config", str(CONFIG), "--data", str(ws.root), "--exports", str(ws.exports)]
    args += ["--dataset", "vn-test"]
    assert main(args + ["status"]) == 0
    out = capsys.readouterr().out
    assert "Batch" in out and "Ổ đĩa" in out
    assert main(args + ["parts"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["created"] == []
    assert list((ws.ledger.parent / "backups").glob("ledger_*_parts.sqlite"))


def test_pool_cleanup_only_touches_decided_speakers(ws, ledger):
    one = add_input(ledger, "u", "1")
    add_raw_video(ws, ledger, "aaaaaaaaaaa", one, 1, seconds=3)
    for sid, review in (("s_done", "done"), ("s_out", "excluded"), ("s_wait", "pending")):
        ledger.insert("speakers", {"speaker_id": sid, "review": review})
    files = {}
    for cid, sid, decision in (
        ("c_keep", "s_done", "keep"),
        ("c_left", "s_done", "pending"),
        ("c_out", "s_out", "reject"),
        ("c_wait", "s_wait", "reject"),
    ):
        path = ws.clips / "aaaaaaaaaaa" / f"{cid}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1000)
        files[cid] = path
        ledger.insert(
            "clips",
            {
                "clip_id": cid,
                "video_id": "aaaaaaaaaaa",
                "speaker_id": sid,
                "decision": decision,
                "batch": 1,
                "file": path.relative_to(ws.root).as_posix(),
                "size_mb": 0.001,
            },
        )
    # Raw batch 1 chưa lên Drive: chưa xoá gì.
    kept = archive.clean_pool(ws, ledger)
    assert kept["deleted_clips"] == 0 and kept["kept_until_raw_uploaded"]["batches"] == [1]
    assert all(p.exists() for p in files.values())
    ledger.update("batches", "batch", 1, {"location": "drive:raw_batch_001.zip"})
    assert archive.clean_pool(ws, ledger)["deleted_clips"] == 2
    assert not files["c_left"].exists() and not files["c_out"].exists()
    assert files["c_keep"].exists() and files["c_wait"].exists()
    where = dict(
        (r["clip_id"], r["location"]) for r in ledger.rows("SELECT clip_id, location FROM clips")
    )
    assert where == {"c_keep": "local", "c_left": "deleted", "c_out": "deleted", "c_wait": "local"}
