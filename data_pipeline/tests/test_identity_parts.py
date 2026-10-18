import json

import cv2
import numpy as np
import pytest
from conftest import add_input, make_video, unit
from fastapi.testclient import TestClient

from vn_av_data.common.runtime import sha
from vn_av_data.contract import validate_bundle
from vn_av_data.data import archive
from vn_av_data.data.export import export_part
from vn_av_data.data.identity import identify_batch, leader_clusters
from vn_av_data.data.parts import form_parts
from vn_av_data.features.face import normalize
from vn_av_data.serving.speakers import create_app


def near(base, sim, seed):
    """Vector có cosine ≈ sim với base."""
    noise = unit(seed)
    noise = normalize(noise - (noise @ base) * base)
    return normalize(sim * base + np.sqrt(1 - sim**2) * noise)


@pytest.fixture
def clip_file(tmp_path):
    return make_video(tmp_path / "template.mp4", 4, size="320x240")


def add_clips(ws, ledger, clip_file, video_id, input_id, faces, channel="ch1", batch=1):
    if not ledger.scalar("SELECT COUNT(*) FROM videos WHERE video_id=?", (video_id,)):
        if not ledger.scalar("SELECT COUNT(*) FROM batches WHERE batch=?", (batch,)):
            ledger.insert("batches", {"batch": batch, "created_at": "now"})
        ledger.insert(
            "videos",
            {
                "video_id": video_id,
                "input_id": input_id,
                "url": "https://www.youtube.com/watch?v=" + video_id,
                "title": video_id,
                "channel": channel,
                "channel_id": channel,
                "status": "accepted",
                "batch": batch,
                "dl_status": "done",
                "cut_status": "done",
                "sha256": "0" * 64,
                "probe": {"fps": 25.0, "width": 1280, "height": 720},
            },
        )
    ids = []
    first = ledger.scalar("SELECT COUNT(*) FROM clips WHERE video_id=?", (video_id,))
    for i, face in enumerate(faces, start=first):
        cid = f"clip_{video_id}_{i:03d}"
        target = ws.clips / video_id / f"{cid}.mp4"
        # Mỗi clip một nội dung khác nhau (bundle chặn video trùng hash).
        add_clips.count = getattr(add_clips, "count", 0) + 1
        make_video(target, 4, size="160x120", freq=200 + 7 * add_clips.count)
        strip = ws.faces / video_id / f"{cid}.jpg"
        strip.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(strip), np.full((96, 480, 3), 100 + i, np.uint8))
        ledger.insert(
            "clips",
            {
                "clip_id": cid,
                "video_id": video_id,
                "input_id": input_id,
                "batch": batch,
                "start_s": 10.0 * i,
                "end_s": 10.0 * i + 4,
                "duration_s": 4.0,
                "file": target.relative_to(ws.root).as_posix(),
                "sha256": sha(target),
                "face": face,
                "face_consistency": 0.9,
            },
        )
        ids.append(cid)
    return ids


def test_leader_clusters_split_two_people():
    a, b = unit(1), unit(2)
    vectors = [near(a, 0.8, i) for i in range(5)] + [near(b, 0.8, 10 + i) for i in range(4)]
    groups = sorted(leader_clusters(vectors, 0.45), key=len)
    assert [len(g) for g in groups] == [4, 5]


def test_identify_matches_registry_by_channel_suggests_and_flags(ws, ledger, cfg, clip_file):
    cfg["identify"]["voice"] = False
    a, b, c, d = unit(1), unit(2), unit(3), unit(4)
    one = add_input(ledger, "u1", "1")
    add_clips(ws, ledger, clip_file, "v1", one, [near(a, 0.85, i) for i in range(5)] + [unit(99)])
    two = add_input(ledger, "u2", "1")
    add_clips(ws, ledger, clip_file, "v2", two, [near(a, 0.85, 20 + i) for i in range(4)])
    many = add_input(ledger, "u3", "n", "video")
    faces = [near(b, 0.85, 30 + i) for i in range(4)] + [near(c, 0.85, 40 + i) for i in range(4)]
    add_clips(ws, ledger, clip_file, "v3", many, faces + [d], channel="ch3")
    four = add_input(ledger, "u4", "1")
    b_like = near(b, 0.5, 77)
    add_clips(
        ws, ledger, clip_file, "v4", four, [near(b_like, 0.95, 50 + i) for i in range(4)], "ch4"
    )
    report = identify_batch(ws, ledger, 1, cfg)
    assert report["new_speakers"] == 4 and report["matched"] == 1
    speakers = {
        r["video_id"]: r["speaker_id"]
        for r in ledger.rows(
            "SELECT DISTINCT video_id, speaker_id FROM clips WHERE speaker_id IS NOT NULL"
        )
    }
    assert speakers["v1"] == speakers["v2"]  # cùng kênh, cùng mặt → gộp tự động
    assert ledger.scalar("SELECT assign FROM clips WHERE video_id='v2' LIMIT 1") == "auto_channel"
    assert ledger.scalar("SELECT COUNT(DISTINCT speaker_id) FROM clips WHERE video_id='v3'") == 2
    assert ledger.scalar("SELECT assign FROM clips WHERE clip_id='clip_v3_008'") == "unassigned"
    suggestion = json.loads(
        ledger.scalar("SELECT suggestion FROM speakers WHERE speaker_id=?", (speakers["v4"],))
    )
    b_speaker = ledger.scalar("SELECT speaker_id FROM clips WHERE clip_id='clip_v3_000'")
    assert suggestion["speaker_id"] == b_speaker and 0.35 <= suggestion["face_sim"] < 0.55
    assert "face_outlier" in ledger.scalar("SELECT flags FROM clips WHERE clip_id='clip_v1_005'")
    photos = ledger.row("SELECT photos FROM speakers WHERE speaker_id=?", (speakers["v1"],))[
        "photos"
    ]
    assert len(photos) == 2 and all((ws.root / p["path"]).is_file() for p in photos)
    assert ledger.scalar("SELECT COUNT(*) FROM videos WHERE id_status='done'") == 4


def make_speaker(ledger, sid, gender, order, review="done"):
    ledger.insert(
        "speakers",
        {
            "speaker_id": sid,
            "gender": gender,
            "review": review,
            "reviewed_at": f"2026-10-15T00:{order:02d}:00",
            "order_key": order,
            "face_sum": unit(order),
            "face_n": 1,
        },
    )


def keep_clips(ws, ledger, clip_file, sid, video_id, count, input_id=1):
    ids = add_clips(ws, ledger, clip_file, video_id, input_id, [unit(i) for i in range(count)])
    for cid in ids:
        ledger.update("clips", "clip_id", cid, {"speaker_id": sid, "decision": "keep"})
    return ids


def small_parts(cfg):
    cfg["parts"].update(
        speakers_per_gender=3,
        clips_per_speaker=2,
        split={"train": [1, 1], "validation": [1, 1], "test": [1, 1]},
    )
    return cfg


def test_parts_balance_gender_split_and_keep_groups_together(ws, ledger, cfg, clip_file):
    cfg = small_parts(cfg)
    add_input(ledger, "u", "n")
    order = 0
    for g in "MF":
        for i in range(4):
            order += 1
            sid = f"spk_{g}{i}"
            make_speaker(ledger, sid, g, order)
            keep_clips(ws, ledger, clip_file, sid, f"v{g}{i}", 2)
    # spk_M0 và spk_F0 cùng xuất hiện trong video vMF (talkshow) → chung split.
    for sid in ("spk_M0", "spk_F0"):
        cid = add_clips(ws, ledger, clip_file, "vMF", 1, [unit(5)])[0]
        ledger.update("clips", "clip_id", cid, {"speaker_id": sid})
    make_speaker(ledger, "spk_wait", "M", 50, review="pending")
    result = form_parts(ledger, cfg)
    assert len(result["created"]) == 1
    part = ledger.row("SELECT * FROM parts WHERE part=1")
    members = part["members"]
    genders = {s["speaker_id"]: s["gender"] for s in ledger.rows("SELECT * FROM speakers")}
    assert sorted(genders[s] for s in members) == ["F"] * 3 + ["M"] * 3
    splits = {
        r["speaker_id"]: r["split"]
        for r in ledger.rows("SELECT DISTINCT speaker_id, split FROM clips WHERE part=1")
    }
    for split in ("train", "validation", "test"):
        assert sorted(genders[s] for s, v in splits.items() if v == split) == ["F", "M"]
    if "spk_M0" in splits and "spk_F0" in splits:
        assert splits["spk_M0"] == splits["spk_F0"]
    assert result["ready_left_over"] == {"M": 1, "F": 1}
    assert form_parts(ledger, cfg)["created"] == []  # phần dư không đủ part mới


def test_export_archive_and_review_api(ws, ledger, cfg, clip_file):
    cfg = small_parts(cfg)
    add_input(ledger, "u", "n")
    order = 0
    for g in "MF":
        for i in range(3):
            order += 1
            make_speaker(ledger, f"spk_{g}{i}", g, order)
            keep_clips(ws, ledger, clip_file, f"spk_{g}{i}", f"v{g}{i}", 2)
    make_speaker(ledger, "spk_new", "", 20, review="pending")
    pending = keep_clips(ws, ledger, clip_file, "spk_new", "vN", 3)
    for cid in pending:
        ledger.update("clips", "clip_id", cid, {"decision": "pending"})
    assert form_parts(ledger, cfg)["created"]
    report = export_part(ws, ledger, 1)
    bundle = ws.part_dir(1)
    rows, summary = validate_bundle(bundle)
    assert summary["clips"] == 12 and set(summary["splits"]) == {"train", "validation", "test"}
    assert rows[0]["gender"] in ("M", "F") and (bundle / "speakers.csv").is_file()
    assert (ws.archive / report["zip"]).is_file()
    # Review API: speaker trong part bị khoá; speaker mới chọn tối đa clips_per_speaker clip.
    client = TestClient(create_app(ws, ledger, cfg))
    state = client.get("/api/state").json()
    assert {s["id"] for s in state["speakers"]} >= {"spk_new", "spk_M0"}
    assert (
        client.post(f"/api/clip/{rows[0]['clip_id']}", json={"decision": "reject"}).status_code
        == 409
    )
    for cid in pending[:2]:
        assert client.post(f"/api/clip/{cid}", json={"decision": "keep"}).status_code == 200
    assert client.post(f"/api/clip/{pending[2]}", json={"decision": "keep"}).status_code == 409
    assert client.post("/api/speaker/spk_new", json={"gender": "F", "review": "done"}).json()["ok"]
    detail = client.get("/api/speaker/spk_new").json()
    assert detail["gender"] == "F" and len(detail["videos"][0]["clips"]) == 3
    assert client.get(f"/media/clip/{pending[0]}").status_code == 200
    assert client.get(f"/media/strip/{pending[0]}").status_code == 200
    moved = client.post(f"/api/clip/{pending[2]}/move", json={"speaker_id": "new"}).json()
    assert (
        ledger.scalar("SELECT speaker_id FROM clips WHERE clip_id=?", (pending[2],))
        == moved["speaker_id"]
    )
    assert client.get(f"/api/clip/{pending[0]}/nearest").status_code == 200
    merged = client.post(
        f"/api/speaker/{moved['speaker_id']}/merge", json={"speaker_id": "spk_new"}
    )
    assert merged.json()["into"] == "spk_new"
    assert "Duyệt speaker" in client.get("/").text
    # 09_clean part: phải --uploaded trước; xoá xong vẫn còn trong ledger.
    with pytest.raises(ValueError):
        archive.delete_local(ws, ledger, "part", 1)
    archive.mark_uploaded(ws, ledger, "part", 1)
    freed = archive.delete_local(ws, ledger, "part", 1)
    assert freed["freed_gb"] >= 0 and not bundle.exists()
    locations = {r["location"] for r in ledger.rows("SELECT location FROM clips WHERE part=1")}
    assert locations == {"drive:" + report["zip"]}
    assert client.get(f"/media/clip/{rows[0]['clip_id']}").status_code == 410
