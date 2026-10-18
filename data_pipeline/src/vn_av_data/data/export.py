"""Bước 07 export: dựng bundle sạch của một part (clip được chọn + manifest có split) và ZIP.

Định dạng bundle giữ nguyên hợp đồng `vn-av-dataset-v1` (contract.py) mà generation/training
đang đọc: `clips/`, `manifest.jsonl`, `dataset_info.json`, `split-lock.json`; thêm
`speakers.csv` (giới, split, kênh). Dựng ở thư mục tạm rồi mới đổi tên, nên bản dở dang không
bao giờ trông như part hoàn chỉnh.

Độ lệch tiếng-hình: tiếng của mỗi clip được dịch theo độ lệch đo ở bước cắt (SyncNet, cả video),
để real và fake (generator luôn vẽ khớp ở độ lệch 0) có cùng mốc. Hình giữ nguyên (copy luồng),
tiếng encode lại AAC; phần hụt ở đầu/cuối (≤ vài frame) đệm lặng. `audio_lag_frames` trong
manifest = độ lệch đã sửa (dương = tiếng gốc trễ hơn hình).
"""

import csv
import io
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes, sha, write_json
from vn_av_data.contract import SCHEMA, validate_bundle
from vn_av_data.data.manifest import write_manifest
from vn_av_data.data.media import ffmpeg, probe
from vn_av_data.ledger import now

REGISTRY_SCHEMA = "vn-av-df-split-registry-v1"


def video_sync(video):
    """(độ lệch cần sửa, trạng thái) của video; không đo được thì không sửa."""
    sync = (video.get("cut_info") or {}).get("sync") or {}
    lag = sync.get("audio_lag_frames", 0) if sync.get("status") in ("ok", "few") else 0
    return int(lag), sync.get("status", "unmeasured")


def place_clip(source, target, lag_frames, duration):
    """Chép clip; lag ≠ 0 thì dời tiếng sớm lên lag frame (copy hình, encode lại tiếng)."""
    from vn_av_data.common.runtime import run

    if not lag_frames:
        shutil.copy2(source, target)
        return
    shift = lag_frames * 0.04
    run(
        [
            ffmpeg(),
            *("-nostdin", "-v", "error", "-y", "-i", str(source)),
            *("-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy"),
            "-af",
            f"asetpts=PTS-{shift:.3f}/TB,aresample=async=1:first_pts=0,apad",
            *("-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-t", f"{duration:.3f}"),
            *("-movflags", "+faststart", str(target)),
        ]
    )


def manifest_row(clip, video, speaker, duration):
    meta = video.get("probe") or {}
    lag, status = video_sync(video)
    return {
        "clip_id": clip["clip_id"],
        "video": f"clips/{clip['clip_id']}.mp4",
        "sha256": clip["sha256"],
        "source_id": "yt_" + clip["video_id"],
        "speaker_id": clip["speaker_id"],
        "gender": speaker["gender"],
        "split": clip["split"],
        "duration_s": duration,
        "source_start_s": clip["start_s"],
        "source_end_s": clip["end_s"],
        "review_decision": "keep",
        "sync_status": "reviewed_match",
        "relation_annotations": {},
        "url": video["url"],
        "channel": video["channel"],
        "source_sha256": video["sha256"],
        "source_fps": meta.get("fps"),
        "source_width": meta.get("width"),
        "source_height": meta.get("height"),
        "source_codec": meta.get("video_codec"),
        "source_bitrate_kbps": meta.get("video_bitrate_kbps") or meta.get("total_bitrate_kbps"),
        "batch": clip["batch"],
        "flags": clip["flags"] or "",
        "audio_lag_frames": lag,
        "av_sync_status": status,
    }


def zip_folder(folder, target):
    """ZIP không nén (mp4 đã nén); ghi ra file tạm rồi đổi tên."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.zip")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                archive.write(path, Path(folder).name + "/" + path.relative_to(folder).as_posix())
    partial.replace(target)
    return {
        "zip": target.name,
        "zip_sha256": sha(target),
        "zip_mb": round(target.stat().st_size / 1e6, 1),
    }


def export_part(ws, ledger, part):
    record = ledger.row("SELECT * FROM parts WHERE part=?", (part,))
    if record is None:
        raise ValueError(f"Part {part} does not exist; run 06_parts first")
    if (record["location"] or "local").startswith("drive:"):
        raise ValueError(f"Part {part} is already on Drive ({record['location']})")
    clips = ledger.rows(
        "SELECT * FROM clips WHERE part=? AND decision='keep' ORDER BY speaker_id, video_id, start_s",
        (part,),
    )
    if len(clips) != record["clips"]:
        raise ValueError(f"Part {part}: expected {record['clips']} clips, found {len(clips)}")
    name = ws.part_name(part)
    output = ws.part_dir(part)
    stage = output.with_name(output.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "clips").mkdir(parents=True)
    videos, speakers, rows = {}, {}, []
    for number, clip in enumerate(clips, 1):
        if clip["location"] != "local":
            raise ValueError(f"{clip['clip_id']} is not local ({clip['location']})")
        source = ws.root / clip["file"]
        if sha(source) != clip["sha256"]:
            raise ValueError(f"Clip changed since cut: {source}")
        video = videos.setdefault(
            clip["video_id"],
            ledger.row("SELECT * FROM videos WHERE video_id=?", (clip["video_id"],)),
        )
        speaker = speakers.setdefault(
            clip["speaker_id"],
            ledger.row("SELECT * FROM speakers WHERE speaker_id=?", (clip["speaker_id"],)),
        )
        target = stage / "clips" / f"{clip['clip_id']}.mp4"
        duration = probe(source)["duration_s"]
        place_clip(source, target, video_sync(video)[0], duration)
        row = manifest_row(clip, video, speaker, duration)
        row["sha256"] = sha(target)
        rows.append(row)
        if number % 100 == 0:
            print(f"  {number}/{len(clips)} clip", flush=True)
    write_manifest(stage / "manifest.jsonl", rows)
    by_speaker = {r["speaker_id"]: r for r in rows}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["speaker_id", "gender", "split", "clips", "channels", "videos"])
    for sid, s in speakers.items():
        mine = [r for r in rows if r["speaker_id"] == sid]
        writer.writerow(
            [
                sid,
                s["gender"],
                by_speaker[sid]["split"],
                len(mine),
                ";".join(sorted({r["channel"] for r in mine})),
                ";".join(sorted({r["source_id"] for r in mine})),
            ]
        )
    atomic_bytes(stage / "speakers.csv", stream.getvalue().encode("utf-8-sig"))
    splits = Counter(r["split"] for r in rows)
    genders = Counter((by_speaker[s]["split"], speakers[s]["gender"]) for s in speakers)
    write_json(
        stage / "split-lock.json",
        {
            "schema": REGISTRY_SCHEMA,
            "data_part": part,
            "seed": None,
            "covered_parts": [part],
            "assignments": [
                {k: r[k] for k in ("clip_id", "split", "speaker_id", "source_id", "sha256")}
                for r in rows
            ],
            "groups": len(speakers),
            "counts": dict(splits),
            "note": "Split assigned per speaker group in 06_parts (gender balanced)",
        },
    )
    info = {
        "schema_version": SCHEMA,
        "dataset_id": name,
        "part": part,
        "clips": len(rows),
        "speakers": len(speakers),
        "manifest_sha256": sha(stage / "manifest.jsonl"),
        "exporter_sha256": sha(__file__),
        "exported_at": now(),
        "split": {
            "counts": dict(splits),
            "speakers_by_split_gender": {f"{k[0]}:{k[1]}": v for k, v in sorted(genders.items())},
        },
    }
    write_json(stage / "dataset_info.json", info)
    _, report = validate_bundle(stage, probe=probe)
    write_json(stage / "quality_report.json", report)
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, output)
    print(f"  ZIP {name}.zip ...", flush=True)
    packed = zip_folder(output, ws.archive / f"{name}.zip")
    ledger.update(
        "parts",
        "part",
        part,
        {"export_dir": output.as_posix(), **packed, "location": "packed"},
    )
    ledger.log("export", f"part {part}: {len(rows)} clips, zip {packed['zip']}")
    return {"part": part, "output": str(output), **report, **packed}
