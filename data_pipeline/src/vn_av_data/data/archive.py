"""Bước 08 archive (đóng gói, đánh dấu đã lên Drive) và bước 09 clean (xoá bản ở máy).

Trình tự cho mỗi batch raw (part tương tự, zip do 07_export tạo):
1. `08_archive.py raw 3`             → ZIP + SHA256SUMS vào `data/archive/`, ledger: packed.
2. bạn up file ZIP (và thư mục `ledger/`) lên Drive.
3. `08_archive.py raw 3 --uploaded`  → ledger: drive:<tên zip>.
4. `09_clean.py raw 3`               → xoá raw + zip ở máy (chỉ khi đã là drive:).
`09_clean.py pool` xoá clip thừa trong pool, chỉ clip mà raw của batch đã lên Drive.
Ledger luôn giữ: video nào, hash gì, nằm trong zip nào trên Drive, clip nào cắt từ đó.
"""

import shutil

from vn_av_data.common.runtime import atomic_bytes
from vn_av_data.data.export import zip_folder


def pack_raw(ws, ledger, batch):
    videos = ledger.rows(
        "SELECT * FROM videos WHERE batch=? AND status='accepted' ORDER BY input_id, rank",
        (batch,),
    )
    if not videos:
        raise ValueError(f"Batch {batch} has no videos")
    unfinished = [
        v["video_id"]
        for v in videos
        if v["dl_status"] not in ("done", "rejected")
        or (v["dl_status"] == "done" and (v["cut_status"] != "done" or v["id_status"] != "done"))
    ]
    if unfinished:
        raise ValueError(
            f"Batch {batch}: {len(unfinished)} videos not downloaded/cut/identified yet "
            f"(e.g. {unfinished[:3]}); finish steps 02-04 first"
        )
    folder = ws.raw(batch)
    lines = [
        f"{v['sha256']}  videos/{v['video_id']}.mp4" for v in videos if v["dl_status"] == "done"
    ]
    atomic_bytes(folder / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode())
    print(f"ZIP batch {batch} ...", flush=True)
    packed = zip_folder(folder, ws.archive / f"raw_batch_{batch:03d}.zip")
    ledger.update("batches", "batch", batch, {**packed, "location": "packed"})
    ledger.log("archive", f"raw batch {batch} packed {packed['zip']}")
    return {
        "batch": batch,
        **packed,
        "next": f"Up {ws.archive / packed['zip']} + thư mục ledger/ lên Drive, rồi chạy với --uploaded",
    }


def mark_uploaded(ws, ledger, kind, number):
    table, key = ("batches", "batch") if kind == "raw" else ("parts", "part")
    record = ledger.row(f"SELECT * FROM {table} WHERE {key}=?", (number,))
    if record is None or not record["zip"]:
        raise ValueError(f"{kind} {number} is not packed yet; run without --uploaded first")
    location = "drive:" + record["zip"]
    with ledger.tx() as conn:
        ledger.update(table, key, number, {"location": location}, conn)
        if kind == "raw":
            conn.execute(
                "UPDATE videos SET raw_location=? WHERE batch=? AND dl_status='done'",
                (location, number),
            )
        ledger.log("archive", f"{kind} {number} uploaded as {record['zip']}", conn)
    return {kind: number, "location": location, "next": f"09_clean.py {kind} {number} để xoá"}


def delete_local(ws, ledger, kind, number):
    table, key = ("batches", "batch") if kind == "raw" else ("parts", "part")
    record = ledger.row(f"SELECT * FROM {table} WHERE {key}=?", (number,))
    if record is None or not (record["location"] or "").startswith("drive:"):
        raise ValueError(
            f"{kind} {number} chưa đánh dấu đã lên Drive; chạy 08_archive.py {kind} {number} "
            "--uploaded trước"
        )
    freed = 0.0
    if kind == "raw":
        targets = [ws.raw(number)]
    else:
        targets = [ws.part_dir(number)]
        # Clip pool của speaker trong part: clip được chọn đã nằm trong zip part; clip không
        # được chọn không còn dùng. Ảnh mặt (pool/faces) giữ lại vì nhỏ và cần khi tra cứu.
        speakers = record["members"] or []
        marks = ",".join("?" for _ in speakers)
        clips = ledger.rows(
            f"SELECT clip_id, file, decision, part FROM clips WHERE speaker_id IN ({marks}) "
            "AND location='local'",
            speakers,
        )
        with ledger.tx() as conn:
            for c in clips:
                path = ws.root / c["file"]
                if path.exists():
                    freed += path.stat().st_size / 1e6
                    path.unlink()
                where = record["location"] if c["part"] == number else "deleted"
                conn.execute("UPDATE clips SET location=? WHERE clip_id=?", (where, c["clip_id"]))
    targets.append(ws.archive / record["zip"])
    for path in targets:
        if path.is_dir():
            freed += sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
            shutil.rmtree(path)
        elif path.exists():
            freed += path.stat().st_size / 1e6
            path.unlink()
    ledger.log("archive", f"{kind} {number} deleted locally, freed {freed:.0f} MB")
    return {kind: number, "freed_gb": round(freed / 1000, 2)}


def pool_leftovers(ledger):
    """Clip ở máy sẽ không bao giờ dùng: không được chọn, của speaker đã chốt (duyệt xong / bị loại),
    và raw của batch đã lên Drive (cần thì cắt lại được).

    Clip của speaker còn chờ duyệt và clip chưa gán speaker được giữ (còn có thể chọn).
    """
    return ledger.rows(
        "SELECT c.clip_id, c.file, c.size_mb, c.batch, b.location AS raw FROM clips c "
        "JOIN speakers s USING(speaker_id) LEFT JOIN batches b ON b.batch = c.batch "
        "WHERE c.location='local' AND c.decision!='keep' AND s.review IN ('done','excluded')"
    )


def clean_pool(ws, ledger):
    """`09_clean.py pool`: xoá clip thừa; clip của batch mà raw chưa lên Drive thì để lại."""
    candidates = pool_leftovers(ledger)
    clips = [c for c in candidates if (c["raw"] or "").startswith("drive:")]
    waiting = sorted({c["batch"] for c in candidates if c not in clips})
    size = sum(c["size_mb"] or 0 for c in clips)
    with ledger.tx() as conn:
        for c in clips:
            (ws.root / c["file"]).unlink(missing_ok=True)
            conn.execute("UPDATE clips SET location='deleted' WHERE clip_id=?", (c["clip_id"],))
        ledger.log("archive", f"pool: deleted {len(clips)} unused clips ({size:.0f} MB)", conn)
    result = {"deleted_clips": len(clips), "freed_gb": round(size / 1000, 2)}
    if waiting:
        result["kept_until_raw_uploaded"] = {
            "clips": len(candidates) - len(clips),
            "batches": waiting,
            "next": "Up raw các batch này lên Drive (08_archive.py raw N --uploaded) rồi chạy lại",
        }
    return result
