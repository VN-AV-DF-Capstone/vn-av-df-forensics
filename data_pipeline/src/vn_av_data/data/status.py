"""Bước 09 status: toàn cảnh từ ledger - batch nào đang ở đâu, việc nào còn dở, dung lượng ở máy."""

import shutil


def collect_status(ws, ledger, per_speaker=20):
    q = ledger.rows
    batches = []
    for b in q("SELECT * FROM batches ORDER BY batch"):
        n = b["batch"]
        v = q(
            "SELECT COUNT(*) AS videos, SUM(duration_s)/3600.0 AS hours, SUM(est_mb)/1000.0 AS est_gb,"
            " SUM(dl_status='done') AS downloaded, SUM(dl_status='failed') AS dl_failed,"
            " SUM(CASE WHEN dl_status='done' THEN size_mb END)/1000.0 AS size_gb,"
            " SUM(cut_status='done') AS cut, SUM(cut_status='failed') AS cut_failed,"
            " SUM(id_status='done') AS identified FROM videos WHERE batch=? AND status='accepted'",
            (n,),
        )[0]
        clips = ledger.scalar("SELECT COUNT(*) FROM clips WHERE batch=?", (n,))
        batches.append(
            {
                "batch": n,
                "videos": v["videos"],
                "hours": round(v["hours"] or 0, 1),
                "est_gb": round(v["est_gb"] or 0, 1),
                "downloaded": v["downloaded"] or 0,
                "dl_failed": v["dl_failed"] or 0,
                "size_gb": round(v["size_gb"] or 0, 2),
                "cut": v["cut"] or 0,
                "cut_failed": v["cut_failed"] or 0,
                "identified": v["identified"] or 0,
                "clips": clips,
                "raw": b["location"],
            }
        )
    keep = {
        r["speaker_id"]: r["n"]
        for r in q(
            "SELECT speaker_id, COUNT(*) AS n FROM clips WHERE decision='keep' GROUP BY speaker_id"
        )
    }
    speakers = q("SELECT speaker_id, gender, review, merged_into, part FROM speakers")
    live = [s for s in speakers if not s["merged_into"]]
    ready = [
        s
        for s in live
        if s["review"] == "done"
        and s["gender"] in ("M", "F")
        and s["part"] is None
        and keep.get(s["speaker_id"], 0) == per_speaker
    ]
    local_mb = ledger.scalar("SELECT SUM(size_mb) FROM clips WHERE location='local'") or 0
    usage = shutil.disk_usage(ws.root if ws.root.exists() else ws.root.parent)
    return {
        "inputs": {
            r["status"]: r["n"]
            for r in q("SELECT status, COUNT(*) AS n FROM inputs GROUP BY status")
        },
        "videos": {
            r["status"]: r["n"]
            for r in q("SELECT status, COUNT(*) AS n FROM videos GROUP BY status")
        },
        "duplicates": ledger.scalar("SELECT COUNT(*) FROM duplicates"),
        "batches": batches,
        "clips": {
            "total": ledger.scalar("SELECT COUNT(*) FROM clips"),
            "local_gb": round(local_mb / 1000, 2),
            "by_decision": {
                r["decision"]: r["n"]
                for r in q("SELECT decision, COUNT(*) AS n FROM clips GROUP BY decision")
            },
            "unassigned": ledger.scalar("SELECT COUNT(*) FROM clips WHERE speaker_id IS NULL"),
        },
        "speakers": {
            "total": len(live),
            "by_review": {
                k: sum(s["review"] == k for s in live) for k in ("pending", "done", "excluded")
            },
            "ready_for_part": {g: sum(s["gender"] == g for s in ready) for g in ("M", "F")},
            "in_parts": sum(s["part"] is not None for s in live),
        },
        "parts": q("SELECT part, clips, location, zip FROM parts ORDER BY part"),
        "disk_free_gb": round(usage.free / 1e9, 1),
    }


def print_status(s):
    print(f"Nguồn: {s['inputs']} | video: {s['videos']} | trùng: {s['duplicates']}")
    print("Batch  video  giờ   ~GB  tải(GB)    cắt  identify  clip  raw")
    for b in s["batches"]:
        print(
            f"{b['batch']:>5}  {b['videos']:>5}  {b['hours']:>4}  {b['est_gb']:>4}  "
            f"{b['downloaded']:>3} ({b['size_gb']:>5})  {b['cut']:>4}  {b['identified']:>8}  "
            f"{b['clips']:>4}  {b['raw']}"
            + (
                f"  [lỗi tải {b['dl_failed']}, lỗi cắt {b['cut_failed']}]"
                if b["dl_failed"] or b["cut_failed"]
                else ""
            )
        )
    c, sp = s["clips"], s["speakers"]
    print(
        f"Clip: {c['total']} (ở máy {c['local_gb']} GB) {c['by_decision']}, chưa gán speaker "
        f"{c['unassigned']}"
    )
    print(
        f"Speaker: {sp['total']} {sp['by_review']} | sẵn sàng vào part: {sp['ready_for_part']} | "
        f"đã trong part: {sp['in_parts']}"
    )
    for p in s["parts"]:
        print(f"Part {p['part']}: {p['clips']} clip, {p['location']} {p['zip'] or ''}")
    print(f"Ổ đĩa còn trống: {s['disk_free_gb']} GB")
