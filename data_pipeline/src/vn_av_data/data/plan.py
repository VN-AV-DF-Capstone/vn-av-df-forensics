"""Bước 01 plan: urls.csv → chọn video theo luật → chia batch raw theo số giờ và dung lượng.

Mỗi dòng urls.csv là một nguồn (video / playlist / kênh) kèm `speakers`:
- `1`: nguồn một người. Lấy video trải theo ngày đăng đến khi tổng ≥ `single.target_minutes`
  (mặc định 20 phút), ít nhất 3 và nhiều nhất 10 video; bước cắt chỉ lấy 40-60 clip rải đều.
- `n`: nguồn nhiều người. Kênh: `multi.max_videos` video mới nhất (mặc định 15); playlist: hỏi
  tối đa `playlist_scan_max` video rồi lấy 15 video mới nhất theo ngày đăng. Bước cắt cắt hết.

Video của cùng một dòng nằm chung một batch (trừ nguồn nhiều người quá lớn). Batch đã bắt đầu
tải thì đóng băng; plan lại chỉ thêm batch mới. Dòng đã plan xong thì không plan lại.
"""

import csv
from collections import defaultdict
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes
from vn_av_data.data import youtube
from vn_av_data.ledger import now

TEMPLATE = (
    "url,speakers,note\n"
    "# Mỗi dòng một URL video / playlist / kênh. speakers: 1 = một người, n = nhiều người.\n"
)
MULTI = {"n", "nhieu", "nhiều", "many", "multi", "2", "2+"}


def read_inputs(path):
    path = Path(path)
    if not path.exists():
        atomic_bytes(path, TEMPLATE.encode("utf-8-sig"))
        raise FileNotFoundError(f"Created {path}; fill URLs then run plan again")
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = [line for line in stream if line.strip() and not line.lstrip().startswith("#")]
    for number, row in enumerate(csv.DictReader(lines), start=2):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        value = (row.get("speakers") or "").strip().lower()
        if value == "1":
            speakers = "1"
        elif value in MULTI:
            speakers = "n"
        else:
            raise ValueError(f"urls.csv row {number}: speakers must be 1 or n ({value!r})")
        rows.append({"url": url, "speakers": speakers, "note": (row.get("note") or "").strip()})
    if not rows:
        raise ValueError(f"No URL in {path}")
    return rows


def spread(candidates, count, prefer_max_s):
    """Chọn `count` video trải theo ngày đăng, ưu tiên video không quá dài."""
    good = [c for c in candidates if (c["duration_s"] or 0) <= prefer_max_s]
    pool = good if len(good) >= count else good + [c for c in candidates if c not in good]
    pool = sorted(pool, key=lambda c: (c["upload_date"] or "", c["rank"]))
    if len(pool) <= count:
        return pool
    if count == 1:
        return [pool[len(pool) // 2]]
    picks = sorted({round(i * (len(pool) - 1) / (count - 1)) for i in range(count)})
    return [pool[i] for i in picks]


def choose_single(candidates, rules):
    """Nguồn một người: số video nhỏ nhất (≥ min_videos) đạt tổng target_minutes, tối đa max."""
    target = rules["target_minutes"] * 60
    chosen = []
    for count in range(rules["min_videos"], rules["max_videos"] + 1):
        chosen = spread(candidates, count, rules["prefer_max_s"])
        if len(chosen) < count or sum(c["duration_s"] or 0 for c in chosen) >= target:
            break
    return chosen


def register_inputs(ledger, rows):
    """Thêm dòng mới; dòng đã có không được đổi số người (đã có thể tải / cắt)."""
    added = 0
    for row in rows:
        kind, url = youtube.classify(row["url"])
        old = ledger.row("SELECT * FROM inputs WHERE url=?", (url,))
        if old:
            if old["speakers"] != row["speakers"]:
                raise ValueError(f"{url}: speakers changed ({old['speakers']} → {row['speakers']})")
            continue
        ledger.insert(
            "inputs", {"url": url, "kind": kind, "speakers": row["speakers"], "note": row["note"]}
        )
        added += 1
    return added


def plan_input(ledger, item, cfg, metadata_loader, flat_loader, require_vietnamese):
    single = item["speakers"] == "1"
    rules = cfg["single"] if single else cfg["multi"]
    if item["kind"] == "video":
        entries = [{"video_id": youtube.video_id(item["url"]), "title": ""}]
    else:
        limit = rules["playlist_scan_max"] if item["kind"] == "playlist" else rules["scan_max"]
        entries = flat_loader(item["url"], limit)
    # Kênh: thứ tự YouTube = mới nhất trước → dừng khi đủ. Playlist: hỏi hết (đến giới hạn).
    enough = (
        1
        if item["kind"] == "video"
        else rules["scan_accepted"]
        if single
        else rules["max_videos"]
        if item["kind"] == "channel"
        else None
    )
    candidates, errors, duplicates = [], 0, 0
    for rank, entry in enumerate(entries):
        if enough is not None and len(candidates) >= enough:
            break
        vid = entry["video_id"]
        old = ledger.row("SELECT * FROM videos WHERE video_id=?", (vid,))
        if old and old["input_id"] != item["input_id"]:
            ledger.insert(
                "duplicates",
                {
                    "video_id": vid,
                    "input_id": item["input_id"],
                    "kept_input_id": old["input_id"],
                    "at": now(),
                },
                replace=True,
            )
            duplicates += 1
            continue
        if old and old["status"] != "error":
            if old["status"] in ("candidate", "accepted", "not_selected"):
                candidates.append(old)
            continue
        print(f"  metadata {rank + 1}/{len(entries)} {vid}", flush=True)
        values = {"video_id": vid, "input_id": item["input_id"], "rank": rank}
        values["url"] = "https://www.youtube.com/watch?v=" + vid
        try:
            meta = youtube.describe(metadata_loader(values["url"]))
        except Exception as exc:  # noqa: BLE001 -- ghi lại, các video khác vẫn chạy
            message = str(exc)
            gone = youtube.UNAVAILABLE.search(message)
            values.update(status="rejected" if gone else "error", reason="youtube: " + message)
            values["title"] = entry.get("title", "")
            errors += not gone
            ledger.insert("videos", values, replace=True)
            continue
        reason = youtube.check(meta, require_vietnamese)
        values.update(
            {k: meta[k] for k in ("title", "channel", "channel_id", "upload_date", "duration_s")},
            est_mb=youtube.estimate_mb(meta, cfg["fallback_gb_per_hour"]),
            status="rejected" if reason else "candidate",
            reason=reason or "",
            meta=meta,
        )
        ledger.insert("videos", values, replace=True)
        if not reason:
            candidates.append(ledger.row("SELECT * FROM videos WHERE video_id=?", (vid,)))
    if errors:
        ledger.update("inputs", "input_id", item["input_id"], {"status": "error"})
        return {"input_id": item["input_id"], "status": "error", "metadata_errors": errors}
    if single:
        chosen = choose_single(candidates, rules)
    elif item["kind"] == "playlist":
        chosen = sorted(candidates, key=lambda c: c["upload_date"] or "", reverse=True)
        chosen = chosen[: rules["max_videos"]]
    else:
        chosen = candidates[: rules["max_videos"]]
    keep = {c["video_id"] for c in chosen}
    with ledger.tx() as conn:
        for c in candidates:
            status = "accepted" if c["video_id"] in keep else "not_selected"
            reason = "" if status == "accepted" else "over per-source limit"
            ledger.update(
                "videos", "video_id", c["video_id"], {"status": status, "reason": reason}, conn
            )
        ledger.update(
            "inputs",
            "input_id",
            item["input_id"],
            {
                "status": "planned" if keep else "empty",
                "reason": "" if keep else "no video passed the checks",
                "entries": len(entries),
                "planned_at": now(),
            },
            conn,
        )
    return {
        "input_id": item["input_id"],
        "status": "planned" if keep else "empty",
        "chosen": len(keep),
        "duplicates": duplicates,
    }


def batch_videos(ledger, ws, cfg):
    """Gán batch cho video đã chọn mà chưa có batch; trả các batch vừa thay đổi."""
    pending = ledger.rows(
        "SELECT video_id, input_id, duration_s, est_mb FROM videos "
        "WHERE status='accepted' AND batch IS NULL ORDER BY input_id, rank"
    )
    if not pending:
        return []
    max_h, max_mb = cfg["batch_max_hours"], cfg["batch_max_gb"] * 1000
    groups = defaultdict(list)
    for v in pending:
        groups[v["input_id"]].append(v)
    # Batch cuối còn mở nếu chưa video nào bắt đầu tải.
    last = ledger.scalar("SELECT MAX(batch) FROM batches")
    load = {"batch": None, "h": 0.0, "mb": 0.0}
    if last is not None and not ledger.scalar(
        "SELECT COUNT(*) FROM videos WHERE batch=? AND dl_status!='pending'", (last,)
    ):
        load["batch"] = last
        load["h"] = ledger.scalar("SELECT SUM(duration_s) FROM videos WHERE batch=?", (last,)) or 0
        load["h"] /= 3600
        load["mb"] = ledger.scalar("SELECT SUM(est_mb) FROM videos WHERE batch=?", (last,)) or 0
    touched = set()

    def new_batch():
        number = (ledger.scalar("SELECT MAX(batch) FROM batches") or 0) + 1
        ledger.insert("batches", {"batch": number, "created_at": now()})
        load.update(batch=number, h=0.0, mb=0.0)

    def place(videos):
        for v in videos:
            ledger.update("videos", "video_id", v["video_id"], {"batch": load["batch"]})
            load["h"] += (v["duration_s"] or 0) / 3600
            load["mb"] += v["est_mb"] or 0
        touched.add(load["batch"])

    for videos in groups.values():
        hours = sum(v["duration_s"] or 0 for v in videos) / 3600
        mb = sum(v["est_mb"] or 0 for v in videos)
        if (
            load["batch"] is None
            or (load["h"] + hours > max_h or load["mb"] + mb > max_mb)
            and (load["h"] or load["mb"])
        ):
            new_batch()
        if hours <= max_h and mb <= max_mb:
            place(videos)
            continue
        # Nguồn quá lớn cho một batch (chỉ xảy ra với nguồn nhiều người): chia theo video.
        for v in videos:
            h, size = (v["duration_s"] or 0) / 3600, v["est_mb"] or 0
            if (load["h"] or load["mb"]) and (load["h"] + h > max_h or load["mb"] + size > max_mb):
                new_batch()
            place([v])
    for number in sorted(touched):
        write_batch_list(ledger, ws, number)
    return sorted(touched)


def write_batch_list(ledger, ws, batch):
    rows = ledger.rows(
        "SELECT v.video_id, v.url, v.title, v.channel, v.duration_s, v.est_mb, i.speakers, "
        "i.input_id FROM videos v JOIN inputs i USING(input_id) WHERE v.batch=? "
        "ORDER BY v.input_id, v.rank",
        (batch,),
    )
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        ["video_id", "url", "title", "channel", "minutes", "est_mb", "speakers", "input"]
    )
    for r in rows:
        writer.writerow(
            [
                r["video_id"],
                r["url"],
                r["title"],
                r["channel"],
                round((r["duration_s"] or 0) / 60, 1),
                r["est_mb"],
                r["speakers"],
                r["input_id"],
            ]
        )
    atomic_bytes(ws.raw(batch) / "videos.csv", stream.getvalue().encode("utf-8-sig"))


def plan_sources(
    ws,
    ledger,
    cfg,
    metadata_loader=None,
    flat_loader=None,
    cookies_from_browser=None,
    force_ipv4=False,
):
    cfg = cfg["plan"]
    added = register_inputs(ledger, read_inputs(ws.inputs))
    todo = ledger.rows("SELECT * FROM inputs WHERE status IN ('pending','error') ORDER BY input_id")
    if todo and (metadata_loader is None or flat_loader is None):
        from vn_av_data.data.acquisition import youtube_options

        network = youtube_options(cookies_from_browser, force_ipv4)
        metadata_loader = metadata_loader or (lambda url: youtube.youtube_metadata(url, network))
        flat_loader = flat_loader or (lambda url, limit: youtube.flat_entries(url, network, limit))
    results = []
    for number, item in enumerate(todo, 1):
        print(f"[{number}/{len(todo)}] {item['url']} (speakers={item['speakers']})", flush=True)
        try:
            results.append(
                plan_input(
                    ledger, item, cfg, metadata_loader, flat_loader, cfg["require_vietnamese"]
                )
            )
        except Exception as exc:  # noqa: BLE001 -- nguồn hỏng không chặn các nguồn khác
            ledger.update(
                "inputs", "input_id", item["input_id"], {"status": "error", "reason": str(exc)}
            )
            results.append({"input_id": item["input_id"], "status": "error", "reason": str(exc)})
    touched = batch_videos(ledger, ws, cfg)
    batches = ledger.rows(
        "SELECT batch, COUNT(*) AS videos, ROUND(SUM(duration_s)/3600.0, 2) AS hours, "
        "ROUND(SUM(est_mb)/1000.0, 2) AS est_gb FROM videos WHERE batch IS NOT NULL "
        "GROUP BY batch ORDER BY batch"
    )
    summary = {
        "inputs_added": added,
        "inputs_processed": len(todo),
        "errors": [r for r in results if r["status"] == "error"],
        "empty": [r["input_id"] for r in results if r["status"] == "empty"],
        "batches_changed": touched,
        "batches": batches,
    }
    ledger.log("plan", f"{len(todo)} inputs, batches changed {touched}")
    if summary["errors"]:
        summary["note"] = "Some inputs failed (network?); run plan again to retry them"
    return summary
