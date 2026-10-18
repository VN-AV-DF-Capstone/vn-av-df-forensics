"""Bước 05 review: duyệt theo speaker, một màn hình (127.0.0.1:8001).

Mỗi speaker: chọn đúng `clips_per_speaker` clip (mặc định 20) và gán giới. Clip nhóm theo video.
Đủ 10 thì có thể bỏ qua toàn bộ clip còn lại của speaker đó. Mọi thao tác ghi thẳng vào ledger.
Speaker đã nằm trong part là đóng băng (chỉ xem).
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from vn_av_data.features.face import normalize
from vn_av_data.ledger import now
from vn_av_data.serving.speaker_page import PAGE


class ClipDecision(BaseModel):
    decision: str


class SpeakerUpdate(BaseModel):
    gender: str | None = None
    review: str | None = None


class Target(BaseModel):
    speaker_id: str


def total(a, b, key, size):
    """Cộng vector tổng của hai speaker (bỏ qua bên chưa có vector)."""
    out = np.zeros(size, np.float32)
    for s in (a, b):
        if s[f"{key}_n"]:
            out = out + s[f"{key}_sum"]
    return out


def photo_urls(speaker):
    return [f"/media/photo/{Path(p['path']).name}" for p in speaker.get("photos") or []]


def create_app(ws, ledger, cfg):
    per_speaker = cfg["parts"]["clips_per_speaker"]
    target = cfg["parts"]["speakers_per_gender"]  # tiến độ tính cho part kế tiếp
    app = FastAPI()

    def speaker(sid):
        s = ledger.row("SELECT * FROM speakers WHERE speaker_id=?", (sid,))
        if s is None:
            raise HTTPException(404, "Speaker not found")
        return s

    def editable(s):
        if s["part"] is not None:
            raise HTTPException(409, f"{s['speaker_id']} đã nằm trong part {s['part']} (khoá)")
        if s["merged_into"]:
            raise HTTPException(409, f"{s['speaker_id']} đã gộp vào {s['merged_into']}")

    def clip(cid):
        c = ledger.row("SELECT * FROM clips WHERE clip_id=?", (cid,))
        if c is None:
            raise HTTPException(404, "Clip not found")
        return c

    def keep_count(sid):
        return ledger.scalar(
            "SELECT COUNT(*) FROM clips WHERE speaker_id=? AND decision='keep'", (sid,)
        )

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.replace("__PER__", str(per_speaker)).replace("__TARGET__", str(target))

    @app.get("/api/state")
    def state():
        counts = {
            r["speaker_id"]: (r["n"], r["k"])
            for r in ledger.rows(
                "SELECT speaker_id, COUNT(*) AS n, SUM(decision='keep') AS k FROM clips "
                "WHERE speaker_id IS NOT NULL GROUP BY speaker_id"
            )
        }
        rows = ledger.rows(
            "SELECT speaker_id, gender, review, part, photos, suggestion, order_key FROM speakers "
            "WHERE merged_into='' ORDER BY order_key"
        )
        speakers = []
        done = {"M": 0, "F": 0}
        for s in rows:
            n, k = counts.get(s["speaker_id"], (0, 0))
            k = k or 0
            ready = s["review"] == "done" and k == per_speaker and s["part"] is None
            if ready and s["gender"] in done:
                done[s["gender"]] += 1
            speakers.append(
                {
                    "id": s["speaker_id"],
                    "gender": s["gender"],
                    "review": s["review"],
                    "part": s["part"],
                    "clips": n,
                    "keep": k,
                    "photo": (photo_urls(s) or [None])[0],
                    "suggestion": bool(s["suggestion"]),
                }
            )
        return {"speakers": speakers, "complete": done, "target": target, "per": per_speaker}

    @app.get("/api/speaker/{sid}")
    def detail(sid: str):
        s = speaker(sid)
        clips = ledger.rows(
            "SELECT clip_id, video_id, start_s, end_s, duration_s, decision, flags, jump, "
            "face_consistency, location, assign FROM clips WHERE speaker_id=? "
            "ORDER BY video_id, start_s",
            (sid,),
        )
        videos = {}
        for c in clips:
            v = videos.get(c["video_id"])
            if v is None:
                meta = (
                    ledger.row(
                        "SELECT title, channel, upload_date, duration_s, cut_info FROM videos "
                        "WHERE video_id=?",
                        (c["video_id"],),
                    )
                    or {}
                )
                sync = (meta.pop("cut_info", None) or {}).get("sync") or {}
                v = videos[c["video_id"]] = {"video_id": c["video_id"], **meta, "clips": []}
                v["sync"] = {
                    "status": sync.get("status", "unmeasured"),
                    "lag": sync.get("audio_lag_frames"),
                }
            v["clips"].append(c)
        suggestion = json.loads(s["suggestion"]) if s["suggestion"] else None
        if suggestion:
            other = ledger.row(
                "SELECT * FROM speakers WHERE speaker_id=?", (suggestion["speaker_id"],)
            )
            if other is None or other["merged_into"]:
                suggestion = None
            else:
                suggestion["photos"] = photo_urls(other)
                suggestion["gender"] = other["gender"]
        return {
            "id": sid,
            "gender": s["gender"],
            "review": s["review"],
            "part": s["part"],
            "photos": photo_urls(s),
            "channels": s["channel_ids"],
            "suggestion": suggestion,
            "videos": list(videos.values()),
        }

    @app.post("/api/clip/{cid}")
    def decide(cid: str, body: ClipDecision):
        if body.decision not in ("keep", "reject", "pending"):
            raise HTTPException(422, "decision must be keep, reject or pending")
        c = clip(cid)
        s = speaker(c["speaker_id"])
        editable(s)
        if body.decision == "keep" and c["decision"] != "keep":
            if keep_count(s["speaker_id"]) >= per_speaker:
                raise HTTPException(409, f"Đã chọn đủ {per_speaker} clip; bỏ chọn một clip trước")
            if c["location"] != "local":
                raise HTTPException(409, "Clip không còn ở máy")
        with ledger.tx() as conn:
            ledger.update("clips", "clip_id", cid, {"decision": body.decision}, conn)
            ledger.log("review", f"{cid} {body.decision}", conn)
        return {"keep": keep_count(s["speaker_id"])}

    @app.post("/api/speaker/{sid}")
    def update(sid: str, body: SpeakerUpdate):
        s = speaker(sid)
        editable(s)
        values = {}
        if body.gender is not None:
            if body.gender not in ("M", "F", ""):
                raise HTTPException(422, "gender must be M, F or empty")
            values["gender"] = body.gender
        if body.review is not None:
            if body.review not in ("pending", "done", "excluded"):
                raise HTTPException(422, "review must be pending, done or excluded")
            values["review"] = body.review
            values["reviewed_at"] = now() if body.review == "done" else None
        if values:
            with ledger.tx() as conn:
                ledger.update("speakers", "speaker_id", sid, values, conn)
                ledger.log("review", f"{sid} {values}", conn)
        return {"ok": True}

    def move_vectors(conn, c, source, dest):
        for key, n in (("face", "face_n"), ("voice", "voice_n")):
            vec = c[key]
            if vec is None:
                continue
            for sid, sign in ((source, -1), (dest, 1)):
                if sid is None:
                    continue
                s = ledger.decode(
                    conn.execute("SELECT * FROM speakers WHERE speaker_id=?", (sid,)).fetchone()
                )
                base = s[f"{key}_sum"] if s[n] else np.zeros_like(vec)
                ledger.update(
                    "speakers",
                    "speaker_id",
                    sid,
                    {f"{key}_sum": base + sign * vec, n: max(0, s[n] + sign)},
                    conn,
                )

    @app.post("/api/speaker/{sid}/merge")
    def merge(sid: str, body: Target):
        s, into = speaker(sid), speaker(body.speaker_id)
        editable(s)
        editable(into)
        if sid == into["speaker_id"]:
            raise HTTPException(422, "Cannot merge a speaker into itself")
        with ledger.tx() as conn:
            values = {
                "channel_ids": sorted(set(s["channel_ids"] or []) | set(into["channel_ids"] or [])),
                "inputs": sorted(set(s["inputs"] or []) | set(into["inputs"] or [])),
                "photos": (into["photos"] or []) + (s["photos"] or []),
                "face_sum": total(into, s, "face", 128),
                "face_n": into["face_n"] + s["face_n"],
            }
            if s["voice_n"] or into["voice_n"]:
                values["voice_sum"] = total(into, s, "voice", 256)
                values["voice_n"] = into["voice_n"] + s["voice_n"]
            ledger.update("speakers", "speaker_id", into["speaker_id"], values, conn)
            ledger.update(
                "speakers",
                "speaker_id",
                sid,
                {"merged_into": into["speaker_id"], "suggestion": ""},
                conn,
            )
            # Clip được chọn của speaker bị gộp quay về chờ duyệt để không vượt 10.
            conn.execute(
                "UPDATE clips SET speaker_id=?, assign='merged', decision=CASE WHEN decision='keep' "
                "THEN 'pending' ELSE decision END WHERE speaker_id=?",
                (into["speaker_id"], sid),
            )
            ledger.log("review", f"merge {sid} -> {into['speaker_id']}", conn)
        return {"into": into["speaker_id"]}

    @app.post("/api/speaker/{sid}/dismiss")
    def dismiss(sid: str):
        editable(speaker(sid))
        ledger.update("speakers", "speaker_id", sid, {"suggestion": ""})
        return {"ok": True}

    @app.get("/api/clip/{cid}/nearest")
    def nearest(cid: str):
        c = clip(cid)
        rows = ledger.rows(
            "SELECT * FROM speakers WHERE merged_into='' AND review!='excluded' AND face_n>0 "
            "AND speaker_id!=? AND part IS NULL",
            (c["speaker_id"],),
        )
        if not rows:
            return []
        sims = np.stack([normalize(s["face_sum"]) for s in rows]) @ c["face"]
        best = np.argsort(-sims)[:3]
        return [
            {
                "id": rows[i]["speaker_id"],
                "sim": round(float(sims[i]), 3),
                "photos": photo_urls(rows[i]),
            }
            for i in best
        ]

    @app.post("/api/clip/{cid}/move")
    def move(cid: str, body: Target):
        c = clip(cid)
        editable(speaker(c["speaker_id"]))
        with ledger.tx() as conn:
            if body.speaker_id == "new":
                number = (ledger.scalar("SELECT MAX(order_key) FROM speakers") or 0) + 1
                dest = f"spk_{number:04d}"
                old = speaker(c["speaker_id"])
                ledger.insert(
                    "speakers",
                    {
                        "speaker_id": dest,
                        "created_at": now(),
                        "channel_ids": old["channel_ids"],
                        "inputs": old["inputs"],
                        "face_n": 0,
                        "order_key": number,
                    },
                    conn=conn,
                )
            else:
                dest = body.speaker_id
                editable(speaker(dest))
            move_vectors(conn, c, c["speaker_id"], dest)
            conn.execute(
                "UPDATE clips SET speaker_id=?, assign='manual', decision='pending' WHERE clip_id=?",
                (dest, cid),
            )
            ledger.log("review", f"move {cid} {c['speaker_id']} -> {dest}", conn)
        return {"speaker_id": dest}

    @app.get("/media/clip/{cid}")
    def media(cid: str):
        c = clip(cid)
        path = ws.root / c["file"]
        if c["location"] != "local" or not path.is_file():
            raise HTTPException(410, f"Clip không còn ở máy ({c['location']})")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/media/strip/{cid}")
    def strip(cid: str):
        c = clip(cid)
        path = ws.faces / c["video_id"] / f"{cid}.jpg"
        if not path.is_file():
            raise HTTPException(404, "No face strip")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/media/photo/{name}")
    def photo(name: str):
        path = (ws.registry / name).resolve()
        if not path.is_relative_to(ws.registry.resolve()) or not path.is_file():
            raise HTTPException(404, "No photo")
        return FileResponse(path, media_type="image/jpeg")

    return app
