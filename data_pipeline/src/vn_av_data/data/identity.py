"""Bước 04 identify: gán speaker cho clip mới cắt bằng face embedding (SFace) + voice (WeSpeaker).

Theo từng dòng nguồn (input):
- nguồn một người: mọi clip là một cụm; clip lệch mặt / lệch giọng so với trung tâm cụm được
  gắn cờ `face_outlier` / `voice_outlier` (không tự loại).
- nguồn nhiều người: gom cụm theo mặt (leader clustering + gộp cụm gần nhau).
Mỗi cụm so với registry đã lưu (vector tổng của từng speaker, không tính lại clip cũ):
1. speaker cùng kênh: sim ≥ `match_high` → gán luôn;
2. toàn bộ registry: sim ≥ `match_high`, cách hạng 2 ≥ `match_margin`, giọng không mâu thuẫn → gán;
3. sim ≥ `match_low` → tạo speaker mới kèm gợi ý gộp (người duyệt quyết định);
4. còn lại → speaker mới.
Ngưỡng mặc định là ước lượng ban đầu; `calibrate` gợi ý ngưỡng từ các speaker đã duyệt.
"""

import json

import numpy as np

from vn_av_data.features.face import normalize
from vn_av_data.ledger import now

STRIP_SIDE = 96


def leader_clusters(vectors, threshold):
    """Gom cụm cosine: gán vào tâm gần nhất nếu ≥ ngưỡng, gộp tâm gần nhau, gán lại một lượt."""
    vectors = np.asarray(vectors, np.float32)
    sums, members = [], []
    for i, v in enumerate(vectors):
        if sums:
            centers = np.stack([normalize(s) for s in sums])
            sims = centers @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                sums[j] = sums[j] + v
                members[j].append(i)
                continue
        sums.append(v.copy())
        members.append([i])
    merged = True
    while merged and len(sums) > 1:
        merged = False
        centers = np.stack([normalize(s) for s in sums])
        sims = centers @ centers.T
        np.fill_diagonal(sims, -1)
        a, b = np.unravel_index(int(np.argmax(sims)), sims.shape)
        if sims[a, b] >= threshold:
            a, b = min(a, b), max(a, b)
            sums[a] = sums[a] + sums[b]
            members[a].extend(members[b])
            del sums[b], members[b]
            merged = True
    centers = np.stack([normalize(s) for s in sums])
    labels = np.argmax(vectors @ centers.T, axis=1)
    return [
        np.flatnonzero(labels == k).tolist() for k in range(len(centers)) if (labels == k).any()
    ]


def save_photo(ws, clip, speaker_id, k):
    """Ảnh đại diện = mặt ở giữa clip (ô thứ 3 của dải 5 mặt)."""
    import cv2

    strip = ws.faces / clip["video_id"] / f"{clip['clip_id']}.jpg"
    image = cv2.imread(str(strip))
    if image is None:
        return None
    tiles = max(1, image.shape[1] // STRIP_SIDE)
    middle = image[:, (tiles // 2) * STRIP_SIDE : (tiles // 2 + 1) * STRIP_SIDE]
    target = ws.registry / f"{speaker_id}_{k}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), middle, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return target.relative_to(ws.root).as_posix()


class Registry:
    def __init__(self, ws, ledger, cfg):
        self.ws, self.ledger, self.cfg = ws, ledger, cfg

    def speakers(self):
        return self.ledger.rows(
            "SELECT * FROM speakers WHERE merged_into='' AND review!='excluded' ORDER BY order_key"
        )

    def match(self, face, voice, channels):
        """(speaker_id | None, cách gán, gợi ý)."""
        cfg = self.cfg
        rows = [s for s in self.speakers() if s["face_n"]]
        if not rows:
            return None, "new", None
        centers = np.stack([normalize(s["face_sum"]) for s in rows])
        sims = centers @ face
        order = np.argsort(-sims)
        best = rows[order[0]]
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        same = [i for i, s in enumerate(rows) if set(s["channel_ids"] or []) & channels]
        if same:
            i = max(same, key=lambda i: sims[i])
            if sims[i] >= cfg["match_high"] and self.voice_ok(rows[i], voice):
                return rows[i]["speaker_id"], "auto_channel", None
        top = float(sims[order[0]])
        if top >= cfg["match_high"] and top - second >= cfg["match_margin"]:
            if self.voice_ok(best, voice):
                return best["speaker_id"], "auto", None
        if top >= cfg["match_low"]:
            return None, "new", {"speaker_id": best["speaker_id"], "face_sim": round(top, 3)}
        return None, "new", None

    def voice_ok(self, speaker, voice):
        if voice is None or not speaker["voice_n"]:
            return True
        return float(normalize(speaker["voice_sum"]) @ voice) >= self.cfg["voice_match_low"]

    def create(self, conn, channels, input_id, suggestion):
        number = (self.ledger.scalar("SELECT MAX(order_key) FROM speakers") or 0) + 1
        speaker_id = f"spk_{number:04d}"
        self.ledger.insert(
            "speakers",
            {
                "speaker_id": speaker_id,
                "created_at": now(),
                "channel_ids": sorted(channels),
                "inputs": [input_id],
                "face_sum": np.zeros(128, np.float32),
                "face_n": 0,
                "suggestion": json.dumps(suggestion) if suggestion else "",
                "order_key": number,
            },
            conn=conn,
        )
        return speaker_id

    def add(self, conn, speaker_id, clips, channels, input_id):
        s = self.ledger.decode(
            conn.execute("SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone()
        )
        faces = np.stack([c["face"] for c in clips])
        voices = [c["voice"] for c in clips if c["voice"] is not None]
        values = {
            "face_sum": (s["face_sum"] if s["face_n"] else 0) + faces.sum(0),
            "face_n": s["face_n"] + len(clips),
            "channel_ids": sorted(set(s["channel_ids"] or []) | channels),
            "inputs": sorted(set(s["inputs"] or []) | {input_id}),
        }
        if voices:
            base = s["voice_sum"] if s["voice_n"] else 0
            values["voice_sum"] = base + np.stack(voices).sum(0)
            values["voice_n"] = s["voice_n"] + len(voices)
        photos = list(s["photos"] or [])
        if len(photos) < self.cfg["photos_per_speaker"]:
            seen = {p["video"] for p in photos}
            ranked = sorted(clips, key=lambda c: -(c["face_consistency"] or 0))
            for clip in ranked:
                if len(photos) >= self.cfg["photos_per_speaker"]:
                    break
                if clip["video_id"] in seen:
                    continue
                path = save_photo(self.ws, clip, speaker_id, len(photos))
                if path:
                    photos.append({"video": clip["video_id"], "path": path})
                    seen.add(clip["video_id"])
            values["photos"] = photos
        self.ledger.update("speakers", "speaker_id", speaker_id, values, conn)


def identify_batch(ws, ledger, batch, cfg, voice_embedder=None):
    from vn_av_data.features.voice import VoiceEmbedder, clip_pcm

    cfg = dict(cfg["identify"])
    undone = ledger.scalar(
        "SELECT COUNT(*) FROM videos WHERE batch=? AND dl_status='done' AND cut_status!='done'",
        (batch,),
    )
    if undone:
        raise ValueError(f"Batch {batch}: {undone} videos not cut yet; run cut first")
    if voice_embedder is None and cfg["voice"]:
        voice_embedder = VoiceEmbedder(cfg["voice_model"])
    registry = Registry(ws, ledger, cfg)
    videos = ledger.rows(
        "SELECT v.video_id, v.input_id, v.channel_id, i.speakers FROM videos v "
        "JOIN inputs i USING(input_id) WHERE v.batch=? AND v.cut_status='done' "
        "AND v.id_status!='done' ORDER BY v.input_id, v.rank",
        (batch,),
    )
    groups = {}
    for v in videos:
        groups.setdefault(v["input_id"], []).append(v)
    totals = {"inputs": 0, "clips": 0, "new_speakers": 0, "matched": 0, "suggestions": 0}
    for input_id, group in groups.items():
        ids = [v["video_id"] for v in group]
        marks = ",".join("?" for _ in ids)
        clips = ledger.rows(
            f"SELECT * FROM clips WHERE video_id IN ({marks}) AND speaker_id IS NULL "
            "ORDER BY video_id, start_s",
            ids,
        )
        if voice_embedder is not None:
            for clip in clips:
                if clip["voice"] is None:
                    clip["voice"] = voice_embedder.embed(clip_pcm(ws.root / clip["file"]))
        channel_of = {v["video_id"]: v["channel_id"] for v in group}
        print(f"[input {input_id}] {len(clips)} clip", flush=True)
        with ledger.tx() as conn:
            for clip in clips:
                if clip["voice"] is not None:
                    ledger.update(
                        "clips", "clip_id", clip["clip_id"], {"voice": clip["voice"]}, conn
                    )
            if not clips:
                clusters = []
            elif group[0]["speakers"] == "1":
                clusters = [list(range(len(clips)))]
            else:
                clusters = leader_clusters([c["face"] for c in clips], cfg["cluster_threshold"])
            for members in clusters:
                part = [clips[i] for i in members]
                face = normalize(np.stack([c["face"] for c in part]).sum(0))
                voices = [c["voice"] for c in part if c["voice"] is not None]
                voice = normalize(np.stack(voices).sum(0)) if voices else None
                channels = {channel_of[c["video_id"]] for c in part if channel_of[c["video_id"]]}
                speaker_id, how, suggestion = registry.match(face, voice, channels)
                if speaker_id is None and len(part) < cfg["min_cluster_clips"]:
                    for c in part:
                        ledger.update(
                            "clips", "clip_id", c["clip_id"], {"assign": "unassigned"}, conn
                        )
                    continue
                if speaker_id is None:
                    speaker_id = registry.create(conn, channels, input_id, suggestion)
                    totals["new_speakers"] += 1
                    totals["suggestions"] += bool(suggestion)
                else:
                    totals["matched"] += 1
                registry.add(conn, speaker_id, part, channels, input_id)
                for c in part:
                    flags = [f for f in (c["flags"] or "").split(",") if f]
                    if float(c["face"] @ face) < cfg["face_outlier"]:
                        flags.append("face_outlier")
                    if voice is not None and c["voice"] is not None:
                        if float(c["voice"] @ voice) < cfg["voice_outlier"]:
                            flags.append("voice_outlier")
                    ledger.update(
                        "clips",
                        "clip_id",
                        c["clip_id"],
                        {
                            "speaker_id": speaker_id,
                            "assign": how,
                            "flags": ",".join(sorted(set(flags))),
                        },
                        conn,
                    )
                totals["clips"] += len(part)
            for vid in ids:
                ledger.update("videos", "video_id", vid, {"id_status": "done"}, conn)
        totals["inputs"] += 1
    ledger.log("identify", f"batch {batch}: {totals}")
    return {"batch": batch, **totals}


def calibrate(ledger):
    """Phân bố sim clip ↔ speaker của chính nó và ↔ speaker khác, trên speaker đã duyệt xong."""
    speakers = ledger.rows(
        "SELECT * FROM speakers WHERE review='done' AND merged_into='' AND face_n>0"
    )
    if len(speakers) < 3:
        raise ValueError("Need at least 3 reviewed speakers to calibrate thresholds")
    centers = np.stack([normalize(s["face_sum"]) for s in speakers])
    index = {s["speaker_id"]: i for i, s in enumerate(speakers)}
    same, other = [], []
    for s in speakers:
        clips = ledger.rows(
            "SELECT face FROM clips WHERE speaker_id=? AND decision='keep'", (s["speaker_id"],)
        )
        for c in clips:
            sims = centers @ c["face"]
            same.append(float(sims[index[s["speaker_id"]]]))
            other.append(float(np.delete(sims, index[s["speaker_id"]]).max()))
    same, other = np.asarray(same), np.asarray(other)
    return {
        "clips": len(same),
        "same_speaker_p1_p5_p50": np.percentile(same, [1, 5, 50]).round(3).tolist(),
        "best_other_p50_p95_p99": np.percentile(other, [50, 95, 99]).round(3).tolist(),
        "suggest_match_high": round(float(max(np.percentile(other, 99) + 0.05, 0.3)), 3),
        "suggest_match_low": round(float(np.percentile(other, 95)), 3),
    }
