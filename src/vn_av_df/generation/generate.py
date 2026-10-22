"""Thực thi kế hoạch v003: job giọng (①②③ + track B cho ⑥) và job lip-sync (④⑤⑥).

Phiên "voice" phải chạy trước phiên lip-sync có cell ⑥ (cần track B). Mỗi job ghi
`records/<key>.json` (danh sách sample, hoặc {"skipped": lý do}); chạy lại thì bỏ qua job đã có
record, xoá output dở của job chưa xong. Mọi sample encode cùng một cấu hình (H.264 CRF, AAC
48 kHz), tiếng luôn lấy từ PCM do pipeline dựng, không lấy tiếng generator.
"""

import queue
import random
import shutil
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from vn_av_df import worker
from vn_av_df.common.runtime import fingerprint, read_json, sha, write_json
from vn_av_df.data.media import decode, ffmpeg, probe
from vn_av_df.data.render import encode
from vn_av_df.generation.cells import labels
from vn_av_df.generation.lipsync import REDO
from vn_av_df.generation.plan import stable_int
from vn_av_df.generation.voice import audio as A
from vn_av_df.generation.voice.asr import text_of
from vn_av_df.generation.voice.edit import choose_edit

SPF = 1920  # mẫu 48 kHz mỗi frame 25 Hz
# Lỗi generator không thấy mặt ở một số frame (bước cắt chỉ kiểm mặt ở vài điểm lấy mẫu).
FACE_ERRORS = ("Face not detected", "Missing MuseTalk face", "Invalid face bbox")
# Bỏ liên tiếp chừng này job thì dừng: lỗi hệ thống (hết VRAM, worker hỏng), không phải clip khó.
MAX_SKIP_STREAK = 10


def skip_reason(exc):
    """Lý do bỏ job, hoặc None nếu phải dừng.

    Bỏ qua: lỗi generator báo cho riêng một job ("Worker failed", worker vẫn sống).
    Dừng: worker chết/timeout và lỗi ở code pipeline (decode, encode, nhãn).
    """
    text = str(exc)
    face = next((m for m in FACE_ERRORS if m in text), None)
    if face:
        return "face", face
    if text.startswith("Worker failed: "):
        lines = [x.strip() for x in text.removeprefix("Worker failed: ").splitlines() if x.strip()]
        # Dòng đầu là "<generator> error: Traceback ...", lỗi thật nằm ở dòng cuối.
        return "generator_error", " … ".join(lines[:1] + lines[1:][-1:])[:300]
    return None


def free_ram():
    """RAM còn trống (Linux); in cùng tiến độ để thấy RAM cạn trước khi kernel chết."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return f", RAM trống {int(line.split()[1]) / 2**20:.1f} GB"
    except OSError:
        pass
    return ""


def retry(record):
    """Record bỏ vì lỗi generator (vd. lỗi code đã sửa): phiên sau làm lại job đó."""
    return read_json(record).get("kind") == "generator_error"


def finished(record):
    return record.exists() and not retry(record)


def has_finished(out, plan, generator):
    """Đã có job xong của generator: chỉ khi đó setup (provenance) của nó mới phải giữ nguyên."""
    keys = (j["key"] for j in plan["jobs"] if j.get("generator") == generator)
    return any(finished(Path(out) / "records" / f"{key}.json") for key in keys)


def adapter_for(name):
    if name == "wav2lip_gan":
        from vn_av_df.generation import wav2lip

        return wav2lip
    if name == "musetalk_1_5":
        from vn_av_df.generation import musetalk

        return musetalk
    from vn_av_df.generation.lipsync import Adapter

    return Adapter(name)


def random_span(rng, frames, options):
    """Đoạn partial dài trong partial_range_s (mặc định 0,3-3 s, lưới frame), cách hai đầu ≥ edge_margin_s, vị trí ngẫu nhiên đều."""
    low, high = options["partial_range_s"]
    margin = round(options["edge_margin_s"] * 25)
    longest = min(round(high * 25), frames - 2 * margin)
    if longest < round(low * 25):
        raise ValueError("Clip too short for a partial segment")
    length = rng.randint(round(low * 25), longest)
    start = rng.randint(margin, frames - margin - length)
    return start, start + length


def blend_compose(original, fake, a, b, rng, options):
    """frames[a:b] = fake; với xác suất blend_probability trộn dần k = 2-6 frame ở hai đầu."""
    frames = original.copy()
    frames[a:b] = fake[a:b]
    k = 0
    if rng.random() < options["blend_probability"]:
        k = min(rng.randint(*options["blend_frames"]), (b - a) // 3)
        for i in range(k):
            alpha = (i + 1) / (k + 1)
            for t in (a + i, b - 1 - i):
                frames[t] = (alpha * fake[t] + (1 - alpha) * original[t]).astype(original.dtype)
    return frames, k


class Context:
    def __init__(self, cfg, plan, out, engine=None, transcriber=None):
        self.cfg, self.plan, self.out = cfg, plan, Path(out)
        self.options = plan["options"]
        self.clean = Path(cfg["clean_dataset"])
        self.engine, self.transcriber = engine, transcriber
        self.lock = threading.Lock()

    def decode(self, row):
        return decode(
            self.clean / row["video"], max_side=self.options["max_side"], sample_rate=48000
        )

    def emit(
        self, key, variant, frames, pcm, parent, cell, kind, span=None, video_from=None, **extra
    ):
        n = len(frames)
        path = self.out / "clips" / f"{key}_{variant}.mp4"
        encode(frames, pcm, path, self.options["crf"], video_from)
        if abs(probe(path)["duration_s"] - n / 25) > 0.1:
            raise ValueError("Encoded duration differs")
        row = {
            "sample_id": f"{key}_{variant}",
            "video": path.relative_to(self.out).as_posix(),
            "sha256": sha(path),
            "duration_s": n / 25,
            "split": parent["split"],
            "source_id": parent["source_id"],
            "source_clip_id": parent["clip_id"],
            "speaker_id": parent.get("speaker_id"),
            "gender": parent.get("gender"),
            "cell": cell,
            "kind": kind,
            "variant": variant,
            "generator": "none",
            "generator_version": "matched-encode-v3",
            "voice_method": None,
            "partial_span": [round(span[0], 4), round(span[1], 4)] if span else None,
            "processing_profile": f"h264-crf{self.options['crf']}-aac48k",
            "review_status": "pending",
            "group_id": parent["clip_id"],
            "parent_ids": [parent["source_id"]],
            "speaker_ids": [parent["speaker_id"]] if parent.get("speaker_id") else [],
            "sync_status": "reviewed_match" if cell == "real" else "unknown",
            **labels(cell, kind, n / 25, span),
            **extra,
        }
        if cell == "sham":
            row["control_type"] = "conventional_audio_splice"
        return row


def run_voice(ctx, job, engine=None):
    """① real, ② sham full/partial, ③ track A (full) / B (partial, ghép) / C (sửa trực tiếp)."""
    engine = engine or ctx.engine
    parent, key, seed = job["parent"], job["key"], ctx.plan["seed"]
    rng = random.Random(stable_int(f"{seed}:{key}"))
    media = ctx.decode(parent)
    frames, pcm = media["frames"], media["pcm"]
    n = len(frames)
    samples = n * SPF
    rows = [ctx.emit(key, "real", frames, pcm, parent, "real", "full")]
    # Mọi sample của job voice cùng hình với real: chép luồng hình, khỏi encode lại.
    same = ctx.out / rows[0]["video"]
    noise = A.noise_floor(pcm)
    if job.get("sham_donor"):
        donor = ctx.decode(job["sham_donor"])["pcm"][:samples]
        if len(donor) < samples:
            raise ValueError("Sham donor shorter than parent")
        extra = {"audio_source_id": job["sham_donor"]["source_id"]}
        extra["parent_ids"] = sorted({parent["source_id"], job["sham_donor"]["source_id"]})
        rows.append(
            ctx.emit(
                key, "sham_full", frames, donor, parent, "sham", "full", video_from=same, **extra
            )
        )
        a, b = random_span(rng, n, ctx.options)
        spliced = A.splice(pcm, donor[a * SPF : b * SPF], a * SPF, b * SPF, noise=noise)
        rows.append(
            ctx.emit(
                key,
                "sham_partial",
                frames,
                spliced,
                parent,
                "sham",
                "partial",
                (a / 25, b / 25),
                video_from=same,
                **extra,
            )
        )
    tracks = ctx.out / "tracks"
    with tempfile.TemporaryDirectory(dir=ctx.out) as tmp:
        tmp = Path(tmp)
        words = ctx.transcriber(A.resample(pcm, 48000, 16000), key=sha(ctx.clean / parent["video"]))
        reference = job["reference"]
        ref_pcm = ctx.decode(reference)["pcm"]
        ref_words = ctx.transcriber(
            A.resample(ref_pcm, 48000, 16000), key=sha(ctx.clean / reference["video"])
        )
        if not words or not ref_words:
            return rows, {"voice_skipped": "no speech recognized"}
        ref_wav = tmp / "reference.wav"
        A.write_wav(ref_wav, ref_pcm)
        method = job["voice"]
        voice = {"voice_method": method, "audio_source_id": parent["source_id"]}
        # Track A: đọc lại cả câu bằng giọng nhân bản → thay toàn bộ tiếng.
        engine.tts(method, text_of(words), ref_wav, text_of(ref_words), tmp / "a.wav", tmp)
        track_a, ratio_a = A.fit_whole(A.read_wav(tmp / "a.wav"), samples, ffmpeg())
        rows.append(
            ctx.emit(
                key,
                "voice_full",
                frames,
                track_a,
                parent,
                "voice",
                "full",
                video_from=same,
                **voice,
                stretch_ratio=round(ratio_a, 3),
                transcript=text_of(words),
            )
        )
        # Track B: đổi một cụm 0,3-3 s, TTS cụm mới, co giãn vào đúng chỗ, ghép như sham.
        edit = choose_edit(
            words,
            n / 25,
            stable_int(f"{seed}:{key}:edit"),
            *ctx.options["partial_range_s"],
            ctx.options["edge_margin_s"],
        )
        if edit is None:
            return rows, {"voice_skipped": "no editable phrase"}
        p, q = round(edit["span"][0] * 25), round(edit["span"][1] * 25)
        engine.tts(method, edit["new"], ref_wav, text_of(ref_words), tmp / "b.wav", tmp)
        insert, ratio_b = A.time_stretch(
            A.trim_silence(A.read_wav(tmp / "b.wav")), (q - p) * SPF, ffmpeg()
        )
        track_b = A.splice(pcm, insert, p * SPF, q * SPF, noise=noise)
        span = (p / 25, q / 25)
        rows.append(
            ctx.emit(
                key,
                "voice_partial",
                frames,
                track_b,
                parent,
                "voice",
                "partial",
                span,
                video_from=same,
                **voice,
                stretch_ratio=round(ratio_b, 3),
                edit=edit,
            )
        )
        A.write_wav(tracks / f"{key}.wav", track_b)
        write_json(
            tracks / f"{key}.json",
            {"span": list(span), "edit": edit, "voice_method": method, "stretch_ratio": ratio_b},
        )
        # Track C (chỉ test): sửa trực tiếp, không có vết ghép kiểu B.
        if job.get("inpaint"):
            whole = tmp / "parent.wav"
            A.write_wav(whole, pcm)
            before = [w["word"] for w in words if w["end"] <= edit["span"][0] + 1e-6]
            after = [w["word"] for w in words if w["start"] >= edit["span"][1] - 1e-6]
            text = " ".join([*before, edit["new"], *after])
            engine.inpaint(job["inpaint"], whole, span, text, tmp / "c.wav")
            painted = A.read_wav(tmp / "c.wav")
            painted = np.pad(painted, (0, max(0, samples - len(painted))))[:samples]
            # Chỉ lấy đoạn đã sửa (vocoder tạo lại cả câu), nối 5 ms để giữ tiếng gốc ngoài đoạn.
            track_c = A.splice(
                pcm,
                painted[p * SPF : q * SPF],
                p * SPF,
                q * SPF,
                crossfade_s=0.005,
                match_loudness=False,
            )
            rows.append(
                ctx.emit(
                    key,
                    "voice_inpaint",
                    frames,
                    track_c,
                    parent,
                    "voice",
                    "partial",
                    span,
                    video_from=same,
                    voice_method=job["inpaint"],
                    audio_source_id=parent["source_id"],
                    edit=edit,
                )
            )
    return rows, {}


def run_lipsync(ctx, job, synthesize, gen_info, gpu=None):
    parent, key, cell = job["parent"], job["key"], job["cell"]
    rng = random.Random(stable_int(f"{ctx.plan['seed']}:{key}"))
    media = ctx.decode(parent)
    frames, pcm = media["frames"], media["pcm"]
    n, samples = len(frames), len(frames) * SPF
    note = {}
    track = None
    if cell == "both":
        wav = ctx.out / "tracks" / f"{job['voice_key']}.wav"
        if not wav.is_file():
            cell, note = "source", {"cell_fallback": "both→source (no voice track)"}
        else:
            track = A.read_wav(wav)[:samples]
            info = read_json(wav.with_suffix(".json"))
    driving = {
        "source": ctx.clean / parent["video"],
        "donor": ctx.clean / job["donor"]["video"] if job.get("donor") else None,
        "both": ctx.out / "tracks" / f"{job['voice_key']}.wav" if track is not None else None,
    }[cell]
    with tempfile.TemporaryDirectory(dir=ctx.out) as tmp:
        target = Path(tmp) / "generated.mkv"
        call = {"gpu": gpu} if gpu is not None else {}
        synthesize(
            ctx.cfg,
            ctx.clean / parent["video"],
            driving,
            target,
            n,
            frames.shape[2],
            frames.shape[1],
            **call,
        )
        fake = decode(target, max_side=ctx.options["max_side"], sample_rate=48000)["frames"]
    if fake.shape != frames.shape:
        raise ValueError(f"Generator changed geometry {fake.shape} != {frames.shape}")
    base = {
        "generator": job["generator"],
        "generator_version": gen_info["version"],
        "generator_checkpoint_sha256": gen_info.get("checkpoint_sha256"),
        **note,
    }
    if cell == "source":
        full_pcm, extra = pcm, {"audio_source_id": parent["source_id"]}
    elif cell == "donor":
        donor = ctx.decode(job["donor"])["pcm"][:samples]
        full_pcm = donor
        extra = {
            "audio_source_id": job["donor"]["source_id"],
            "parent_ids": sorted({parent["source_id"], job["donor"]["source_id"]}),
        }
    else:
        full_pcm = track
        extra = {"audio_source_id": parent["source_id"], "voice_method": info["voice_method"]}
    rows = [
        ctx.emit(
            key,
            "full",
            fake,
            full_pcm,
            parent,
            cell,
            "full",
            info["span"] if cell == "both" else None,
            **base,
            **extra,
        )
    ]
    if cell == "both":
        a, b = round(info["span"][0] * 25), round(info["span"][1] * 25)
        part_pcm = track
    else:
        a, b = random_span(rng, n, ctx.options)
        part_pcm = (
            pcm
            if cell == "source"
            else A.splice(
                pcm, full_pcm[a * SPF : b * SPF], a * SPF, b * SPF, noise=A.noise_floor(pcm)
            )
        )
    mixed, k = blend_compose(frames, fake, a, b, rng, ctx.options)
    rows.append(
        ctx.emit(
            key,
            "partial",
            mixed,
            part_pcm,
            parent,
            cell,
            "partial",
            (a / 25, b / 25),
            **base,
            **extra,
            blend_frames=k,
        )
    )
    return rows, {}


def batch_clips(plan, batch):
    """Clip cha của đợt k / n (batch = (k, n)): clip theo thứ tự trong plan, cắt n phần liền nhau.

    Thứ tự plan cũng là thứ tự phiên voice chạy, nên đợt đầu có giọng trước.
    """
    k, n = batch
    if not 1 <= k <= n:
        raise ValueError(f"Batch must be k/n with 1 <= k <= n, got {k}/{n}")
    clips = list(dict.fromkeys(j["parent"]["clip_id"] for j in plan["jobs"]))
    return set(clips[(k - 1) * len(clips) // n : k * len(clips) // n])


def run_generation(cfg, phase, engine=None, transcriber=None, synthesizers=None):
    """phase: "voice" | "lipsync". cfg["session_generators"] giới hạn generator của phiên,
    cfg["session_batch"] = (k, n) giới hạn ở đợt k / n clip cha."""
    plan = read_json(cfg["plan"])
    out = Path(cfg["generated_dataset"])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in plan["jobs"] if j["type"] == phase]
    active = None
    if phase == "lipsync":
        active = set(cfg.get("session_generators") or {j["generator"] for j in jobs})
        jobs = [j for j in jobs if j["generator"] in active]
    if cfg.get("session_batch"):
        keep = batch_clips(plan, cfg["session_batch"])
        jobs = [j for j in jobs if j["parent"]["clip_id"] in keep]
    if phase == "lipsync":
        # Thiếu record giọng thì run_lipsync lặng lẽ đổi ⑥ thành ④; bắt chạy / gộp voice trước.
        waiting = sum(
            j["cell"] == "both" and not finished(out / "records" / f"{j['voice_key']}.json")
            for j in jobs
        )
        if waiting:
            raise RuntimeError(f"{waiting} job ⑥ chưa có record giọng: chạy hoặc gộp voice trước")
    lock = out / "generation.json"
    log = (
        read_json(lock)
        if lock.exists()
        else {"plan_sha": fingerprint(plan), "sessions": [], "provenance": {}}
    )
    if log["plan_sha"] != fingerprint(plan):
        raise ValueError("Output belongs to another plan; use a new output folder")
    infos, synth = {}, {}
    if phase == "lipsync":
        for name in sorted(active):
            if synthesizers and name in synthesizers:
                synth[name], infos[name] = (
                    synthesizers[name],
                    {"version": "test", "generator": name},
                )
            else:
                adapter = adapter_for(name)
                synth[name], infos[name] = adapter.synthesize, adapter.provenance(cfg)
            old = log["provenance"].get(name, infos[name])
            if old != infos[name] and has_finished(out, plan, name):
                if name not in REDO or old.get("adapter_sha256") != REDO[name]:
                    raise ValueError(f"{name} differs from earlier sessions")
                # Adapter cũ có lỗi: job đã xong của generator thành job lỗi, phiên này làm lại.
                for job in plan["jobs"]:
                    record = out / "records" / f"{job['key']}.json"
                    if job.get("generator") == name and finished(record):
                        write_json(record, {"skipped": "adapter cũ lỗi", "kind": "generator_error"})
            log["provenance"][name] = infos[name]
    else:
        if transcriber is None:
            from vn_av_df.generation.voice.asr import Transcriber

            transcriber = Transcriber(cfg["generation"]["asr_model"], cfg["device"], out / "asr")
    write_json(lock, log)
    ctx = Context(cfg, plan, out, engine, transcriber)
    records = out / "records"
    records.mkdir(exist_ok=True)
    todo = [j for j in jobs if not finished(records / f"{j['key']}.json")]
    print(f"{phase}: {len(jobs)} job, còn {len(todo)}", flush=True)
    for leftover in out.glob("tmp*"):
        shutil.rmtree(leftover, ignore_errors=True)
    gpus = list(cfg.get("generation_gpus") or [None])
    engines = {gpu: engine for gpu in gpus}
    if phase == "voice" and engine is None:
        from vn_av_df.generation.voice.engines import Engine

        # Mỗi luồng một GPU: worker giọng nạp model trên GPU của luồng gọi nó.
        engines = {gpu: Engine(cfg, gpu) for gpu in gpus}
    deadline = cfg.get("generation_deadline")
    began = time.monotonic()
    state = {"done": 0, "skipped": 0, "streak": 0}
    timing = Counter()
    count = Counter()

    def one(job, gpu):
        if deadline is not None and time.time() > deadline:
            return None
        for stale in (out / "clips").glob(f"{job['key']}_*"):
            stale.unlink()
        tick = time.monotonic()
        try:
            if phase == "voice":
                rows, note = run_voice(ctx, job, engines[gpu])
            else:
                name = job["generator"]
                rows, note = run_lipsync(ctx, job, synth[name], infos[name], gpu)
        except Exception as exc:
            reason = skip_reason(exc)
            if reason is None:
                raise
            write_json(records / f"{job['key']}.json", {"skipped": reason[1], "kind": reason[0]})
            with ctx.lock:
                state["skipped"] += 1
                state["streak"] += 1
                if state["streak"] >= MAX_SKIP_STREAK:
                    raise RuntimeError(
                        f"{state['streak']} jobs skipped in a row: {reason}"
                    ) from exc
            print(f"[{phase}] {job['key']} bỏ qua ({reason[0]}): {reason[1]}", flush=True)
            return []
        write_json(records / f"{job['key']}.json", {"rows": rows, **note})
        with ctx.lock:
            state["done"] += 1
            state["streak"] = 0
            label = job.get("generator", job.get("voice"))
            timing[label] += time.monotonic() - tick
            count[label] += 1
            spent = time.monotonic() - began
            left = spent / state["done"] * (len(todo) - state["done"] - state["skipped"])
        print(
            f"[{phase}] {job['key']} {job.get('generator') or job.get('voice')} "
            f"{job['parent']['split']} → {len(rows)} sample | {time.monotonic() - tick:.0f}s, "
            f"xong {state['done']}/{len(todo)}, còn ~{left / 60:.0f} phút{free_ram()}",
            flush=True,
        )
        return rows

    free = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def on_free_gpu(job):
        # Mỗi job đang chạy giữ riêng một GPU, nên hai job không gọi cùng lúc một worker.
        gpu = free.get()
        try:
            return one(job, gpu)
        finally:
            free.put(gpu)

    # Chạy hết một hệ giọng / generator rồi tắt worker của nó: mỗi lúc chỉ vài process model
    # nằm trong RAM (Kaggle T4×2 ~30 GB; giữ mọi hệ trên cả 2 GPU thì hết RAM).
    groups = {}
    # Lip-sync chạy theo thứ tự GENERATORS của phiên: generator cần thử đặt đầu, lỗi hiện sớm.
    order = list(cfg.get("session_generators") or [])
    for job in sorted(
        todo, key=lambda j: order.index(j["generator"]) if j.get("generator") in order else 0
    ):
        groups.setdefault(job.get("voice") or job.get("generator"), []).append(job)
    failed = {}
    try:
        for name, group in groups.items():
            try:
                with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                    futures = [pool.submit(on_free_gpu, job) for job in group]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        # Không huỷ thì pool vẫn chạy hết job còn lại và nuốt lỗi của chúng.
                        for future in futures:
                            future.cancel()
                        raise
            except Exception as exc:
                # Một hệ lỗi thì dừng hệ đó, các hệ khác vẫn chạy và phiên kết thúc bình thường:
                # Output của phiên Kaggle lỗi không gắn được vào phiên sau.
                text = f"{type(exc).__name__}: {exc}"
                # Giữ cả dòng cuối traceback: log upstream ở đầu dễ chiếm hết chỗ.
                failed[name] = text if len(text) <= 2000 else f"{text[:300]}\n...\n{text[-1700:]}"
                print(
                    f"[{phase}] {name} DỪNG, job chưa làm để phiên sau: {failed[name]}", flush=True
                )
            worker.close_all()
    finally:
        worker.close_all()
        log["sessions"].append(
            {
                "phase": phase,
                "started_unix": round(time.time() - (time.monotonic() - began)),
                "elapsed_s": round(time.monotonic() - began, 1),
                "done": state["done"],
                "skipped": state["skipped"],
                "seconds_per_job": {k: round(timing[k] / count[k], 1) for k in count},
            }
        )
        write_json(lock, log)
    remaining = sum(not finished(records / f"{j['key']}.json") for j in jobs)
    return {
        "phase": phase,
        "done": state["done"],
        "skipped": state["skipped"],
        "remaining": remaining,
        "failed": failed,
    }


def finalize(cfg, audit=True):
    """Gom record → manifest.jsonl (vn-av-df-v1); review.csv (nếu có) chỉ để loại mẫu lỗi.

    audit: kiểm lối tắt "đặc trưng ngây thơ" (khoảng lặng, độ to, độ dài) → shortcut_audit.json.
    """
    import csv

    from vn_av_df.data.groups import write_manifest
    from vn_av_df.dataset import SCHEMA, validate_rows

    plan = read_json(cfg["plan"])
    out = Path(cfg["generated_dataset"])
    missing = [
        j["key"] for j in plan["jobs"] if not (out / "records" / f"{j['key']}.json").exists()
    ]
    if missing:
        raise ValueError(f"{len(missing)} jobs not generated yet (e.g. {missing[:3]})")
    rows = []
    for job in plan["jobs"]:
        record = read_json(out / "records" / f"{job['key']}.json")
        rows.extend(record.get("rows", []))
    decisions = {}
    review = out / "review.csv"
    if review.exists():
        with review.open(encoding="utf-8-sig", newline="") as f:
            decisions = {r["sample_id"]: r["decision"] for r in csv.DictReader(f)}
    kept = [
        {**r, "review_status": "keep"}
        for r in rows
        if decisions.get(r["sample_id"], "keep") == "keep"
    ]
    validate_rows(kept, out, verify_media=True)
    write_manifest(out / "manifest.jsonl", kept)
    info = {
        "schema_version": SCHEMA,
        "samples": len(kept),
        "manifest_sha256": sha(out / "manifest.jsonl"),
        "plan_sha": fingerprint(plan),
        "cells": dict(Counter(r["cell"] for r in kept)),
        "splits": dict(Counter(r["split"] for r in kept)),
    }
    write_json(out / "dataset_info.json", info)
    if audit:
        from vn_av_df.generation.audit import audit as shortcut_audit

        report = shortcut_audit(out, kept)
        write_json(out / "shortcut_audit.json", report)
        info["shortcut_flags"] = report["flags"]
        if report["flags"]:
            print("CẢNH BÁO lối tắt (xem shortcut_audit.json):", report["flags"], flush=True)
    return info


def merge_previous(cfg, sources):
    """Gộp output các phiên trước (mỗi phiên một thư mục vn-av-df-forensics/ đã attach).

    Chép plan (phải giống nhau) và mọi file record / clip / track / asr còn thiếu; record đã có
    thì phải trùng nội dung. Phiên này sau đó chỉ làm các job chưa có record.
    """
    out, plan_path = Path(cfg["generated_dataset"]), Path(cfg["plan"])
    copied = 0
    for source in map(Path, sources):
        other_plan = source / plan_path.relative_to(cfg["root"])
        if not other_plan.is_file():
            raise FileNotFoundError(f"No plan in previous output: {other_plan}")
        if plan_path.exists():
            if read_json(plan_path) != read_json(other_plan):
                raise ValueError("Previous output used another plan")
        else:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(other_plan, plan_path)
        folder = source / out.relative_to(cfg["root"])
        log = folder / "generation.json"
        take = {}
        if log.is_file() and (out / "generation.json").exists():
            # Cùng generator chạy ở hai nơi với setup khác → record lẫn hai phiên bản; dừng trước khi chép.
            # Bên chưa xong job nào của generator đó (vd. phiên lỗi trước khi sửa code) thì nhường.
            mine, theirs, plan = (
                read_json(out / "generation.json"),
                read_json(log),
                read_json(plan_path),
            )
            for name, info in theirs["provenance"].items():
                if mine["provenance"].get(name, info) == info or not has_finished(out, plan, name):
                    take[name] = info
                elif has_finished(folder, plan, name):
                    raise ValueError(f"{name} ran with another setup in {source}")
        # Job ở đây bỏ vì lỗi generator nhưng phiên kia đã làm lại xong: lấy mọi file của job đó.
        here = out / "records"
        redone = {
            p.stem
            for p in (folder / "records").glob("*.json")
            if (here / p.name).exists() and retry(here / p.name) and not retry(p)
        }
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name == "generation.json":
                continue
            target = out / path.relative_to(folder)
            if target.exists() and path.stem.split("_")[0] not in redone:
                if (
                    path.parent.name == "records"
                    and not retry(path)
                    and read_json(target) != read_json(path)
                ):
                    raise ValueError(f"Conflicting record {path.name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
        if log.is_file():
            mine = (
                read_json(out / "generation.json") if (out / "generation.json").exists() else None
            )
            theirs = read_json(log)
            if mine is None:
                write_json(out / "generation.json", theirs)
            else:
                mine["sessions"] = mine["sessions"] + [
                    s for s in theirs["sessions"] if s not in mine["sessions"]
                ]
                mine["provenance"].update(take)
                write_json(out / "generation.json", mine)
    return {"files_copied": copied}
