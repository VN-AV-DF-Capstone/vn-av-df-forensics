"""Demo: tạo MỘT video fake từ một video real, cùng cách và cùng nhãn với bộ dữ liệu v003.

Dùng lại các hàm của pipeline sinh (ghép tiếng, co giãn, chọn cụm sửa, trộn mép partial, nhãn
6 ô) nên một mẫu tạo ở đây giống một mẫu trong bộ dữ liệu. Khác: một video lẻ, không có plan;
giọng mẫu để nhân bản lấy từ chính video đó; tiếng thật khác (⑤ donor) do người dùng đưa.

Ô hỗ trợ: ③ voice, ④ source, ⑤ donor, ⑥ both; kind = full | partial.
"""

import random
import tempfile
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import write_json
from vn_av_df.data.media import decode, ffmpeg
from vn_av_df.data.render import encode
from vn_av_df.generation.cells import labels
from vn_av_df.generation.generate import SPF, adapter_for, blend_compose, random_span
from vn_av_df.generation.voice import audio as A
from vn_av_df.generation.voice.asr import text_of
from vn_av_df.generation.voice.edit import choose_edit

CELLS = ("voice", "source", "donor", "both")


def frame_span(span, n, options):
    """Đoạn partial (giây) → frame; kiểm cách hai đầu ≥ edge_margin_s và trong 0,2-10 s."""
    a, b = round(span[0] * 25), round(span[1] * 25)
    margin = round(options["edge_margin_s"] * 25)
    if not (margin <= a < b <= n - margin) or not 5 <= b - a <= 250:
        raise ValueError(
            f"Đoạn {span[0]:.2f}-{span[1]:.2f} s không hợp lệ: cần dài 0,2-10 s và cách hai đầu "
            f"≥ {options['edge_margin_s']} s (video {n / 25:.1f} s)"
        )
    return a, b


def make_edit(words, n, options, seed, span=None, new_text=None):
    """Cụm sửa cho track B: người dùng cho (đoạn + câu mới) hoặc tự chọn như pipeline."""
    if span is not None and new_text:
        a, b = frame_span(span, n, options)
        old = [
            w["word"] for w in words if w["start"] >= a / 25 - 0.05 and w["end"] <= b / 25 + 0.05
        ]
        return {"span": [a / 25, b / 25], "old": " ".join(old), "new": new_text, "kind": "user"}
    edit = choose_edit(words, n / 25, seed, *options["partial_range_s"], options["edge_margin_s"])
    if edit is None:
        raise ValueError("Không tìm được cụm từ để sửa (ASR không nhận được lời?)")
    return edit


def donor_audio(path, samples):
    pcm = decode(path, max_duration=120, max_side=64, sample_rate=48000)["pcm"]
    if len(pcm) < samples:
        pcm = np.pad(pcm, (0, samples - len(pcm)))
    return pcm[:samples].astype(np.float32)


def generate_one(
    cfg,
    video,
    cell,
    kind="full",
    generator="wav2lip_gan",
    voice="f5_vi",
    span=None,
    new_text=None,
    donor=None,
    output=None,
    seed=0,
    engine=None,
    transcriber=None,
    synthesize=None,
):
    """→ (đường dẫn mp4, nhãn + thông tin). `video` nên đã chuẩn hoá (demo.media.normalize)."""
    if cell not in CELLS or kind not in ("full", "partial"):
        raise ValueError(f"cell ∈ {CELLS}, kind ∈ full/partial")
    if cell == "donor" and donor is None:
        raise ValueError("Ô ⑤ donor cần một đoạn tiếng thật khác (file video / audio)")
    options = cfg["generation"]
    rng = random.Random(seed)
    media = decode(video, max_duration=61, max_side=options["max_side"], sample_rate=48000)
    frames, pcm = media["frames"], media["pcm"]
    n, samples = len(frames), len(frames) * SPF
    output = Path(output or Path(video).with_name(f"fake_{cell}_{kind}.mp4"))
    output.parent.mkdir(parents=True, exist_ok=True)
    info = {"cell": cell, "kind": kind, "generator": None, "voice_method": None}
    noise = A.noise_floor(pcm)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        track, edit = None, None
        if cell in ("voice", "both"):
            if transcriber is None:
                from vn_av_df.generation.voice.asr import Transcriber

                transcriber = Transcriber(options["asr_model"], cfg["device"])
            if engine is None:
                from vn_av_df.generation.voice.engines import Engine

                engine = Engine(cfg)
            words = transcriber(A.resample(pcm, 48000, 16000))
            if not words:
                raise ValueError("ASR không nhận được lời trong video")
            reference = tmp / "reference.wav"  # nhân bản giọng từ chính video
            A.write_wav(reference, pcm)
            info["voice_method"] = voice
            info["transcript"] = text_of(words)
            if cell == "voice" and kind == "full":  # track A: đọc lại cả câu
                engine.tts(voice, text_of(words), reference, text_of(words), tmp / "a.wav", tmp)
                track, ratio = A.fit_whole(A.read_wav(tmp / "a.wav"), samples, ffmpeg())
                info["stretch_ratio"] = round(ratio, 3)
            else:  # track B: sửa một cụm, ghép vào tiếng gốc
                edit = make_edit(words, n, options, seed, span, new_text)
                p, q = round(edit["span"][0] * 25), round(edit["span"][1] * 25)
                engine.tts(voice, edit["new"], reference, text_of(words), tmp / "b.wav", tmp)
                insert, ratio = A.time_stretch(
                    A.trim_silence(A.read_wav(tmp / "b.wav")), (q - p) * SPF, ffmpeg()
                )
                track = A.splice(pcm, insert, p * SPF, q * SPF, noise=noise)
                info["stretch_ratio"] = round(ratio, 3)
                info["edit"] = edit
        if cell == "voice":
            out_frames, out_pcm = frames, track
            region = tuple(edit["span"]) if edit else None
        else:
            info["generator"] = generator
            if cell == "source":
                driving, full_pcm = Path(video), pcm
            elif cell == "donor":
                full_pcm = donor_audio(donor, samples)
                driving = tmp / "donor.wav"
                A.write_wav(driving, full_pcm)
            else:
                full_pcm = track
                driving = tmp / "track_b.wav"
                A.write_wav(driving, track)
            run = synthesize or adapter_for(generator).synthesize
            target = tmp / "generated.mkv"
            run(cfg, Path(video), driving, target, n, frames.shape[2], frames.shape[1])
            fake = decode(target, max_duration=61, max_side=options["max_side"], sample_rate=48000)[
                "frames"
            ]
            if fake.shape != frames.shape:
                raise ValueError(f"Generator đổi kích thước {fake.shape} != {frames.shape}")
            if kind == "full":
                out_frames, out_pcm = fake, full_pcm
                region = tuple(edit["span"]) if cell == "both" else None
            else:
                if cell == "both":
                    a, b = round(edit["span"][0] * 25), round(edit["span"][1] * 25)
                    out_pcm = track
                else:
                    a, b = frame_span(span, n, options) if span else random_span(rng, n, options)
                    out_pcm = (
                        pcm
                        if cell == "source"
                        else A.splice(
                            pcm, full_pcm[a * SPF : b * SPF], a * SPF, b * SPF, noise=noise
                        )
                    )
                out_frames, info["blend_frames"] = blend_compose(frames, fake, a, b, rng, options)
                region = (a / 25, b / 25)
    encode(out_frames, out_pcm, output, options["crf"])
    result = {**info, "duration_s": n / 25, **labels(cell, kind, n / 25, region)}
    write_json(output.with_suffix(".json"), result)
    return output, result
