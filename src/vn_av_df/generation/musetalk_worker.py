"""Bridge lossless dùng API inference chính thức MuseTalk 1.5, không gọi shell upstream.

Một lần: `python musetalk_worker.py request.json`. Chạy liên tục: `--serve '<config JSON>'`,
nạp VAE/UNet/Whisper/bộ dò mặt một lần rồi nhận từng yêu cầu (video, audio_video, output,
frames, width, height) qua stdin.
"""

import json
import sys
import tempfile
from pathlib import Path

# Python tự thêm thư mục script (src/vn_av_df/generation) vào sys.path; ở đó có musetalk.py che mất
# package musetalk/ của upstream (không có __init__.py nên thua module thường). Bỏ thư mục đó.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vn_av_df.common.runtime import read_json, run  # noqa: E402


def load(cfg):
    """Nạp model một lần; module preprocessing của MuseTalk tự nạp DWPose/bộ dò mặt khi import."""
    import torch
    from transformers import WhisperModel

    sys.path.insert(0, str(Path(cfg["repo"]).resolve()))
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model

    device = torch.device(cfg.get("device", "cuda"))
    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vae.vae.to(device=device, dtype=dtype).eval()
    unet.model.to(device=device, dtype=dtype).eval()
    pe.to(device=device, dtype=dtype).eval()
    whisper = (
        WhisperModel.from_pretrained("models/whisper", local_files_only=True)
        .to(device=device, dtype=dtype)
        .eval()
    )
    return dict(
        cfg=cfg,
        device=device,
        dtype=dtype,
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        processor=AudioProcessor(feature_extractor_path="models/whisper"),
        parser=FaceParsing(left_cheek_width=90, right_cheek_width=90),
    )


def synthesize(models, request):
    """Sinh đúng số frame nguồn; fail khi thiếu face/audio, không loop nguồn."""
    import cv2
    import numpy as np
    import torch
    from musetalk.utils.blending import get_image
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox
    from musetalk.utils.utils import datagen
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    from vn_av_df.data.media import decode, ffmpeg

    cfg, device, dtype = models["cfg"], models["device"], models["dtype"]
    vae, unet, pe = models["vae"], models["unet"], models["pe"]
    # Seed lại mỗi video: kết quả không phụ thuộc thứ tự/số video worker đã xử lý trước đó.
    torch.manual_seed(int(cfg.get("seed", 42)))
    n, w, h = request["frames"], request["width"], request["height"]
    source = decode(request["video"], max_side=max(w, h), sample_rate=48000)
    if Path(request["audio_video"]).suffix == ".wav":
        # Ô ⑥ lái bằng track giọng AI (chỉ có tiếng, 48 kHz), không phải video donor.
        from vn_av_df.generation.voice.audio import read_wav

        donor = {"pcm": read_wav(request["audio_video"])}
    else:
        donor = decode(request["audio_video"], max_side=max(w, h), sample_rate=48000)
    if len(source["frames"]) != n or len(donor["pcm"]) < n * 1920:
        raise ValueError("MuseTalk source/donor timeline differs from plan")
    with tempfile.TemporaryDirectory(dir=Path(request["output"]).parent) as folder:
        folder = Path(folder)
        images = []
        for i, frame in enumerate(source["frames"]):
            path = folder / f"{i:08d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Failed to write input frame")
            images.append(str(path))
        audio = donor["pcm"][: n * 1920]
        # MuseTalk đếm frame = floor(số mẫu / 16000 * 25) bằng số thực: với một số độ dài
        # (vd. 201, 203 frame) sai số làm tròn làm thiếu 1 frame. Thêm 10 ms im lặng ở cuối để
        # đủ n frame; chỉ dùng chunks[:n] nên không đổi tiếng của các frame đã sinh.
        audio16 = np.concatenate([resample_poly(audio, 1, 3), np.zeros(160)])
        wavfile.write(folder / "audio16.wav", 16000, audio16.astype(np.float32))
        wavfile.write(
            folder / "audio48.wav", 48000, (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        )
        coords, frames = get_landmark_and_bbox(images, 0)
        if len(coords) != n or any(tuple(b) == tuple(coord_placeholder) for b in coords):
            raise ValueError("Missing MuseTalk face; cannot label full clip as manipulated")
        latents, boxes = [], []
        with torch.no_grad():
            for frame, box in zip(frames, coords):
                x1, y1, x2, y2 = map(int, box)
                y2 = min(y2 + 10, h)
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                    raise ValueError("Invalid face bbox")
                boxes.append((x1, y1, x2, y2))
                patch = cv2.resize(
                    frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4
                )
                latents.append(vae.get_latents_for_unet(patch))
            inputs, length = models["processor"].get_audio_feature(str(folder / "audio16.wav"))
            chunks = models["processor"].get_whisper_chunk(
                inputs,
                device,
                dtype,
                models["whisper"],
                length,
                fps=25,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
            )
            if len(chunks) < n:
                raise ValueError("Whisper produced fewer frames than source")
            writer = cv2.VideoWriter(
                str(folder / "video.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot write FFV1")
            index = 0
            try:
                for audio_batch, latent_batch in datagen(
                    chunks[:n], latents, cfg.get("batch_size", 4), device=device
                ):
                    features = pe(audio_batch.to(device=device, dtype=dtype))
                    predicted = unet.model(
                        latent_batch.to(device=device, dtype=dtype),
                        torch.tensor([0], device=device),
                        encoder_hidden_states=features,
                    ).sample
                    for face in vae.decode_latents(predicted):
                        x1, y1, x2, y2 = boxes[index]
                        face = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                        composite = get_image(
                            frames[index].copy(),
                            face,
                            [x1, y1, x2, y2],
                            mode="jaw",
                            fp=models["parser"],
                        )
                        writer.write(composite)
                        index += 1
            finally:
                writer.release()
            if index != n:
                raise ValueError("Generated frame count differs")
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                folder / "video.avi",
                "-i",
                folder / "audio48.wav",
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                "-t",
                n / 25,
                request["output"],
            ]
        )


def main(request):
    synthesize(load(request["config"]), request)


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        from vn_av_df.worker import serve

        loaded = load(json.loads(sys.argv[2]))
        serve(lambda request: synthesize(loaded, request))
    else:
        main(read_json(sys.argv[1]))
