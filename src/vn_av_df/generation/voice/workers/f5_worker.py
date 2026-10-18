"""F5-TTS tiếng Việt (hynt/F5-TTS-Vietnamese-ViVoice, CC-BY-NC-SA-4.0), 24 kHz.

mode "tts": nhân bản giọng (ref_audio + ref_text). mode "inpaint": sửa trực tiếp đoạn [span] của
audio theo văn bản mới của cả câu (cách làm của speech_edit.py trong repo F5-TTS).
"""

from pathlib import Path

from _common import main, models_dir, save_wav

REPO = "hynt/F5-TTS-Vietnamese-ViVoice"
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def load():
    import torch
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(REPO, cache_dir=str(models_dir("f5_vi"))))
    checkpoint = sorted([*folder.rglob("*.pt"), *folder.rglob("*.safetensors")])[-1]
    vocab = next(folder.rglob("vocab.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        DiT, MODEL_CFG, str(checkpoint), mel_spec_type="vocos", vocab_file=str(vocab), device=device
    )
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    return {"model": model, "vocoder": vocoder, "device": device}


def tts(state, request):
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    ref_audio, ref_text = preprocess_ref_audio_text(request["ref_audio"], request["ref_text"])
    wav, rate, _ = infer_process(
        ref_audio,
        ref_text,
        request["text"],
        state["model"],
        state["vocoder"],
        mel_spec_type="vocos",
        nfe_step=32,
    )
    return save_wav(request["out"], wav, rate)


def inpaint(state, request):
    import torch
    import torch.nn.functional as F
    import torchaudio
    from f5_tts.model.utils import convert_char_to_pinyin

    rate, hop, target_rms = 24000, 256, 0.1
    audio, sr = torchaudio.load(request["audio"])
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    level = torch.sqrt(torch.mean(audio**2))
    if level < target_rms:
        audio = audio * target_rms / level
    if sr != rate:
        audio = torchaudio.transforms.Resample(sr, rate)(audio)
    start, end = request["span"]
    begin, stop = round(start * rate), round(end * rate)
    gap = stop - begin
    body = torch.cat([audio[:, :begin], torch.zeros(1, gap), audio[:, stop:]], dim=-1)
    mask = torch.cat(
        [
            torch.ones(1, round(begin / hop), dtype=torch.bool),
            torch.zeros(1, round(gap / hop), dtype=torch.bool),
        ],
        dim=-1,
    )
    mask = F.pad(mask, (0, body.shape[-1] // hop - mask.shape[-1] + 1), value=True)
    text = convert_char_to_pinyin([request["text"]])
    with torch.inference_mode():
        generated, _ = state["model"].sample(
            cond=body.to(state["device"]),
            text=text,
            duration=body.shape[-1] // hop,
            steps=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            seed=0,
            edit_mask=mask.to(state["device"]),
        )
        mel = generated.to(torch.float32).permute(0, 2, 1)
        wave = state["vocoder"].decode(mel).squeeze().cpu()
    if level < target_rms:
        wave = wave * level / target_rms
    return save_wav(request["out"], wave.numpy(), rate)


def handle(state, request):
    if request["mode"] == "tts":
        return tts(state, request)
    if request["mode"] == "inpaint":
        return inpaint(state, request)
    raise ValueError(f"Unsupported mode {request['mode']}")


if __name__ == "__main__":
    main(load, handle)
