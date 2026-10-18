"""viXTTS (capleaf/viXTTS: XTTS-v2 tinh chỉnh tiếng Việt, Coqui Public Model License), 24 kHz."""

from pathlib import Path

from _common import main, models_dir, save_wav


def load():
    import torch
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    folder = Path(snapshot_download("capleaf/viXTTS", cache_dir=str(models_dir("vixtts"))))
    config = XttsConfig()
    config.load_json(str(folder / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(folder),
        vocab_path=str(folder / "vocab.json"),
        use_deepspeed=False,
    )
    if torch.cuda.is_available():
        model.cuda()
    return model


def handle(model, request):
    if request["mode"] != "tts":
        raise ValueError(f"Unsupported mode {request['mode']}")
    latent, speaker = model.get_conditioning_latents(
        audio_path=[request["ref_audio"]], gpt_cond_len=30, max_ref_length=60
    )
    out = model.inference(request["text"], "vi", latent, speaker, temperature=0.3)
    return save_wav(request["out"], out["wav"], 24000)


if __name__ == "__main__":
    main(load, handle)
