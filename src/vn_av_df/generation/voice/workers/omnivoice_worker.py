"""OmniVoice (k2-fsa, >600 ngôn ngữ), 24 kHz: generate(text, ref_audio, ref_text)."""

from _common import main, models_dir, save_wav


def load():
    import os

    os.environ.setdefault("HF_HOME", str(models_dir("omnivoice")))
    from omnivoice import OmniVoice

    return OmniVoice.from_pretrained("k2-fsa/OmniVoice")


def handle(model, request):
    if request["mode"] != "tts":
        raise ValueError(f"Unsupported mode {request['mode']}")
    audio = model.generate(
        text=request["text"], ref_audio=request["ref_audio"], ref_text=request["ref_text"]
    )
    return save_wav(request["out"], audio[0], 24000)


if __name__ == "__main__":
    main(load, handle)
