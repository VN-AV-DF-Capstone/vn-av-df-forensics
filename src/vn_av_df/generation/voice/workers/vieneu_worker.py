"""VieNeu-TTS v3 Turbo (Apache-2.0, 48 kHz): `Vieneu().infer(text, ref_audio=...)`.

mode "tts": nhân bản giọng từ ref_audio; mode "tts_default": giọng mặc định (đầu vào cho Seed-VC).
"""

from _common import main


def load():
    from vieneu import Vieneu

    return Vieneu()


def handle(model, request):
    if request["mode"] == "tts":
        audio = model.infer(request["text"], ref_audio=request["ref_audio"], denoise=True)
    elif request["mode"] == "tts_default":
        audio = model.infer(request["text"])
    else:
        raise ValueError(f"Unsupported mode {request['mode']}")
    model.save(audio, request["out"])
    return {"out": request["out"]}


if __name__ == "__main__":
    main(load, handle)
