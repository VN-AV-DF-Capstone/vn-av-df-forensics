"""VoxCPM2 (OpenBMB, Apache-2.0, 48 kHz): prompt_wav + prompt_text (+ reference_wav)."""

from _common import main, models_dir, save_wav


def load():
    import os

    os.environ.setdefault("HF_HOME", str(models_dir("voxcpm2")))
    from voxcpm import VoxCPM

    # torch.compile (mặc định bật) hỏng khi trace einops trên Kaggle (torch 2.8): chạy eager.
    return VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)


def handle(model, request):
    if request["mode"] != "tts":
        raise ValueError(f"Unsupported mode {request['mode']}")
    wav = model.generate(
        text=request["text"],
        prompt_wav_path=request["ref_audio"],
        prompt_text=request["ref_text"],
        reference_wav_path=request["ref_audio"],
    )
    rate = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)
    return save_wav(request["out"], wav, rate)


if __name__ == "__main__":
    main(load, handle)
