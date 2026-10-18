"""Registry phương pháp giọng AI (mỗi hệ một model family) và cách gọi worker của chúng.

| method      | họ kỹ thuật                         | dùng ở        | env worker    |
|-------------|-------------------------------------|---------------|---------------|
| vieneu      | codec LM, Vietnamese-first (v3)     | train/val/test| tts_vieneu    |
| f5_vi       | flow matching + vocoder             | train/val/test| tts_f5        |
| omnivoice   | diffusion language model            | train/val/test| tts_omnivoice |
| seedvc      | voice conversion (VieNeu giọng mặc  | train/val/test| tts_vieneu +  |
|             | định → Seed-VC sang giọng người đó) |               | vc_seedvc     |
| voxcpm2     | tokenizer-free                      | chỉ test      | tts_voxcpm    |
| vixtts      | GPT + HiFi-GAN (XTTS-v2 tiếng Việt) | chỉ test      | tts_xtts      |
| f5_inpaint  | sửa trực tiếp (speech inpainting)   | chỉ test (C)  | tts_f5        |

Worker nhận JSON: {"mode": "tts", "text", "ref_audio", "ref_text", "out"} |
{"mode": "vc", "source", "target", "out"} | {"mode": "inpaint", "audio", "span", "text", "out"}.
"""

from pathlib import Path

from vn_av_df import worker

WORKERS = Path(__file__).with_name("workers")
ENGINES = {
    "vieneu": [("tts_vieneu", "vieneu_worker.py", "tts")],
    "f5_vi": [("tts_f5", "f5_worker.py", "tts")],
    "omnivoice": [("tts_omnivoice", "omnivoice_worker.py", "tts")],
    "seedvc": [
        ("tts_vieneu", "vieneu_worker.py", "tts_default"),
        ("vc_seedvc", "seedvc_worker.py", "vc"),
    ],
    "voxcpm2": [("tts_voxcpm", "voxcpm_worker.py", "tts")],
    "vixtts": [("tts_xtts", "xtts_worker.py", "tts")],
    "f5_inpaint": [("tts_f5", "f5_worker.py", "inpaint")],
}


def envs_for(methods):
    return sorted({env for m in methods for env, _, _ in ENGINES[m]})


def python_of(cfg, env):
    path = (cfg.get("envs") or {}).get(env)
    if not path:
        raise FileNotFoundError(
            f"No python for voice env {env}; run environments/worker_envs.py {env} and set "
            f"cfg['envs']['{env}']"
        )
    return path


def open_worker(cfg, env, script, gpu=None):
    import os

    command = [python_of(cfg, env), "-B", WORKERS / script, "--serve", Path(cfg["root"]).resolve()]
    environ = os.environ.copy()
    environ["VNAV_MODELS"] = str(Path(cfg["generation"]["voice_models"]).resolve())
    environ["VNAV_SEEDVC_REPO"] = str((Path(cfg["root"]) / "external/seed-vc").resolve())
    if gpu is not None:
        environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return worker.get(
        ("voice", env, script, gpu),
        lambda: worker.Worker(command, env=environ, startup_timeout=3600),
    )


class Engine:
    """Gọi một phương pháp giọng; `call` thay bằng hàm giả trong test."""

    def __init__(self, cfg, gpu=None, call=None):
        self.cfg, self.gpu = cfg, gpu
        self.call = call or self._call

    def _call(self, env, script, request):
        return open_worker(self.cfg, env, script, self.gpu).call(request, timeout=1800)

    def tts(self, method, text, ref_audio, ref_text, out, scratch):
        steps = ENGINES[method]
        if steps[0][2] == "inpaint":
            raise ValueError(f"{method} is an inpainting method")
        if len(steps) == 1:
            env, script, _ = steps[0]
            return self.call(
                env,
                script,
                {
                    "mode": "tts",
                    "text": text,
                    "ref_audio": str(ref_audio),
                    "ref_text": ref_text,
                    "out": str(out),
                },
            )
        # Chuỗi TTS giọng mặc định → VC sang giọng người tham chiếu.
        (tts_env, tts_script, _), (vc_env, vc_script, _) = steps
        neutral = Path(scratch) / "neutral.wav"
        self.call(tts_env, tts_script, {"mode": "tts_default", "text": text, "out": str(neutral)})
        return self.call(
            vc_env,
            vc_script,
            {"mode": "vc", "source": str(neutral), "target": str(ref_audio), "out": str(out)},
        )

    def inpaint(self, method, audio, span, text, out):
        env, script, mode = ENGINES[method][0]
        if mode != "inpaint":
            raise ValueError(f"{method} is not an inpainting method")
        return self.call(
            env,
            script,
            {
                "mode": "inpaint",
                "audio": str(audio),
                "span": list(span),
                "text": text,
                "out": str(out),
            },
        )
