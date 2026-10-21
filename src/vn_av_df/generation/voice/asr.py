"""PhoWhisper (VinAI, Whisper fine-tune tiếng Việt) chép lời kèm mốc thời gian từng từ.

Chạy trong môi trường chính (transformers). Kết quả lưu cache theo hash file tiếng.
"""

import threading
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import read_json, write_json

MODEL = "vinai/PhoWhisper-small"


class Transcriber:
    def __init__(self, model=MODEL, device="cuda", cache=None):
        import torch
        from transformers import pipeline

        self.cache = Path(cache) if cache else None
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            chunk_length_s=30,
        )
        self.model = model
        self.lock = threading.Lock()

    def __call__(self, pcm16k, key=None):
        """pcm 16 kHz float → [{"word", "start", "end"}] (giây)."""
        path = self.cache / f"{key}.json" if self.cache and key else None
        if path and path.is_file():
            return read_json(path)["words"]
        import torch

        # Mỗi GPU một luồng job nhưng chỉ một pipeline (trên GPU 0 của process chính): gọi lần
        # lượt và trả cache sau mỗi lần, không thì cache PyTorch phình và worker giọng cùng GPU 0
        # hết VRAM (VoxCPM2 trên Kaggle: một process khác giữ 12 GB của GPU 0).
        with self.lock:
            out = self.pipe(
                {"raw": np.asarray(pcm16k, np.float32), "sampling_rate": 16000},
                return_timestamps="word",
                generate_kwargs={"language": "vi", "task": "transcribe"},
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        words = []
        for chunk in out.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            text = (chunk.get("text") or "").strip()
            if text and start is not None and end is not None and end > start:
                words.append({"word": text, "start": float(start), "end": float(end)})
        if path:
            write_json(path, {"model": self.model, "text": out.get("text", ""), "words": words})
        return words


def text_of(words):
    return " ".join(w["word"] for w in words).strip()
