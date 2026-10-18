"""Sáu cell của v003 và nhãn ba đầu ra cho từng biến thể (variant) sinh từ một clip real.

                      ORIGINAL AUDIO     SPLICED REAL AUDIO     AI VOICE
    REAL VIDEO        ① real             ② sham                 ③ fake voice
    LIP-SYNCED VIDEO  ④ source           ⑤ donor                ⑥ both

Nhãn mỗi sample (khoảng thời gian, giây):
- visual_fake_intervals: miệng bị generator vẽ lại;
- audio_fake_intervals: tiếng do TTS / VC tạo;
- av_mismatch_intervals: tiếng không khớp môi (đã biết chắc).
`fake_intervals` = hợp của visual và audio (nhãn "bị AI"), đúng hợp đồng vn-av-df-v1.
"""

CELLS = ("real", "sham", "voice", "source", "donor", "both")
LIPSYNC_CELLS = ("source", "donor", "both")
CELL_NUMBER = {name: i + 1 for i, name in enumerate(CELLS)}


def union(*lists):
    merged = []
    for a, b in sorted(span for spans in lists for span in spans):
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[round(a, 4), round(b, 4)] for a, b in merged]


def labels(cell, kind, duration, span=None):
    """Nhãn ba đầu ra. kind: full | partial. span: [a, b] (đoạn bị thay khi partial / đoạn
    tiếng bị sửa của ⑥)."""
    whole = [[0.0, round(duration, 4)]]
    part = [[round(span[0], 4), round(span[1], 4)]] if span else []
    visual, audio, mismatch = [], [], []
    if cell == "real":
        pass
    elif cell == "sham":
        mismatch = whole if kind == "full" else part
    elif cell == "voice":
        audio = whole if kind == "full" else part
        mismatch = audio  # môi gốc không khớp giọng mới
    elif cell == "source":
        visual = whole if kind == "full" else part
    elif cell == "donor":
        visual = whole if kind == "full" else part
    elif cell == "both":
        visual = whole if kind == "full" else part
        audio = part  # chỉ cụm từ bị sửa là giọng AI; generator vẽ môi theo đúng tiếng đó
    else:
        raise ValueError(f"Unknown cell {cell}")
    fake = union(visual, audio)
    return {
        "label": int(bool(fake)),
        "fake_intervals": fake,
        "visual_fake_intervals": visual,
        "audio_fake_intervals": audio,
        "av_mismatch_intervals": mismatch,
        "synthetic_visual": bool(visual),
        "synthetic_audio": bool(audio),
    }
