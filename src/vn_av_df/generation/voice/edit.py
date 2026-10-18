"""Chọn một cụm từ 0,3-3 s trong lời nói và đổi nghĩa của nó (kiểu tin giả: có ↔ không, số).

Thứ tự ưu tiên: (1) cụm có từ đổi nghĩa được theo bảng (có/không, tăng/giảm, ...);
(2) cụm có số; (3) thay một âm tiết bằng từ phổ biến khác. Mọi lựa chọn theo seed để tái lập.
"""

import random
import re

SWAPS = {
    "có": "không",
    "không": "có",
    "được": "bị",
    "bị": "được",
    "tăng": "giảm",
    "giảm": "tăng",
    "đúng": "sai",
    "sai": "đúng",
    "nhiều": "ít",
    "ít": "nhiều",
    "trước": "sau",
    "sau": "trước",
    "lớn": "nhỏ",
    "nhỏ": "lớn",
    "cao": "thấp",
    "thấp": "cao",
    "nên": "không nên",
    "đã": "chưa",
    "chưa": "đã",
    "đồng ý": "phản đối",
    "thắng": "thua",
    "thua": "thắng",
    "mua": "bán",
    "bán": "mua",
}
NUMBERS = ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "trăm", "nghìn"]
FILLERS = [
    "tiền",
    "nhà",
    "xe",
    "việc",
    "người",
    "trường",
    "bệnh viện",
    "công ty",
    "ngân hàng",
    "thành phố",
    "hôm qua",
    "ngày mai",
]


def clean(word):
    return re.sub(r"[^\wÀ-ỹ]", "", word.lower())


def phrases(words, duration, low=0.5, high=3.0, margin=0.5):
    """Mọi dãy từ liên tiếp có tổng độ dài trong [low, high], nằm trong [margin, duration-margin]."""
    out = []
    for i in range(len(words)):
        for j in range(i, len(words)):
            a, b = words[i]["start"], words[j]["end"]
            if b - a > high:
                break
            if b - a >= low and a >= margin and b <= duration - margin:
                out.append((i, j))
    return out


def edit_words(tokens, rng):
    """(văn bản mới, loại sửa) hoặc None nếu cụm không sửa được theo bảng / số."""
    lowered = [clean(t) for t in tokens]
    joined = " ".join(lowered)
    for src in sorted(SWAPS, key=len, reverse=True):
        if re.search(rf"(^| ){re.escape(src)}( |$)", joined):
            new = re.sub(rf"(^| ){re.escape(src)}( |$)", rf"\g<1>{SWAPS[src]}\g<2>", joined, 1)
            return new, f"swap:{src}->{SWAPS[src]}"
    for k, w in enumerate(lowered):
        if w.isdigit():
            value = str(int(w) + rng.choice([-3, -2, -1, 1, 2, 3, 5, 10]))
            return " ".join(lowered[:k] + [value.lstrip("-")] + lowered[k + 1 :]), "number"
        if w in NUMBERS:
            other = rng.choice([n for n in NUMBERS if n != w])
            return " ".join(lowered[:k] + [other] + lowered[k + 1 :]), "number"
    return None


def choose_edit(words, duration, seed, low=0.5, high=3.0, margin=0.5):
    """words: [{"word", "start", "end"}] (giây). Trả {"span", "old", "new", "kind"} hoặc None."""
    rng = random.Random(seed)
    words = [w for w in words if w["end"] > w["start"] and clean(w["word"])]
    candidates = phrases(words, duration, low, high, margin)
    if not candidates:
        return None
    rng.shuffle(candidates)
    for i, j in candidates:
        tokens = [w["word"] for w in words[i : j + 1]]
        edited = edit_words(tokens, rng)
        if edited:
            return {
                "span": [words[i]["start"], words[j]["end"]],
                "old": " ".join(clean(t) for t in tokens),
                "new": edited[0],
                "kind": edited[1],
            }
    i, j = candidates[0]
    tokens = [clean(w["word"]) for w in words[i : j + 1]]
    k = rng.randrange(len(tokens))
    replaced = tokens[:k] + [rng.choice([f for f in FILLERS if f != tokens[k]])] + tokens[k + 1 :]
    return {
        "span": [words[i]["start"], words[j]["end"]],
        "old": " ".join(tokens),
        "new": " ".join(replaced),
        "kind": "replace",
    }
