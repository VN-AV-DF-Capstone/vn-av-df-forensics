"""Kế hoạch sinh v003 cho một part sạch: mỗi clip real → 1 job giọng + job lip-sync cho MỘT PHẦN
số generator (`generators_per_clip` theo split; mặc định mọi generator).

Gán generator và ô ④⑤⑥ (`assign_generators`, xét cả split theo thứ tự người → clip):
- chia clip thành "vòng" n/m clip (n generator, m generator mỗi clip); mỗi vòng mỗi generator chạy
  đúng 1 clip → số lần chạy mỗi generator bằng nhau;
- ô của generator j ở vòng r = (r + j) mod 3 → mỗi generator chia đều 3 ô (số vòng chia hết cho 3);
- trong một clip các generator luôn ở ô khác nhau; cặp / bộ generator chung clip rút ngẫu nhiên
  theo seed mỗi vòng (không cố định).
Căn cứ chọn 2/4 (train, validation) và 3/6 (test): docs/v003_model_research.md §10.
Phương pháp giọng của clip thứ i: methods[(i + offset_speaker) mod n] (1 hệ giọng mỗi clip).
Donor (cho ② sham và ⑤): clip khác của CÙNG người, CÙNG split, dài ≥ clip gốc, không chồng thời
gian nguồn. Không có donor thì ⑤ đổi thành ④ và bỏ ②. Tham chiếu giọng cho TTS: clip khác của
cùng người (ưu tiên khác video).
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from vn_av_data.contract import validate_bundle

from vn_av_df.common.runtime import fingerprint, write_json
from vn_av_df.data.media import probe
from vn_av_df.generation.cells import LIPSYNC_CELLS

SCHEMA = "vn-av-df-plan-v3"


def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def donors_for(row, rows):
    return sorted(
        (
            d
            for d in rows
            if d["clip_id"] != row["clip_id"]
            and d["split"] == row["split"]
            and d.get("speaker_id") == row.get("speaker_id")
            and d["duration_s"] >= row["duration_s"] - 1e-6
            and (
                d["source_id"] != row["source_id"]
                or d["source_end_s"] <= row["source_start_s"]
                or row["source_end_s"] <= d["source_start_s"]
            )
        ),
        key=lambda d: d["clip_id"],
    )


def reference_for(row, rows):
    """Clip tham chiếu giọng: cùng người, khác clip; ưu tiên khác video nguồn, dài nhất."""
    same = [d for d in rows if d.get("speaker_id") == row.get("speaker_id") and d is not row]
    same = [d for d in same if d["split"] == row["split"]]
    if not same:
        return row
    return sorted(
        same, key=lambda d: (d["source_id"] == row["source_id"], -d["duration_s"], d["clip_id"])
    )[0]


def assign_generators(count, generators, per_clip, seed):
    """→ list dài `count` (số clip của split), mỗi phần tử [(generator, ô 0/1/2), ...] dài per_clip.

    per_clip phải chia hết số generator. Mỗi vòng (n/m clip) mỗi generator dùng đúng một lần, ô
    của generator j ở vòng r là (r + j) mod 3; các bộ m generator trong vòng có ô khác nhau.
    """
    import random

    n = len(generators)
    if per_clip < 1 or n % per_clip:
        raise ValueError(f"generators_per_clip={per_clip} phải chia hết {n} generator")
    groups = n // per_clip
    out = []
    for round_index in range((count + groups - 1) // groups):
        rng = random.Random(stable_int(f"{seed}:round:{round_index}"))
        buckets = defaultdict(list)
        for j, generator in enumerate(generators):
            buckets[(round_index + j) % 3].append((generator, (round_index + j) % 3))
        for members in buckets.values():
            rng.shuffle(members)
        sets = []
        for _ in range(groups):
            # Lấy từ các ô còn nhiều generator nhất trước → bộ cuối vẫn đủ ô khác nhau.
            cells = sorted((c for c in buckets if buckets[c]), key=lambda c: (-len(buckets[c]), c))
            sets.append([buckets[c].pop() for c in cells[: min(per_clip, 3)]])
            while len(sets[-1]) < per_clip:  # per_clip > 3: cho phép trùng ô
                c = next(c for c in sorted(buckets, key=lambda c: -len(buckets[c])) if buckets[c])
                sets[-1].append(buckets[c].pop())
        rng.shuffle(sets)
        out.extend(sets)
    return out[:count]


def make_plan(clean_dir, plan_path, options, seed=42):
    """options = cfg["generation"]: generators/voices theo split, inpaint (chỉ test)."""
    plan_path = Path(plan_path)
    if plan_path.exists():
        raise FileExistsError(f"{plan_path} exists; reuse it to resume or use a new name")
    rows, info = validate_bundle(clean_dir, probe=probe)
    if any("split" not in r or not r.get("speaker_id") for r in rows):
        raise ValueError("Clean part needs split and speaker_id for every clip")
    jobs = []
    for split in ("train", "validation", "test"):
        generators = list(options["generators"][split])
        voices = list(options["voices"][split])
        if not generators or not voices or len(set(generators)) != len(generators):
            raise ValueError(f"Need unique generators and voices for {split}")
        inpaint = (options.get("inpaint") or {}).get(split)
        in_split = [r for r in rows if r["split"] == split]
        by_speaker = defaultdict(list)
        for r in in_split:
            by_speaker[r["speaker_id"]].append(r)
        per_clip = (options.get("generators_per_clip") or {}).get(split, len(generators))
        assignment = iter(assign_generators(len(in_split), generators, per_clip, f"{seed}:{split}"))
        for k, (speaker, clips) in enumerate(sorted(by_speaker.items())):
            clips.sort(key=lambda r: r["clip_id"])
            # Lệch vòng theo thứ tự người: cả split cân bằng (không lệch thì ô / giọng đầu vòng
            # luôn dư một lần ở mỗi người khi số clip không chia hết cho số hệ giọng).
            offset = (k + stable_int(f"{seed}:voices")) % len(voices)
            for i, row in enumerate(clips):
                candidates = donors_for(row, in_split)
                donor = (
                    candidates[stable_int(f"{seed}:{row['clip_id']}") % len(candidates)]
                    if (candidates)
                    else None
                )
                voice_key = fingerprint(["voice", row["clip_id"]])[:20]
                jobs.append(
                    {
                        "type": "voice",
                        "key": voice_key,
                        "parent": row,
                        "voice": voices[(i + offset) % len(voices)],
                        "inpaint": inpaint,
                        "sham_donor": donor,
                        "reference": reference_for(row, in_split),
                    }
                )
                for generator, index in next(assignment):
                    cell = LIPSYNC_CELLS[index]
                    if cell == "donor" and donor is None:
                        cell = "source"
                    jobs.append(
                        {
                            "type": "lipsync",
                            "key": fingerprint(["lipsync", row["clip_id"], generator])[:20],
                            "parent": row,
                            "generator": generator,
                            "cell": cell,
                            "donor": donor if cell == "donor" else None,
                            "voice_key": voice_key if cell == "both" else None,
                        }
                    )
    plan = {
        "schema": SCHEMA,
        "clean_manifest_sha256": info["manifest_sha256"],
        "seed": seed,
        "options": options,
        "jobs": jobs,
    }
    write_json(plan_path, plan)
    return summarize(plan)


def summarize(plan):
    lip = [j for j in plan["jobs"] if j["type"] == "lipsync"]
    voice = [j for j in plan["jobs"] if j["type"] == "voice"]
    return {
        "parents": len(voice),
        "voice_jobs": len(voice),
        "lipsync_jobs": len(lip),
        "cells": dict(Counter(j["cell"] for j in lip)),
        "generators": dict(Counter(j["generator"] for j in lip)),
        "voices": dict(Counter(j["voice"] for j in voice)),
        "without_donor": sum(j["sham_donor"] is None for j in voice),
        "by_split": dict(Counter(j["parent"]["split"] for j in voice)),
    }
