"""Các phần nhỏ của v003: sửa cụm từ, ghép tiếng, encoder mới, lệnh generator."""

import numpy as np
import pytest
import torch

from vn_av_df.data.media import ffmpeg
from vn_av_df.generation import lipsync
from vn_av_df.generation.voice import audio as A
from vn_av_df.generation.voice.edit import choose_edit, edit_words


def words(text, step=0.4):
    return [
        {"word": w, "start": 0.6 + i * step, "end": 0.6 + i * step + 0.35}
        for i, w in enumerate(text.split())
    ]


def test_choose_edit_prefers_meaning_flip_then_numbers():
    edit = choose_edit(words("hôm nay giá xăng không tăng nhiều lắm"), 6.0, seed=1)
    assert edit is not None and 0.5 <= edit["span"][1] - edit["span"][0] <= 3.0
    assert edit["kind"].startswith("swap") and edit["new"] != edit["old"]
    import random

    new, kind = edit_words(["giảm", "hai", "nghìn"], random.Random(0))
    assert kind == "swap:giảm->tăng" and new.startswith("tăng")
    assert edit_words(["tốn", "3", "triệu"], random.Random(0))[1] == "number"
    assert choose_edit(words("a"), 6.0, seed=0) is None  # quá ngắn để có cụm 0,5 s


def test_splice_and_stretch_keep_length_and_level():
    rate = A.RATE
    base = 0.1 * np.sin(2 * np.pi * 200 * np.arange(rate * 3) / rate).astype(np.float32)
    insert = 0.8 * np.sin(2 * np.pi * 300 * np.arange(rate) / rate).astype(np.float32)
    stretched, ratio = A.time_stretch(insert, rate // 2, ffmpeg())
    assert len(stretched) == rate // 2 and ratio == pytest.approx(2.0)
    out = A.splice(base, stretched, rate, rate + rate // 2)
    assert len(out) == len(base)
    assert A.rms(out[rate + 2000 : rate + rate // 2 - 2000]) == pytest.approx(
        A.rms(base[rate : rate + rate // 2]), rel=0.15
    )
    assert np.allclose(out[: rate - 1000], base[: rate - 1000])
    fitted, _ = A.fit_whole(insert, rate * 2, ffmpeg())
    assert len(fitted) == rate * 2


def test_mouth_encoder_crops_native_resolution(tmp_path):
    from vn_av_df.common.runtime import run, save_npz, sha, write_json
    from vn_av_df.features.encoders import MouthEncoder, normalized_lips

    video = tmp_path / "v.mp4"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1280x720:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            video,
        ]
    )
    n = 50
    boxes = np.tile([[320.0, 180.0, 50.0]], (n, 1)).astype(np.float32)
    box = tmp_path / "box.npz"
    save_npz(
        box,
        mouth_boxes=boxes,
        landmarks=np.random.default_rng(0).normal(100, 3, (n, 68, 2)),
        visual_valid=np.ones(n, bool),
    )
    write_json(box.with_suffix(".json"), {"feature_sha256": sha(box)})
    meta = MouthEncoder({"crop": 64}).extract(
        {"video": str(video), "box_cache": str(box)}, tmp_path / "m.npz"
    )
    with np.load(tmp_path / "m.npz") as z:
        assert z["mouth"].shape == (n, 64, 64) and z["lips"].shape == (n, 20, 2)
        assert float(z["native_side"]) == pytest.approx(100.0)  # 50 px ở 640 → 100 px ở 1280
    assert meta["box_sha256"] == sha(box)
    lips = normalized_lips(np.random.default_rng(1).normal(0, 5, (3, 68, 2)))
    assert np.allclose(lips.astype(np.float32).mean(1), 0, atol=1e-2)


def test_xlsr_encoder_pools_layers_to_25hz(tmp_path):
    from types import SimpleNamespace

    from vn_av_df.common.runtime import run
    from vn_av_df.features.encoders import XLSREncoder

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, output_hidden_states=True):
            frames = x.shape[1] // 320
            return SimpleNamespace(
                hidden_states=[torch.full((1, frames, 1024), float(i)) for i in range(25)]
            )

    video = tmp_path / "v.mp4"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x64:r=25:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            video,
        ]
    )
    encoder = XLSREncoder({"layer_low": 6, "layer_high": 12}, model=Fake())
    encoder.extract({"video": str(video)}, tmp_path / "x.npz")
    with np.load(tmp_path / "x.npz") as z:
        assert z["ssl"].shape[0] in (50, 51) and np.allclose(z["ssl"].astype(np.float32), 9.0)


def test_generator_commands(tmp_path):
    repo = tmp_path / "diff2lip"
    (repo / "scripts").mkdir(parents=True)
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints/model.pt").write_bytes(b"x")
    (repo / "scripts/inference_single_video.sh").write_text(
        'video_path="X"\naudio_path="Y"\nout_path="Z"\nmodel_path="M"\npython generate.py $video_path\n',
        encoding="utf-8",
    )
    cfg = {"envs": {"gen_diff2lip": "/env/python", "gen_iplap": "/env/py37"}, "generation": {}}
    args, out = lipsync.command(
        cfg, "diff2lip", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    script = (tmp_path / "run_diff2lip.sh").read_text(encoding="utf-8")
    assert (
        args[0] == "bash"
        and f'video_path="{tmp_path / "face.mp4"}"' in script
        and "model.pt" in script
    )
    args, out = lipsync.command(
        cfg, "iplap", repo, tmp_path / "face.mp4", tmp_path / "a.wav", tmp_path
    )
    assert out.name == "result.avi" and args[1].name == "iplap_worker.py"
    with pytest.raises(FileNotFoundError, match="envs"):
        lipsync.command({"envs": {}, "generation": {}}, "videoretalking", repo, "f", "a", tmp_path)


def test_iplap_worker_keeps_result_half(tmp_path):
    """Upstream giả ghi [sketch | kết quả] rồi mux; worker giữ nửa phải, bỏ mux, dọn thư mục tạm."""
    import os
    import subprocess
    import sys
    import textwrap

    import cv2

    from vn_av_df.generation import iplap_worker

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text("def manual_seed(s):\n    pass\n", encoding="utf-8")
    (stubs / "torchvision").mkdir()
    (stubs / "torchvision/__init__.py").write_text("from . import models\n", encoding="utf-8")
    (stubs / "torchvision/models.py").write_text(
        "def vgg19(**kw):\n    assert 'pretrained' not in kw\n    return kw\n", encoding="utf-8"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference_single.py").write_text(
        textwrap.dedent(
            """
            import argparse, os, subprocess
            import cv2, numpy as np, torchvision
            p = argparse.ArgumentParser()
            for k in ("--input", "--audio", "--output_dir"):
                p.add_argument(k)
            a = p.parse_args()
            assert os.path.isfile(a.input) and torchvision.models.vgg19(pretrained=True) == {}
            temp = "tempfile_of_" + a.output_dir.split("/")[-1]
            os.makedirs(temp)
            out = cv2.VideoWriter(temp + "/result.avi", cv2.VideoWriter_fourcc(*"DIVX"), 25, (64, 32))
            for i in range(5):
                frame = np.zeros((32, 64, 3), np.uint8)
                frame[:, 32:] = 200
                out.write(frame)
            out.release()
            assert subprocess.call("ffmpeg -i x " + temp + "/result.avi y.mp4", shell=True) == 0
            """
        ),
        encoding="utf-8",
    )
    video = tmp_path / "face.mp4"
    video.write_bytes(b"v")
    (tmp_path / "audio.wav").write_bytes(b"a")
    out = tmp_path / "result.avi"
    env = {**os.environ, "PYTHONPATH": str(stubs)}
    subprocess.run(
        [
            sys.executable,
            iplap_worker.__file__,
            "--video",
            video,
            "--audio",
            tmp_path / "audio.wav",
            "--output",
            out,
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    stream = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        frames.append(frame)
    assert len(frames) == 5 and frames[0].shape == (32, 32, 3) and frames[0].min() > 150
    assert sorted(p.name for p in repo.iterdir()) == ["inference_single.py"]


def test_env_recipes_cover_every_worker():
    import sys
    from pathlib import Path

    from vn_av_df.generation.voice.engines import ENGINES

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environments"))
    from worker_envs import RECIPES

    needed = {env for steps in ENGINES.values() for env, _, _ in steps} | set(lipsync.ENVS.values())
    assert needed <= set(RECIPES)


@pytest.mark.parametrize("lag", [2, -1])
def test_lag_audio_shifts_audio_features_and_masks_edges(lag):
    from vn_av_df.training.data import lag_audio

    t = np.arange(10, dtype=np.float32)[:, None]
    item = {"a": t.copy(), "ssl": t.copy() * 2, "v": t.copy()}
    out = lag_audio(item, lag)
    valid = out["lag_valid"]
    assert valid.sum() == 10 - abs(lag)
    # Ở frame còn hợp lệ: tiếng tại t là tiếng gốc của t - lag (tiếng trễ lag frame).
    assert np.all(out["a"][valid, 0] == np.arange(10)[valid] - lag)
    assert np.all(out["ssl"][valid, 0] == 2 * (np.arange(10)[valid] - lag))
    assert np.all(out["v"] == t)  # hình giữ nguyên


@pytest.mark.parametrize(
    "clips,generators,per_clip",
    [(600, 4, 2), (240, 6, 3), (12, 3, 3), (160, 4, 2)],  # train, test, cũ (đủ), validation
)
def test_generator_assignment_balanced_and_varied(clips, generators, per_clip):
    from collections import Counter

    from vn_av_df.generation.plan import assign_generators

    names = [f"g{j}" for j in range(generators)]
    plan = assign_generators(clips, names, per_clip, "7:train")
    assert len(plan) == clips and all(len(s) == per_clip for s in plan)
    runs = Counter(g for s in plan for g, _ in s)
    assert set(runs.values()) == {clips * per_clip // generators}  # mỗi generator chạy bằng nhau
    for s in plan:  # trong một clip: generator khác nhau, ô khác nhau
        assert len({g for g, _ in s}) == per_clip and len({c for _, c in s}) == min(per_clip, 3)
    cells = Counter((g, c) for s in plan for g, c in s)
    for g in names:
        counts = [cells[(g, c)] for c in range(3)]
        assert max(counts) - min(counts) <= 1  # mỗi generator chia đều 3 ô
    if per_clip < generators:
        pairs = Counter(tuple(sorted(g for g, _ in s)) for s in plan)
        assert len(pairs) > generators // per_clip  # bộ generator chung clip không cố định


def test_temporal_metrics_perfect_and_empty():
    from vn_av_df.training.evaluate import proposals, temporal_metrics

    class Model:
        heads = {"score": "ai"}

    rows, predictions = [], {}
    for i, span in enumerate([[1.0, 2.0], [2.4, 3.0], None]):
        scores = np.zeros(125)
        if span:
            scores[int(span[0] * 25) : int(span[1] * 25)] = 1.0
        rows.append({"sample_id": f"s{i}", "fake_intervals": [span] if span else []})
        predictions[f"s{i}"] = {"frames": {"score": scores}, "video": {}}
    out = temporal_metrics(Model(), rows, predictions, "ai")
    assert out["segments"] == 2 and out["ap@0.5"] == pytest.approx(1.0)
    assert out["ap@0.95"] == pytest.approx(1.0) and out["ar@1"] == pytest.approx(1.0)
    assert proposals(np.zeros(50)) == [] or all(b > a for _, a, b in proposals(np.zeros(50)))
    flat = {"s0": {"frames": {"score": np.random.default_rng(0).random(125)}, "video": {}}}
    noisy = temporal_metrics(Model(), rows[:1], flat, "ai")
    assert 0 <= noisy["ap@0.5"] < 1
