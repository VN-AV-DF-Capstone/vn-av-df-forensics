"""Giao diện demo (Gradio, 2 tab): tạo fake từ video real · phân tích video.

Chạy trong notebooks/demo.ipynb (Kaggle, `share=True` → link gradio.live). Tab "Tạo fake" trả
video + nhãn thật; nút "Phân tích video vừa tạo" chuyển sang tab phân tích và vẽ chồng nhãn thật
lên timeline điểm của từng model.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np

CELL_CHOICES = {
    "③ voice: giữ hình, thay tiếng bằng giọng AI": "voice",
    "④ source: vẽ lại môi theo tiếng gốc": "source",
    "⑤ donor: vẽ lại môi theo tiếng thật khác (tải lên bên dưới)": "donor",
    "⑥ both: sửa một cụm thành giọng AI rồi vẽ môi theo": "both",
}
TRUTH_KEYS = {
    "ai": "fake_intervals",
    "visual": "visual_fake_intervals",
    "audio": "audio_fake_intervals",
    "mismatch": "av_mismatch_intervals",
}


def summary_rows(result):
    rows = []
    for name, out in result["models"].items():
        for task, score in out["video"].items():
            threshold = out["thresholds"].get(task)
            decision = out["decision"].get(task)
            rows.append(
                [
                    name,
                    task,
                    round(float(score), 4),
                    None if threshold is None else round(float(threshold), 4),
                    "—" if decision is None else ("NGHI NGỜ" if decision else "bình thường"),
                ]
            )
    return rows


def timeline_figure(result, truth=None):
    """Mỗi tác vụ một hàng: điểm frame − ngưỡng của từng model (> 0 = nghi ngờ); vùng tô = nhãn thật."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [
        t
        for t in ("ai", "visual", "audio", "mismatch")
        if any(t in m["frames"] for m in result["models"].values())
    ]
    fig, axes = plt.subplots(
        len(tasks) or 1, 1, figsize=(11, 2.1 * max(1, len(tasks))), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        for span in (truth or {}).get(TRUTH_KEYS[task]) or []:
            ax.axvspan(span[0], span[1], color="#f2b8b5", alpha=0.6, lw=0)
        for name, out in result["models"].items():
            curve = out["frames"].get(task)
            if curve is None:
                continue
            threshold = out["thresholds"].get(task)
            values = curve - threshold if threshold is not None else curve
            ax.plot(np.arange(len(values)) / 25, values, lw=1.2, label=name)
        ax.axhline(0, color="#555", lw=0.8, ls="--")
        ax.set_ylabel(task)
        ax.grid(alpha=0.25)
    axes[0].set_title(
        "Điểm theo thời gian (điểm − ngưỡng validation; > 0 = nghi ngờ); vùng đỏ = nhãn thật"
    )
    axes[-1].set_xlabel("giây")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def sync_text(sync):
    if sync.get("status") not in ("ok", "few"):
        return (
            "Không đo được độ lệch tiếng–hình (mặt không rõ / không phải người nói?): giữ nguyên."
        )
    ms = sync["audio_lag_ms"]
    if not ms:
        return f"Tiếng và hình khớp (độ lệch 0 ms, độ tin {sync['conf']:.1f})."
    text = (
        f"Độ lệch tiếng–hình đo được: {'tiếng trễ' if ms >= 0 else 'tiếng sớm'} {abs(ms)} ms "
        f"(độ tin {sync['conf']:.1f}); đã dịch về 0 trước khi chấm."
    )
    if sync.get("noticeable"):
        text += " ⚠ Vượt ngưỡng người xem nhận ra (BT.1359): video lệch thời gian, không nhất thiết là AI."
    return text


def build_app(cfg, detector, generators, voices, work=None):
    import gradio as gr

    from vn_av_df.demo.fake import generate_one
    from vn_av_df.demo.media import normalize

    work = Path(work or tempfile.mkdtemp(prefix="vnav-app-"))

    def make_fake(video, cell_label, kind, generator, voice, start, end, text, donor):
        if not video:
            raise gr.Error("Chưa có video real")
        folder = work / f"fake-{int(time.time() * 1000)}"
        source = normalize(video, folder / "source.mp4")
        span = (float(start), float(end)) if start is not None and end else None
        path, label = generate_one(
            cfg,
            source,
            CELL_CHOICES[cell_label],
            kind=kind,
            generator=generator,
            voice=voice,
            span=span,
            new_text=(text or "").strip() or None,
            donor=getattr(donor, "name", donor),
            output=folder / "fake.mp4",
        )
        return str(path), label, {"video": str(path), "truth": label}

    def analyze(video, references, truth):
        if not video:
            raise gr.Error("Chưa có video")
        refs = [getattr(r, "name", r) for r in references or []]
        result = detector.analyze(video, refs, work / f"detect-{int(time.time() * 1000)}")
        return (
            summary_rows(result),
            timeline_figure(result, truth),
            sync_text(result["sync"]),
            {
                "duration_s": result["duration_s"],
                "references": result["references"],
                "sync": result["sync"],
            },
        )

    with gr.Blocks(title="VN-AV-DF v003 demo") as app:
        last = gr.State({})
        gr.Markdown("# VN-AV-DF v003 · Demo phát hiện lip-sync deepfake tiếng Việt")
        with gr.Tab("1. Tạo fake từ video real"):
            with gr.Row():
                with gr.Column():
                    real = gr.Video(label="Video real (≤ 60 s, một người nói, thấy rõ mặt)")
                    cell = gr.Radio(list(CELL_CHOICES), value=list(CELL_CHOICES)[1], label="Ô")
                    kind = gr.Radio(
                        ["full", "partial"], value="partial", label="Giả cả clip / một đoạn"
                    )
                    generator = gr.Dropdown(
                        generators, value=generators[0], label="Generator (④⑤⑥)"
                    )
                    voice = gr.Dropdown(voices, value=voices[0], label="Hệ giọng (③⑥)")
                    with gr.Row():
                        start = gr.Number(label="Đoạn partial: từ giây (để trống = tự chọn)")
                        end = gr.Number(label="đến giây")
                    text = gr.Textbox(label="Câu mới cho cụm bị sửa (③ partial, ⑥); trống = tự đổi")
                    donor = gr.File(label="Tiếng thật khác cho ⑤ donor (video/audio)")
                    run_fake = gr.Button("Tạo video fake", variant="primary")
                with gr.Column():
                    fake_video = gr.Video(label="Video fake")
                    fake_label = gr.JSON(label="Nhãn thật (đoạn giả, ô, generator, giọng)")
                    send = gr.Button("Phân tích video vừa tạo →")
        with gr.Tab("2. Phân tích video"):
            with gr.Row():
                with gr.Column():
                    target = gr.Video(label="Video cần kiểm tra (≤ 60 s)")
                    references = gr.File(
                        label="Video real tham chiếu của cùng người (tuỳ chọn, 1-3 clip)",
                        file_count="multiple",
                    )
                    truth = gr.JSON(label="Nhãn thật (tự điền khi chuyển từ tab 1)", value=None)
                    run_detect = gr.Button("Phân tích", variant="primary")
                with gr.Column():
                    sync = gr.Markdown()
                    table = gr.Dataframe(
                        headers=["model", "tác vụ", "điểm video", "ngưỡng", "kết luận"],
                        label="Điểm cả video (ngưỡng chọn trên validation)",
                    )
            plot = gr.Plot(label="Timeline")
            details = gr.JSON(label="Chi tiết")
        run_fake.click(
            make_fake,
            [real, cell, kind, generator, voice, start, end, text, donor],
            [fake_video, fake_label, last],
        )
        send.click(lambda s: (s.get("video"), s.get("truth")), [last], [target, truth])
        run_detect.click(analyze, [target, references, truth], [table, plot, sync, details])
    return app


def dump(result):
    """Kết quả phân tích → JSON (bỏ mảng frame) để lưu / so sánh."""
    return json.dumps(
        {
            **{k: v for k, v in result.items() if k != "models"},
            "models": {
                n: {k: v for k, v in m.items() if k != "frames"}
                for n, m in result["models"].items()
            },
        },
        ensure_ascii=False,
        indent=2,
        default=float,
    )
