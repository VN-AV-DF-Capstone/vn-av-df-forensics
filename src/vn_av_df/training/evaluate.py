"""Đánh giá v003: một lần trên test, ngưỡng chọn trên validation.

Tác vụ (điểm video; điểm cao = nghi ngờ):
- ai: sample có bị AI không (label). Model 3 đầu: max(visual, audio).
- visual / audio / mismatch: cờ synthetic_visual / synthetic_audio / có av_mismatch.
Model một điểm (baseline) dùng cùng điểm cho mọi tác vụ, để thấy nó thực ra bắt cái gì.

Báo cáo:
- tổng: AUC, AP, EER, F1 / FAR tại ngưỡng validation;
- phân rã: AUC từng nhóm fake so với real (theo cell, generator, voice_method, kind, đã thấy / chưa
  thấy khi train), AUC trong từng speaker;
- partial: AUC frame trong đoạn vs ngoài đoạn trong CÙNG clip, ô sát chỗ nối (±2 frame) vs ô giữa;
- định vị chuẩn LAV-DF / AV-Deepfake1M: AP theo IoU 0,5 / 0,75 / 0,95 và AR@1/5/10 (IoU 0,5-0,95);
- ref-norm: z-score điểm theo R clip real tham chiếu của chính người đó (bỏ các clip tham chiếu VÀ
  mọi file sinh từ chúng khỏi phép đo; điểm thô báo trên đúng tập còn lại để so công bằng);
- điểm vận hành theo tỉ lệ báo nhầm cố định trên real validation (FPR 1 %, 5 %) và precision / báo
  nhầm ở nhiều tỉ lệ fake ngoài thực tế (50 %, 10 %, 1 %), theo khuyến nghị Layton et al. (USENIX
  Security 2024): F1 tối đa trên validation (~70 % fake) cho ngưỡng hợp với môi trường nhiều fake.
Kết quả: `<run>/<model>_seed<seed>/metrics.json`, `predictions-test.npz`; khoá test-lock.json.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from vn_av_df import tracking
from vn_av_df.common.runtime import fingerprint, read_json, sha, write_json
from vn_av_df.dataset import training_dataset
from vn_av_df.training.data import BenchData, collate, enrollment, to_device
from vn_av_df.training.models import pool_scores
from vn_av_df.training.train import load_model

TASKS = ("ai", "visual", "audio", "mismatch")
TARGET_KEY = {"ai": "label", "visual": "synthetic_visual", "audio": "synthetic_audio"}


def target(row, task):
    if task == "mismatch":
        return int(bool(row.get("av_mismatch_intervals")))
    return int(bool(row.get(TARGET_KEY[task])))


def auc(labels, scores):
    labels = np.asarray(labels)
    return float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None


def eer(labels, scores):
    if len(set(labels)) != 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def best_threshold(labels, scores):
    """Max F1 trên validation, cho điểm thực bất kỳ (không cần trong [0, 1])."""
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    f1 = 2 * tp / np.maximum(1, tp + labels.sum() + fp)
    last = np.r_[np.diff(sorted_scores) != 0, True]
    candidates = np.flatnonzero(last)
    if not len(candidates):
        return float(scores.mean())
    return float(sorted_scores[candidates[np.argmax(f1[candidates])]])


def at_threshold(labels, scores, threshold):
    labels, predicted = np.asarray(labels), np.asarray(scores) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "far": fp / max(1, int((labels == 0).sum())),
        "recall": tp / max(1, int((labels == 1).sum())),
    }


@torch.no_grad()
def predict(model, cfg, rows, device):
    data = BenchData(cfg, rows, model.needs, references=model.references, fixed_references=True)
    loader = torch.utils.data.DataLoader(
        data, batch_size=cfg["training"]["eval_batch_size"], collate_fn=collate
    )
    out = {}
    model.eval()
    for batch in loader:
        batch = to_device(batch, device)
        heads = model(batch)
        mask = batch["mask"]
        for i, index in enumerate(batch["index"].tolist()):
            sid = rows[index]["sample_id"]
            length = int(mask[i].sum()) if mask[i].any() else mask.shape[1]
            frames = {
                h: v[i, :length].float().cpu().numpy()
                for h, v in heads.items()
                if not h.startswith("_")
            }
            video = {
                h: float(pool_scores(v[i : i + 1], mask[i : i + 1], model.pool)[0])
                for h, v in heads.items()
                if not h.startswith("_")
            }
            out[sid] = {"frames": frames, "video": video, "mask": mask[i, :length].cpu().numpy()}
    return out


def task_scores(model, prediction):
    """Điểm video theo tác vụ từ các đầu của model."""
    video = prediction["video"]
    by_task = {}
    for head, task in model.heads.items():
        by_task[task] = video[head]
    if "score" in video:
        return {t: video["score"] for t in TASKS}
    if "ai" not in by_task:
        parts = [by_task[t] for t in ("visual", "audio") if t in by_task]
        if parts:
            by_task["ai"] = max(parts)
    return by_task


def frame_head(model, prediction, task):
    frames = prediction["frames"]
    if "score" in frames:
        return frames["score"]
    for head, t in model.heads.items():
        if t == task:
            return frames[head]
    if task == "ai":
        parts = [frames[h] for h, t in model.heads.items() if t in ("visual", "audio")]
        return np.maximum.reduce(parts) if parts else None
    return None


def localization(model, rows, predictions, task):
    """Partial: AUC frame trong / ngoài đoạn (cùng clip) và ô sát chỗ nối vs ô giữa đoạn."""
    key = {
        "visual": "visual_fake_intervals",
        "audio": "audio_fake_intervals",
        "ai": "fake_intervals",
    }[task]
    inside_auc, top_inside, edge, middle = [], [], [], []
    for row in rows:
        if row.get("kind") != "partial" or not row.get(key):
            continue
        scores = frame_head(model, predictions[row["sample_id"]], task)
        if scores is None:
            continue
        n = len(scores)
        truth = np.zeros(n, bool)
        for a, b in row[key]:
            truth[int(round(a * 25)) : int(round(b * 25))] = True
        if truth.all() or not truth.any():
            continue
        inside_auc.append(roc_auc_score(truth, scores))
        top_inside.append(bool(truth[int(np.argmax(scores))]))
        idx = np.flatnonzero(truth)
        near = np.zeros(n, bool)
        for e in (idx[0], idx[-1]):
            near[max(0, e - 2) : e + 3] = True
        edge.append(float(scores[near & truth].mean()))
        if (truth & ~near).any():
            middle.append(float(scores[truth & ~near].mean()))
    if not inside_auc:
        return None
    return {
        "clips": len(inside_auc),
        "auc_inside_vs_outside": float(np.mean(inside_auc)),
        "top_frame_inside": float(np.mean(top_inside)),
        "mean_score_edge": float(np.mean(edge)),
        "mean_score_middle": float(np.mean(middle)) if middle else None,
    }


IOU_AP = (0.5, 0.75, 0.95)
IOU_AR = tuple(np.round(np.arange(0.5, 0.951, 0.05), 2))
AR_AT = (1, 5, 10)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def proposals(scores, step=0.04, levels=(50, 60, 70, 80, 90, 95), keep=10):
    """Đoạn ứng viên từ điểm frame: các khúc liên tiếp vượt ngưỡng ở nhiều mức phân vị của chính
    video; điểm đoạn = điểm frame trung bình trong đoạn; bỏ trùng bằng NMS (IoU ≥ 0,7)."""
    scores = np.asarray(scores, float)
    found = []
    for level in levels:
        above = scores >= np.percentile(scores, level)
        edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start >= 2:
                found.append((float(scores[start:end].mean()), start * step, end * step))
    found.sort(reverse=True)
    kept = []
    for score, a, b in found:
        if all(iou((a, b), (x, y)) < 0.7 for _, x, y in kept):
            kept.append((score, a, b))
        if len(kept) >= keep:
            break
    return kept


def temporal_metrics(model, rows, predictions, task):
    """Định vị kiểu LAV-DF / AV-Deepfake1M: AP theo IoU (0,5 / 0,75 / 0,95) trên mọi video test
    (video real chỉ góp báo nhầm) và AR@N (recall với N đoạn đầu, trung bình IoU 0,5-0,95)."""
    key = {
        "visual": "visual_fake_intervals",
        "audio": "audio_fake_intervals",
        "ai": "fake_intervals",
    }[task]
    pool, truth = [], {}
    for row in rows:
        scores = frame_head(model, predictions[row["sample_id"]], task)
        if scores is None:
            return None
        truth[row["sample_id"]] = [tuple(span) for span in row.get(key) or []]
        for score, a, b in proposals(scores):
            pool.append((score, row["sample_id"], a, b))
    total = sum(len(v) for v in truth.values())
    if not total:
        return None
    pool.sort(key=lambda p: -p[0])
    out = {"segments": total}
    for threshold in IOU_AP:
        used, hits = set(), []
        for _, sid, a, b in pool:
            best, match = threshold, None
            for k, span in enumerate(truth[sid]):
                overlap = iou((a, b), span)
                if (sid, k) not in used and overlap >= best:
                    best, match = overlap, (sid, k)
            hits.append(match is not None)
            if match is not None:
                used.add(match)
        hits = np.asarray(hits, float)
        if not len(hits):
            out[f"ap@{threshold:g}"] = 0.0
            continue
        tp = np.cumsum(hits)
        precision, recall = tp / np.arange(1, len(hits) + 1), tp / total
        # AP kiểu VOC: precision bao (giảm dần) lấy tại mỗi điểm recall tăng.
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        steps = np.diff(np.r_[0.0, recall])
        out[f"ap@{threshold:g}"] = float((envelope * steps).sum())
    by_video = {}
    for score, sid, a, b in pool:
        by_video.setdefault(sid, []).append((a, b))
    for n in AR_AT:
        recalls = []
        for threshold in IOU_AR:
            found = 0
            for sid, spans in truth.items():
                top = by_video.get(sid, [])[:n]
                found += sum(any(iou(p, span) >= threshold for p in top) for span in spans)
            recalls.append(found / total)
        out[f"ar@{n}"] = float(np.mean(recalls))
    return out


def breakdown(rows, scores, task, seen):
    """AUC từng nhóm sample dương so với clip real (cell real) cùng tập."""
    reals = [r for r in rows if r.get("cell") == "real"]
    real_scores = [scores[r["sample_id"]] for r in reals]
    groups = defaultdict(list)
    for r in rows:
        if not target(r, task):
            continue
        groups[f"cell/{r['cell']}"].append(r)
        groups[f"kind/{r.get('kind')}"].append(r)
        if r.get("generator") not in (None, "none"):
            groups[f"generator/{r['generator']}"].append(r)
            groups[f"generator_seen/{r['generator'] in seen['generators']}"].append(r)
        if r.get("voice_method"):
            groups[f"voice/{r['voice_method']}"].append(r)
            groups[f"voice_seen/{r['voice_method'] in seen['voices']}"].append(r)
    out = {}
    for name, members in sorted(groups.items()):
        labels = [1] * len(members) + [0] * len(reals)
        values = [scores[r["sample_id"]] for r in members] + real_scores
        out[name] = {"n": len(members), "auc": auc(labels, values)}
    return out


def per_speaker(rows, scores, task):
    by = defaultdict(list)
    for r in rows:
        by[r.get("speaker_id")].append(r)
    values = {}
    for speaker, members in by.items():
        value = auc([target(r, task) for r in members], [scores[r["sample_id"]] for r in members])
        if value is not None:
            values[speaker] = value
    return {
        "speakers": len(values),
        "mean_within_speaker_auc": float(np.mean(list(values.values()))) if values else None,
        "min": float(min(values.values())) if values else None,
    }


def ref_norm(rows, scores, task, count):
    """z-score theo clip real tham chiếu của từng người; trả (metric ref, metric thô cùng tập)."""
    references = enrollment(rows, count)
    used = {r["source_clip_id"] for refs in references.values() for r in refs}
    stats = {}
    for key, refs in references.items():
        values = np.array([scores[r["sample_id"]] for r in refs])
        stats[key] = (values.mean(), values.std() + 1e-3)
    kept = [
        r
        for r in rows
        if r.get("source_clip_id") not in used and (r["split"], r.get("speaker_id")) in stats
    ]
    labels = [target(r, task) for r in kept]
    raw = [scores[r["sample_id"]] for r in kept]
    normed = [
        (scores[r["sample_id"]] - stats[(r["split"], r["speaker_id"])][0])
        / stats[(r["split"], r["speaker_id"])][1]
        for r in kept
    ]
    return {
        "samples": len(kept),
        "auc_raw_same_set": auc(labels, raw),
        "auc_ref_norm": auc(labels, normed),
    }


FPR_TARGETS = (0.01, 0.05)
BASE_RATES = (0.5, 0.1, 0.01)


def operating_points(val_labels, val_scores, labels, scores):
    """Ngưỡng = phân vị điểm real validation (FPR mục tiêu); TPR / FPR test và precision theo tỉ lệ
    fake π: TPR·π / (TPR·π + FPR·(1 − π))."""
    val_labels, val_scores = np.asarray(val_labels), np.asarray(val_scores, float)
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    reals = val_scores[val_labels == 0]
    out = {}
    if not len(reals) or len(set(labels.tolist())) < 2:
        return out
    for target in FPR_TARGETS:
        threshold = float(np.quantile(reals, 1 - target))
        predicted = scores > threshold
        tpr = float(predicted[labels == 1].mean())
        fpr = float(predicted[labels == 0].mean())
        out[f"fpr_{target:g}"] = {
            "threshold": threshold,
            "tpr": tpr,
            "fpr": fpr,
            "base_rates": {
                f"{rate:g}": {
                    "precision": tpr * rate / max(1e-12, tpr * rate + fpr * (1 - rate)),
                    "false_alarms_per_1000_real": fpr * 1000,
                }
                for rate in BASE_RATES
            },
        }
    return out


def evaluate_model(cfg, model_dir, rows, device, seen):
    model = load_model(Path(model_dir) / "best.pt", device)
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    val_pred = predict(model, cfg, val_rows, device)
    test_pred = predict(model, cfg, test_rows, device)
    metrics = {"model": Path(model_dir).name, "tasks": {}}
    for task in TASKS:
        val_scores = {s: task_scores(model, p).get(task) for s, p in val_pred.items()}
        test_scores = {s: task_scores(model, p).get(task) for s, p in test_pred.items()}
        if any(v is None for v in test_scores.values()):
            continue
        v_labels = [target(r, task) for r in val_rows]
        if len(set(v_labels)) < 2:
            continue
        threshold = best_threshold(v_labels, [val_scores[r["sample_id"]] for r in val_rows])
        labels = [target(r, task) for r in test_rows]
        values = [test_scores[r["sample_id"]] for r in test_rows]
        metrics["tasks"][task] = {
            "auc": auc(labels, values),
            "ap": float(average_precision_score(labels, values)) if any(labels) else None,
            "eer": eer(labels, values),
            "threshold": threshold,
            **at_threshold(labels, values, threshold),
            "operating_points": operating_points(
                v_labels, [val_scores[r["sample_id"]] for r in val_rows], labels, values
            ),
            "fpr_by_cell": {
                cell: float(
                    np.mean(
                        [
                            test_scores[r["sample_id"]] >= threshold
                            for r in test_rows
                            if r.get("cell") == cell and not target(r, task)
                        ]
                    )
                )
                for cell in ("real", "sham")
                if any(r.get("cell") == cell and not target(r, task) for r in test_rows)
            },
            "breakdown": breakdown(test_rows, test_scores, task, seen),
            "per_speaker": per_speaker(test_rows, test_scores, task),
            "ref_norm": ref_norm(test_rows, test_scores, task, cfg["generation"]["references"]),
            "localization": localization(model, test_rows, test_pred, task)
            if task != "mismatch"
            else None,
            "temporal": temporal_metrics(model, test_rows, test_pred, task)
            if task != "mismatch"
            else None,
        }
    np.savez_compressed(
        Path(model_dir) / "predictions-test.npz",
        sample_id=np.array(list(test_pred)),
        **{
            f"video_{t}": np.array(
                [task_scores(model, p).get(t, np.nan) for p in test_pred.values()], float
            )
            for t in TASKS
        },
    )
    write_json(Path(model_dir) / "metrics.json", metrics)
    return metrics


def evaluate_all(cfg, force=False):
    rows, selection = training_dataset(cfg)
    runs = Path(cfg["runs"])
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    lock = runs / "test-lock.json"
    models = sorted(p for p in runs.iterdir() if (p / "best.pt").is_file())
    stamp = {"dataset": selection, "checkpoints": {p.name: sha(p / "best.pt") for p in models}}
    if lock.exists() and not force:
        previous = read_json(lock)
        if previous["stamp"] != fingerprint(stamp):
            raise ValueError("Test already run with other checkpoints/data; use a new run or force")
    train_rows = [r for r in rows if r["split"] == "train"]
    seen = {
        "generators": {
            r["generator"] for r in train_rows if r.get("generator") not in (None, "none")
        },
        "voices": {r["voice_method"] for r in train_rows if r.get("voice_method")},
    }
    results = {}
    for folder in models:
        print(f"Evaluate {folder.name}", flush=True)
        results[folder.name] = evaluate_model(cfg, folder, rows, device, seen)
    write_json(lock, {"stamp": fingerprint(stamp), "details": stamp})
    write_json(
        runs / "evaluation.json",
        {"seen": {k: sorted(v) for k, v in seen.items()}, "results": results},
    )
    log_evaluation(cfg, runs, rows, results, seen, stamp, models)
    return {name: {t: m["tasks"][t]["auc"] for t in m["tasks"]} for name, m in results.items()}


def log_evaluation(cfg, runs, rows, results, seen, stamp, models):
    """Run "evaluate" (một lần): mọi bảng số liệu test + ROC + histogram điểm + artifact khoá."""
    run = tracking.open_run(
        cfg,
        "evaluate",
        f"{runs.name}-test",
        group=runs.name,
        config={"seen": {k: sorted(v) for k, v in seen.items()}, "stamp": fingerprint(stamp)},
        folder=runs,
    )
    if not run.active:
        return
    summary, parts, speakers, refs, locs = [], [], [], [], []
    for folder, metrics in results.items():
        model, seed = folder.rsplit("_seed", 1)
        for task, m in metrics["tasks"].items():
            summary.append(
                [model, seed, task, m["auc"], m["ap"], m["eer"], m["f1"], m["far"], m["recall"]]
            )
            for group, value in m["breakdown"].items():
                parts.append([model, seed, task, group, value["n"], value["auc"]])
            ps = m["per_speaker"]
            speakers.append(
                [model, seed, task, ps["speakers"], ps["mean_within_speaker_auc"], ps["min"]]
            )
            rn = m["ref_norm"]
            refs.append(
                [model, seed, task, rn["samples"], rn["auc_raw_same_set"], rn["auc_ref_norm"]]
            )
            lo = m.get("localization")
            if lo:
                locs.append(
                    [
                        model,
                        seed,
                        task,
                        lo["clips"],
                        lo["auc_inside_vs_outside"],
                        lo["top_frame_inside"],
                        lo["mean_score_edge"],
                        lo["mean_score_middle"],
                    ]
                )
    run.table(
        "test/summary",
        ["model", "seed", "task", "auc", "ap", "eer", "f1", "far", "recall"],
        summary,
    )
    run.table("test/breakdown", ["model", "seed", "task", "group", "n", "auc"], parts)
    run.table(
        "test/per_speaker", ["model", "seed", "task", "speakers", "mean_auc", "min_auc"], speakers
    )
    run.table(
        "test/ref_norm", ["model", "seed", "task", "samples", "auc_raw", "auc_ref_norm"], refs
    )
    run.table(
        "test/localization",
        ["model", "seed", "task", "clips", "auc_in_vs_out", "top_frame_inside", "edge", "middle"],
        locs,
    )
    test_rows = {r["sample_id"]: r for r in rows if r["split"] == "test"}
    first = {}
    for folder in sorted(results):
        first.setdefault(folder.rsplit("_seed", 1)[0], folder)
    series = {}
    for model, folder in first.items():
        with np.load(runs / folder / "predictions-test.npz", allow_pickle=False) as z:
            ids, scores = z["sample_id"].tolist(), z["video_ai"]
        keep = [i for i, s in enumerate(ids) if s in test_rows and np.isfinite(scores[i])]
        labels = [target(test_rows[ids[i]], "ai") for i in keep]
        if len(set(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, scores[keep])
            step = max(1, len(fpr) // 200)
            series[model] = (fpr[::step], tpr[::step])
        for cell in ("real", "sham", "voice", "source", "donor", "both"):
            run.hist(
                f"test_scores_ai/{model}/{cell}",
                [scores[i] for i in keep if test_rows[ids[i]].get("cell") == cell],
            )
    tracking.curves(run, "test/roc_ai", series, title="ROC test, tác vụ ai (seed đầu)")
    run.artifact(
        f"evaluation-{runs.name}",
        "evaluation",
        [runs / "evaluation.json", runs / "test-lock.json"]
        + [p / "metrics.json" for p in models]
        + [p / "predictions-test.npz" for p in models],
        {"stamp": fingerprint(stamp)},
    )
    run.finish()
