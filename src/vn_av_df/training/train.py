"""Train model v003 trên cache: mỗi (model, seed) một thư mục `runs/<RUN>/<model>_seed<seed>/`.

- Model không train (speechforensics) chỉ ghi checkpoint cấu hình.
- avh_align (không giám sát) chỉ dùng clip real của train / validation.
- Chọn checkpoint theo validation loss; early stopping `patience` epoch.
- Đã có `best.pt` + `history.json` hoàn chỉnh thì bỏ qua (chạy lại sau khi ngắt sẽ tiếp tục
  các model chưa xong; model đang dở train lại từ đầu).
"""

import io
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from vn_av_df import tracking
from vn_av_df.common.runtime import atomic_bytes, read_json, write_json
from vn_av_df.dataset import training_dataset
from vn_av_df.training.data import BenchData, collate, to_device
from vn_av_df.training.models import build


def feature_dim(cfg, rows):
    for row in rows:
        path = Path(cfg["cache"]) / "avhubert" / f"{row['sample_id']}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                return int(z["audio"].shape[1])
    return 1024


def pos_weights(rows):
    out = {}
    for task, key in (
        ("ai", "label"),
        ("visual", "synthetic_visual"),
        ("audio", "synthetic_audio"),
    ):
        counts = Counter(bool(r.get(key)) for r in rows)
        out[task] = min(5.0, max(0.2, counts[False] / max(1, counts[True])))
    counts = Counter(bool(r.get("av_mismatch_intervals")) for r in rows)
    out["mismatch"] = min(5.0, max(0.2, counts[False] / max(1, counts[True])))
    return out


def save(path, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())


def loaders(cfg, model, rows, seed, train):
    options = cfg["training"]
    data = BenchData(
        cfg,
        rows,
        model.needs,
        augment=train and model.augment,
        lag_augment=train and getattr(model, "lag_augment", False),
        references=model.references,
        fixed_references=not train,
        seed=seed,
        crop_frames=model.crop_frames if train else None,
    )
    return data, torch.utils.data.DataLoader(
        data,
        batch_size=options["batch_size"],
        shuffle=train,
        collate_fn=collate,
        num_workers=options.get("workers", 0),
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, optimizer, device, weights, stats=None):
    """Loss trung bình; `stats` (dict) nhận thêm trung bình từng thành phần loss + grad norm."""
    total, count = 0.0, 0
    parts, norms = Counter(), []
    model.train(optimizer is not None)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            out = model(batch)
            loss = model.loss(batch, out, weights)
        if optimizer is not None and loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
        size = len(batch["index"])
        total += float(loss.detach()) * size
        count += size
        for key, value in (getattr(model, "loss_parts", None) or {}).items():
            parts[key] += value * size
    if stats is not None:
        stats.update({k: v / max(1, count) for k, v in parts.items()})
        if norms:
            stats["grad_norm"] = float(np.mean(norms))
    return total / max(1, count)


def validation_metrics(model, cfg, rows, device):
    """AUC validation theo tác vụ và theo ô / generator / giọng (tác vụ ai), mỗi epoch."""
    from vn_av_df.training.evaluate import TASKS, auc, predict, target, task_scores

    predictions = predict(model, cfg, rows, device)
    out = {}
    scores_by_task = {}
    for task in TASKS:
        scores = {s: task_scores(model, p).get(task) for s, p in predictions.items()}
        if any(v is None for v in scores.values()):
            continue
        scores_by_task[task] = scores
        out[f"val_auc/{task}"] = auc(
            [target(r, task) for r in rows], [scores[r["sample_id"]] for r in rows]
        )
    ai = scores_by_task.get("ai")
    if ai:
        reals = [r for r in rows if r.get("cell") == "real"]
        for key in ("cell", "generator", "voice_method", "kind"):
            groups = {}
            for r in rows:
                if r.get("label") and r.get(key) not in (None, "none"):
                    groups.setdefault(r[key], []).append(r)
            for value, members in groups.items():
                labels = [1] * len(members) + [0] * len(reals)
                values = [ai[r["sample_id"]] for r in members + reals]
                out[f"val_auc_ai_by_{key}/{value}"] = auc(labels, values)
    model.train()
    return out


def train_one(cfg, name, seed, rows, out):
    out = Path(out)
    if (out / "best.pt").is_file() and (out / "history.json").is_file():
        if read_json(out / "history.json").get("complete"):
            return read_json(out / "history.json")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    options = cfg["training"]
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    dim = feature_dim(cfg, rows)
    model = build(name, dim, options["hidden"], options["dropout"]).to(device)
    config = {"name": name, "dim": dim, "hidden": options["hidden"], "dropout": options["dropout"]}
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if name == "avh_align":  # chỉ học từ real (đúng paper)
        train_rows = [r for r in train_rows if r["label"] == 0 and r.get("cell") == "real"]
        val_rows = [r for r in val_rows if r["label"] == 0 and r.get("cell") == "real"]
    history = {
        "model": name,
        "seed": seed,
        "epochs": [],
        "complete": False,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
    }
    run = tracking.open_run(
        cfg,
        "train",
        f"{name}_seed{seed}",
        group=Path(cfg["runs"]).name,
        config={
            "model": name,
            "seed": seed,
            **config,
            "training": options,
            "parameters": sum(p.numel() for p in model.parameters()),
            "needs": sorted(model.needs),
            "augment_mouth": model.augment,
            "augment_lag": getattr(model, "lag_augment", False),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
        },
        folder=out,
        tags=[name],
    )
    if not model.trainable:
        save(out / "best.pt", {"config": config, "state": model.state_dict(), "epoch": 0})
        history["complete"] = True
        history["validation"] = validation_metrics(model, cfg, val_rows, device)
        write_json(out / "history.json", history)
        run.log(history["validation"])
        run.summary({"trainable": False})
        run.finish()
        return history
    weights = {k: torch.tensor(v, device=device) for k, v in pos_weights(train_rows).items()}
    _, train_loader = loaders(cfg, model, train_rows, seed, True)
    _, val_loader = loaders(cfg, model, val_rows, seed, False)
    lr = options["lr"] if name != "avh_align" else options.get("avh_align_lr", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=options["weight_decay"])
    best, patience, stop = None, 0, "max_epochs"
    began = time.monotonic()
    for epoch in range(1, options["epochs"] + 1):
        tick = time.monotonic()
        train_loader.dataset.epoch = epoch
        train_stats, val_stats = {}, {}
        train_loss = run_epoch(model, train_loader, optimizer, device, weights, train_stats)
        val_loss = run_epoch(model, val_loader, None, device, weights, val_stats)
        metrics = validation_metrics(model, cfg, val_rows, device)
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.monotonic() - tick, 1),
                "train_parts": train_stats,
                "val_parts": val_stats,
                "validation": metrics,
            }
        )
        run.log(
            {
                "epoch": epoch,
                "loss/train": train_loss,
                "loss/val": val_loss,
                **{f"loss_train/{k}": v for k, v in train_stats.items() if k != "grad_norm"},
                **{f"loss_val/{k}": v for k, v in val_stats.items() if k != "grad_norm"},
                "grad_norm": train_stats.get("grad_norm"),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.monotonic() - tick,
                **metrics,
                **(
                    {"gpu/max_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
                    if torch.cuda.is_available()
                    else {}
                ),
            },
            step=epoch,
        )
        print(
            f"{name} seed{seed} epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}",
            flush=True,
        )
        if best is None or val_loss < best - 1e-4:
            best, patience = val_loss, 0
            save(
                out / "best.pt",
                {
                    "config": config,
                    "state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
            )
        else:
            patience += 1
            if patience >= options["patience"]:
                stop = f"early_stop (patience {options['patience']})"
                break
        write_json(out / "history.json", history)
    history["complete"] = True
    history["best_val_loss"] = best
    history["stop"] = stop
    write_json(out / "history.json", history)
    best_epoch = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["epoch"]
    chosen = next(e for e in history["epochs"] if e["epoch"] == best_epoch)
    run.summary(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "stop": stop,
            "epochs_run": len(history["epochs"]),
            "train_hours": (time.monotonic() - began) / 3600,
            **{f"best/{k}": v for k, v in chosen["validation"].items()},
        }
    )
    run.artifact(
        f"model-{Path(cfg['runs']).name}-{name}-seed{seed}",
        "model",
        [out / "best.pt", out / "history.json"],
        {"model": name, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best},
    )
    run.finish()
    return history


def load_model(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    config = state["config"]
    model = build(config["name"], config["dim"], config["hidden"], config["dropout"])
    model.load_state_dict(state["state"])
    return model.to(device).eval()


def train_all(cfg):
    rows, selection = training_dataset(cfg)
    runs = Path(cfg["runs"])
    write_json(
        runs / "training.json",
        {
            "selection": selection,
            "models": cfg["models"],
            "seeds": cfg["seeds"],
            "options": cfg["training"],
        },
    )
    report = {}
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            history = train_one(cfg, name, seed, rows, runs / f"{name}_seed{seed}")
            report[f"{name}_seed{seed}"] = {
                "epochs": len(history["epochs"]),
                "best_val_loss": history.get("best_val_loss"),
            }
    return report
