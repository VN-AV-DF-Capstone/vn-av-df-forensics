"""Lệnh của pipeline v003 (gọi qua generation/0x_*.py, training/0x_*.py hoặc notebook).

setup [--generators ...] [--encoders]   tải source / weight generator và encoder
plan                                     kế hoạch sinh cho part sạch đang chọn
voice [--batch k/n]                      phiên giọng (①②③ + track B)
lipsync [--generators ...] [--batch k/n] phiên lip-sync (④⑤⑥) cho các generator chọn
finalize                                 gom record → manifest
voice / lipsync / finalize [--previous DIR ...]  gộp output phiên khác (vd. Kaggle) trước khi chạy
prepare [--encoders ...]                 trích feature (avhubert, mouth, xlsr)
train [--models ...] [--seeds ...]       train model
evaluate [--force]                       test một lần
report                                   bảng kết quả + cổng
"""

import argparse
import json


def setup(cfg, generators=None, encoders=False):
    from vn_av_df.generation.generate import adapter_for

    result = {}
    names = (
        generators
        if generators is not None
        else sorted({g for split in cfg["generation"]["generators"].values() for g in split})
    )
    for name in names:
        adapter = adapter_for(name)
        result[name] = adapter.setup(cfg)
    if encoders:
        from vn_av_df.assets import hf_snapshot, setup_avhubert

        result["avhubert"] = setup_avhubert(cfg["encoders"]["avhubert"])
        x = cfg["encoders"]["xlsr"]
        result["xlsr"] = hf_snapshot(
            x["hf_id"],
            x["model"],
            ["config.json", "*.bin", "*.safetensors", "preprocessor_config.json"],
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-df")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--generators", nargs="*")
    p.add_argument("--encoders", action="store_true")
    sub.add_parser("plan")
    for name in ("voice", "lipsync", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--previous", nargs="*", default=[])
        if name != "finalize":
            p.add_argument("--batch", help="k/n: chỉ chạy đợt k trong n đợt clip")
    sub.choices["lipsync"].add_argument("--generators", nargs="*")
    sub.add_parser("prepare").add_argument("--encoders", nargs="*")
    p = sub.add_parser("train")
    p.add_argument("--models", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    sub.add_parser("evaluate").add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    from settings import config

    cfg = config()
    if getattr(args, "previous", None):
        from vn_av_df.generation.generate import merge_previous

        print(merge_previous(cfg, args.previous))
    if getattr(args, "batch", None):
        cfg["session_batch"] = tuple(int(x) for x in args.batch.split("/"))
    if args.command == "setup":
        result = setup(cfg, args.generators, args.encoders)
    elif args.command == "plan":
        from vn_av_df.generation.plan import make_plan

        result = make_plan(cfg["clean_dataset"], cfg["plan"], cfg["generation"], cfg["seed"])
    elif args.command in ("voice", "lipsync"):
        from vn_av_df.generation.generate import run_generation

        if args.command == "lipsync" and args.generators:
            cfg["session_generators"] = args.generators
        result = run_generation(cfg, args.command)
    elif args.command == "finalize":
        from vn_av_df.generation.generate import finalize

        result = finalize(cfg)
    elif args.command == "prepare":
        from vn_av_df.training.prepare import ORDER, prepare

        result = prepare(cfg, args.encoders or ORDER)
    elif args.command == "train":
        from vn_av_df.training.train import train_all

        if args.models:
            cfg["models"] = args.models
        if args.seeds:
            cfg["seeds"] = args.seeds
        result = train_all(cfg)
    elif args.command == "evaluate":
        from vn_av_df.training.evaluate import evaluate_all

        result = evaluate_all(cfg, args.force)
    else:
        from vn_av_df.training.report import build_report

        result = build_report(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
