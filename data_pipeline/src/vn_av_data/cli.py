"""CLI của pipeline real v003 (không import torch). Mọi lệnh đọc/ghi một ledger chung.

Lệnh có ghi vào ledger sẽ tự sao lưu ledger (ledger/backups/) và xuất CSV sau khi chạy, kể cả
khi bị lỗi hay Ctrl+C giữa chừng.
"""

import argparse
import json
import sys

READ_ONLY = {"status", "validate", "calibrate", "jumpcut-eval"}


def pick_batch(ledger, step):
    """Batch đầu tiên còn việc cho bước này."""
    sql = {
        "download": "SELECT MIN(batch) FROM videos WHERE status='accepted' AND batch IS NOT NULL "
        "AND dl_status IN ('pending','failed') AND batch IN "
        "(SELECT batch FROM batches WHERE location='local')",
        "cut": "SELECT MIN(batch) FROM videos WHERE dl_status='done' AND cut_status!='done'",
        "identify": "SELECT MIN(batch) FROM videos WHERE cut_status='done' AND id_status!='done'",
    }[step]
    batch = ledger.scalar(sql)
    if batch is None:
        raise ValueError(f"No batch has pending work for {step}")
    return batch


def build_parser():
    parser = argparse.ArgumentParser(prog="vn-av-data")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--data", default="data", help="workspace (ledger, raw, pool, ...)")
    parser.add_argument("--exports", default="exports")
    parser.add_argument("--dataset", default="vn-av-df-data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    for name in ("plan", "download"):
        p = sub.add_parser(name)
        p.add_argument(
            "--cookies-from-browser",
            choices=("firefox", "chrome", "edge", "brave"),
            help="Use a browser session when YouTube rejects anonymous requests",
        )
        p.add_argument("--force-ipv4", action="store_true")
        if name == "download":
            p.add_argument("--batch", default="auto")
    for name in ("cut", "identify"):
        sub.add_parser(name).add_argument("--batch", default="auto")
    p = sub.add_parser("jumpcut-eval")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--windows", type=int, default=60)
    sub.add_parser("review").add_argument("--port", type=int, default=8001)
    sub.add_parser("parts")
    sub.add_parser("export").add_argument("--part", type=int, required=True)
    p = sub.add_parser("archive")
    p.add_argument("kind", choices=("raw", "part"))
    p.add_argument("number", type=int)
    p.add_argument("--uploaded", action="store_true", help="đã up file zip lên Drive")
    p = sub.add_parser("clean")
    p.add_argument("kind", choices=("raw", "part", "pool"))
    p.add_argument("number", type=int, nargs="?")
    sub.add_parser("status")
    sub.add_parser("calibrate")
    sub.add_parser("validate").add_argument("--dataset-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from vn_av_data.common.runtime import load_config

    cfg = load_config(args.config)
    if args.command == "setup":
        from vn_av_data.assets import setup_assets

        print(json.dumps(setup_assets(cfg), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        from vn_av_data.contract import validate_bundle

        print(json.dumps(validate_bundle(args.dataset_dir)[1], ensure_ascii=False, indent=2))
        return 0
    from vn_av_data.ledger import Ledger
    from vn_av_data.workspace import Workspace

    ws = Workspace.at(args.data, args.exports, args.dataset)
    ledger = Ledger(ws.ledger)
    try:
        result = run(args, cfg, ws, ledger)
    finally:
        if args.command not in READ_ONLY:
            ledger.backup(args.command)
        ledger.close()
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


def run(args, cfg, ws, ledger):
    command = args.command
    if command == "plan":
        from vn_av_data.data.plan import plan_sources

        return plan_sources(
            ws,
            ledger,
            cfg,
            cookies_from_browser=args.cookies_from_browser,
            force_ipv4=args.force_ipv4,
        )
    if command in ("download", "cut", "identify"):
        batch = pick_batch(ledger, command) if args.batch == "auto" else int(args.batch)
        print(f"Batch {batch}", flush=True)
        if command == "download":
            from vn_av_data.data.acquisition import download_batch

            return download_batch(ws, ledger, batch, args.cookies_from_browser, args.force_ipv4)
        if command == "cut":
            from vn_av_data.data.cut import cut_batch

            return cut_batch(ws, ledger, batch, cfg)
        from vn_av_data.data.identity import identify_batch

        return identify_batch(ws, ledger, batch, cfg)
    if command == "jumpcut-eval":
        from vn_av_data.data.cut import jumpcut_eval

        return jumpcut_eval(ws, ledger, cfg, args.batch, args.windows)
    if command == "review":
        import uvicorn

        from vn_av_data.serving.speakers import create_app

        print(f"Mở http://127.0.0.1:{args.port} - Ctrl+C để dừng (mọi thao tác đã tự lưu).")
        uvicorn.run(create_app(ws, ledger, cfg), host="127.0.0.1", port=args.port)
        return None
    if command == "parts":
        from vn_av_data.data.parts import form_parts

        return form_parts(ledger, cfg)
    if command == "export":
        from vn_av_data.data.export import export_part

        return export_part(ws, ledger, args.part)
    if command == "clean":
        from vn_av_data.data import archive

        if args.kind == "pool":
            return archive.clean_pool(ws, ledger)
        if args.number is None:
            raise ValueError(f"Cần số {args.kind}, ví dụ: 09_clean.py {args.kind} 1")
        return archive.delete_local(ws, ledger, args.kind, args.number)
    if command == "archive":
        from vn_av_data.data import archive

        if args.uploaded:
            return archive.mark_uploaded(ws, ledger, args.kind, args.number)
        if args.kind == "raw":
            return archive.pack_raw(ws, ledger, args.number)
        record = ledger.row("SELECT zip, location FROM parts WHERE part=?", (args.number,))
        if record is None or not record["zip"]:
            raise ValueError("Part not exported yet; run 07_export.py first (it creates the zip)")
        return {"part": args.number, **record, "next": "Up zip + ledger/ lên Drive, rồi --uploaded"}
    if command == "status":
        from vn_av_data.data.status import collect_status, print_status

        print_status(collect_status(ws, ledger, cfg["parts"]["clips_per_speaker"]))
        return None
    if command == "calibrate":
        from vn_av_data.data.identity import calibrate

        return calibrate(ledger)
    raise ValueError(f"Unknown command {command}")


def entrypoint():
    try:
        return main()
    except KeyboardInterrupt:
        print("Đã dừng; chạy lại cùng lệnh để tiếp tục.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
