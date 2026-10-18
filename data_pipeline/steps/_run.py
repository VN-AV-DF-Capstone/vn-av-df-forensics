import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from settings import DATA, DATASET_NAME, EXPORTS, ROOT

sys.path.insert(0, str(ROOT / "src"))
from vn_av_data.cli import entrypoint


def run(args, batch_step=False):
    """Chạy một lệnh CLI. batch_step: tham số đầu tiên (nếu có) là số batch, vd. `03_cut.py 2`."""
    extra = sys.argv[1:]
    if batch_step and extra and extra[0].isdigit():
        extra = ["--batch", extra[0], *extra[1:]]
    print(f"Workspace: {ROOT / DATA} | Step: {args[0]}", flush=True)
    os.chdir(ROOT)
    sys.argv = ["vn-av-data", "--data", DATA, "--exports", EXPORTS, "--dataset", DATASET_NAME]
    sys.argv += [str(x) for x in args] + extra
    raise SystemExit(entrypoint())
