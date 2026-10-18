"""Export một part ra exports/<dataset>/<dataset>-partN + ZIP trong data/archive: `07_export.py 1`."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        raise SystemExit("Usage: 07_export.py <part>")
    part = sys.argv.pop(1)
    run(["export", "--part", part])
