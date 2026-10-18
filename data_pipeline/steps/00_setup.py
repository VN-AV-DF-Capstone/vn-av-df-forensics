"""Chạy một lần: tải YuNet, Silero VAD, SFace (mặt), WeSpeaker (giọng), SyncNet v2 (độ lệch tiếng-hình) vào checkpoints/media."""

import sys

sys.dont_write_bytecode = True
from _run import run

if __name__ == "__main__":
    run(["setup"])
