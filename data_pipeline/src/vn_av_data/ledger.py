"""Sổ cái (ledger) của toàn bộ dữ liệu real: một file SQLite, chỉ chứa metadata, không chứa media.

Media (raw, clip, part) có thể đã lên Drive và bị xoá ở máy; ledger vẫn ghi đủ nguồn gốc, kiểm tra,
quyết định duyệt và vị trí lưu (`local` / `drive:<file zip>` / `deleted`) của từng thứ.

- Mỗi đơn vị việc (một video, một clip, một quyết định duyệt) được commit riêng: tắt ngang giữa
  chừng thì chạy lại sẽ tiếp tục từ đơn vị chưa xong.
- Sau mỗi lệnh có ghi: chép một bản sao lưu có thời gian vào `ledger/backups/` và xuất các bảng ra
  `ledger/csv/` để mở bằng Excel.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vn_av_data.common.runtime import atomic_bytes

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS inputs(
  input_id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, kind TEXT, speakers TEXT, note TEXT,
  status TEXT DEFAULT 'pending', reason TEXT DEFAULT '', entries INTEGER, planned_at TEXT);
CREATE TABLE IF NOT EXISTS videos(
  video_id TEXT PRIMARY KEY, input_id INTEGER, rank INTEGER, url TEXT, title TEXT, channel TEXT,
  channel_id TEXT, upload_date TEXT, duration_s REAL, est_mb REAL,
  status TEXT, reason TEXT DEFAULT '', batch INTEGER, meta TEXT,
  dl_status TEXT DEFAULT 'pending', dl_error TEXT DEFAULT '', file TEXT, sha256 TEXT,
  size_mb REAL, probe TEXT,
  cut_status TEXT DEFAULT 'pending', cut_info TEXT, id_status TEXT DEFAULT 'pending',
  raw_location TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS duplicates(
  video_id TEXT, input_id INTEGER, kept_input_id INTEGER, at TEXT,
  PRIMARY KEY(video_id, input_id));
CREATE TABLE IF NOT EXISTS batches(
  batch INTEGER PRIMARY KEY, created_at TEXT, zip TEXT, zip_sha256 TEXT, zip_mb REAL,
  location TEXT DEFAULT 'local');
CREATE TABLE IF NOT EXISTS clips(
  clip_id TEXT PRIMARY KEY, video_id TEXT, input_id INTEGER, batch INTEGER,
  start_s REAL, end_s REAL, duration_s REAL, file TEXT, sha256 TEXT, size_mb REAL,
  face BLOB, face_consistency REAL, voice BLOB, flags TEXT DEFAULT '', jump TEXT DEFAULT '',
  speaker_id TEXT, assign TEXT DEFAULT '', decision TEXT DEFAULT 'pending',
  part INTEGER, split TEXT, location TEXT DEFAULT 'local', created_at TEXT);
CREATE INDEX IF NOT EXISTS clips_speaker ON clips(speaker_id);
CREATE INDEX IF NOT EXISTS clips_video ON clips(video_id);
CREATE TABLE IF NOT EXISTS speakers(
  speaker_id TEXT PRIMARY KEY, created_at TEXT, channel_ids TEXT DEFAULT '[]',
  inputs TEXT DEFAULT '[]', face_sum BLOB, face_n INTEGER DEFAULT 0, voice_sum BLOB,
  voice_n INTEGER DEFAULT 0, photos TEXT DEFAULT '[]', gender TEXT DEFAULT '',
  review TEXT DEFAULT 'pending', suggestion TEXT DEFAULT '', merged_into TEXT DEFAULT '',
  part INTEGER, reviewed_at TEXT, order_key INTEGER);
CREATE TABLE IF NOT EXISTS parts(
  part INTEGER PRIMARY KEY, created_at TEXT, members TEXT, clips INTEGER, export_dir TEXT,
  zip TEXT, zip_sha256 TEXT, zip_mb REAL, location TEXT DEFAULT 'local');
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, at TEXT, step TEXT, message TEXT);
"""
# Cột dạng JSON (đọc ra thành dict/list) và dạng vector (BLOB float32).
JSON_COLUMNS = {"meta", "probe", "cut_info", "channel_ids", "inputs", "photos", "members"}
VECTOR_COLUMNS = {"face", "voice", "face_sum", "voice_sum"}
CSV_TABLES = ("inputs", "videos", "duplicates", "batches", "clips", "speakers", "parts")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_blob(vector):
    return None if vector is None else np.asarray(vector, np.float32).tobytes()


def from_blob(value):
    return None if value is None else np.frombuffer(value, np.float32).copy()


class Ledger:
    """Truy cập SQLite có khoá (server duyệt gọi từ nhiều luồng)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.lock = threading.RLock()
        self.depth = 0
        self.conn.executescript(SCHEMA)
        version = self.get_meta("schema_version")
        if version is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        elif int(version) != SCHEMA_VERSION:
            raise ValueError(f"Ledger schema {version} != {SCHEMA_VERSION}; migrate first")

    def close(self):
        self.conn.close()

    @contextmanager
    def tx(self):
        """Một giao dịch: hoặc ghi hết, hoặc không ghi gì (kể cả khi bị Ctrl+C).

        Lồng nhau thì phần trong thuộc về giao dịch ngoài cùng."""
        with self.lock:
            if self.depth:
                self.depth += 1
                try:
                    yield self.conn
                finally:
                    self.depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self.depth = 1
            try:
                yield self.conn
            except BaseException:
                self.depth = 0
                self.conn.execute("ROLLBACK")
                raise
            self.depth = 0
            self.conn.execute("COMMIT")

    # ---- đọc ----
    @staticmethod
    def decode(row):
        if row is None:
            return None
        out = dict(row)
        for key, value in out.items():
            if key in JSON_COLUMNS and isinstance(value, str) and value:
                out[key] = json.loads(value)
            elif key in VECTOR_COLUMNS:
                out[key] = from_blob(value)
        return out

    def rows(self, sql, params=()):
        with self.lock:
            return [self.decode(r) for r in self.conn.execute(sql, params).fetchall()]

    def row(self, sql, params=()):
        with self.lock:
            return self.decode(self.conn.execute(sql, params).fetchone())

    def scalar(self, sql, params=()):
        with self.lock:
            value = self.conn.execute(sql, params).fetchone()
        return None if value is None else value[0]

    # ---- ghi ----
    @staticmethod
    def encode(values):
        out = {}
        for key, value in values.items():
            if key in VECTOR_COLUMNS:
                out[key] = to_blob(value)
            elif key in JSON_COLUMNS and not isinstance(value, (str, type(None))):
                out[key] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            else:
                out[key] = value
        return out

    def insert(self, table, values, conn=None, replace=False):
        values = self.encode(values)
        keys = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        sql = f"{verb} INTO {table}({keys}) VALUES({marks})"
        if conn is not None:
            conn.execute(sql, tuple(values.values()))
            return
        with self.tx() as c:
            c.execute(sql, tuple(values.values()))

    def update(self, table, key, ident, values, conn=None):
        values = self.encode(values)
        sets = ", ".join(f"{k}=?" for k in values)
        sql = f"UPDATE {table} SET {sets} WHERE {key}=?"
        params = (*values.values(), ident)
        if conn is not None:
            conn.execute(sql, params)
            return
        with self.tx() as c:
            c.execute(sql, params)

    def get_meta(self, key):
        return self.scalar("SELECT value FROM meta WHERE key=?", (key,))

    def set_meta(self, key, value, conn=None):
        self.insert("meta", {"key": key, "value": str(value)}, conn=conn, replace=True)

    def log(self, step, message, conn=None):
        self.insert("events", {"at": now(), "step": step, "message": message}, conn=conn)

    # ---- sao lưu ----
    def backup(self, label="manual", keep=30):
        """Bản sao nhất quán (sqlite backup API) + xuất CSV; giữ `keep` bản gần nhất."""
        folder = self.path.parent / "backups"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = folder / f"ledger_{stamp}_{label}.sqlite"
        with self.lock:
            copy = sqlite3.connect(target)
            try:
                self.conn.backup(copy)
            finally:
                copy.close()
        for old in sorted(folder.glob("ledger_*.sqlite"))[:-keep]:
            old.unlink(missing_ok=True)
        self.export_csv()
        return target

    def export_csv(self):
        folder = self.path.parent / "csv"
        folder.mkdir(parents=True, exist_ok=True)
        for table in CSV_TABLES:
            with self.lock:
                cursor = self.conn.execute(f"SELECT * FROM {table}")
                fields = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            # Vector không có ích khi mở bằng Excel: chỉ ghi có/không.
            keep = [i for i, f in enumerate(fields) if f not in ("meta", "probe")]
            writer.writerow([fields[i] for i in keep])
            for row in rows:
                writer.writerow(
                    [
                        ("yes" if row[i] is not None else "")
                        if fields[i] in VECTOR_COLUMNS
                        else row[i]
                        for i in keep
                    ]
                )
            atomic_bytes(folder / f"{table}.csv", stream.getvalue().encode("utf-8-sig"))
