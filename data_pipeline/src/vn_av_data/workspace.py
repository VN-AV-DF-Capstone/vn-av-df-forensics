"""Thư mục làm việc của pipeline real v003 (mặc định `data_pipeline/data/`).

    inputs/urls.csv                  danh sách URL bạn đưa (url, speakers = 1 | n, note)
    ledger/ledger.sqlite             sổ cái; backups/ và csv/ đi kèm
    raw/batch_001/videos/*.mp4       video gốc của từng batch (lên Drive xong thì xoá được)
    raw/batch_001/videos.csv         danh sách link của batch để xem / tải tay
    pool/clips/<video_id>/*.mp4      clip ứng viên của mọi batch (chung một chỗ)
    pool/faces/<clip_id>.jpg         5 mặt của clip (đầu, ngẫu nhiên, giữa, ngẫu nhiên, cuối)
    registry/<speaker_id>_<k>.jpg    ảnh mặt đại diện của speaker
    archive/                         file zip chờ up lên Drive
Part đã export nằm ở `data_pipeline/exports/<dataset>/<dataset>-partN/` (generation đọc từ đó).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path
    exports: Path
    dataset: str

    @classmethod
    def at(cls, root, exports, dataset):
        return cls(Path(root).resolve(), Path(exports).resolve(), dataset)

    @property
    def inputs(self):
        return self.root / "inputs/urls.csv"

    @property
    def ledger(self):
        return self.root / "ledger/ledger.sqlite"

    def raw(self, batch):
        return self.root / "raw" / f"batch_{batch:03d}"

    @property
    def clips(self):
        return self.root / "pool/clips"

    @property
    def faces(self):
        return self.root / "pool/faces"

    @property
    def registry(self):
        return self.root / "registry"

    @property
    def archive(self):
        return self.root / "archive"

    def part_name(self, part):
        return f"{self.dataset}-part{part}"

    def part_dir(self, part):
        return self.exports / self.dataset / self.part_name(part)
