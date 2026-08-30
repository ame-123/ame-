"""把本地路径伪装成 MaxKB 处理器需要的 file 对象。"""

from __future__ import annotations

from pathlib import Path


class LocalFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.name

    def read(self) -> bytes:
        return self.path.read_bytes()

    def chunks(self, size: int = 8192):
        with self.path.open("rb") as handle:
            while True:
                data = handle.read(size)
                if not data:
                    break
                yield data
