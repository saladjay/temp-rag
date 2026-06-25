"""定长吸附切块器（中文感知，确定性）。"""
from __future__ import annotations
import re
from .interfaces import Chunk

# 句末边界：中文句号/问号/叹号/分号/换行
_BOUNDARY = re.compile(r"[。！？；\n]")


class FixedChunker:
    def __init__(self, size: int = 500, overlap: int = 80, tolerance: int = 50):
        if overlap >= size:
            raise ValueError("overlap 必须小于 size")
        self.size = size
        self.overlap = overlap
        self.tolerance = tolerance

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        text = text or ""
        n = len(text)
        if n == 0:
            return []
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        ordinal = 0
        while start < n:
            target = start + self.size
            end = self._snap(text, start, target)
            piece = text[start:end]
            chunks.append(Chunk(text=piece, doc_id=doc_id,
                                segment_id=f"{doc_id}#{ordinal:04d}", ordinal=ordinal))
            ordinal += 1
            # 下一块起点：保证至少前进 step
            next_start = start + step
            if next_start >= end:        # 边界吸附把 end 拉到了 step 之前，强制前进
                next_start = end
            start = next_start if next_start > start else end
        return chunks

    def _snap(self, text: str, start: int, target: int) -> int:
        """在 [target-tolerance, target+tolerance] 范围内往回找最近句末标点。"""
        n = len(text)
        target = min(target, n)
        lo = max(start + 1, target - self.tolerance)
        hi = min(n, target + self.tolerance)
        # 从 target 往后找第一个边界
        for i in range(target, hi):
            if _BOUNDARY.match(text[i]):
                return i + 1
        # 再从 target 往前找
        for i in range(target - 1, lo - 1, -1):
            if _BOUNDARY.match(text[i]):
                return i + 1
        return target  # 找不到 → 硬切
