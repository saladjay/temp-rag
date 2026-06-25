"""入库流水线各阶段接口（Protocol）。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass
class Chunk:
    text: str
    doc_id: str
    segment_id: str
    ordinal: int
    # 结构感知切块新增字段（都有默认值，向后兼容 4 字段构造）
    doc_name: str = ""
    kb: str = ""
    heading: str | None = None
    source_chunk_ids: list = field(default_factory=list)


class Chunker(Protocol):
    def chunk(self, text: str, doc_id: str, doc_name: str = "", kb: str = "") -> list[Chunk]: ...


class Parser(Protocol):
    def parse(self, file_path: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]):  # -> np.ndarray (n, dim)
        ...
    @property
    def dim(self) -> int: ...


class Writer(Protocol):
    def write(self, kb_name: str, chunks: list[Chunk], vectors) -> int: ...
