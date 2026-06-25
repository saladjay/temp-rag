"""入库流水线各阶段接口（Protocol）。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class Chunk:
    text: str
    doc_id: str
    segment_id: str
    ordinal: int


class Chunker(Protocol):
    def chunk(self, text: str, doc_id: str) -> list[Chunk]: ...


class Parser(Protocol):
    def parse(self, file_path: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]):  # -> np.ndarray (n, dim)
        ...
    @property
    def dim(self) -> int: ...


class Writer(Protocol):
    def write(self, kb_name: str, chunks: list[Chunk], vectors) -> int: ...
