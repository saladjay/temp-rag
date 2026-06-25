"""入库流水线编排 + CLI 入口。"""
from __future__ import annotations
import argparse
from pathlib import Path

from app.config import settings
from app.utils.log import get_logger

logger = get_logger(__name__)


def run_ingest(kb_name: str, file_path: str, parser, chunker, embedder, writer) -> int:
    text = parser.parse(file_path)
    chunks = chunker.chunk(text, doc_id=Path(file_path).stem,
                           doc_name=Path(file_path).stem, kb=kb_name)
    if not chunks:
        return 0
    vectors = embedder.embed([c.text for c in chunks])
    n = writer.write(kb_name, chunks, vectors)
    logger.info("ingest_done", kb=kb_name, file=file_path, chunks=n)
    return n


def _make_components():
    from app.ingest.chunker import StructuralChunker
    from app.ingest.embedder import CloudEmbedder
    from app.ingest.parser import MinerUParser
    from app.ingest.writer import MilvusWriter
    chunker = StructuralChunker()
    return MinerUParser(), chunker, CloudEmbedder(), MilvusWriter()


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    parser, chunker, embedder, writer = _make_components()
    total = 0
    for p in Path(args.dir).glob("**/*"):
        if p.is_file():
            total += run_ingest(args.kb, str(p), parser, chunker, embedder, writer)
    print(f"入库完成 kb={args.kb} 总分段={total}")


if __name__ == "__main__":
    cli()
