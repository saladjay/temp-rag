"""MilvusWriter：按 doc_id 幂等写入（删旧再插）。"""
from __future__ import annotations
import hashlib
from typing import Optional

from app.config import settings
from app.ingest.interfaces import Chunk


class MilvusWriter:
    def __init__(self, store=None):
        self._store = store

    def _store_(self):
        if self._store is None:
            from app.store.milvus_store import MilvusStore
            self._store = MilvusStore()
        return self._store

    def write(self, kb_name: str, chunks: list[Chunk], vectors) -> int:
        store = self._store_()
        full = f"{settings.milvus_collection_prefix}{kb_name}"
        rows = []
        for ch, vec in zip(chunks, vectors):
            content_hash = hashlib.sha1(ch.text.encode("utf-8")).hexdigest()[:16]
            rows.append({
                "embedding": list(vec),
                "text": ch.text,
                "doc_id": ch.doc_id,
                "doc_name": ch.doc_id,
                "segment_id": ch.segment_id,
                "source": kb_name,
                "embedding_model": getattr(self, "_embedding_model", "bge-m3"),
                # content_hash 进 dynamic field 不在 schema 内 → 改用 doc_id 命名空间去重
            })
        client = store._connect()
        # 幂等：删同 doc_id 旧分段再插
        for doc_id in {c.doc_id for c in chunks}:
            client.delete(full, filter=f'doc_id == "{doc_id}"')
        client.insert(collection_name=full, data=rows)
        return len(rows)
