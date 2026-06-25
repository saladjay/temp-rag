"""真 Milvus 写入/检索集成测试。

单测的 FakeClient 无法覆盖 pymilvus 真实 API（曾因此漏掉 create_index/search_params
的 orm-vs-MilvusClient 不兼容，见 commit 1632e55）。本测试连真 Milvus
（localhost:19530），用确定性 embedder 跑 ensure_collection→insert→search 全路径；
Milvus 不可达时整体 skip。不依赖云端 embedding/completion。
"""
import json
import pathlib

import numpy as np
import pytest

KB = "itest_real"          # 不含 kb_ 前缀（register_kb/write/search 内部自动加）
FULL = f"kb_{KB}"
DIM = 1024
REG = pathlib.Path("kb_registry.json")


def _milvus_reachable() -> bool:
    try:
        from pymilvus import MilvusClient
        from app.config import settings
        c = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        c.list_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _milvus_reachable(), reason="Milvus 不可达 (localhost:19530)")


@pytest.fixture()
def clean_kb():
    """每个用例前后保证 Milvus 集合与 kb_registry.json 干净。"""
    from app.store.milvus_store import MilvusStore
    backup = REG.read_text("utf-8") if REG.exists() else None
    store = MilvusStore()
    yield store
    # 清理：删集合 + 还原 registry
    try:
        c = store._connect()
        if FULL in c.list_collections():
            c.drop_collection(FULL)
    except Exception:
        pass
    if backup is None:
        if REG.exists():
            data = json.loads(REG.read_text("utf-8"))
            data.pop(KB, None)
            if data:
                REG.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            else:
                REG.unlink()
    else:
        REG.write_text(backup, "utf-8")


def _embedder():
    from app.ingest.embedder import CloudEmbedder
    from app.services.cloud_embedding_service import MockCloudEmbeddingService
    return CloudEmbedder(service=MockCloudEmbeddingService(dimension=DIM))


def test_ensure_collection_creates_index(clean_kb):
    """register_kb 走 create_schema/create_collection/create_index/load（pymilvus 风险点）。"""
    store = clean_kb
    store.register_kb(KB, DIM, "mock-bge-m3")   # 若 create_index API 错，此处抛
    c = store._connect()
    assert FULL in c.list_collections()


def test_insert_and_search_roundtrip(clean_kb):
    """insert（writer + numpy 向量）→ flush → search → 命中字段解析。"""
    from app.ingest.writer import MilvusWriter
    from app.ingest.interfaces import Chunk
    store = clean_kb
    store.register_kb(KB, DIM, "mock-bge-m3")

    chunks = [Chunk(text=f"常见问题片段 {i}", doc_id="d1", segment_id=f"d1#{i:04d}",
                    ordinal=i, doc_name="d1", kb=KB) for i in range(5)]
    vecs = _embedder().embed([c.text for c in chunks])
    n = MilvusWriter(store=store).write(KB, chunks, vecs)
    assert n == 5

    store._connect().flush(FULL)   # 让 HNSW 增长段可搜
    q = np.random.RandomState(7).randn(DIM).astype("float32")
    q = (q / (np.linalg.norm(q) + 1e-8)).tolist()
    hits = store.search(q, [KB], top_k=3, ef=64)
    assert len(hits) == 3
    h = hits[0]
    assert h.text and h.doc_id and h.source == KB
