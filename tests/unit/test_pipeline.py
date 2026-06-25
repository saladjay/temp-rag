"""入库流水线单元测试：用桩件验证 run_ingest 编排与 MilvusWriter 幂等。"""
import numpy as np
from app.ingest.chunker import FixedChunker
from app.ingest.embedder import CloudEmbedder
from app.ingest.writer import MilvusWriter
from app.ingest.pipeline import run_ingest
from app.ingest.interfaces import Chunk


class StubParser:
    def parse(self, file_path):
        return "第一句内容。第二句内容。"


class StubEmbedder:
    def __init__(self):
        self._n = 0
    @property
    def dim(self):
        return 4
    def embed(self, texts):
        # 确定性向量：按内容哈希
        return np.array([[float(len(t) % 7)] * 4 for t in texts], dtype="float32")


class SpyWriter:
    def write(self, kb_name, chunks, vectors):
        return len(chunks)


def test_run_ingest_calls_chunk_embed_write(monkeypatch):
    out = run_ingest("faq", "x.txt",
                     parser=StubParser(), chunker=FixedChunker(size=20, overlap=5, tolerance=10),
                     embedder=StubEmbedder(), writer=SpyWriter())
    assert out >= 1


class FakeMilvusClient:
    """记录 delete/insert，用于验证真实 MilvusWriter 的幂等行为。"""
    def __init__(self):
        self.deleted = []
        self.inserted = []
    def delete(self, collection_name, filter=None, **kw):
        self.deleted.append(filter)
    def insert(self, collection_name, data, **kw):
        self.inserted.append((collection_name, len(data)))


def test_writer_idempotent_deletes_before_insert():
    import numpy as np
    from app.ingest.writer import MilvusWriter
    from app.ingest.interfaces import Chunk
    fake = FakeMilvusClient()
    w = MilvusWriter()
    # 注入 store：store._connect() 返回 fake
    w._store = type("S", (), {"_connect": lambda self: fake})()
    chunks = [Chunk(text="a", doc_id="d1", segment_id="d1#0000", ordinal=0)]
    vecs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype="float32")
    n1 = w.write("faq", chunks, vecs)
    n2 = w.write("faq", chunks, vecs)
    assert n1 == n2 == 1
    # 每次写入前都按 doc_id 删旧 → 两次 delete、两次 insert（幂等：终态等于一次）
    assert len(fake.deleted) == 2
    assert all('doc_id == "d1"' in f for f in fake.deleted)
    assert len(fake.inserted) == 2
