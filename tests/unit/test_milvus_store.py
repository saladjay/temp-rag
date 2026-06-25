import pytest
from app.store.milvus_store import MilvusStore, MilvusSearchHit


class FakeRes:
    def __init__(self, rows):
        self._rows = rows  # list[dict]
    def __iter__(self):
        return iter(self._rows)


class FakeClient:
    def __init__(self):
        self.searched = []
        self.kbs = []
    def list_collections(self):
        return list(self.kbs)
    def create_collection(self, name, schema, **kw):
        self.kbs.append(name)
    def create_index(self, name, **kw):
        pass
    def get_collection_schema(self, name):
        class S:
            fields = {"embedding": type("F", (), {"params": {"dim": 1024}})()}
        return S()
    def load_collection(self, name):
        pass
    def search(self, collection_name, data, anns_field, param, limit, output_fields, **kw):
        self.searched.append((collection_name, param, limit))
        # 构造两行同分但 pk 不同的命中，验证二级排序
        rows = [
            {"id": 2, "distance": 0.9, "entity": {"text": "b", "doc_id": "d2",
              "doc_name": "n2", "segment_id": "d2#0001", "source": "kb_a"}},
            {"id": 1, "distance": 0.9, "entity": {"text": "a", "doc_id": "d1",
              "doc_name": "n1", "segment_id": "d1#0000", "source": "kb_a"}},
            {"id": 3, "distance": 0.5, "entity": {"text": "c", "doc_id": "d3",
              "doc_name": "n3", "segment_id": "d3#0002", "source": "kb_a"}},
        ]
        # 真实 Milvus 会按 limit 截断；fake 在此对齐
        return [rows[:limit]]


def test_search_tiebreak_score_desc_then_pk_asc():
    store = MilvusStore(client=FakeClient())
    hits = store.search([0.1]*1024, ["kb_a"], top_k=3, ef=64)
    # 两个 0.9 同分 → pk 升序：1 在 2 前；0.5 最后
    assert [h.pk for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == [0.9, 0.9, 0.5]


def test_search_respects_per_kb_limit():
    store = MilvusStore(client=FakeClient())
    hits = store.search([0.1]*1024, ["kb_a"], top_k=2, ef=64)
    assert len(hits) == 2


def test_search_multi_kb_not_globally_truncated():
    """多库检索：合并后不全局截断，2 库 × top_k=2 → 4 条。"""
    store = MilvusStore(client=FakeClient())
    hits = store.search([0.1]*1024, ["kb_a", "kb_b"], top_k=2, ef=64)
    assert len(hits) == 4


def test_uses_fixed_ef_param():
    fake = FakeClient()
    store = MilvusStore(client=fake)
    store.search([0.1]*1024, ["kb_a"], top_k=3, ef=128)
    assert fake.searched[0][1]["params"]["ef"] == 128


def test_hits_are_dataclass():
    store = MilvusStore(client=FakeClient())
    h = store.search([0.1]*1024, ["kb_a"], top_k=1, ef=64)[0]
    assert isinstance(h, MilvusSearchHit)
    assert h.source == "kb_a"
