"""本地 GPU Milvus 直连存储：固定搜索参数 + 确定排序。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class MilvusSearchHit:
    pk: int
    score: float
    text: str
    doc_id: str
    doc_name: str
    segment_id: str
    source: str


class MilvusStore:
    def __init__(self, client=None):
        # client 可注入：生产用 pymilvus，测试用 fake
        self._client = client

    def _connect(self):
        if self._client is not None:
            return self._client
        from pymilvus import MilvusClient
        self._client = MilvusClient(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            db_name=settings.milvus_db,
        )
        return self._client

    def search(self, query_vector, kb_names: list[str], top_k: int, ef: int) -> list[MilvusSearchHit]:
        client = self._connect()
        collected: list[MilvusSearchHit] = []
        for kb in kb_names:
            name = f"{settings.milvus_collection_prefix}{kb}"
            res = client.search(
                collection_name=name,
                data=[query_vector],
                anns_field="embedding",
                search_params={"metric_type": settings.milvus_metric, "params": {"ef": ef}},
                limit=top_k,
                output_fields=["text", "doc_id", "doc_name", "segment_id", "source"],
            )
            for r in res[0]:
                ent = r["entity"]
                collected.append(MilvusSearchHit(
                    pk=r["id"], score=float(r["distance"]), text=ent["text"],
                    doc_id=ent["doc_id"], doc_name=ent["doc_name"],
                    segment_id=ent["segment_id"], source=ent["source"],
                ))
        # 确定排序：score DESC → pk ASC；稳定排序保证同分顺序确定
        collected.sort(key=lambda h: (-h.score, h.pk))
        # 每个 KB 已由上面 limit=top_k 限定；合并后不再全局截断，
        # 最终 top_n 由 rerank 节点负责（避免强 KB 挤占弱 KB 的召回）。
        return collected

    def ensure_collection(self, kb_name: str, dim: int) -> None:
        from pymilvus import DataType
        client = self._connect()
        full = f"{settings.milvus_collection_prefix}{kb_name}"
        if full in client.list_collections():
            return
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_name", DataType.VARCHAR, max_length=512)
        schema.add_field("segment_id", DataType.VARCHAR, max_length=128)
        schema.add_field("source", DataType.VARCHAR, max_length=64)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=128)
        client.create_collection(collection_name=full, schema=schema)
        # MilvusClient 建索引需 IndexParams 对象（2.4+ 起 index_type/metric_type/params
        # 不再作为 create_index 的 kwargs，老 orm Collection 写法对 MilvusClient 无效）。
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type=settings.milvus_metric,
            params={"M": settings.milvus_hnsw_m,
                    "efConstruction": settings.milvus_ef_construction},
        )
        client.create_index(collection_name=full, index_params=index_params)
        client.load_collection(full)
        logger.info("milvus_collection_created", name=full, dim=dim)

    def register_kb(self, kb_name: str, dim: int, embedding_model: str) -> None:
        self.ensure_collection(kb_name, dim)
        # KB 清单写到本地 json（init_milvus 与问答侧共用）
        import json, pathlib
        p = pathlib.Path("kb_registry.json")
        data = json.loads(p.read_text("utf-8")) if p.exists() else {}
        data[kb_name] = {"dim": dim, "embedding_model": embedding_model}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def list_kbs(self) -> list[str]:
        import json, pathlib
        p = pathlib.Path("kb_registry.json")
        if not p.exists():
            return []
        return list(json.loads(p.read_text("utf-8")).keys())
