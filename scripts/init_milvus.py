"""按 schema 契约建 collection 并注册 KB 清单。

用法:
  python scripts/init_milvus.py --kb faq --dim 1024 --model bge-m3
"""
import argparse
from app.config import settings
from app.store.milvus_store import MilvusStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="知识库名（不含 kb_ 前缀）")
    ap.add_argument("--dim", type=int, default=None, help="向量维度，缺省探测")
    ap.add_argument("--model", default="bge-m3", help="embedding 模型名")
    args = ap.parse_args()

    dim = args.dim
    if dim is None:
        from app.services import CloudEmbeddingService
        dim = CloudEmbeddingService().get_dimension()
        print(f"探测到 embedding 维度 dim={dim}")

    store = MilvusStore()
    store.register_kb(args.kb, dim, args.model)
    print(f"已创建/确认 collection: {settings.milvus_collection_prefix}{args.kb}")


if __name__ == "__main__":
    main()
