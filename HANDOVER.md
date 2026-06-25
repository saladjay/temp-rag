# 交接文档 · 多轮对话智能客服系统

> 接手人：在新机器上从本文第 5 节开始。最后更新 2026-06-25。

## 1. 这是什么

LangGraph + 多知识库 RAG 的多轮对话智能客服，FastAPI HTTP + SSE 流式交付。
核心硬指标：**同一问题多次提问 → 检索严格一致 + 答案语义稳定**。

- 设计 spec：`docs/superpowers/specs/2026-06-25-langgraph-multikb-customer-service-design.md`
- 实现计划：`docs/superpowers/plans/2026-06-25-langgraph-multikb-customer-service.md`

## 2. 当前状态

- 分支：`main`（HEAD `2b9df37`）。原特性分支 `feat/multikb-customer-service` 已合并删除。
- 12 个实现任务**全部完成并合入 main**，每任务经实现+评审两轮。端到端代码 39 测试曾全绿。
- **kbmap（知识库分类工具链）已由另一会话合入 main**（`5e90d7b Merge branch 'kbmap-reconcile'` 系列），且**动了 `app/ingest/`（把原 `FixedChunker` 换成"结构感知 chunker"并重接了 pipeline）**。
- ⚠️ **合并 kbmap 后 pytest 尚未复跑确认**。新机器第一件事就是跑 `pytest -q`——`tests/unit/test_chunker.py`、`tests/unit/test_pipeline.py`、`tests/stability/` 最可能受影响。

## 3. 迁移到新机器：这些被 gitignore，必须手动带/重建

| 项 | 说明 | 怎么办 |
|---|---|---|
| `.env` | 真实云服务令牌与地址（**最关键**，没有它服务全连不上） | 从旧机器拷贝；或向运维获取（见第 11 节键名） |
| `.venv/` | Python 虚拟环境 | 新机器 `python -m venv .venv` + `pip install -r requirements.txt` |
| Docker 卷 | Milvus 数据（etcd/minio/milvus_data） | 不迁移；新机器重新建库 + 重灌语料 |
| `kb_registry.json` | 已注册的知识库清单 | 由 `scripts/init_milvus.py` 重新生成 |
| `.superpowers/sdd/` | SDD 进度账本/任务简报 | 不随 clone 过去；内容已并入本文与 spec/plan |

## 4. 环境要求

- Python **3.10+**
- Docker（跑 Milvus；CPU 镜像即可，GPU 需较强显卡，见第 6 节）
- Redis（会话历史 + 确定性缓存；单测用 fakeredis，**端到端需真实 Redis**，本机此前没起）
- 代码注释/文档中文，标识符英文

## 5. 新机器上从零跑起来（按顺序）

```bash
# 0) 拿到代码 + .env（务必）
git clone <repo> && cd langgraph
#   把旧机器的 .env 放到项目根

# 1) Python 环境（Windows 用 Scripts，Linux/Mac 用 bin）
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2) 先确认测试绿（合并 kbmap 后尚未验证）
.venv/Scripts/python -m pytest -q
#   若 test_chunker / test_pipeline / stability 挂了 → kbmap 的结构感知 chunker 改了契约，需要同步更新这些测试或回看 kbmap 那几个提交

# 3) 起 Milvus（CPU 版；GPU 见第 6 节）
docker compose -f docker/docker-compose.yml up -d
curl -sf http://localhost:9092/healthz && echo OK   # 等 30~60s

# 4) 建知识库 collection（dim 自动探测；bge-m3=1024）
.venv/Scripts/python scripts/init_milvus.py --kb demo --dim 1024 --model bge-m3

# 5) 灌语料（二选一）
.venv/Scripts/python -m app.ingest --kb demo --dir ./your_docs
#   或 HTTP：POST /api/v1/ingest  (multipart: kb=..., file=...)

# 6) 起 Redis（任一方式）
docker run -d -p 6379:6379 --name redis redis:7

# 7) 起服务
.venv/Scripts/python -m uvicorn app.main:app --port 8000

# 8) 试问
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","question":"你好"}'
```

## 6. Milvus 避坑（已踩过的坑）

- **GPU 镜像 `milvusdb/milvus:v2.4.10-gpu` 在入门级 GPU 上必崩**：Knowhere RAFT 初始化 `raft_initialization.cc:73` SIGABRT（旧机器 GTX 1650/4GB 实测）。新机器若显卡强（≥8GB、计算能力 7.5+）可试 `docker/docker-compose-gpu.yml`；否则用 CPU 的 `docker/docker-compose.yml`，稳。
- **国内拉 `docker.io` 需代理或镜像**：
  - 代理：Docker Desktop → Settings → Resources → Proxies（填本地代理，如 `http://127.0.0.1:7890`）。
  - 镜像：`docker.1panel.live` / `docker.m.daocloud.io` 可达。但 **>1GB 的单层在镜像上易 EOF**（GPU 镜像的两个 1.4GB 层就是死在这）；CPU 镜像层较小，镜像通常拉得下来。
  - `quay.io`（etcd）国内一般可直连。
- 端口：19530(gRPC，pymilvus 连这个)、9092→9091(metrics/healthz)。

## 7. 代码地图（`app/`）

```
app/
  main.py                # FastAPI app 工厂
  config.py              # 全部配置（含稳定性默认档）
  api/                   # /chat(SSE) /ingest /sessions /health
  graph/                 # LangGraph: state.py / nodes.py / builder.py
    节点: load_history → rewrite → retrieve → rerank → generate → save_history
  store/                 # milvus_store / session_store(Redis) / cache(确定性缓存)
  ingest/                # 【注意】kbmap 合并后: chunker 换成结构感知版 + pipeline 重接
  services/              # 复用的 4 个云服务: CloudEmbedding(bge-m3) / CloudRerank / CloudCompletion(deepseek_v4) / MinerU
tests/
  unit/      每个模块；stability/ 固化验收口径；integration/ API
```

## 8. 关键设计点（接手别踩坏）

- **稳定性命脉**：
  - 检索：Milvus 固定 `ef/top_k`，应用层 `score DESC → pk ASC` 排序后返回（多库合并不再全局截断，交由 rerank）。
  - 重排：`score DESC → doc_id ASC → segment_id ASC`，全序、稳定排序。
  - 生成：`temperature=0, top_p=1.0`；确定性缓存命中 → 逐字一致。
  - 稳定性自动测试：`tests/stability/test_retrieval_stability.py`（检索 10× 全等 + 缓存 `llm.calls==1`）。
- **一个知识库 = 一个 collection**，命名 `kb_<name>`。
- 生成/重写节点**必须用 `llm.chat(messages=...)`**，不能用 `.complete(prompt=...)`——`deepseek_v4` 端点要求 messages 体（否则 `error parsing the body`）。
- 所有影响结果的参数在 `app/config.py`，默认即稳定档；换模型只改 `*_BACKEND`/`*_MODEL`。

## 9. 未验证 / 已知问题（接手优先级）

1. **合并 kbmap 后测试是否还绿**（最高优先，先跑 pytest）。kbmap 改了 chunker/pipeline，`test_chunker.py`/`test_pipeline.py`/`tests/stability/` 可能要同步更新。
2. **pymilvus 客户端 ↔ 真实 Milvus 端到端**未在本机验过（旧机 RAM 不足，`import pymilvus` 都卡死）。新机器内存够的话补上：连 19530 + `list_collections` + `init_milvus` 建库。
3. **真实后端联调**未做：bge-m3 嵌入、embed_rerank 重排、deepseek_v4 生成、MinerU 解析——代码层用 fake 过了单测，真实链路待联。
4. 既往 Minor（非阻断，可日后清理）：`writer.py` 死的 `content_hash`；若干未用 `import pytest`；`kb_registry.json` CWD 相对 + 非原子写；Starlette/httpx 弃用警告；`build_graph` 每请求编译；`/ingest` 失败无结构化日志。

## 10. 文档位置

- 设计：`docs/superpowers/specs/2026-06-25-langgraph-multikb-customer-service-design.md`
- 计划：`docs/superpowers/plans/2026-06-25-langgraph-multikb-customer-service.md`
- SDD 本地账本（gitignored，不随 clone）：`.superpowers/sdd/progress.md`（含每任务的 Minor 记录）

## 11. `.env` 需要的键（值是内网/机密，向原作者或运维取）

```
# 云服务（内网 128.23.74.3:9091 等；具体地址/token 见旧机器 .env）
CLOUD_AUTH_TOKEN=...
CLOUD_EMBEDDING_URL=...        # bge-m3
CLOUD_RERANK_URL=...           # embed_rerank
CLOUD_COMPLETION_URL=...       # deepseek_v4, chat-completions 端点
CLOUD_COMPLETION_MODEL=deepseek_v4
MINERU_URL=...
MINERU_AUTH_TOKEN=...

# 本地基础设施（默认值见 app/config.py，按需覆盖）
# MILVUS_HOST=localhost  MILVUS_PORT=19530
# REDIS_URL=redis://localhost:6379/0
```

> 其余键都有合理默认（稳定性档），不覆盖也能跑。真实 `.env` 请从旧机器拷贝，勿提交进库。
