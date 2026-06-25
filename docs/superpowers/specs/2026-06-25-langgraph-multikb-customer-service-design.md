# 多轮对话智能客服系统设计（LangGraph + 多知识库 RAG）

- 状态：草案（待评审）
- 日期：2026-06-25
- 技术栈：Python 3.10+ / FastAPI / LangChain (core) / LangGraph / pymilvus / Redis / Docker GPU Milvus
- 复用资产：现有 `services/` 中的 `CloudEmbeddingService`(bge-m3)、`CloudRerankService`(embed_rerank)、`CloudCompletionService`/`GLMCompletionService`(Qwen3-32B)、`MinerUService`

---

## 1. 目标与非目标

### 目标
1. 基于 LangChain + LangGraph 的多轮对话智能客服，FastAPI HTTP + SSE 流式交付。
2. 横跨多个知识库的智能问答，并行检索 + 重排合并。
3. **稳定性硬指标**：同一问题多次提问——
   - 检索结果（doc_id + segment_id 序列）**严格一致**；
   - 生成答案**语义稳定**（事实/结论/引用来源一致，允许个别字差异）。
4. 文件→入库全流程**模型可插拔**：换 embedding/解析/切块/生成模型时只动配置。
5. 复用现有云端服务接口方式（HTTP + Basic 认证 + 重试 + Mock 体系）。

### 非目标
- 不做 Web UI（仅 HTTP + SSE 接口）。
- 不做用户体系/权限/审计；会话以 `session_id` 标识，不绑定账号。
- 不替代现有远程 OA 知识库 HTTP 客户端（`MilvusKnowledgeClient`），二者独立。

---

## 2. 关键决策汇总（已确认）

| 维度 | 决策 |
|------|------|
| 范围 | 问答侧 + 入库流水线（模型可插拔） |
| 多库检索 | 并行检索全库 + 重排合并（不引入 LLM 路由，避免非确定性） |
| 多轮深度 | 历史 + 查询重写（LLM `t=0` 把追问重写成独立问题） |
| 稳定性口径 | 检索严格一致 + 答案语义稳定 |
| 交付形态 | FastAPI HTTP + SSE 流式 |
| 会话状态 | 服务端按 `session_id` 存 Redis |
| 默认切块 | 定长吸附（按字符滑动 + 句末边界吸附，中文感知），确定可复现 |
| 确定性缓存 | 默认开启；命中即逐字一致 |
| 语义相似阈值 | 0.98（同问题多次生成答案两两 embedding 余弦下限） |
| 宿主机 | 已具备 NVIDIA GPU + CUDA 11.8；需另装 NVIDIA Container Toolkit |

---

## 3. 整体架构

```
客户端(SSE)
   │  POST /api/v1/chat  { session_id, question, kb_names? }
   ▼
FastAPI 应用层 (pydantic 请求/响应, 与现有风格一致)
   │
   ▼
LangGraph StateGraph (async, 编译一次常驻)
   load_history ──▶ rewrite ──▶ retrieve ──▶ rerank ──▶ generate ──▶ save_history
   │
   ▼  服务层 (复用 + 新增)
 ┌───────────────┬────────────────┬───────────────┬──────────────────┐
 │ CloudEmbed    │ MilvusStore    │ CloudRerank   │ CloudCompletion  │
 │ bge-m3 (复用) │ pymilvus→本地  │ embed_rerank  │ Qwen3-32B (复用) │
 │               │ GPU Milvus(新) │ (复用)        │                  │
 └───────────────┴────────────────┴───────────────┴──────────────────┘
   │
   ▼
Redis: SessionStore(会话历史) + DeterministicCache(可选, 命中即逐字一致)
```

入库流水线作为离线/批量动作独立运行，与问答链路仅在 Milvus schema 契约处耦合。

---

## 4. 组件清单

| 组件 | 路径 | 状态 | 职责 |
|------|------|------|------|
| FastAPI 路由 | `app/api/chat.py`、`app/api/ingest.py` | 新 | `/chat`(SSE)、`/sessions`、`/ingest`、`/health`、`/metrics` |
| LangGraph 图 | `app/graph/` | 新 | `StateGraph` 定义、`ChatState`、6 个节点函数 |
| Milvus 存储 | `app/store/milvus_store.py` | 新 | pymilvus 直连本地 GPU Milvus；固定搜索参数 + 确定排序 |
| 会话存储 | `app/store/session_store.py` | 新 | Redis 会话历史，有界窗口 + TTL |
| 确定性缓存 | `app/store/cache.py` | 新 | key=query+sources+模型哈希 |
| 入库流水线 | `app/ingest/` | 新 | Parser/Chunker/Embedder/Writer 四段可插拔 |
| 配置 | `app/config.py`、`.env` | 扩展 | 新增 milvus/redis/缓存/稳定性/入库参数 |
| 建库脚本 | `scripts/init_milvus.py` | 新 | 按 schema 契约建 collection + 注册 KB 清单 |
| 部署 | `docker/docker-compose-gpu.yml`、`docker/verify_gpu.sh` | 新 | etcd+minio+milvus-standalone(GPU) + GPU 探测 |
| 现有云服务 | `services/*` | 复用 | 认证/重试/Mock 体系沿用 |

**取舍：** 现有云服务原样复用，在 LangGraph 节点里用 `asyncio.to_thread` 包异步；LangChain 用其 `core`（`Document`、消息、prompt、输出解析），LangGraph 负责状态机编排——两者都真正用上，但不让框架遮住稳定性参数。

---

## 5. Milvus Schema 契约

物理模型：**一个知识库 = 一个 collection**，命名 `kb_<name>`（如 `kb_faq`、`kb_product_manual`）。`kb_names` 请求参数可选，缺省 = 查全部已注册 collection。

### 单 collection schema（所有库统一字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `pk` | Int64 主键, 自增 | 确定排序末位 tie-breaker |
| `embedding` | FloatVector, dim=1024 | bge-m3 dense 维度（换模型时自动对齐） |
| `text` | VarChar(65535) | 切块正文（生成时拼入上下文） |
| `doc_id` | VarChar(128) | 文档唯一ID |
| `doc_name` | VarChar(512) | 文档名（引用展示） |
| `segment_id` | VarChar(128) | 段落ID（检索稳定性断言用） |
| `source` | VarChar(64) | = KB 名，回显来源 |
| `embedding_model` | VarChar(128) | 入库所用模型（可追溯） |

### 索引与搜索参数
- 索引：HNSW（`M=16, efConstruction=200`），度量 `COSINE`；`doc_id`/`segment_id` 加标量索引。
- 固定搜索参数（写入 config）：`top_k_per_kb=10`，`ef=128`，`offset=0`，`nq=1`。
- **确定排序规则：** Milvus 返回后，应用层按 `score DESC → pk ASC` 二级排序后截断 → 命中的 doc_id+segment_id 序列严格确定。

> `init_milvus.py` 提供建库/建索引/注册；含最小写入示例。embedding 维度在首次运行时由 Embedder 探测，避免写死出错。

---

## 6. 入库流水线（模型可插拔）

四阶段，每段都是接口 + 默认实现，按 config 的 `*_BACKEND` 字段切换（沿用现有 `completion_backend: default|glm|mock` 风格）。

| 阶段 | 接口 | 默认实现 | 可替换为 |
|------|------|----------|----------|
| 解析 Parser | `parse(path)->Doc` | `MinerUParser`（复用 MinerU HTTP） | `LocalParser`（PyMuPDF/python-docx，离线） |
| 切块 Chunker | `chunk(text)->List[Chunk]` | `FixedChunker`（定长吸附，中文感知） | `LangChainSplitter`（RecursiveCharacter） |
| 向量化 Embedder | `embed(list)->Vec` | `CloudEmbedder`（复用 bge-m3） | 任意 OpenAI 兼容 embedding |
| 写入 Writer | `write(kb, chunks, vecs)` | `MilvusWriter`（pymilvus，按 §5 schema） | — |

### 默认切块策略：定长吸附（已确认）
- 按字符数滑动窗口：`chunk_size`（默认 500），步长 = `chunk_size - overlap`（默认 420）。
- "中文感知" = 边界吸附：窗口累积到接近 `chunk_size` 时，在容差带（默认 ±50）内回找最近句末标点（`。！？；\n`）切下；找不到才硬切。计量单位是字符（1 汉字 = 1）。
- **不是逐句判归属**（那属于语义切块，依赖 embedding 聚类，会引入非确定性，与稳定性硬指标冲突）。
- 确定性保证：同一文本永远切成相同的块序列。

### 模型可插拔保证
1. 维度不写死：Embedder 暴露 `dim`，`init_milvus.py` 按当前 Embedder 探测的 `dim` 建库。
2. 模型可追溯：每条向量带 `embedding_model`；collection 属性登记 `embedding_model / dim / chunker`，防"换了一半"。
3. 入口两种：`POST /api/v1/ingest`（文件上传 + 目标 KB，异步任务）和 CLI `python -m app.ingest --kb kb_faq --dir ./docs`。
4. 幂等去重：按 `doc_id + content_hash` 去重，重灌不翻倍。
5. 换 LLM/Rerank：由 config 驱动（`CLOUD_COMPLETION_MODEL` 等），无需改代码。

---

## 7. 稳定性机制（核心硬指标）

### A. 检索严格一致（硬保证）
| 来源 | 处理 |
|------|------|
| Embedding 非确定 | bge-m3 同输入同向量（确定性），无需额外处理 |
| Milvus HNSW 同分浮动 | 固定 `ef=128, top_k=10, offset=0, nq=1`；应用层二级排序 `score DESC → pk ASC` 后截断 |
| 查询重写漂移 | 重写节点 `temperature=0`（贪心）→ 重写结果确定 |
| 多库合并顺序漂移 | 并行结果合并后过重排；cross-encoder 无采样，按 `relevance_score DESC → doc_id+segment_id ASC` 二级排序 |

→ 同一问题（同一重写结果）→ 检索片段序列逐条一致，可断言。

### B. 答案语义稳定（尽力而为 + 缓存兜底）
- 生成：`temperature=0, top_p=1.0, max_tokens` 固定；上下文按 A 的确定顺序拼接、固定截断规则（每段上限 + 总段数上限）；系统 prompt 固定。
- Qwen3-32B 贪心解码近乎确定，偶发浮点级字面差异 → 符合"语义稳定"口径。
- **`DeterministicCache`（默认开）：** key = `hash(rewritten_query + 命中 sources 序列 + 模型)`，命中 → 回放缓存答案，逐字一致 + 零延迟。重灌库/换模型时缓存自动失效（key 含 sources 与模型）。

### C. 把口径做成自动测试（防回归）
`tests/stability/`：同一问题跑 N=10 次——断言检索序列完全一致；断言生成答案两两 embedding 余弦 ≥ `STABILITY_SEMANTIC_THRESHOLD`(0.98)。

### D. 参数全进 config
所有稳定性参数 `STABILITY_*` 前缀，默认值即稳定档，可环境变量覆盖。

---

## 8. LangGraph 状态机与数据流

### 状态 `ChatState`（TypedDict）
`session_id, question, kb_names, history, rewritten_query, retrieved[], sources[], context_text, answer, cache_key, cache_hit, error`

### 图（线性 + 流式输出）
```
load_history ─▶ rewrite ─▶ retrieve ─▶ rerank ─▶ generate ─▶ save_history
   (Redis)     (LLM t=0)  (bge-m3 +   (重排+   (缓存/LLM t=0,    (Redis)
                          并行Milvus)  确定排序)  SSE流式token)
```

| 节点 | 行为 | 稳定性 |
|------|------|--------|
| `load_history` | 从 Redis 读最近 N 条消息（默认 6）作为上下文窗口；首轮为空 | 确定 |
| `rewrite` | 有历史才调 LLM(`t=0`)重写成独立问题；无历史透传原问题（省一次调用、更稳） | 确定 |
| `retrieve` | bge-m3 编码 rewritten_query → `asyncio.gather` 并行查各 collection（`asyncio.to_thread` 包同步客户端）→ 每库取 `top_k_per_kb` | 确定 |
| `rerank` | 合并去重 → CloudRerankService → `score DESC → doc_id+segment_id ASC` 截 `top_n`(默认5) | 确定（序列可断言） |
| `generate` | 查 `DeterministicCache`；未命中则按确定顺序拼 `context_text` + 系统 prompt + 历史 → Qwen3-32B(`t=0`)流式生成；结束后写缓存 | 命中逐字一致 / 未命中语义稳定 |
| `save_history` | (原问题, 答案) 追加到 Redis | — |

### SSE 事件流
```
event: rewrite   data: {rewritten_query}
event: sources   data: {[{doc_name, score, source}]}
event: token     data: {text}                 // 仅未命中缓存时
event: done      data: {answer, sources, cache_hit}
event: error     data: {msg}
```
命中缓存时直接发 `done`（无 token 事件）。

生命周期：图编译一次常驻；每请求独立 state，跨请求无共享内存状态——全部状态落 Redis，进程可多副本水平扩展。

---

## 9. 部署：本地 GPU Milvus

`docker/docker-compose-gpu.yml`：etcd + minio + milvus-standalone(GPU)。

```yaml
services:
  etcd:        # quay.io/coreos/etcd  (元数据)
  minio:       # minio/minio          (对象存储)
  milvus-standalone:
    image: milvusdb/milvus:2.4.x-gpu-latest   # 规划阶段核对并固定稳定 tag
    command: ["milvus","run","standalone"]
    environment: { ETCD_ENDPOINTS, MINIO_ADDRESS }
    ports: ["19530:19530","9092:9091"]        # gRPC; metrics 映射 9092
    deploy:
      resources:
        reservations:
          devices:
            - { driver: nvidia, capabilities: [gpu] }
```

- 前提：宿主机 NVIDIA GPU 驱动（已具备，CUDA 11.8）+ **NVIDIA Container Toolkit**（待安装，`docker/verify_gpu.sh` 探测并提示）。
- `pymilvus` 连 `MILVUS_HOST=localhost MILVUS_PORT=19530`。
- GPU 主要加速建索引与大库检索；小库兼容 CPU 回退，不影响稳定性口径。
- 规划阶段会核对当前 Milvus GPU 镜像 tag 与 CUDA 兼容矩阵，固定一个版本。

---

## 10. 配置（`app/config.py` / `.env` 扩展）

```
# Milvus
MILVUS_HOST=localhost  MILVUS_PORT=19530  MILVUS_DB=default
MILVUS_COLLECTION_PREFIX=kb_  MILVUS_METRIC=COSINE
MILVUS_TOP_K_PER_KB=10  MILVUS_EF=128

# Redis (会话 + 缓存)
REDIS_URL=redis://localhost:6379/0

# 入库（模型可插拔）
PARSER_BACKEND=mineru            # mineru | local
CHUNKER_BACKEND=fixed            # fixed | recursive
EMBEDDING_BACKEND=cloud          # cloud (bge-m3)
CHUNK_SIZE=500  CHUNK_OVERLAP=80  CHUNK_TOLERANCE=50

# 生成 / 重写 / 稳定性
GEN_TEMPERATURE=0.0  GEN_TOP_P=1.0  GEN_MAX_TOKENS=1024
GEN_TOP_N_CONTEXT=5  GEN_CONTEXT_CHAR_PER_SEG=800  GEN_CONTEXT_TOTAL_CHARS=4000  GEN_HISTORY_TURNS=3
#   GEN_HISTORY_TURNS=3 表示生成时拼入最近 3 轮（=6 条消息）对话，与下方 SESSION_HISTORY_MESSAGES=6 对齐
REWRITE_TEMPERATURE=0.0
SESSION_HISTORY_MESSAGES=6  SESSION_TTL=86400   # 存/读窗口 6 条消息 = 3 轮
STABILITY_CACHE_ENABLED=true  STABILITY_CACHE_TTL=604800  STABILITY_SEMANTIC_THRESHOLD=0.98
```

所有影响检索/生成结果的参数集中于此，默认值即稳定档；换模型只改 `*_BACKEND`/`*_MODEL`，代码不动。

---

## 11. 错误处理 / 可观测性 / 测试

### 错误处理（分层降级，不裸抛）
| 场景 | 处理 |
|------|------|
| 单个云服务调用失败 | 复用 `*_with_retry`（指数退避），终态失败 → 记日志 + SSE `error` 并关闭流 |
| 重写节点失败 | 降级：用原问题继续 |
| 检索某 collection 失败 | 跳过该库，其余照常合并；日志标记 |
| 检索全部为空 | 仍进入生成，给"知识库无相关内容"兜底提示（不幻觉） |
| 生成失败 | SSE `error`；已检索 `sources` 照样回发 |
| Redis 不可用 | 会话失败→本次按首轮处理；缓存失败→跳过缓存直生成 |
| 整体超时 | FastAPI 请求级 `timeout`，触发后取消图执行 |

### 可观测性
- `structlog`（config 已有）输出 JSON，每请求带 `session_id / request_id`；每节点记录耗时、检索命中 `doc_id+segment_id` 序列、`cache_hit`、生成参数。
- `/metrics`（Prometheus，config 已有开关）：节点耗时、缓存命中率、检索条数、错误率。
- **检索序列写进日志**，便于出问题回放对比一致性。

### 测试金字塔
1. **单元**：每节点/服务用现有 `Mock*` 测；`FixedChunker` 确定性测试（同文本→同切片序列）。
2. **稳定性测试（核心）** `tests/stability/`：见 §7-C。
3. **集成**：`docker compose-gpu` 起本地 Milvus + 固定 fixture 小库，端到端跑 `/chat`，含多轮追问（验证重写）与命中缓存（验证逐字一致）。
4. **契约测试**：`init_milvus.py` 建库与检索端读取的 schema 断言一致，防两侧 schema 漂移。

---

## 12. 验收口径

1. 同一问题对同一库连续提问 10 次：检索 `doc_id+segment_id` 序列**逐条完全一致**（自动测试断言）。
2. 同一问题生成答案两两 embedding 余弦 ≥ 0.98（自动测试断言）。
3. 命中确定性缓存时：答案逐字一致。
4. 多轮追问经重写后检索到正确独立语义；首轮无历史跳过重写。
5. 换 embedding 模型：改 config → 重建库 → 重灌 → 问答侧自动跟随，无需改代码。
6. 文件→入库端到端可用（HTTP + CLI 两种入口）。

---

## 13. 风险与未决

| 风险 | 缓解 |
|------|------|
| Qwen3-32B 后端 `t=0` 仍有极小字面浮动 | 已用 `DeterministicCache` 兜底；口径定为"语义稳定"而非逐字 |
| Milvus GPU 镜像 tag / CUDA 兼容 | 规划阶段核对固定版本；`verify_gpu.sh` 前置探测 |
| 大库下 HNSW 并行检索偶发同分 | 应用层 `score DESC → pk ASC` 二级排序兜底 |
| 并行查多 collection 耗时 | `asyncio.gather` 并行 + 每库 top_k 限 10；必要时按库分片 |
| 宿主机缺 NVIDIA Container Toolkit | 部署步骤含安装指引，`verify_gpu.sh` 探测提示 |

- 未决：`kb_names` 是否需要在请求中强制指定（目前可选，缺省全查）—— 暂定可选，视压测结果再收紧。
