# 交接文档（迁移到新机器）

- 日期：2026-06-25
- 仓库：`git@github.com:saladjay/temp-rag.git`，分支 `main`
- 本文档目的：在新机器上把这个项目跑起来。

---

## 1. 项目是什么

基于 **LangGraph + 多知识库 RAG** 的多轮对话智能客服系统，外加两个离线工具链：

| 组件 | 位置 | 作用 |
|------|------|------|
| 主客服服务 | `app/{config,services,ingest,graph,store,api,main}.py` | FastAPI + SSE 流式问答；LangGraph 6 节点（load_history→rewrite→retrieve→rerank→generate→save_history）；Milvus 向量库 + Redis 会话/缓存 |
| **kbmap**（知识库分类工具链）| `app/kbmap/` | 离线把混乱的知识库目录清洗归类成 6 个 KB，embedding 验证分类质量，产出 manifest + 分类器 |
| **结构感知 chunker** | `app/ingest/chunker.py` | 入库切块：清洗 coreagent 噪声 → 按文档结构（标题/章节/条）切 + 字数护栏 + 确定性 segment_id |

设计/计划文档在 `docs/superpowers/{specs,plans}/`（含主客服、kbmap、chunker 三套 spec+plan）。

## 2. 仓库 & 分支

- 远程：`git@github.com:saladjay/temp-rag.git`（`origin`）
- 主分支：`main`（**唯一 source of truth**，所有成果已合并到此）
- 历史 worktree 分支（`worktree-kbmap` / `gen-deepseek` / `kbmap-reconcile`）：内容已全部并入 main，可清理。

```bash
git clone git@github.com:saladjay/temp-rag.git
cd temp-rag
```

## 3. 新机器环境搭建

### 3.1 Python（uv 管理）
- Python **3.12**（项目用 3.12.12 验证）。
- 用 [uv](https://github.com/astral-sh/uv) 建 venv：
  ```bash
  uv venv .venv --python 3.12
  # 激活：Windows `.venv\Scripts\activate`；bash `source .venv/bin/activate`
  ```

### 3.2 依赖
```bash
uv pip install -r requirements.txt   # 或 pip install -r requirements.txt
```
关键依赖：fastapi、uvicorn、sse-starlette、langgraph、langchain-core、pymilvus、redis、pydantic v2、structlog、numpy、pyyaml。

### 3.3 外部服务
- **Milvus**（向量库，GPU）：原机器用 docker-compose 起 GPU Milvus（`docker/` 下，**注意**：docker-compose 文件在原机器是未提交的 WIP，未进 git —— 见 §8 末尾，需从原机器拷过来或重做）。
- **Redis**（会话 + 稳定性缓存）：本地起一个即可。
- **云端模型服务**（原机器内网，新机器要能访问）：
  - embedding：bge-m3（`CLOUD_EMBEDDING_URL`，见 .env）。
  - completion：**deepseek_v4**（`http://128.23.74.3:9091/AIAPLLM/chat/max/v1/chat/completions`，OpenAI 兼容 chat-completions + Basic auth）。
  - rerank：`CLOUD_RERANK_URL`。
  - 文档解析：MinerU（`MINERU_URL`）。

### 3.4 `.env`（**关键 —— gitignored，不会随 clone 过来**）
`.env` 不在 git 里（含密钥）。新机器必须**从原机器拷贝 `.env`**，或按下表重建。键（值在原机器 `.env` 里）：

| 键 | 用途 |
|----|------|
| `CLOUD_EMBEDDING_URL` / `CLOUD_EMBEDDING_MODEL=bge-m3` / `CLOUD_EMBEDDING_TIMEOUT` | 嵌入服务 |
| `CLOUD_AUTH_TOKEN` | 云端服务通用 Basic auth token |
| `CLOUD_COMPLETION_URL=http://128.23.74.3:9091/AIAPLLM/chat/max/v1/chat/completions` / `CLOUD_COMPLETION_MODEL=deepseek_v4` | 生成模型（deepseek_v4）|
| `CLOUD_RERANK_URL` / `CLOUD_RERANK_MODEL=embed_rerank` | 重排序 |
| `MINERU_URL` / `MINERU_AUTH_TOKEN` | 文档解析 |
| `REDIS_URL=redis://localhost:6379/0` | Redis |
| `MILVUS_HOST=localhost` / `MILVUS_PORT=19530` | Milvus |

> **注意（deepseek_v4 接线）**：该端点是 OpenAI 兼容 chat-completions，要求 `messages` 体。生成节点必须调 `CloudCompletionService.chat(messages=[...])`，**不能用 `.complete(prompt=...)`**（发 `prompt` 字段会被端点拒绝 "error parsing the body"）。

## 4. 代码结构

```
app/
├── config.py            # 全部配置（pydantic-settings），全局 settings
├── main.py              # FastAPI app 工厂
├── services/            # 复用的云端服务（embedding/rerank/completion/mineru）
├── ingest/              # 入库流水线
│   ├── chunker.py       # ★ 结构感知切块（chunk_document + StructuralChunker）
│   ├── pipeline.py      # 入库编排 + CLI（run_ingest）
│   ├── interfaces.py    # Chunk dataclass（8字段）+ Protocol
│   ├── embedder/parser/writer.py
├── kbmap/               # ★ 知识库分类工具链（离线）
│   ├── scanner/manifest/project_serializer/embed/metrics/centroids/classifier
│   └── __main__.py      # CLI（scan/verify/build-centroids/classify/assign）
├── graph/               # LangGraph 状态机 + 节点
├── store/               # Milvus 存储 + Redis 会话/缓存
├── api/                 # HTTP schemas + routes
└── utils/log.py
tests/{unit,integration,stability,fixtures,kbmap,ingest}/
docs/                    # 本文档 + kbmap 真实数据产物 + spec/plan
```

## 5. 关键配置（`app/config.py` 的默认值）

```python
chunker_backend = "structural"     # 结构感知切块（默认）
chunk_target_max = 1500            # chunk 硬上限
chunk_target_min = 200             # chunk 软下限
chunk_whole_doc_max = 800          # 清洗后整篇 ≤ 此值则不切分（整篇一个 chunk）
cloud_completion_model = "deepseek_v4"
gen_temperature = 0.0              # 稳定性：全链路 temperature=0
milvus_metric = "COSINE"
```

## 6. 常用命令

```bash
# 跑客服服务
uvicorn app.main:app --reload    # 或 app/main.py 里 create_app()

# 入库（按 KB 目录灌 Milvus）
python -m app.ingest.pipeline --kb kb_faq --dir ./docs/某库

# kbmap 工具链（离线分类）
python -m app.kbmap scan --root <merged目录> --out docs/kb_manifest.yaml --draft docs/kb_taxonomy_draft.md
python -m app.kbmap verify --manifest docs/kb_manifest.yaml --root <merged目录> --out docs/kb_verify_report.md
python -m app.kbmap build-centroids --manifest docs/kb_manifest.yaml --root <merged目录> --out-dir docs/kb_centroids
python -m app.kbmap classify "问题" --centroids-dir docs/kb_centroids
# 测试时用 mock embedder：设环境变量 KBMAP_EMBEDDER=mock

# 测试（迁移时如需验证）
pytest                            # 全量（当前 95 测试）
pytest tests/unit/kbmap/          # 只 kbmap
pytest tests/unit/ingest/         # 只 chunker
```

## 7. 知识库数据

- **真实知识库**（200 文件，10 个原目录）：在原机器 `D:/project/kxx/04需求开发/005其他/download_chunk_from_coreagent/merged/` —— **项目外、不在 git**，新机器要单独拷贝。
- 已产出的分类产物（在 git 里，`docs/`）：
  - `docs/kb_manifest.yaml`：200 文件 → 6 个 KB（kb_policy_national 164 / kb_ops 16 / kb_regulation 12 / kb_policy_group 5 / kb_template 2 / kb_project 1），已按正文哈希去重 18 个，5 个集团级文档 re-tag。
  - `docs/kb_taxonomy_draft.md`、`docs/kb_verify_report.md`：分类草案 + embedding 验证报告。

## 8. 当前状态 & 已知问题

### 已完成且在 main 上
- 主客服 spec 全套（graph/store/api/main/docker）—— 由另一会话实现。
- kbmap 工具链（9 模块 + 33 测试 + 真实数据产物）。
- 结构感知 chunker（替换原 FixedChunker，已接 pipeline/interfaces，24+测试）。
- chunker 真实数据验证后的 4 个修复：①图片块 OCR 文字保留 ②chunk_target_min 120→200 ③短文档整篇化(≤800) ④`_index.md` 目录不切碎。

### 已知遗留（低优先）
1. **chunker 4 个 Minor finding**（记在原 worktree-kbmap 的 `.superpowers/sdd/progress-chunker.md`，gitignored）：`_APPENDIX_RE` 不识别裸"附件"标题、硬切路径首块 heading 丢失、段落切分边界丢 1 个 `\n\n`、`import re` 未在文件顶。
2. **碎片保留决定**：长文档的结构碎片（制度第X条、操作步骤、表单字段）保留不合并 —— 它们是正确检索单元。kb_template 表单字段（~80 个，极短）是唯一残留弱点。
3. **docker-compose 未入 git**：原机器 `docker/docker-compose.yml`、`docker/docker-compose.mirror.override.yml` 是另一会话的未提交 WIP（GPU Milvus 编排 + 镜像加速覆盖）。**新机器起 Milvus 要从原机器拷这两个文件，或重做。**
4. **`kb_policy_group` 质心不稳**（仅 5 文件）：会把"科技创新规划"类查询/政府科技政策误吸为异常 —— 集团文档扩充后自然好转。

### 并行协作历史
本仓库经历过两个 claude 会话并行（一个做主客服 spec，一个做 kbmap/chunker），最后受控调和进 main（见 main 的 merge commit `5e90d7b`）。新机器上单会话操作即可，无并发问题。

## 9. spec / plan 文档索引（`docs/superpowers/`）

| 文档 | 内容 |
|------|------|
| `specs/2026-06-25-langgraph-multikb-customer-service-design.md` | 主客服 spec |
| `plans/2026-06-25-langgraph-multikb-customer-service.md` | 主客服实施计划 |
| `specs/2026-06-25-kb-taxonomy-and-tooling-design.md` | kbmap 设计 |
| `plans/2026-06-25-kb-taxonomy-and-tooling.md` | kbmap 实施计划 |
| `specs/2026-06-25-chunker-structural-design.md` | 结构感知 chunker 设计 |
| `plans/2026-06-25-chunker-structural.md` | chunker 实施计划 |

## 10. 迁移检查清单

- [ ] `git clone` 到新机器。
- [ ] Python 3.12 + venv + `pip install -r requirements.txt`。
- [ ] **从原机器拷 `.env`**（含 CLOUD_*_URL/AUTH_TOKEN、MINERU_* 等）。
- [ ] **从原机器拷 `merged/` 知识库数据**（项目外，200 文件）。
- [ ] **从原机器拷 `docker/docker-compose*.yml`**（GPU Milvus 编排，未入 git），起 Milvus + Redis。
- [ ] 验证云端模型服务可达（embedding / deepseek_v4 completion / rerank）。
- [ ] （可选）`pytest` 跑一遍验证（95 测试应全绿；5 个 Windows 编码 warning 可忽略）。
- [ ] （可选）清理已合并的 worktree 分支：`worktree-kbmap` / `gen-deepseek` / `kbmap-reconcile`。
