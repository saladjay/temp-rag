# 知识库分类体系与离线整理工具链设计

- 状态：草案（待评审）
- 日期：2026-06-25
- 关联文档：`docs/superpowers/specs/2026-06-25-langgraph-multikb-customer-service-design.md`（以下简称「主 spec」）
- 原始知识库路径：`D:/project/kxx/04需求开发/005其他/download_chunk_from_coreagent/merged/`
- 知识库文件夹中\*.md可以理解成markdown化的原文档，包含了原始chunk信息，\*.txt是文本化的原文档。

---

## 1. 目标与非目标

### 目标
1. 把现有 10 个混乱知识库目录（5 个政策类重复、命名带 UUID 前缀）整理成清晰的、可供入库流水线消费的 KB 分类体系。
2. 用 embedding 对整理结果做**客观验证**（类内凝聚度 / 类间分离度 / 异常文件标记），让分类决策可量化、可回溯。
3. 产出一个**可复用的开发工具**（`app/kbmap/`），让开发 agent 能对任意问题或文件快速判断归属 KB（用于调试、评测集构建、测试用例生成）。

### 非目标
- **不进运行时问答链路**。主 spec §2、§7、§8 的"并行检索全库 + 重排合并、不引入路由、保证确定性"决策**原样保留**。本设计产出的分类器只服务于离线整理和开发辅助，不在 `/chat` 请求路径上出现。
- 不替代主 spec 的入库流水线（`app/ingest/`），只产出 manifest 数据供其消费。
- 不重新切块、不重新解析；原始 md 文件的 `<!-- chunk_idx chunk_id -->` 标记原样保留，切块策略由入库流水线决定。

---

## 2. 整体架构与三阶段流程

三阶段管线，每阶段独立产出、可单独审阅。P1 是人类决策，P2/P3 是机器执行。

```
[原始 10 目录: merged/*_<中文名>/]
      │
      ▼  P1: 人工草案（Claude 扫文件起草，用户编辑裁决）
[docs/kb_taxonomy_draft.md]
      │     人类可读：推荐 KB 清单 + 每库 scope + 合并/拆分/去重建议（附证据）+ 含混文件标记
      │
      ▼  P2: 映射 + embedding 验证（草案冻结后跑）
[docs/kb_manifest.yaml]       ← 机读：file → kb 映射（已按 content-hash 去重）
[docs/kb_verify_report.md]    ← 人类可读：凝聚度/分离度/异常文件
      │
      ▼  P3: 开发工具（基于 final manifest 建分类器）
[app/kbmap/]                  ← dev agent 复用的 Python 模块 + CLI
[app/kbmap/data/kb_centroids.npy]
```

**关键原则**：
- P1 人类决策，P2/P3 机器执行。
- embedding 只在 P2/P3 出现，**不进运行时**（呼应非目标）。
- **复用主 spec 的 `CloudEmbeddingService`（bge-m3）**，不引入新模型。
- 每阶段产出物可独立审阅、可回滚。

---

## 3. P1：分类体系草案

### 3.1 草案文档 `docs/kb_taxonomy_draft.md` 结构

```markdown
# KB 分类体系草案（待用户编辑）

## 1. 推荐 KB 清单
| KB 名称 | collection 名 | scope 描述 | 来源目录 | 文件数（去重后） |
|---------|---------------|-----------|----------|------------------|
| 国家级政策库 | kb_policy_national | … | 政策文件+政策二+规划政策文件+政策_会议_讲话汇编 | ~178 |
| 集团级政策库 | kb_policy_group | … | 集团规划（+ 含「集团/粤交集」关键词文件） | ~2–5 |

## 2. 合并/拆分/去重建议（附证据）
- ✅ 合并 `政策文件` + `政策二` → `kb_policy_national`：15 与 14 个 md 中 12 个完全同名（diff 实证）
- ✅ 拆分 `kb_policy` → national / group：按发文层级，`集团规划` 整体归 group
- ⚠️ `规划政策文件` 中含「集团/粤交集」关键词的文件建议移到 `kb_policy_group`（P2 标记）

## 3. 含混文件标记（需用户裁决）
- `2000-01-09_印发《关于加速…》.md`：标题跨"政策"与"制度"，归 ?

## 4. 特殊处理说明
- `集团历史项目-v2`：入 `kb_project`，jsonl 一项目一 chunk，自然语言序列化后 embed（见 §3.3-3）
```

### 3.2 推荐分类（Claude 起草，供用户改）

基于已扫数据（每个目录的实际 md 数，已实证 `政策文件` 与 `政策二` 80% 重复）：

| # | KB | collection | scope | 来源目录 | md 数 |
|---|----|------------|-------|----------|-------|
| 1 | 国家级政策库 | `kb_policy_national` | 国家/部委/省的政策、规划、讲话 | 政策文件+政策二+规划政策文件+政策_会议_讲话汇编 | ~178（去重后） |
| 2 | 集团级政策库 | `kb_policy_group` | 集团内部规划、办法 | 集团规划（+ P2 标记的含「集团/粤交集」关键词文件） | ~2–5 |
| 3 | 平台操作库 | `kb_ops` | 信息平台操作指引、常见问题、功能说明 | 操作指引_常见问题+科小星-操作文档 | ~16 |
| 4 | 科研制度库 | `kb_regulation` | 集团/部门科研管理制度、办法、通知 | 科研管理制度 | ~12 |
| 5 | 表单模板库 | `kb_template` | 可研报告、预算表等申报模板 | 模版 | ~2 |
| 6 | 历史项目库 | `kb_project` | 集团历史科研项目（结构化，jsonl） | 集团历史项目-v2（一项目一 chunk） | ~N 项目 |

### 3.3 关键合并建议（附证据）

1. **5 个政策类目录 → 按发文层级拆成 `kb_policy_national` + `kb_policy_group`**
   - `政策文件`（15 md）与 `政策二`（14 md）：`diff` 实证 12/15 完全同名，差异仅 `_1` 后缀 → 合并去重后归国家级。
   - `规划政策文件`（150 md）：按日期序（2000-01-09 ~ 2025）的国家/部委政策档案 → 归国家级。
   - `政策_会议_讲话汇编`（6 md）：含习近平总书记讲话摘编、国家部委政策汇编 → 归国家级。
   - `集团规划`（2 md）：集团内部发文（粤交集基[2022]404 号等）→ 归集团级。
   - **拆分落地**：默认按来源目录（`集团规划`→group，其余 4 目录→national）；P2 验证会标记国家级目录中含「集团/粤交集/广东省交通集团」关键词的文件，由你裁决是否移到 `kb_policy_group`。
2. **2 个操作类目录 → 合成 `kb_ops`**
   - `操作指引_常见问题`（18 md）与 `科小星-操作文档`（14 md）：均为信息平台操作说明，无清晰边界。
3. **`历史项目-v2` → `kb_project`（jsonl 入向量库）**
   - 内容是 Excel 导出转 md（`项目自定义导出数据 (基础)-v3.xlsx.md`），原文已是 **JSON-per-line** 结构（每行一个项目 JSON，含立项年份 / 项目编号 / 项目名称 / 承担单位 / 项目负责人 / 技术领域 / 主要研究内容 / 预期成果等字段）。
   - **切块**：不走主 spec 的 `FixedChunker`，而是**一个 JSON 对象 = 一个 chunk**（尊重原结构，已是天然切分）。
   - **embedding 输入**：不嵌裸 JSON 字符串（字段名会干扰召回），而是把每条项目**序列化成自然语言描述**再 embed，例如：
     > `2019年立项的【桥梁工程】类项目《大跨径悬索桥缆索系统全寿命周期腐蚀与防护关键技术研究》（编号 JT2019YB15），由广东省公路建设有限公司承担，项目负责人熊锋。主要研究内容：…；预期成果：…`
   - **为何不入 SQL**：客服场景的查询是语义型（"我们做过哪些桥梁项目"、"谁负责过悬索桥研究"），不是聚合分析（"2019 年立项多少个"）；SQL 方案需新增 SQL 基建 + 在运行时加路由判断（违反主 spec"不引入运行时路由"）。聚合分析走原系统 BI 更合适。

### 3.4 开放点裁决结果（已确认）

| 开放点 | 裁决 |
|--------|------|
| `kb_policy` 拆分 | **拆 2 个**：`kb_policy_national`（国家级）+ `kb_policy_group`（集团级），按来源目录默认拆，P2 标记关键词文件给用户微调 |
| `模版` 是否并入 `kb_regulation` | **不合并**，独立成 `kb_template` |
| `历史项目-v2` 是否入库 | **入库**，走 jsonl → `kb_project`，一项目一 chunk，自然语言序列化后 embed |

---

## 4. P2：Manifest 格式 + Embedding 验证

### 4.1 `docs/kb_manifest.yaml`（机读，入库流水线吃这个）

```yaml
version: 1
embedding_model: bge-m3          # 追溯用，换模型时 manifest 整体失效需重跑 P2

kb_defines:                       # KB 元信息
  kb_policy_national:
    description: "国家/部委/省的政策、规划、讲话"
  kb_policy_group:
    description: "集团内部规划、办法"
  kb_ops:
    description: "信息平台操作指引、常见问题、功能说明"
  kb_regulation:
    description: "科研管理制度、办法、通知"
  kb_template:
    description: "申报表单模板"
  kb_project:
    description: "集团历史科研项目（jsonl，一项目一 chunk）"
    chunker: jsonl                       # 例外：不走 FixedChunker
    serializer: project_natural_language # 例外：embed 前序列化为自然语言

files:                            # 文件→KB 映射（已按 content-hash 去重）
  - path: "merged/0352..._政策文件/十四五...doc.md"
    kb: kb_policy_national
    doc_name: "十四五交通领域科技创新规划.doc"
    duplicates:                   # 同 content-hash 被丢弃的文件（保留 provenance）
      - "merged/2022..._政策二/十四五...doc.md"
  - path: "merged/d216..._集团规划/粤交集基[2022]404号...pdf.md"
    kb: kb_policy_group
    doc_name: "粤交集基[2022]404号：关于印发《广东省交通集团科技创新"十四五"发展纲要》的通知.pdf"
  - path: "merged/0969..._操作指引/常见问题解答（一）.pdf.md"
    kb: kb_ops
    doc_name: "广东省交通科技协同创新信息平台常见问题解答（一）.pdf"
  - path: "merged/1aa4..._集团历史项目-v2/项目自定义导出数据 (基础)-v3.xlsx.md"
    kb: kb_project
    doc_name: "项目自定义导出数据 (基础)-v3.xlsx"
    # kb_project 特殊：入库时按 jsonl 拆，一项目一 chunk；embed 前序列化为自然语言
```

**字段约定**：
- `path`：相对于 `merged/` 的相对路径；只列 `.md`（`.txt` 是 fallback，入库时自动找同名 `.md`，缺失再回退 `.txt`）。
- `doc_id`：不在 manifest 写死；入库时按 `path` 哈希自动生成（保证可重现，对齐主 spec §6 的 `doc_id + content_hash` 幂等去重）。
- `duplicates`：保留被去重文件的 provenance，便于回溯；入库流水线只入库主 `path`。
- `kb_defines.<kb>.chunker`：缺省 = 走主 spec 的 `FixedChunker`；`kb_project` 设为 `jsonl`（一项目一 chunk）作为例外。
- 所有入库的 KB 都进 manifest（含 `kb_project`）。明确**不**入库的内容才不进 manifest，并在草案文档 §4 单独说明。

### 4.2 验证报告 `docs/kb_verify_report.md`

三个核心指标，全部基于 bge-m3 embedding（每文件取标题 + 首个 chunk 的前 500 字嵌入，不嵌整篇以省时且稳定）：

| 指标 | 计算法 | 阈值参考 |
|------|--------|----------|
| **类内凝聚度** | 每文件 embedding 与其 KB 质心（KB 内文件 embedding 平均向量）的余弦，取 KB 内均值 | > 0.5 为合格；< 0.5 该 KB 可能太杂，考虑拆 |
| **类间分离度** | KB 质心两两余弦矩阵 | < 0.85 为区分良好；> 0.85 的两个 KB 考虑合并 |
| **异常文件** | 每文件找最近质心；若不是其分配的 KB、或 margin（到本库质心余弦 − 到最近他库质心余弦）< 0.05 | 标 ⚠️ 给用户裁决 |

**报告形态**（示例片段）：

```markdown
## 总览
| KB | 文件数 | 类内凝聚度 |
|----|--------|------------|
| kb_policy_national | 178 | 0.71 |
| kb_policy_group | 3 | 0.74 |
| kb_ops | 16 | 0.83 |
| kb_regulation | 12 | 0.79 |
| kb_template | 2 | 0.66 |
| kb_project | N | 0.69 |

## 类间分离度矩阵（KB 质心两两余弦）
| | kb_policy_national | kb_policy_group | kb_ops | kb_regulation | kb_template | kb_project |
|---|---|---|---|---|---|---|
| kb_policy_national | 1.00 | 0.62 | 0.34 | 0.52 | 0.48 | 0.41 |
| ... |

## 异常文件（margin < 0.05，按置信度升序）
| 文件 | 当前归属 | 最近他库 | margin | 建议 |
|------|----------|----------|--------|------|
| 粤交集基[2022]404号...pdf.md | kb_policy_national | kb_policy_group | −0.02 | ⚠️ 建议改归 kb_policy_group |
```

### 4.3 验证脚本行为

- 读 `kb_manifest.yaml`（**只读**，不改 manifest）。
- 每文件取标题 + 首个 chunk 的前 N 字符（默认 `CHUNK_PREVIEW_CHARS=500`）。
- 复用 `CloudEmbeddingService` 批量 embed（按 `EMBED_BATCH_SIZE=32` 分批）。
- 算三个指标 → 写 `kb_verify_report.md` + 终端打印 top-10 异常文件。
- 异常处理闭环：你看报告 → 改 manifest（`kb` 字段）或改草案 → 重跑验证。这就是"embedding 验证"的具体形式。

---

## 5. P3：开发工具 `app/kbmap/`

### 5.1 模块结构

```
app/kbmap/
├── __init__.py
├── classifier.py      # KBClassifier：classify(text) / assign_file(path)
├── centroids.py       # build_centroids()：从 manifest + 文件算质心
├── verify.py          # run_verify()：P2 验证脚本主逻辑
├── __main__.py        # CLI 入口
└── data/
    ├── kb_centroids.npy    # build-centroids 产出（dim=1024 × KB 数）
    └── kb_names.json       # centroid 行号 → kb_name 映射
```

### 5.2 `KBClassifier` API

```python
class KBClassifier:
    def __init__(self, centroids_path: Path, manifest_path: Path): ...

    def classify(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """对一段文本（如用户问题）返回 top-k 候选 KB 及与其质心的余弦得分。"""

    def assign_file(self, path: str, top_k: int = 3) -> list[tuple[str, float]]:
        """读文件标题 + 首个 chunk 前 500 字，embed 后分类。用于离线给单个文件判归属。"""
```

**确定性保证**：质心是 manifest 冻结后一次性算出并落盘的；classify 仅做一次 embed + 余弦比较，无采样、无随机性 → 同输入同输出。

### 5.3 CLI

```bash
# 给开发 agent 用：判一个问题该归哪个库（调试 / 评测集构建）
python -m app.kbmap classify "如何申报科技项目？"

# 给开发 agent 用：判一个文件该归哪个库（建测试集 / 校验 manifest）
python -m app.kbmap assign "merged/.../xxx.md"

# manifest 改动后重建质心
python -m app.kbmap build-centroids --manifest docs/kb_manifest.yaml

# 跑 P2 验证（生成 kb_verify_report.md）
python -m app.kbmap verify-manifest --manifest docs/kb_manifest.yaml
```

### 5.4 质心构建算法

1. 读 `kb_manifest.yaml` → 得 KB 清单与每 KB 的文件列表。
2. 每文件取标题（文件名去掉 UUID 前缀和 `.md`）+ 首个 `<!-- chunk_idx=0 -->` 之后的前 500 字。
   - **`kb_project` 例外**：文件是 JSONL，取首个 JSON 对象，按 §3.3-3 的自然语言模板序列化后再 embed（与入库时的 embed 输入保持一致）。
3. 复用 `CloudEmbeddingService` 批量 embed。
4. 按 KB 聚合：每 KB 内文件 embedding 取均值 → 质心向量（dim=1024，bge-m3）。
5. 落盘 `kb_centroids.npy`（shape = [KB 数, 1024]）+ `kb_names.json`。

---

## 6. 与主 spec 的集成

### 6.1 不改动（关键）

| 主 spec 章节 | 原决策 | 保持 |
|--------------|--------|------|
| §2 关键决策 | 并行检索全库 + 重排合并，不引入 LLM 路由 | ✅ 不动 |
| §5 Milvus schema | 一个 KB = 一个 collection `kb_<name>` | ✅ 不动（manifest 的 `kb_defines` 即 collection 清单） |
| §7 稳定性机制 | temperature=0 / 固定搜索参数 / 二级排序 / 确定性缓存 | ✅ 不动 |
| §8 LangGraph 图 | load_history → rewrite → retrieve → rerank → generate → save_history | ✅ 不动 |

### 6.2 增量改动（非破坏性）

| 主 spec 章节 | 改动 |
|--------------|------|
| §6 入库流水线 CLI | **新增 `--manifest` 模式**：`python -m app.ingest --manifest docs/kb_manifest.yaml` → 读 manifest，循环每个 KB 调用现有入库逻辑。原 `--kb X --dir Y` 模式保留不变。 |
| §6 KB 清单来源 | 从"硬编码 / 命令行参数"变成"data-driven（来自 manifest）"。collection 命名仍遵循 `kb_<name>` 规则。 |
| §6 切块策略 | 新增**一条例外**（由 manifest `kb_defines.<kb>.chunker` 字段驱动）：`kb_project` 用 `jsonl` 切块器（一项目一 chunk）+ 自然语言序列化；其余 KB 默认仍走 `FixedChunker`。 |

### 6.3 契约约束

- manifest 的 `kb_defines.<name>` 必须是合法的 collection 名（`kb_` 前缀 + 小写字母/数字/下划线），与主 spec §5 一致。
- manifest 的 `embedding_model` 字段必须与入库所用 embedder 一致；不一致时入库流水线报错（防"换了一半"），对齐主 spec §6 的模型可追溯保证。

---

## 7. 文件结构与产物清单

```
langgraph/                                # 项目根
├── app/
│   └── kbmap/                            # P3 产出（新）
│       ├── __init__.py
│       ├── classifier.py
│       ├── centroids.py
│       ├── verify.py
│       ├── __main__.py
│       └── data/
│           ├── kb_centroids.npy
│           └── kb_names.json
├── scripts/
│   └── build_manifest.py                 # P2：从草案 + 目录扫描生成 manifest（含 content-hash 去重）
└── docs/
    ├── kb_taxonomy_draft.md              # P1 产出（人类编辑）
    ├── kb_manifest.yaml                  # P2 产出（机读）
    └── kb_verify_report.md               # P2 产出（验证报告）
```

**产物依赖链**：
```
原始目录 ──P1──▶ kb_taxonomy_draft.md ──P2──▶ kb_manifest.yaml ──P3──▶ kb_centroids.npy
                                                  │
                                                  └──▶ kb_verify_report.md（反馈到草案编辑循环）
```

---

## 8. 开放问题与风险

| 项 | 决定 | 备注 / 触发重评 |
|----|------|------------------|
| 原文件已含 `chunk_id` 标记如何处理 | **其他 KB 不保留原 chunk**，入库走主 spec 的 `FixedChunker` 重新切（确定性优先）；**`kb_project` 例外**，按 jsonl 一项目一 chunk | 若未来需要保留原 chunk_id 追溯，再设计 chunk-preserving 入库模式 |
| `md` vs `txt` | manifest 只列 `.md`；入库时若 `.md` 缺失自动回退同名 `.txt`（`.md` 是 markdown 化原文含 chunk 信息，`.txt` 是纯文本化原文） | — |
| `kb_policy` 拆分 | **已拆**：`kb_policy_national` + `kb_policy_group` | 若 `kb_policy_national` 凝聚度 < 0.5，再按主题细分 |
| `历史项目-v2` | **入 `kb_project`**（jsonl，一项目一 chunk，自然语言序列化后 embed） | 聚合分析需求走原系统 BI，不入 RAG |
| `kb_template` 独立 | **独立**，不并入 `kb_regulation` | 文件少（~2）但语义边界清晰 |
| content-hash 去重的哈希算法 | SHA-256，按整文件 md 内容 | — |
| manifest `version` 字段 | 当前 `1`；未来 schema 变动时递增 | — |

---

## 9. 验收口径

1. **P1**：`kb_taxonomy_draft.md` 含完整 KB 清单、每条合并/去重建议附证据、含混文件明确标 ⚠️；用户已签字裁决开放点。
2. **P2**：
   - `kb_manifest.yaml` 覆盖 `merged/` 下所有入库目录的 md 文件；`kb_project` 的 JSONL 文件也以单条 path 列入（入库时按 jsonl 拆）。
   - content-hash 重复文件只列一条主 path，其余进 `duplicates`。
   - `kb_verify_report.md` 含三个指标 + 异常文件清单；用户已看过并确认无未裁决异常。
3. **P3**：
   - `KBClassifier.classify` 对同一问题多次调用结果完全一致（确定性）。
   - CLI `classify` / `assign` / `build-centroids` / `verify-manifest` 四个子命令可用且有单元测试。
   - 质心文件 `kb_centroids.npy` 与 manifest 的 `embedding_model` 一致。
4. **集成**：主 spec 入库 CLI 的 `--manifest` 模式能用 manifest 把全部 KB 灌入 Milvus，且不破坏原 `--kb X --dir Y` 模式。

---

## 10. 后续步骤

本设计获批后，进入实施计划（writing-plans）阶段。实施顺序按 P1 → P2 → P3 串行：

1. **P1 执行**（人工为主）：Claude 扫描全部文件生成 `kb_taxonomy_draft.md` 初稿 → 用户编辑裁决 → 草案冻结。
2. **P2 实现**：`scripts/build_manifest.py`（去重 + 生成 manifest）+ `app/kbmap/verify.py`（验证脚本）→ 跑出 `kb_verify_report.md` → 用户按报告调 manifest → 重跑直至无异常。
3. **P3 实现**：`app/kbmap/classifier.py` + `centroids.py` + CLI → 单元测试 → 质心构建。
4. **集成**：给主 spec 入库 CLI 加 `--manifest` 模式。
