# 结构感知切块（Structural Chunking）设计

- 状态：草案（待评审）
- 日期：2026-06-25
- 关联：主 spec `2026-06-25-langgraph-multikb-customer-service-design.md` §6（入库流水线 / 切块策略）；kbmap 真实数据调研结论
- 适用：`app/ingest/chunker.py`（主 spec 待实现）

---

## 1. 背景：现有数据告诉我们什么

对 `merged/` 真实文件的调研（规划政策文件/科研管理制度/操作指引 三类采样）：

| 指标 | 规划政策 | 制度 | 操作指引 |
|------|---------|------|---------|
| chunk 边界总数 | 70 | 73 | 92 |
| 语义边界（下块以标题/编号/条款开头 **且** 上块以句末终止）| **0%** | **0%** | **0%** |
| 截断边界（上块未句末终止 + 下块非结构开头）| 65% | 72% | 72% |
| 模糊边界 | 34% | 27% | 27% |

**结论 1：coreagent 的 chunk 边界绝大多数是按字数截断，不是语义切分。** 少数看似语义的边界，也因 chunk 开头夹带版式噪声（`<图片内容|start>...`）而不在真正的结构起始位置。

**结论 2：chunk 正文里混着版式噪声** —— `<图片内容|start>…<图片内容|end>` 块、`<img src="…长hashURL…"/>` 标签。这些 URL 的 hash 会主导向量，污染 embedding。

**结论 3：大小方差极大** —— 中位 655 字（尚可），但有 33 个 <100 字碎片（目录/页眉残留）和 8 个 >1500 字巨块（max 9222）。

## 2. 为什么 FixedChunker 是降级

主 spec §6 默认 `FixedChunker(chunk_size=500, overlap=80)`：
- 在 coreagent **已经按 ~655 字切过**的基础上再硬切 500 字 → 把本可保持完整的一段话再切一刀，**截断问题加倍**。
- 丢掉原 `chunk_id` → 跨系统追溯断裂，重灌无法按原 id 幂等。
- 仍然把图片 URL 混进 chunk 文本。

固定切块在这里没有任何优势，只有劣势。

## 3. 设计目标

1. **段落连贯**：一个 chunk 不应在一句话中间断开；同一观点/条款尽量在一个 chunk 里。
2. **结构对齐**：尽可能在标题/章节/条款边界切，而非任意字数。
3. **噪声剥离**：图片占位、URL、版式标记不进 embedding 文本。
4. **确定性与幂等**：同输入→同 chunk 序列→同 segment_id（重灌不翻倍，对齐主 spec §6 的 `doc_id + content_hash` 幂等）。
5. **可追溯**：保留贡献每个 chunk 的原 coreagent `chunk_id` 集合。
6. **per-KB 差异**：`kb_project` 维持一项目一 chunk（jsonl），其他 KB 走本设计。

## 4. 总体管线（6 阶段）

```
原始 md（含 <!-- chunk_idx chunk_id --> 标记 + <图片内容> 噪声）
   │
   ▼ ① 解析：按标记拆成 raw chunks，记下每个的 chunk_id
[raw chunks: (chunk_id, dirty_text)]
   │
   ▼ ② 清洗：剥噪声（图片块/img/纯URL/归一化空白）
[cleaned spans: (chunk_id, clean_text)]
   │
   ▼ ③ 重建：把所有 span 按顺序拼回成完整 clean 文本；coreagent 边界不再作切点
   │        只保留 chunk_id → 文本区间的映射（供追溯）
[full_text, span_index]
   │
   ▼ ④ 结构切分：在标题/章节/条款/段落边界切，产出 logical sections
[logical sections]
   │
   ▼ ⑤ 字数护栏：>MAX 按→段落→句子再切；<MIN 并入邻居
[final chunks]
   │
   ▼ ⑥ 派 id：segment_id = 确定性 hash(doc_id + 序号 + 文本前缀)
[Chunk(segment_id, text, source_chunk_ids, heading, ...)]
```

**核心思想**：**不信任 coreagent 的边界**（多为截断），改由**文档自身的结构**重新决定边界；coreagent 的 chunk_id 只作为 source 追溯保留。

## 5. 各阶段细则

### ① 解析 raw chunks
- 按 `<!--\s*chunk_idx=(\d+)\s+chunk_id=([0-9a-f]+)` 拆分。
- 每个 raw chunk = 标记行之后、下一标记行之前的文本，附 `(idx, chunk_id)`。
- 无标记的文件（少数）：整文件作单个 raw chunk，chunk_id 退化为 `doc_id+0`。

### ② 清洗规则（逐 chunk 应用）
按顺序剥除：
1. `<图片内容|start>.*?<图片内容|end>`（整块，DOTALL；含其中的 `<img>`）。
2. 残留的独立 `<img\s[^>]*?/>` 标签。
3. coreagent 注入的元数据表（仅当位于 chunk 开头且表头为 `|字段|值|` 形态 —— 这是导出注入，非文档内容）。**文档自带的表格保留**。
4. 连续空行归一为单个段落分隔（`\n{3,}` → `\n\n`）；行内空白归一。
5. 去首尾空白。

**保守原则**：拿不准是不是噪声的（如普通表格），一律保留。宁可多留不要误删文档内容。

### ③ 重建完整文本
- `full_text = "\n".join(span.clean_text for span in spans)`
- 建立 `span_index`：记录每个原 chunk_id 在 `full_text` 中的 `[start, end)` 区间。后续 final chunk 落在 [a,b) → 其 `source_chunk_ids` = 所有与 [a,b) 相交的 span 的 chunk_id。

### ④ 结构切分（核心）
在 `full_text` 中识别**硬边界**并据此切：

| 边界类型 | 识别（行首） | 说明 |
|---------|-------------|------|
| Markdown 标题 | `^#{1,6}\s` | 文档层级 |
| 中文编号章 | `^[一二三四五六七八九十]+、` | "一、""二、" |
| 中文编号项 | `^（[一二三四五六七八九十]）` | "（一）""（二）" |
| 阿拉伯编号 | `^\d+[、.]` | "1.""2、" |
| 法条 | `^第[一二三四五六七八九十百千\d]+条` | 制度/法规 |
| 附件/附表 | `^附件|^附表` | 附件清单 |

算法：
1. 找出所有硬边界位置 → 切成 `logical_sections`。
2. 文档开头到第一个硬边界也是一段（抬头/发文单位/文号）。

### ⑤ 字数护栏
阈值（进 `app/config.py`，可调）：
- `chunk_target_max: int = 1500`（硬上限）
- `chunk_target_min: int = 120`（软下限）
- `chunk_target_ideal: int = 700`（参考目标，不强制）

对每个 `logical_section`：
- 若 `len <= MAX`：作为候选 chunk。
- 若 `len > MAX`：按**段落**（`\n\n`）切；段落仍 > MAX 再按**句子**（`[。！？；]`）切；句子仍 > MAX 按 MAX 硬回车切（最后兜底，极少）。
- 切完后逐个候选 chunk 扫一遍：若 `< MIN` 且有邻居，并入邻居（优先并入上一段，保上下文）。

**kb_project 例外**：不进 ④⑤，走 jsonl 路径（一项目一 chunk），见 §7。

### ⑥ segment_id 派生
```
segment_id = sha256(f"{doc_id}|{ordinal}|{text[:64]}").hexdigest()[:16]
```
- `doc_id`：manifest 派生（path 哈希，主 spec §6）。
- `ordinal`：该 chunk 在本 doc 的序号（0-based）。
- `text[:64]`：前缀防顺序碰撞。
- **确定性**：同 doc 同文本 → 同 segment_id → 重灌幂等（命中既有 segment 即跳过/覆盖，不翻倍）。

## 6. 接口（主 spec §6 集成）

```python
# app/ingest/chunker.py
from dataclasses import dataclass

@dataclass
class Chunk:
    segment_id: str              # 确定性 id（见 §5-⑥）
    text: str                    # 清洗+切分后文本（embedding 输入）
    doc_id: str
    doc_name: str
    kb: str
    heading: str | None          # 所属最近标题（给 LLM 上下文 grounding）
    source_chunk_ids: list[str]  # 贡献此 chunk 的原 coreagent chunk_id（追溯）
    ordinal: int                 # doc 内序号

def chunk_document(
    doc_id: str, doc_name: str, kb: str, raw_text: str
) -> list[Chunk]:
    """主入口。根据 kb 选策略：kb_project→jsonl；其他→结构感知（本设计）。"""
    ...
```

`app/config.py` 新增：
```python
chunker_backend: str = "structural"   # 原 "fixed" 默认改为 "structural"；fixed 保留作回退
chunk_target_max: int = 1500
chunk_target_min: int = 120
```

主 spec §6 入库流水线：`chunker_backend="structural"` 时调 `chunk_document`；`="fixed"` 时保留原 FixedChunker（回退/对比用）。

## 7. per-KB 策略

| KB | 切块策略 | embedding 输入 |
|----|---------|---------------|
| kb_policy_national / kb_policy_group | 结构感知（本设计） | 清洗后 chunk 文本 |
| kb_regulation | 结构感知（法条边界优先） | 同上 |
| kb_ops | 结构感知（步骤/章节边界） | 同上 |
| kb_template | 结构感知 | 同上 |
| kb_project | **jsonl**：一项目一 chunk | 自然语言序列化（kbmap project_serializer） |

`kb_project` 完全旁路 ④⑤，直接每行 JSON → 一个 Chunk（chunk_id 用原 jsonl 行号派生）。

## 8. 确定性与幂等

- 全流程无随机：清洗是正则替换，切分是结构匹配，id 是 hash。同输入→同输出。
- 重灌幂等：`segment_id` 命中既有 → 覆盖（content_hash 不变则跳过）。对齐主 spec §6。
- 与 bge-m3 的确定性呼应：embedding 确定输入 → 确定向量。

## 9. 边界情况

| 情况 | 处理 |
|------|------|
| 文件无 chunk 标记 | 整文件作单 raw chunk，仍走清洗+结构切分 |
| 全文无任何硬边界（纯散文长文） | ④ 切不出 section → 整篇作一段，⑤ 按段落→句子切到 MAX 内 |
| 单句 > MAX（罕见） | 硬回车切，末段标记 `truncated=true` |
| 清洗后 chunk 为空 | 丢弃（不产出空 segment） |
| 表格内容（文档自带） | 保留；不识别为硬边界；若超 MAX 按行切 |
| 重建后单 chunk `< MIN` 且无邻居（孤段） | 保留（不强行合并到虚无） |

## 10. 测试策略

fixture（`tests/fixtures/ingest/`）覆盖：
1. **基础结构**：含 `#` 标题 + `一、二、` 章节的 doc → 断言按章节切。
2. **截断愈合**：两段 prose 被 coreagent 在句中切断 → 重建后按段落切，不断在句中。
3. **噪声剥离**：含 `<图片内容>` + `<img/>` → 清洗后文本不含这些；且不含 URL hash。
4. **大段切分**：一个 3000 字无结构段落 → 按 `。` 切到 <=1500。
5. **小段合并**：连续两个 50 字碎片 → 合并成一个 >=120 的 chunk。
6. **幂等**：同 doc 跑两次 → segment_id 集合完全一致。
7. **kb_project**：jsonl 文件 → 一项目一 chunk，序列化文本含项目字段。
8. **追溯**：final chunk 的 source_chunk_ids 覆盖其文本区间对应的所有原 chunk_id。

## 11. 与主 spec 的衔接

- **§6 切块策略**：`chunker_backend` 默认从 `"fixed"` 改 `"structural"`；`FixedChunker` 保留作 `"fixed"` 回退。
- **§6 幂等**：`segment_id` 纳入 `doc_id + content_hash` 幂等去重的 segment 维度（原 spec 是 doc 维度；本设计把 segment 也做成确定性 id，重灌可按 segment 精确覆盖）。
- **§5 Milvus schema**：segment 字段加 `source_chunk_ids`（array）和 `heading`（varchar），便于回显追溯。
- **不影响**：§2/§7/§8（检索/稳定性/问答图）零改动。

## 12. 待裁决 / 开放

| 项 | 默认 | 触发重评 |
|----|------|---------|
| `chunk_target_max=1500` | 取原数据 p95 附近 | 若召回粒度过粗，降到 1000 |
| `chunk_target_min=120` | 去碎片 | 若丢信息，降到 80 |
| 元数据表剥离只剥 `|字段|值|` 头 | 保守 | 若发现其他注入模式，扩规则 |
| 是否给 chunk 追加上文窗口（前后各 1 句） | 暂不加（YAGNI） | 若 LLM 答案缺上下文，再加 |
| kb_project 是否也过清洗 | 不过（已是结构化 JSON） | 若 JSON 里也混入噪声再说 |

## 13. 后续

本设计获批后，进入实施计划（writing-plans）。实施位置：`app/ingest/chunker.py` + 测试 + `app/config.py` 字段 + 主 spec §5/§6 文案更新。`kb_project` 序列化复用 kbmap 的 `project_serializer.py`。
