# 结构感知切块（Structural Chunking）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `app/ingest/chunker.py` —— 清洗 coreagent 噪声、忽略其多为截断的边界、按文档自身结构（标题/章节/条/段落）+ 字数护栏重切，产出确定性的 `Chunk`，供主 spec 入库流水线和 kbmap 复用。

**Architecture:** 单模块 `app/ingest/chunker.py`，6 阶段纯函数管线：解析 raw chunks → 清洗噪声 → 重建完整文本（忽略 coreagent 边界）→ 结构切分（硬边界）→ 字数护栏（大切/小并）→ 派确定性 segment_id。`kb_project` 旁路走 jsonl（复用 kbmap 的 `serialize_project`）。全流程无随机 → 确定性 + 重灌幂等。

**Tech Stack:** Python ≥3.10，标准库 `re`/`hashlib`/`json`/`dataclasses`，复用 `app.config.settings` 与 `app.kbmap.project_serializer.serialize_project`；pytest + pytest-asyncio。

**Spec:** `docs/superpowers/specs/2026-06-25-chunker-structural-design.md`

## Global Constraints

- Python ≥3.10。新代码放 `app/ingest/` 包；复用 `app.config.settings` 与 `app.kbmap.project_serializer.serialize_project`。
- 工作目录：`D:/project/kxx/04需求开发/005其他/langgraph/.claude/worktrees/kbmap`（隔离 worktree）。所有操作只在此目录内，不碰主工作树。
- Python 解释器（路径含中文，必须引号、不可存入 bash 变量）：`D:/project/kxx/04需求开发/005其他/langgraph/.venv/Scripts/python.exe`。跑测试：`"<该路径>" -m pytest <path> -v`。
- **确定性**：所有切块操作必须确定性（正则替换/结构匹配/hash），无随机。同输入→同 chunk 序列→同 segment_id（重灌幂等）。
- 代码注释/文档用中文，标识符用英文。
- TDD：每任务先写失败测试，再最小实现，再提交。
- 不改 kbmap 已有代码（`app/kbmap/`），只新增 `app/ingest/` 和改 `app/config.py`。

## File Structure

```
langgraph/.claude/worktrees/kbmap/
├── app/ingest/                       # 新增包
│   ├── __init__.py
│   └── chunker.py                    # Chunk + 全部管线函数 + chunk_document
├── app/config.py                     # 改：chunker_backend 默认 + chunk_target_max/min
└── tests/unit/ingest/                # 新增
    ├── __init__.py
    └── test_chunker.py
```

`chunker.py` 公开符号（后续任务逐步产出，最终汇总）：
- `Chunk`（dataclass）、`RawChunk`（dataclass）、`SpanInterval`（dataclass）、`Section`（dataclass）
- `assign_segment_id(doc_id, ordinal, text) -> str`
- `parse_raw_chunks(raw_text) -> list[RawChunk]`
- `clean_span(text) -> str`
- `find_hard_boundaries(full_text) -> list[int]`
- `reconstruct(raw_chunks) -> tuple[str, list[SpanInterval]]`
- `split_by_structure(full_text) -> list[Section]`
- `apply_size_guardrails(sections, max_size, min_size) -> list[Section]`
- `chunk_document(doc_id, doc_name, kb, raw_text) -> list[Chunk]`

---

## Task 1: 包骨架 + 配置字段 + Chunk/segment_id

**Files:**
- Create: `app/ingest/__init__.py`, `app/ingest/chunker.py`（仅 Chunk + assign_segment_id）
- Modify: `app/config.py:114-120`（改 chunker_backend 默认 + 加 chunk_target_max/min）
- Test: `tests/unit/ingest/__init__.py`, `tests/unit/ingest/test_chunker.py`

**Interfaces:**
- Consumes: `app.config.settings`
- Produces: `Chunk`（字段：segment_id, text, doc_id, doc_name, kb, heading, source_chunk_ids, ordinal）、`assign_segment_id(doc_id, ordinal, text) -> str`

- [ ] **Step 1: 写失败测试**

`tests/unit/ingest/__init__.py`（空）。`tests/unit/ingest/test_chunker.py`:

```python
from app.config import settings
from app.ingest.chunker import Chunk, assign_segment_id


def test_config_chunker_defaults():
    assert settings.chunker_backend == "structural"
    assert settings.chunk_target_max == 1500
    assert settings.chunk_target_min == 120


def test_assign_segment_id_is_deterministic():
    a = assign_segment_id("doc1", 0, "一段文本内容")
    b = assign_segment_id("doc1", 0, "一段文本内容")
    assert a == b
    assert len(a) == 16


def test_assign_segment_id_differs_by_inputs():
    d = assign_segment_id("doc1", 0, "abc")
    assert assign_segment_id("doc1", 1, "abc") != d   # 序号不同
    assert assign_segment_id("doc2", 0, "abc") != d   # doc_id 不同
    assert assign_segment_id("doc1", 0, "xyz") != d   # 文本不同


def test_chunk_dataclass_fields():
    c = Chunk(segment_id="x", text="t", doc_id="d", doc_name="n",
              kb="kb_policy_national", heading="一、总则",
              source_chunk_ids=["a", "b"], ordinal=0)
    assert c.segment_id == "x"
    assert c.source_chunk_ids == ["a", "b"]
    assert c.heading == "一、总则"
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: 改 `app/config.py`**

把第 114-120 行的"入库（模型可插拔）"段替换为：

```python
    # ========== 入库（模型可插拔） ==========
    parser_backend: str = "mineru"        # mineru | local
    chunker_backend: str = "structural"   # structural（结构感知，默认）| fixed（回退）
    embedding_backend: str = "cloud"      # cloud (bge-m3)
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_tolerance: int = 50
    # 结构感知切块护栏
    chunk_target_max: int = 1500
    chunk_target_min: int = 120
```

- [ ] **Step 4: 建 `app/ingest/__init__.py`（空）**

```python
"""入库流水线。"""
```

- [ ] **Step 5: 建 `app/ingest/chunker.py`（仅 Chunk + assign_segment_id）**

```python
"""结构感知切块（Structural Chunking）。

清洗 coreagent 导出噪声 → 忽略其多为截断的边界 → 按文档自身结构
（标题/章节/条/段落）+ 字数护栏重切 → 派确定性 segment_id。
设计见 docs/superpowers/specs/2026-06-25-chunker-structural-design.md。
"""
import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """切块产物。"""
    segment_id: str              # 确定性 id（重灌幂等）
    text: str                    # 清洗+切分后文本（embedding 输入）
    doc_id: str
    doc_name: str
    kb: str
    heading: str | None          # 所属最近硬边界标题（LLM 上下文 grounding）
    source_chunk_ids: list[str] = field(default_factory=list)  # 贡献此 chunk 的原 coreagent chunk_id
    ordinal: int = 0             # doc 内序号


def assign_segment_id(doc_id: str, ordinal: int, text: str) -> str:
    """确定性 segment id：sha256(doc_id|ordinal|text[:64])[:16]。"""
    raw = f"{doc_id}|{ordinal}|{text[:64]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 6: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（4 passed）

- [ ] **Step 7: 提交**

```bash
git add app/ingest/__init__.py app/ingest/chunker.py app/config.py tests/unit/ingest/__init__.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务1 chunker 骨架+配置字段+Chunk/segment_id"
```

---

## Task 2: parse_raw_chunks（解析 coreagent 标记）

**Files:**
- Modify: `app/ingest/chunker.py`（加 `RawChunk` + `parse_raw_chunks`）
- Test: `tests/unit/ingest/test_chunker.py`（追加）

**Interfaces:**
- Produces: `RawChunk(idx, chunk_id, text)`、`parse_raw_chunks(raw_text) -> list[RawChunk]`

- [ ] **Step 1: 追加失败测试**

在 `tests/unit/ingest/test_chunker.py` 末尾追加：

```python
from app.ingest.chunker import parse_raw_chunks, RawChunk


def test_parse_raw_chunks_splits_by_markers():
    raw = ("<!-- chunk_idx=0 chunk_id=aaa123 -->\n第一段\n"
           "<!-- chunk_idx=1 chunk_id=bbb456 -->\n第二段\n"
           "<!-- chunk_idx=2 chunk_id=ccc789 occurTime=1 -->\n第三段")
    chunks = parse_raw_chunks(raw)
    assert len(chunks) == 3
    assert chunks[0].idx == 0 and chunks[0].chunk_id == "aaa123"
    assert "第一段" in chunks[0].text
    assert chunks[1].idx == 1 and chunks[1].chunk_id == "bbb456"
    assert chunks[2].chunk_id == "ccc789" and "第三段" in chunks[2].text


def test_parse_raw_chunks_no_markers_returns_single_chunk():
    chunks = parse_raw_chunks("没有任何标记的纯文本")
    assert len(chunks) == 1
    assert chunks[0].idx == 0
    assert "纯文本" in chunks[0].text
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_parse_raw_chunks_splits_by_markers -v`
Expected: FAIL — `ImportError: cannot import name 'parse_raw_chunks'`

- [ ] **Step 3: 加 `RawChunk` + `parse_raw_chunks` 到 chunker.py**

在 `Chunk` dataclass 之后、`assign_segment_id` 之前插入：

```python
@dataclass
class RawChunk:
    """coreagent 原始 chunk（标记行之后的未清洗文本）。"""
    idx: int
    chunk_id: str
    text: str


import re

# coreagent chunk 标记：<!-- chunk_idx=N chunk_id=hex ... -->
_RAW_CHUNK_RE = re.compile(r"<!--\s*chunk_idx=(\d+)\s+chunk_id=([0-9a-fA-F]+)[^>]*-->")


def parse_raw_chunks(raw_text: str) -> list[RawChunk]:
    """按 chunk 标记拆分。无标记时整文件作单个 RawChunk（chunk_id='nomarker'）。"""
    matches = list(_RAW_CHUNK_RE.finditer(raw_text))
    if not matches:
        return [RawChunk(idx=0, chunk_id="nomarker", text=raw_text)]
    out: list[RawChunk] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        out.append(RawChunk(idx=int(m.group(1)), chunk_id=m.group(2),
                            text=raw_text[start:end]))
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务2 parse_raw_chunks 解析 coreagent 标记"
```

---

## Task 3: clean_span（剥离版式噪声）

**Files:**
- Modify: `app/ingest/chunker.py`（加 `clean_span` + `_strip_leading_meta_table` + 正则常量）
- Test: `tests/unit/ingest/test_chunker.py`（追加）

**Interfaces:**
- Produces: `clean_span(text) -> str`

- [ ] **Step 1: 追加失败测试**

```python
from app.ingest.chunker import clean_span


def test_clean_span_strips_image_blocks_and_img_tags():
    text = ('前言\n<图片内容|start>\n<img src="https://x/abc123def456.jpeg"/>\n'
            '图片说明\n<图片内容|end>\n正文部分\n<img src="https://y/xyz.png"/>')
    cleaned = clean_span(text)
    assert "<图片内容" not in cleaned
    assert "<img" not in cleaned
    assert "https://" not in cleaned
    assert "前言" in cleaned and "正文部分" in cleaned


def test_clean_span_strips_leading_meta_table_only():
    # 开头的注入元表（|字段|值|）被剥；文档自带的普通表保留
    text = ("|字段|值|\n|---|---|\n|发布日期|2021-11|\n\n正文开始\n"
            "|列1|列2|\n|---|---|\n|a|b|")
    cleaned = clean_span(text)
    assert "字段" not in cleaned      # 注入表头被剥
    assert "发布日期" not in cleaned
    assert "正文开始" in cleaned
    assert "|列1|列2|" in cleaned     # 文档自带表保留
    assert "|a|b|" in cleaned


def test_clean_span_normalizes_blank_lines():
    text = "行1\n\n\n\n\n行2"
    assert "\n\n\n" not in clean_span(text)
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_clean_span_strips_image_blocks_and_img_tags -v`
Expected: FAIL — `ImportError: cannot import name 'clean_span'`

- [ ] **Step 3: 加清洗逻辑到 chunker.py**

在 `parse_raw_chunks` 之后追加：

```python
# 噪声正则
_IMG_BLOCK_RE = re.compile(r"<图片内容\|start>.*?<图片内容\|end>", re.DOTALL)
_IMG_TAG_RE = re.compile(r"<img\s[^>]*?/?>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _strip_leading_meta_table(text: str) -> str:
    """剥离开头的 coreagent 注入元表（|字段|值| 头）。文档自带的表不剥。"""
    t = text.lstrip()
    if not t.startswith("|字段|值|"):
        return text
    lines = t.splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        i += 1
    return "\n".join(lines[i:])


def clean_span(text: str) -> str:
    """剥离版式噪声：图片块、img 标签、开头注入元表、归一化空行。"""
    text = _IMG_BLOCK_RE.sub("", text)
    text = _IMG_TAG_RE.sub("", text)
    text = _strip_leading_meta_table(text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()
```

- [ ] **Step 4: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务3 clean_span 剥离图片/URL/注入元表噪声"
```

---

## Task 4: find_hard_boundaries（结构边界检测）

**Files:**
- Modify: `app/ingest/chunker.py`（加 `find_hard_boundaries` + `_looks_like_heading` + 边界正则）
- Test: `tests/unit/ingest/test_chunker.py`（追加）

**Interfaces:**
- Produces: `find_hard_boundaries(full_text) -> list[int]`（升序位置，不含 0）、`_looks_like_heading(line) -> bool`

- [ ] **Step 1: 追加失败测试**

```python
from app.ingest.chunker import find_hard_boundaries, _looks_like_heading


def test_find_hard_boundaries_detects_all_kinds():
    text = ("抬头内容\n"
            "# 一级标题\n正文A\n"
            "一、第一章\n正文B\n"
            "（二）项\n正文C\n"
            "第3条\n条款内容\n"
            "1. 阿拉伯项\n正文D\n"
            "附件1\n附件内容")
    positions = find_hard_boundaries(text)
    # 6 个硬边界（# 一级标题 / 一、 / （二） / 第3条 / 1. / 附件1）
    assert len(positions) == 6
    assert positions == sorted(positions)
    assert 0 not in positions


def test_find_hard_boundaries_empty_when_no_structure():
    assert find_hard_boundaries("纯散文没有任何标题或编号。") == []


def test_looks_like_heading():
    assert _looks_like_heading("# 标题")
    assert _looks_like_heading("一、总则")
    assert _looks_like_heading("（三）具体要求")
    assert _looks_like_heading("第5条")
    assert _looks_like_heading("2. 子项")
    assert _looks_like_heading("附件1")
    assert not _looks_like_heading("这是普通正文句子。")
    assert not _looks_like_heading("")
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_find_hard_boundaries_detects_all_kinds -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 加边界检测到 chunker.py**

在 `clean_span` 之后追加：

```python
# 硬边界正则（行首）
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}\s", re.MULTILINE)
_NUM_CN_RE = re.compile(r"^[ \t]*[一二三四五六七八九十]+、", re.MULTILINE)
_NUM_CN_PAREN_RE = re.compile(r"^[ \t]*（[一二三四五六七八九十]+）", re.MULTILINE)
_NUM_AR_RE = re.compile(r"^[ \t]*\d+[、.][^0-9]", re.MULTILINE)  # 排除"2021年"这种
_CLAUSE_RE = re.compile(r"^[ \t]*第[一二三四五六七八九十百千\d]+条", re.MULTILINE)
_APPENDIX_RE = re.compile(r"^[ \t]*(附件|附表)", re.MULTILINE)

_BOUNDARY_RES = (_HEADING_RE, _NUM_CN_RE, _NUM_CN_PAREN_RE, _NUM_AR_RE, _CLAUSE_RE, _APPENDIX_RE)


def find_hard_boundaries(full_text: str) -> list[int]:
    """返回硬边界行首 offset（升序、去重、不含 0）。"""
    positions: set[int] = set()
    for rx in _BOUNDARY_RES:
        for m in rx.finditer(full_text):
            positions.add(m.start())
    return sorted(p for p in positions if p > 0)


def _looks_like_heading(line: str) -> bool:
    """该行是否是硬边界起始行（标题/编号/条款/附件）。"""
    stripped = line.lstrip()
    if not stripped:
        return False
    return any(rx.match(stripped) for rx in _BOUNDARY_RES)
```

- [ ] **Step 4: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务4 find_hard_boundaries 结构边界检测"
```

---

## Task 5: reconstruct + split_by_structure

**Files:**
- Modify: `app/ingest/chunker.py`（加 `SpanInterval` + `Section` + `reconstruct` + `split_by_structure`）
- Test: `tests/unit/ingest/test_chunker.py`（追加）

**Interfaces:**
- Produces: `SpanInterval(chunk_id, start, end)`、`Section(heading, text, start)`、`reconstruct(raw_chunks) -> tuple[str, list[SpanInterval]]`、`split_by_structure(full_text) -> list[Section]`

- [ ] **Step 1: 追加失败测试**

```python
from app.ingest.chunker import reconstruct, split_by_structure, RawChunk, SpanInterval, Section


def test_reconstruct_concatens_and_tracks_intervals():
    rcs = [
        RawChunk(0, "id0", "段落甲。"),
        RawChunk(1, "id1", "段落乙。"),  # 与甲是截断连续
        RawChunk(2, "id2", "二、新章节\n内容"),  # 结构边界
    ]
    full, intervals = reconstruct(rcs)
    assert "段落甲" in full and "段落乙" in full and "二、新章节" in full
    assert len(intervals) == 3
    # 区间互不重叠且覆盖（首区间 start=0）
    assert intervals[0].start == 0
    assert intervals[0].chunk_id == "id0"
    assert intervals[0].end <= intervals[1].start
    assert intervals[-1].end == len(full)


def test_reconstruct_drops_empty_cleaned_spans():
    rcs = [RawChunk(0, "id0", "<图片内容|start><图片内容|end>"), RawChunk(1, "id1", "有效文本")]
    full, intervals = reconstruct(rcs)
    assert len(intervals) == 1  # 清洗后空的被丢
    assert intervals[0].chunk_id == "id1"


def test_split_by_structure_cuts_at_hard_boundaries():
    full = "抬头说明\n一、第一章\n正文甲\n二、第二章\n正文乙"
    sections = split_by_structure(full)
    # 抬头 / 一、 / 二、 三段
    assert len(sections) == 3
    assert sections[0].heading is None and "抬头说明" in sections[0].text
    assert sections[1].heading == "一、第一章"
    assert sections[2].heading == "二、第二章"
    # start offset 升序
    starts = [s.start for s in sections]
    assert starts == sorted(starts)
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_reconstruct_concatens_and_tracks_intervals -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 加 reconstruct + split_by_structure 到 chunker.py**

在 `_looks_like_heading` 之后追加：

```python
@dataclass
class SpanInterval:
    """清洗后某原 chunk_id 在 full_text 中的文本区间。"""
    chunk_id: str
    start: int
    end: int


@dataclass
class Section:
    """结构切分产物（未经字数护栏）。"""
    heading: str | None
    text: str
    start: int   # 在 full_text 中的起始 offset


def reconstruct(raw_chunks: list[RawChunk]) -> tuple[str, list[SpanInterval]]:
    """清洗每个 raw chunk，按顺序拼回成完整文本；忽略 coreagent 边界作切点。
    返回 (full_text, span_intervals)，区间用于后续 source_chunk_ids 追溯。"""
    cleaned = [(rc.chunk_id, clean_span(rc.text)) for rc in raw_chunks]
    cleaned = [(cid, t) for cid, t in cleaned if t]  # 丢清洗后为空的
    full = "\n".join(t for _, t in cleaned)
    intervals: list[SpanInterval] = []
    pos = 0
    for cid, t in cleaned:
        intervals.append(SpanInterval(cid, pos, pos + len(t)))
        pos += len(t) + 1  # +1 为 "\n" 分隔符
    return full, intervals


def split_by_structure(full_text: str) -> list[Section]:
    """按硬边界切 full_text 成 sections。每段 heading 取首行（若是边界起始）。"""
    if not full_text:
        return []
    bounds = find_hard_boundaries(full_text)
    cut_points = [0] + bounds + [len(full_text)]
    sections: list[Section] = []
    for i in range(len(cut_points) - 1):
        s, e = cut_points[i], cut_points[i + 1]
        chunk_text = full_text[s:e].strip()
        if not chunk_text:
            continue
        first_line = chunk_text.split("\n", 1)[0].strip()
        heading = first_line if _looks_like_heading(first_line) else None
        sections.append(Section(heading=heading, text=chunk_text, start=s))
    return sections
```

- [ ] **Step 4: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（15 passed）

- [ ] **Step 5: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务5 reconstruct+split_by_structure 重建段落按结构切"
```

---

## Task 6: apply_size_guardrails（大切/小并）

**Files:**
- Modify: `app/ingest/chunker.py`（加 `apply_size_guardrails` + `_split_oversized` + `_merge_undersized`）
- Test: `tests/unit/ingest/test_chunker.py`（追加）

**Interfaces:**
- Produces: `apply_size_guardrails(sections, max_size, min_size) -> list[Section]`

- [ ] **Step 1: 追加失败测试**

```python
from app.ingest.chunker import apply_size_guardrails, Section


def _sec(text, heading=None, start=0):
    return Section(heading=heading, text=text, start=start)


def test_guardrails_split_oversized_by_paragraph():
    long_para = "句一。" * 400  # 1200 字，无段落分隔
    sec = _sec(long_para)
    out = apply_size_guardrails([sec], max_size=500, min_size=50)
    assert all(len(s.text) <= 500 for s in out)
    assert len(out) > 1
    # 不丢内容（拼接后等于原文）
    assert "".join(s.text for s in out) == long_para


def test_guardrails_split_uses_paragraph_boundaries_first():
    text = "段一有够多字。" * 50 + "\n\n" + "段二有够多字。" * 50  # 两个大段
    out = apply_size_guardrails([_sec(text)], max_size=300, min_size=20)
    # 应在段落 \n\n 处优先切
    assert len(out) >= 2
    assert all(len(s.text) <= 300 for s in out)


def test_guardrails_merge_undersized_into_previous():
    secs = [_sec("正常长度的段落，足够超过下限。"), _sec("短。"), _sec("另一正常段落内容。")]
    out = apply_size_guardrails(secs, max_size=1500, min_size=30)
    # "短。" 应被并入前一段，不再单独存在
    assert not any(s.text == "短。" for s in out)
    assert len(out) < len(secs)


def test_guardrails_keeps_solo_short_section():
    # 孤段且短：保留（不强行合并到虚无）
    out = apply_size_guardrails([_sec("孤短。")], max_size=1500, min_size=120)
    assert len(out) == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_guardrails_split_oversized_by_paragraph -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 加护栏逻辑到 chunker.py**

在 `split_by_structure` 之后追加：

```python
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])")


def _split_oversized(sec: Section, max_size: int) -> list[Section]:
    """对超过 max_size 的 section：先按段落，再按句子，最后硬切。保留 heading 在首块。"""
    text = sec.text
    if len(text) <= max_size:
        return [sec]
    # 1) 按段落
    parts = [p for p in text.split("\n\n") if p.strip()]
    if len(parts) == 1:
        # 2) 按句子
        parts = [p for p in _SENT_SPLIT_RE.split(text) if p]
    # 3) 聚合到 max_size
    out: list[Section] = []
    buf = ""
    for p in parts:
        if len(p) > max_size:
            # 单段/单句仍超：硬切
            if buf:
                out.append(Section(sec.heading if not out else None, buf, sec.start)); buf = ""
            for i in range(0, len(p), max_size):
                out.append(Section(None, p[i:i + max_size], sec.start + i))
            continue
        if len(buf) + len(p) + (1 if buf else 0) <= max_size:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                out.append(Section(sec.heading if not out else None, buf, sec.start))
            buf = p
    if buf:
        out.append(Section(sec.heading if not out else None, buf, sec.start))
    return out if out else [sec]


def _merge_undersized(sections: list[Section], min_size: int) -> list[Section]:
    """小于 min_size 的 section 并入前一个（首段孤立则保留）。"""
    out: list[Section] = []
    for sec in sections:
        if out and len(sec.text) < min_size:
            prev = out[-1]
            out[-1] = Section(prev.heading, prev.text + "\n\n" + sec.text, prev.start)
        else:
            out.append(sec)
    return out


def apply_size_guardrails(sections: list[Section], max_size: int, min_size: int) -> list[Section]:
    """大切/小并。先切超限，再并过小。"""
    split: list[Section] = []
    for sec in sections:
        split.extend(_split_oversized(sec, max_size))
    return _merge_undersized(split, min_size)
```

- [ ] **Step 4: 运行确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（19 passed）

- [ ] **Step 5: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务6 字数护栏（大切/小并）"
```

---

## Task 7: chunk_document 编排 + kb_project 旁路 + 集成测试

**Files:**
- Modify: `app/ingest/chunker.py`（加 `chunk_document` + `_chunk_jsonl`）
- Test: `tests/unit/ingest/test_chunker.py`（追加集成测试）

**Interfaces:**
- Consumes: `app.config.settings.chunk_target_max/min`、`app.kbmap.project_serializer.serialize_project`
- Produces: `chunk_document(doc_id, doc_name, kb, raw_text) -> list[Chunk]`

- [ ] **Step 1: 追加集成测试**

```python
import json
from app.ingest.chunker import chunk_document


def test_chunk_document_full_pipeline_cleans_and_splits():
    raw = ("<!-- chunk_idx=0 chunk_id=aaa -->\n"
           "<图片内容|start><img src=\"https://x/hash.jpeg\"/><图片内容|end>\n"
           "抬头说明文字。\n"
           "<!-- chunk_idx=1 chunk_id=bbb -->\n"
           "一、总则\n本章节讲总体要求，内容比较丰富。\n"
           "<!-- chunk_idx=2 chunk_id=ccc -->\n"
           "二、附则\n第二条的具体规定。")
    chunks = chunk_document("doc1", "测试.doc", "kb_policy_national", raw)
    assert len(chunks) >= 2
    # 无噪声
    for c in chunks:
        assert "<图片内容" not in c.text and "<img" not in c.text
        assert "https://" not in c.text
    # segment_id 确定性、长度 16
    for c in chunks:
        assert len(c.segment_id) == 16
    # source_chunk_ids 非空（追溯原 chunk_id）
    all_src = {cid for c in chunks for cid in c.source_chunk_ids}
    assert "aaa" in all_src or "bbb" in all_src
    # heading 至少有一个被识别
    assert any(c.heading for c in chunks)


def test_chunk_document_is_idempotent():
    raw = "<!-- chunk_idx=0 chunk_id=a -->\n一、章\n正文内容。\n二、章\n正文。"
    a = chunk_document("docX", "x.doc", "kb_regulation", raw)
    b = chunk_document("docX", "x.doc", "kb_regulation", raw)
    assert [c.segment_id for c in a] == [c.segment_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_document_truncation_heals_into_paragraph():
    # 两块被 coreagent 在句中截断（无结构边界）：重建后合成一段连贯文本
    raw = ("<!-- chunk_idx=0 chunk_id=a -->\n这是一段连续的散文内容被截"
           "<!-- chunk_idx=1 chunk_id=b -->\n断了，但其实属于同一句话。")
    chunks = chunk_document("docT", "t.doc", "kb_policy_national", raw)
    # 拼起来应是连贯的一句（不被句中切开）
    joined = "".join(c.text for c in chunks)
    assert "同一句话" in joined
    assert "被截断了" in joined or "被截\n断了" in joined or "被截断" in joined.replace("\n","")


def test_chunk_document_kb_project_uses_jsonl_serializer():
    obj1 = {"立项年份": "2019", "项目名称": "桥A", "技术领域": "桥梁", "承担单位": "X公司",
            "项目负责人": "张三", "主要研究内容": "研究一"}
    obj2 = {"立项年份": "2020", "项目名称": "隧B", "技术领域": "隧道", "承担单位": "Y公司",
            "项目负责人": "李四", "主要研究内容": "研究二"}
    raw = ("<!-- chunk_idx=0 chunk_id=p0 -->\n" + json.dumps(obj1, ensure_ascii=False) + "\n"
           "<!-- chunk_idx=1 chunk_id=p1 -->\n" + json.dumps(obj2, ensure_ascii=False))
    chunks = chunk_document("docP", "项目.xlsx", "kb_project", raw)
    # 一项目一 chunk
    assert len(chunks) == 2
    assert "桥A" in chunks[0].text and "张三" in chunks[0].text
    assert "隧B" in chunks[1].text and "李四" in chunks[1].text
    # 序号递增
    assert chunks[0].ordinal == 0 and chunks[1].ordinal == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py::test_chunk_document_full_pipeline_cleans_and_splits -v`
Expected: FAIL — ImportError（chunk_document 未定义）

- [ ] **Step 3: 加 chunk_document + _chunk_jsonl 到 chunker.py**

在文件顶部 `import hashlib` 之后加：

```python
import json
from app.config import settings
```

在 `apply_size_guardrails` 之后追加：

```python
def _source_chunk_ids_for(intervals: list[SpanInterval], start: int, end: int) -> list[str]:
    """返回与 [start, end) 文本区间相交的原 chunk_id（去重、保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for iv in intervals:
        if iv.end > start and iv.start < end and iv.chunk_id not in seen:
            seen.add(iv.chunk_id)
            out.append(iv.chunk_id)
    return out


def _chunk_jsonl(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:
    """kb_project 旁路：每行 JSON 一个 chunk，自然语言序列化后作 embedding 输入。"""
    from app.kbmap.project_serializer import serialize_project
    chunks: list[Chunk] = []
    ordinal = 0
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = serialize_project(obj)
        if not text:
            continue
        chunks.append(Chunk(
            segment_id=assign_segment_id(doc_id, ordinal, text),
            text=text, doc_id=doc_id, doc_name=doc_name, kb=kb,
            heading=None, source_chunk_ids=[], ordinal=ordinal,
        ))
        ordinal += 1
    return chunks


def chunk_document(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:
    """主入口：根据 kb 选策略。kb_project 走 jsonl；其他走结构感知管线。"""
    if kb == "kb_project":
        return _chunk_jsonl(doc_id, doc_name, kb, raw_text)

    raw_chunks = parse_raw_chunks(raw_text)
    full, intervals = reconstruct(raw_chunks)
    if not full:
        return []
    sections = split_by_structure(full)
    sections = apply_size_guardrails(
        sections, settings.chunk_target_max, settings.chunk_target_min
    )
    chunks: list[Chunk] = []
    for ordinal, sec in enumerate(sections):
        seg_id = assign_segment_id(doc_id, ordinal, sec.text)
        src = _source_chunk_ids_for(intervals, sec.start, sec.start + len(sec.text))
        chunks.append(Chunk(
            segment_id=seg_id, text=sec.text, doc_id=doc_id, doc_name=doc_name,
            kb=kb, heading=sec.heading, source_chunk_ids=src, ordinal=ordinal,
        ))
    return chunks
```

- [ ] **Step 4: 运行全量 ingest 测试确认通过**

Run: `"<venv python>" -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: PASS（23 passed，含 4 个集成测试）

- [ ] **Step 5: 跑全量回归（确保没碰坏 kbmap）**

Run: `"<venv python>" -m pytest tests/unit/ -q`
Expected: 全部通过（kbmap 33 + ingest 23 = 56 passed 左右），0 失败。

- [ ] **Step 6: 提交**

```bash
git add app/ingest/chunker.py tests/unit/ingest/test_chunker.py
git commit -m "feat(ingest): 任务7 chunk_document 编排+kb_project 旁路+集成测试"
```

---

## 完工验收（对应 spec §10 测试清单）

1. **基础结构**：含 `#`/`一、` 章节的 doc → 按章节切（Task 5 测试覆盖）。
2. **截断愈合**：句中截断的两块 → 重建后连贯（Task 7 `test_chunk_document_truncation_heals_into_paragraph`）。
3. **噪声剥离**：`<图片内容>`/`<img/>`/注入表 → 清洗后不含（Task 3 + Task 7 集成）。
4. **大段切分**：3000 字无结构段 → 按句切到 <=max（Task 6）。
5. **小段合并**：碎片 → 并入邻居（Task 6）。
6. **幂等**：同 doc 两次 → 同 segment_id 集合（Task 7 `test_chunk_document_is_idempotent`）。
7. **kb_project**：jsonl 一项目一 chunk（Task 7）。
8. **追溯**：final chunk 的 source_chunk_ids 覆盖其文本区间（Task 7 集成 + `_source_chunk_ids_for`）。

## 范围外（推迟）

- **与主 spec 入库流水线的接线**（`app/ingest/pipeline.py` 调 `chunk_document`）：主 spec 的 pipeline 由另一会话实现，本计划只交付独立可测的 `chunker.py`。pipeline 接线、Milvus 写入、`chunker_backend="fixed"` 回退分支均待 pipeline 就位后做。
- **主 spec §5 Milvus schema 加 `source_chunk_ids`/`heading` 字段**：待 pipeline 实现时一并加。
- **真实数据批量验证**（跑 chunker 看 200 文件的 chunk 大小分布、噪声剥离效果）：工具链就位后用 `scripts/` 脚本做数据验证，非 TDD 任务。
