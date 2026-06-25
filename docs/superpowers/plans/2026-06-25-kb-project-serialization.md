# kb_project 序列化改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 kb_project 的序列化在真实数据上产出干净、以项目主题为主导的自然语言文本，并捞回当前被静默丢弃的 5 个切断项目——全部集中在 `app/kbmap/project_serializer.py` 单一文件。

**Architecture:** 三层纯函数管线 `parse_projects`（容错解析，字段正则替代 json.loads，切断单元+续片回接）→ `clean_project`（实际产出成果剥套话+去重全留；预期成果指标清垃圾）→ `serialize_project`（头部散文+字段取舍+原样输出保确定性）。入库 `chunker._chunk_jsonl` 与验证 `kbmap.embed` 都改为调用这条管线，`iter_jsonl_chunks` 对外签名不变。全链路无随机 → 确定性/重灌幂等不变。

**Tech Stack:** Python ≥3.10，标准库 `re`/`pathlib`；pytest。复用 `app.config.settings`、`app.kbmap.project_serializer`；不引入新依赖。

**Spec:** `docs/superpowers/specs/2026-06-25-kb-project-serialization-design.md`

**运行环境假设：**
- 所有命令在 worktree 根 `D:\project\temp-rag\.claude\worktrees\kb-project` 下、项目 venv 激活后执行（Windows：`.venv/Scripts/python -m pytest ...`；见 HANDOFF §3.1-3.2）。
- **本计划只改 `project_serializer.py` + 两处调用方接线 + 测试**。MinerU 解析对 JSONL 的端到端影响（HANDOFF §9 缺口 #1）**不在本计划范围**——`_chunk_jsonl` 收到的 `raw_text` 在测试里直接是文件原文，生产链路里是 MinerU 输出（假设其透传 JSON 行，与现状一致）。
- 真实数据在 `D:/project/temp-rag/merged/merged/1aa48..._集团历史项目-v2/项目自定义导出数据 (基础)-v3.xlsx.md`（gitignored），供手动验证用；测试用合成数据自包含。

---

## File Structure

| 文件 | 责任 | 本计划改动 |
|------|------|-----------|
| `app/kbmap/project_serializer.py` | kb_project 的解析/清洗/序列化（单一源） | 新增 `parse_projects`/`clean_project`/`_split_units`/`_clean_achievements`/`_clean_garbage`；改写 `serialize_project`、`iter_jsonl_chunks`；删 `_FIELDS_ORDER`、`import json`，加 `import re` |
| `app/ingest/chunker.py` | 入库切块；`_chunk_jsonl` 是 kb_project 旁路 | 改 `_chunk_jsonl` 走新管线；删未用的 `import json`（line 8） |
| `tests/unit/kbmap/test_project_serializer.py` | 序列化器单测 | 新增 parse/clean/恢复/字段排除用例；保留并确认旧用例仍绿 |
| `tests/unit/ingest/test_chunker.py` | chunker 回归 | 确认 `test_chunk_document_kb_project_uses_jsonl_serializer` 仍绿（无需改） |
| `tests/unit/kbmap/test_embed.py` | embed 回归 | 确认 `test_extract_kb_project_*` 仍绿（无需改） |

---

## Task 1: `parse_projects` 容错解析（含切断项目恢复）

> **实施后修订（代码评审驱动）：** Step 3 的「纯正则提取」在 real-data 上会因转义 `\"` 截断字段值（JT2022YB27 的 `项目概况` 丢 ~88%）且不解码 `\n`/`\t`。**实际实现改为 `_parse_blob`：优先 `json.loads`（完整记录精确还原转义），仅切断记录回退正则。** 见 fix 提交 `cec289e` + 新增 2 个转义回归测试；spec §4 已同步。下方 Step 3 代码保留作历史记录，以 as-built 为准。

**Files:**
- Modify: `app/kbmap/project_serializer.py`（顶部加 `import re`，文件末尾追加新函数）
- Test: `tests/unit/kbmap/test_project_serializer.py`

- [ ] **Step 1: 写失败测试（追加到 test_project_serializer.py 顶部 import 之后）**

```python
from app.kbmap.project_serializer import parse_projects


def test_parse_projects_complete_records():
    raw = ('<!-- chunk_idx=0 chunk_id=a occurTime=1 -->\n'
           '{"立项年份":"2019","项目名称":"完整项目","技术领域":"桥梁"}\n\n---\n\n'
           '<!-- chunk_idx=1 chunk_id=b occurTime=2 -->\n'
           '{"立项年份":"2020","项目名称":"另一项目","技术领域":"隧道"}\n\n---\n\n')
    projects = parse_projects(raw)
    assert len(projects) == 2
    assert projects[0]["项目名称"] == "完整项目"
    assert projects[1]["技术领域"] == "隧道"


def test_parse_projects_recovers_truncated_record_with_fragments():
    # 切断：实际产出成果 中途截断（无闭合 " 与 }），后跟 2 个裸续片
    trunc = ('{"立项年份":"2020","项目名称":"切断项目",'
             '"实际产出成果":"成果类型：专利，成果名称：P2 ；')
    raw = ('<!-- chunk_idx=0 chunk_id=a -->\n'
           '{"立项年份":"2019","项目名称":"完整项目"}\n\n---\n\n'
           '<!-- chunk_idx=1 chunk_id=b -->\n' + trunc + '\n\n---\n\n'
           '<!-- chunk_idx=2 chunk_id=c -->\n'
           '成果类型：论文，成果名称：P3 ；\n\n---\n\n'
           '<!-- chunk_idx=3 chunk_id=d -->\n'
           '成果类型：指南，成果名称：P4 ；\n\n---\n\n')
    projects = parse_projects(raw)
    assert len(projects) == 2
    assert projects[1]["项目名称"] == "切断项目"
    # 切断项目的 实际产出成果 应含母项目成果 + 全部续片（回接恢复）
    ach = projects[1]["实际产出成果"]
    assert "P2" in ach and "P3" in ach and "P4" in ach


def test_parse_projects_tolerates_json_dumps_spacing():
    # test_chunker.py 用 json.dumps 造数据，字段间是 "： "（冒号后空格），必须兼容
    import json
    obj = {"立项年份": "2019", "项目名称": "间距项目"}
    raw = '<!-- chunk_idx=0 chunk_id=a -->\n' + json.dumps(obj, ensure_ascii=False)
    projects = parse_projects(raw)
    assert len(projects) == 1
    assert projects[0]["项目名称"] == "间距项目"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py::test_parse_projects_complete_records -v`
Expected: FAIL（`ImportError: cannot import name 'parse_projects'`）

- [ ] **Step 3: 实现 `parse_projects`（追加到 project_serializer.py；并在文件顶部 `import json` 之外加 `import re`）**

在文件顶部把 `import json` 那行下方加一行 `import re`（json 暂保留，Task 4 才删）。然后在文件末尾追加：

```python
_CHUNK_MARKER_RE = re.compile(r"<!--\s*chunk_idx=\d+[^>]*-->")
# 字段提取：容忍 "k":"v" 与 json.dumps 的 "k": "v"（冒号两侧空格）
_FIELD_RE = re.compile(r'"([^"\\]+)"\s*:\s*"([^"]*)"')


def _split_units(raw_text: str) -> list[str]:
    """按 chunk_idx 标记切成单元正文，剥掉 --- 分隔行与首尾空白。"""
    parts = _CHUNK_MARKER_RE.split(raw_text)
    units: list[str] = []
    for body in parts[1:]:  # parts[0] 是首个标记前的内容（通常为空），跳过
        cleaned = "\n".join(
            ln for ln in body.splitlines() if ln.strip() != "---"
        ).strip()
        if cleaned:
            units.append(cleaned)
    return units


def parse_projects(raw_text: str) -> list[dict]:
    """容错解析 kb_project 文件文本。每个逻辑项目一个 dict（字段名→值）。

    用字段正则提取，不依赖 json.loads，故未闭合的切断记录也能恢复：
    切断单元（{ 开头但不闭合）+ 紧随其后的裸续片会被合并成一个逻辑项目，
    其 实际产出成果 字段值由续片补全。返回顺序与文件一致。
    """
    units = _split_units(raw_text)
    blobs: list[str] = []
    for u in units:
        if u.startswith("{"):
            blobs.append(u)
        elif blobs:
            # 裸续片：回接到当前（最后一个）逻辑项目
            blobs[-1] += u
        # 没有前置 { 单元的孤立续片：丢弃
    projects: list[dict] = []
    for blob in blobs:
        # 切断记录的 实际产出成果 未闭合（无结尾 "）→ 补 " 让正则能匹配到值尾
        if not blob.endswith('"'):
            blob = blob + '"'
        d = {k: v for k, v in _FIELD_RE.findall(blob)}
        if d:
            projects.append(d)
    return projects
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py -k parse_projects -v`
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/project_serializer.py tests/unit/kbmap/test_project_serializer.py
git commit -m "feat(kbmap): parse_projects 容错解析（字段正则+切断项目回接恢复）"
```

---

## Task 2: `clean_project` 清洗（成果去重全留 + 清垃圾）

**Files:**
- Modify: `app/kbmap/project_serializer.py`（追加函数）
- Test: `tests/unit/kbmap/test_project_serializer.py`

- [ ] **Step 1: 写失败测试（追加）**

```python
from app.kbmap.project_serializer import clean_project


def test_clean_project_strips_achievement_boilerplate_and_dedupes():
    d = {"实际产出成果":
         "成果类型：专利，成果名称：自锁式行走轮 ；"
         "成果类型：专利，成果名称：自锁式行走轮 ；"   # 重复
         "成果类型：论文，成果名称：除湿微机控制系统 ；"}
    out = clean_project(d)
    assert out["实际产出成果"] == "自锁式行走轮；除湿微机控制系统"  # 剥套话+去重+；连接
    assert "成果类型" not in out["实际产出成果"]
    assert "成果名称" not in out["实际产出成果"]


def test_clean_project_preserves_non_boilerplate_achievements():
    # 无 成果名称： 套话模式时，原样返回去空白文本（不丢数据）
    d = {"实际产出成果": "  自由文本成果描述  "}
    assert clean_project(d)["实际产出成果"] == "自由文本成果描述"


def test_clean_project_cleans_garbage_in_expected_indicator():
    d = {"预期成果指标": "1项专利授权发明专利(项)\t\t,;\n9篇形成研究报告数(篇)\t\t,;"}
    out = clean_project(d)["预期成果指标"]
    assert "(项)" not in out and "(篇)" not in out
    assert "\t" not in out


def test_clean_project_does_not_mutate_input():
    d = {"实际产出成果": "成果类型：专利，成果名称：A ；"}
    original = dict(d)
    clean_project(d)
    assert d == original  # 原 dict 不被修改
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py -k clean_project -v`
Expected: FAIL（`ImportError: cannot import name 'clean_project'`）

- [ ] **Step 3: 实现 `clean_project`（追加到 project_serializer.py）**

```python
_ACHIEVEMENT_NAME_RE = re.compile(r"成果名称：([^；]+)")


def _clean_achievements(value: str) -> str:
    """剥 '成果类型：X，成果名称：' 套话 → 取名称 → 去重（保序）→ 全留，；连接。
    无套话模式时返回去空白后的原文（不丢数据）。"""
    if not value:
        return ""
    names = _ACHIEVEMENT_NAME_RE.findall(value)
    if not names:
        return value.strip()
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            unique.append(n)
    return "；".join(unique)


def _clean_garbage(value: str) -> str:
    """清 (项)/(本)/(篇) 等单位标记与 \\t,; 计数噪声，折叠空白。"""
    if not value:
        return ""
    s = re.sub(r"\([^)]*\)", " ", value)   # 去 (项)(本)(篇)
    s = re.sub(r"[\t,;]+", " ", s)          # 去制表/逗号/分号噪声
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_project(d: dict) -> dict:
    """清洗：实际产出成果（剥套话+去重全留）、预期成果指标（清垃圾）。
    返回新 dict，不改入参。"""
    out = dict(d)
    out["实际产出成果"] = _clean_achievements(d.get("实际产出成果", ""))
    out["预期成果指标"] = _clean_garbage(d.get("预期成果指标", ""))
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py -k clean_project -v`
Expected: 4 PASS

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/project_serializer.py tests/unit/kbmap/test_project_serializer.py
git commit -m "feat(kbmap): clean_project 成果去重全留+套话剥离+垃圾清理"
```

---

## Task 3: 改写 `serialize_project`（头部散文 + 字段取舍 + 删 `_FIELDS_ORDER`）

**Files:**
- Modify: `app/kbmap/project_serializer.py`（替换 `serialize_project` 函数体与 `_FIELDS_ORDER`）
- Test: `tests/unit/kbmap/test_project_serializer.py`

- [ ] **Step 1: 写新测试（追加）；旧 `test_serialize_project_basic` / `test_serialize_project_skips_empty_fields` 保留不动（仍应绿）**

```python
def test_serialize_project_excludes_contact_and_budget_fields():
    obj = {
        "技术领域": "桥梁工程", "项目名称": "某项目", "项目编号": "JT0001",
        "立项年份": "2019", "承担单位": "某公司", "项目负责人": "张三",
        "项目负责人电话": "13900000000", "项目负责人电子邮箱": "secret@example.com",
        "项目预算总经费": "99999", "项目总决算": "88888",
    }
    s = serialize_project(obj)
    # 联系/经费字段不进入序列化文本
    assert "13900000000" not in s
    assert "secret@example.com" not in s
    assert "99999" not in s
    assert "88888" not in s
    # 主题字段在
    assert "桥梁工程" in s and "某项目" in s and "张三" in s


def test_serialize_project_head_is_prose_with_category_in_parens():
    obj = {"技术领域": "桥梁工程", "项目名称": "缆索研究", "项目编号": "JT1",
           "业务类别": "重点科技项目", "立项年份": "2019",
           "承担单位": "某公司", "项目负责人": "熊锋"}
    s = serialize_project(obj)
    assert s.startswith("桥梁工程领域的科研项目《缆索研究》（编号 JT1，重点科技项目）")
    assert "2019年立项" in s
    assert "由某公司承担" in s
    assert "项目负责人熊锋" in s
    assert s.endswith("。")


def test_clean_then_serialize_full_pipeline():
    d = {"技术领域": "桥梁工程", "项目名称": "X", "项目编号": "JT1", "立项年份": "2019",
         "承担单位": "某公司", "项目负责人": "张三",
         "实际产出成果": "成果类型：专利，成果名称：A ；成果类型：专利，成果名称：A ；成果类型：论文，成果名称：B ；"}
    s = serialize_project(clean_project(d))
    assert "已产出成果：A；B。" in s          # 去重+全留+句末句号
    assert "成果类型" not in s and "成果名称" not in s
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py -k "head_is_prose or excludes_contact or full_pipeline" -v`
Expected: 新 3 个 FAIL（输出格式不符；`head_is_prose` 因当前模板开头是"2019年立项的【桥梁工程】类项目"而失败）

- [ ] **Step 3: 改写 `serialize_project` 并删除 `_FIELDS_ORDER`**

先把文件顶部的 `_FIELDS_ORDER = [...]` 整块（约 4 行，含其下空行）**删除**。再把现有 `def serialize_project(obj: dict) -> str:` 整个函数体替换为：

```python
def serialize_project(d: dict) -> str:
    """把（已清洗的）项目 dict 序列化成自然语言文本。

    头部一句散文（技术领域打头），随后主要研究内容/关键技术问题/成果分句。
    字段值原样输出（不做改写/概括，保证确定性）；空字段与弃用字段不出现。
    成果需先经 clean_project 清洗，本函数只负责排版。
    """
    def g(key: str) -> str:
        return (d.get(key) or "").strip()

    tech, name = g("技术领域"), g("项目名称")
    number, category = g("项目编号"), g("业务类别")
    year, org, leader = g("立项年份"), g("承担单位"), g("项目负责人")
    content = g("主要研究内容")
    problem = g("拟解决的关键技术问题")
    achievements = g("实际产出成果")

    sentences: list[str] = []

    # 头部：[技术领域]领域的科研项目《项目名称》（编号 X，业务类别），年份立项，由X承担，项目负责人X。
    head_left = f"{tech}领域的科研项目" if tech else ("科研项目" if name else "")
    name_part = f"《{name}》" if name else ""
    paren_inner = "，".join(p for p in [f"编号 {number}" if number else "", category] if p)
    paren = f"（{paren_inner}）" if paren_inner else ""
    core = (head_left + name_part + paren).strip()
    head_bits: list[str] = []
    if core:
        head_bits.append(core)
    if year:
        head_bits.append(f"{year}年立项")
    if org:
        head_bits.append(f"由{org}承担")
    if leader:
        head_bits.append(f"项目负责人{leader}")
    if head_bits:
        sentences.append("，".join(head_bits) + "。")

    if content:
        sentences.append(f"主要研究内容：{content}。")
    if problem:
        sentences.append(f"拟解决的关键技术问题：{problem}。")
    if achievements:
        sentences.append(f"已产出成果：{achievements}。")

    return "".join(sentences)
```

- [ ] **Step 4: 运行测试确认通过（含旧用例回归）**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py -v`
Expected: 全部 PASS（新 3 + 旧 `test_serialize_project_basic` / `_skips_empty_fields`，以及 Task 1/2 的 parse/clean 用例）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/project_serializer.py tests/unit/kbmap/test_project_serializer.py
git commit -m "feat(kbmap): serialize_project 改写为头部散文+字段取舍（删 _FIELDS_ORDER）"
```

---

## Task 4: `iter_jsonl_chunks` 接线新管线 + 删 `import json`

**Files:**
- Modify: `app/kbmap/project_serializer.py`（替换 `iter_jsonl_chunks`；删顶部 `import json`）
- Test: `tests/unit/kbmap/test_project_serializer.py`、`tests/unit/kbmap/test_embed.py`（回归）

- [ ] **Step 1: 写新测试（追加到 test_project_serializer.py）—— 在文件级切断恢复**

```python
def test_iter_jsonl_chunks_recovers_truncated_project_from_file(tmp_path):
    trunc = ('{"立项年份":"2020","项目名称":"切断项目",'
             '"实际产出成果":"成果类型：专利，成果名称：P2 ；')
    content = ('<!-- chunk_idx=0 chunk_id=a -->\n'
               '{"立项年份":"2019","项目名称":"完整项目"}\n\n---\n\n'
               '<!-- chunk_idx=1 chunk_id=b -->\n' + trunc + '\n\n---\n\n'
               '<!-- chunk_idx=2 chunk_id=c -->\n'
               '成果类型：论文，成果名称：P3 ；\n\n---\n\n')
    f = tmp_path / "项目导出.xlsx.md"
    f.write_text(content, encoding="utf-8")
    chunks = iter_jsonl_chunks(f)
    assert len(chunks) == 2                    # 切断项目被恢复，不是 1
    assert "切断项目" in chunks[1]
    assert "P2" in chunks[1] and "P3" in chunks[1]
    assert "成果类型" not in chunks[1]         # 已清洗
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py::test_iter_jsonl_chunks_recovers_truncated_project_from_file -v`
Expected: FAIL（当前 `iter_jsonl_chunks` 逐行 `startswith("{")`+`json.loads`，切断行解析失败被跳过 → 返回 1 条）

- [ ] **Step 3: 改写 `iter_jsonl_chunks` 并删除 `import json`**

删除文件顶部的 `import json` 行（此时已无任何代码用到 json）。把现有 `def iter_jsonl_chunks(path: Path) -> list[str]:` 整个函数替换为：

```python
def iter_jsonl_chunks(path: Path) -> list[str]:
    """读 kb_project 文件 → 解析 → 清洗 → 序列化，返回每项目一段文本（按文件顺序）。

    复用 parse_projects（容错，含切断项目恢复）+ clean_project + serialize_project，
    与入库 chunker._chunk_jsonl 共用同一管线。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return [serialize_project(clean_project(d)) for d in parse_projects(text)]
```

- [ ] **Step 4: 运行测试确认通过 + embed 回归**

Run: `python -m pytest tests/unit/kbmap/test_project_serializer.py tests/unit/kbmap/test_embed.py -v`
Expected: 全部 PASS
- 新 `test_iter_jsonl_chunks_recovers_truncated_project_from_file` PASS
- 旧 `test_iter_jsonl_chunks_one_project_per_chunk`（fixture 2 项目）仍 PASS
- `test_extract_kb_project_uses_serializer` / `test_extract_kb_project_falls_back_to_doc_name_when_no_json` 仍 PASS

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/project_serializer.py tests/unit/kbmap/test_project_serializer.py
git commit -m "feat(kbmap): iter_jsonl_chunks 接线 parse+clean+serialize 管线（删 import json）"
```

---

## Task 5: `chunker._chunk_jsonl` 接线新管线 + 删未用 `import json` + 回归

**Files:**
- Modify: `app/ingest/chunker.py`（替换 `_chunk_jsonl` 函数体；删顶部 `import json` line 8）
- Test: `tests/unit/ingest/test_chunker.py`（回归，确认 `test_chunk_document_kb_project_uses_jsonl_serializer` 仍绿）

- [ ] **Step 1: 先跑现有 kb_project 回归用例，记录当前基线（应绿）**

Run: `python -m pytest tests/unit/ingest/test_chunker.py::test_chunk_document_kb_project_uses_jsonl_serializer -v`
Expected: PASS（改前基线；改后须仍 PASS）

- [ ] **Step 2: 改写 `_chunk_jsonl` 并删除 `import json`**

删除 `app/ingest/chunker.py` 第 8 行 `import json`（grep 已确认仅 `_chunk_jsonl` 用到 json）。把 `def _chunk_jsonl(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:` 整个函数替换为：

```python
def _chunk_jsonl(doc_id: str, doc_name: str, kb: str, raw_text: str) -> list[Chunk]:
    """kb_project 旁路：每行 JSON 一个 chunk，自然语言序列化后作 embedding 输入。

    复用 app.kbmap.project_serializer 的 parse_projects（容错，含切断项目恢复）
    + clean_project + serialize_project 管线，与 kbmap.embed 共用单一源。
    ordinal 按 emitted chunk 递增 → segment_id 确定性。
    """
    from app.kbmap.project_serializer import parse_projects, clean_project, serialize_project
    chunks: list[Chunk] = []
    ordinal = 0
    for obj in parse_projects(raw_text):
        text = serialize_project(clean_project(obj))
        if not text:
            continue
        chunks.append(Chunk(
            segment_id=assign_segment_id(doc_id, ordinal, text),
            text=text, doc_id=doc_id, doc_name=doc_name, kb=kb,
            heading=None, source_chunk_ids=[], ordinal=ordinal,
        ))
        ordinal += 1
    return chunks
```

- [ ] **Step 3: 运行 chunker 全量 + kb_project 回归**

Run: `python -m pytest tests/unit/ingest/test_chunker.py -v`
Expected: 全部 PASS，含 `test_chunk_document_kb_project_uses_jsonl_serializer`（2 项目、桥A/张三、ordinal 0/1）——`parse_projects` 兼容 `json.dumps` 的冒号空格。

- [ ] **Step 4: 跑 kbmap + ingest 全量回归**

Run: `python -m pytest tests/unit/kbmap tests/unit/ingest -v`
Expected: 全部 PASS

- [ ] **Step 5: （可选）真实数据手动验证切断项目恢复**

Run（一行，确认真实文件解析出 230 个项目而非 225）:
```bash
python -c "from app.kbmap.project_serializer import parse_projects; t=open(r'D:/project/temp-rag/merged/merged/1aa4823513ea407e93a5af1594122da0_集团历史项目-v2/项目自定义导出数据 (基础)-v3.xlsx.md',encoding='utf-8').read(); ps=parse_projects(t); print('项目数:',len(ps)); print('含高速公路改扩建:', any('高速公路改扩建' in (p.get('项目名称') or '') for p in ps))"
```
Expected: `项目数: 230`（225 完整 + 5 切断恢复）；`含高速公路改扩建: True`（之前被丢的切断项目已恢复）

- [ ] **Step 6: 提交**

```bash
git add app/ingest/chunker.py
git commit -m "feat(ingest): _chunk_jsonl 接线 project_serializer 管线（删未用 import json）"
```

---

## 收尾：全量回归

- [ ] **Run:** `python -m pytest -q`
Expected: 全绿（HANDOFF 记 95 测试基线；本计划净增约 9 个用例，无删除）。5 个 Windows 编码 warning 可忽略。

---

## Self-Review

**1. Spec coverage：**
- §3 三层管线 parse/clean/serialize → Task 1/2/3 ✓
- §4 容错解析（字段正则、切断回接、补 `"`、json.dumps 兼容）→ Task 1 ✓
- §5.1 成果去重全留 → Task 2 `_clean_achievements` ✓
- §5.2 预期成果指标清垃圾 → Task 2 `_clean_garbage` ✓
- §5.3 字段取舍（弃联系/经费）→ Task 3 `serialize_project` 仅取留用字段 ✓
- §6 模板（头部散文、业务类别入括号、原样输出）→ Task 3 ✓
- §7 集成：`iter_jsonl_chunks` 签名不变、`_chunk_jsonl`/`embed` 共用管线 → Task 4/5 ✓
- §8 测试（完整/切断恢复/去重/清垃圾/字段取舍/空字段）→ Task 1-4 用例 ✓
- §10 验收（230 项目、0 套话残留、不含联系经费、签名不变）→ Task 3/4/5 + 收尾 ✓

**2. Placeholder scan：** 无 TBD/TODO/"适当处理"；每步含完整代码与确切命令。✓

**3. Type/name 一致性：** `parse_projects(raw_text)->list[dict]`、`clean_project(d)->dict`、`serialize_project(d)->str`、`iter_jsonl_chunks(path)->list[str]` 全计划一致；`_FIELD_RE`/`_ACHIEVEMENT_NAME_RE`/`_split_units`/`_clean_achievements`/`_clean_garbage` 定义与引用一致；`Chunk` 字段（segment_id/text/doc_id/doc_name/kb/heading/source_chunk_ids/ordinal）与 `chunker.py` 现有 dataclass 一致。✓
