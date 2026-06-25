# 知识库分类体系与离线整理工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现离线知识库整理工具链（`app/kbmap/`），把 `merged/` 下 10 个混乱目录清洗成清晰的 KB 分类（manifest），用 embedding 验证分类质量，并产出可复用的 KB 分类器供开发 agent 使用。

**Architecture:** 三阶段管线折叠为 9 个 TDD 任务：①目录扫描器（含 content-hash 去重 + 默认 dir→KB 映射）产出 manifest 草案 → ②manifest pydantic 模型 + IO → ③kb_project JSONL 序列化器 → ④文件嵌入助手（复用 `CloudEmbeddingService`）→ ⑤验证（凝聚度/分离度/异常）→ ⑥质心构建 → ⑦分类器 → ⑧CLI 串联。**不进运行时问答链路**；ingest `--manifest` 集成因依赖主 spec 的 `app/ingest/`（尚未实现）而**推迟**，本计划只交付自包含的工具链。

**Tech Stack:** Python ≥3.10, pydantic v2, pydantic-settings, numpy, pyyaml（新增）, 复用 `app.services.CloudEmbeddingService`(bge-m3) / `MockCloudEmbeddingService`；pytest + pytest-asyncio。

**Spec:** `docs/superpowers/specs/2026-06-25-kb-taxonomy-and-tooling-design.md`

## Global Constraints

- Python ≥3.10。新代码全部放 `app/kbmap/` 包内。
- 复用现有 `app.services.CloudEmbeddingService`（bge-m3, dim=1024）；测试用 `MockCloudEmbeddingService(dimension=1024)`。
- 所有影响结果的参数进 `app/config.py` 的 `Settings` 类，前缀 `kbmap_`。
- 不进运行时问答链路（spec §1 非目标）；不修改主 spec 的 LangGraph 图 / Milvus schema / 稳定性机制。
- 代码注释/文档用中文，标识符用英文。
- TDD：每个任务先写失败测试，再最小实现，再提交。
- `merged/` 目录在项目外（`D:/project/kxx/04需求开发/005其他/download_chunk_from_coreagent/merged/`），其路径由 config `kbmap_merged_root` 指定；**测试一律用 `tests/fixtures/kbmap/` 下的小型 fixture，不依赖真实 merged/**。

## File Structure

```
langgraph/
├── app/kbmap/                         # 全部新代码
│   ├── __init__.py
│   ├── scanner.py          # Task 2: 扫描 merged/ + 去重 + 默认映射 + 产 manifest 草案
│   ├── manifest.py         # Task 3: Manifest/FileEntry pydantic 模型 + load/save
│   ├── project_serializer.py # Task 4: kb_project JSONL → 自然语言序列化
│   ├── embed.py            # Task 5: 文件→向量（含 md 标题+首 chunk、kb_project 例外）
│   ├── metrics.py          # Task 6: 凝聚度/分离度/异常 + 报告
│   ├── centroids.py        # Task 7: KB 质心构建 + 落盘
│   ├── classifier.py       # Task 8: KBClassifier.classify / assign_file
│   └── __main__.py         # Task 9: CLI 入口（scan/verify/build-centroids/classify/assign）
├── tests/fixtures/kbmap/               # 测试用小型知识库 fixture
│   ├── merged/              # 模拟 merged 结构（小）
│   └── expected/            # 期望产物片段
└── tests/unit/kbmap/                    # 每个模块一个 test 文件
```

**注：** Spec §7 的 `scripts/build_manifest.py` 职责并入 `app/kbmap/scanner.py` + CLI `scan` 子命令（DRY，避免双入口）。

---

## Task 1: kbmap 包骨架与配置扩展

**Files:**
- Create: `app/kbmap/__init__.py`
- Modify: `app/config.py`（在末尾 `stability_semantic_threshold` 后追加 kbmap 字段）
- Modify: `requirements.txt`（新增 `pyyaml>=6.0`）
- Test: `tests/unit/kbmap/__init__.py`, `tests/unit/kbmap/test_config.py`

**Interfaces:**
- Consumes: `app.config.Settings`
- Produces: `app.kbmap` 包（空）、`settings.kbmap_*` 字段、`pyyaml` 依赖

- [ ] **Step 1: 新建测试目录与失败测试**

`tests/unit/kbmap/__init__.py`（空文件）。`tests/unit/kbmap/test_config.py`:

```python
from app.config import settings


def test_kbmap_defaults():
    assert settings.kbmap_embed_batch_size == 32
    assert settings.kbmap_chunk_preview_chars == 500
    assert settings.kbmap_outlier_margin == 0.05
    assert settings.kbmap_cohesion_min == 0.5
    assert settings.kbmap_separation_max == 0.85
    assert settings.kbmap_merged_root == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'kbmap_embed_batch_size'`

- [ ] **Step 3: 创建 `app/kbmap/__init__.py`（空）**

```python
"""知识库分类与离线整理工具链。"""
```

- [ ] **Step 4: 扩展 `app/config.py`**

在 `stability_semantic_threshold: float = 0.98` 这一行之后、`model_config = ...` 之前插入：

```python

    # ========== KB 分类工具链 (kbmap) ==========
    kbmap_merged_root: str = ""              # 原始 merged/ 目录绝对路径
    kbmap_embed_batch_size: int = 32         # CloudEmbeddingService 批大小
    kbmap_chunk_preview_chars: int = 500     # 嵌入时每文件取首 chunk 字数
    kbmap_outlier_margin: float = 0.05       # 异常文件判定阈值
    kbmap_cohesion_min: float = 0.5          # 类内凝聚度下限
    kbmap_separation_max: float = 0.85       # 类间分离度上限（超过考虑合并）
```

- [ ] **Step 5: 加 pyyaml 依赖**

在 `requirements.txt` 末尾追加一行：

```
pyyaml>=6.0
```

- [ ] **Step 6: 安装依赖并跑测试确认通过**

Run: `pip install pyyaml>=6.0 && pytest tests/unit/kbmap/test_config.py -v`
Expected: PASS（1 passed）

- [ ] **Step 7: 提交**

```bash
git add app/kbmap/__init__.py app/config.py requirements.txt tests/unit/kbmap/__init__.py tests/unit/kbmap/test_config.py
git commit -m "feat(kbmap): 任务1 包骨架与 kbmap 配置字段"
```

---

## Task 2: 目录扫描器（去重 + 默认映射 + 草案）

**Files:**
- Create: `app/kbmap/scanner.py`
- Test: `tests/unit/kbmap/test_scanner.py`, `tests/fixtures/kbmap/merged/`（fixture）

**Interfaces:**
- Consumes: `settings.kbmap_merged_root`
- Produces:
  - `DEFAULT_DIR_TO_KB: dict[str, str]` — 目录后缀名（去掉 UUID 前缀）→ kb 名
  - `FileRecord`（dataclass）：`path: str`（相对 merged 根）、`doc_name: str`、`content_hash: str`、`kb: str`
  - `KbInventory`（dataclass）：`files: list[FileRecord]`、`duplicates: dict[str, list[str]]`（主 path → 同 hash 的其他 path）
  - `scan_kb_dirs(root: Path) -> KbInventory`
  - `write_draft_md(inventory, out_path: Path) -> None`（写人类可读草案）
  - `strip_uuid_prefix(dir_name: str) -> str`（`"0352..._政策文件"` → `"政策文件"`）

- [ ] **Step 1: 建 fixture 目录结构**

建以下空目录与小文件（每个 `.md` 放 1–2 行内容，两个 `_政策文件`/`_政策二` 放**相同**内容以模拟重复）：

```
tests/fixtures/kbmap/merged/
├── aaa1_政策文件/
│   ├── 十四五规划.md            （内容："政策内容A"）
│   └── 交通强国纲要.pdf.md      （内容："政策内容B"）
├── bbb2_政策二/
│   └── 十四五规划.md            （内容："政策内容A"  ← 与上者重复）
├── ccc3_集团规划/
│   └── 集团科技创新纲要.pdf.md   （内容："集团规划内容"）
└── ddd4_集团历史项目-v2/
    └── 项目导出.xlsx.md          （内容：见 Step 2 注释，含两行 JSON）
```

`项目导出.xlsx.md` 内容：

```
<!-- chunk_idx=0 chunk_id=x occurTime=1 -->
{"立项年份":"2019","项目编号":"JT2019YB15","项目名称":"测试项目一","承担单位":"测试公司","技术领域":"桥梁工程","主要研究内容":"内容一"}
<!-- chunk_idx=1 chunk_id=y occurTime=2 -->
{"立项年份":"2020","项目编号":"JT2020YB16","项目名称":"测试项目二","承担单位":"测试公司","技术领域":"隧道工程","主要研究内容":"内容二"}
```

- [ ] **Step 2: 写失败测试**

`tests/unit/kbmap/test_scanner.py`:

```python
from pathlib import Path
from app.kbmap.scanner import (
    scan_kb_dirs, strip_uuid_prefix, DEFAULT_DIR_TO_KB,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "kbmap" / "merged"


def test_strip_uuid_prefix():
    assert strip_uuid_prefix("aaa1_政策文件") == "政策文件"
    assert strip_uuid_prefix("集团规划") == "集团规划"


def test_default_dir_to_kb_covers_all_needed():
    assert DEFAULT_DIR_TO_KB["政策文件"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["政策二"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["规划政策文件"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["政策_会议_讲话汇编"] == "kb_policy_national"
    assert DEFAULT_DIR_TO_KB["集团规划"] == "kb_policy_group"
    assert DEFAULT_DIR_TO_KB["操作指引_常见问题"] == "kb_ops"
    assert DEFAULT_DIR_TO_KB["科小星-操作文档"] == "kb_ops"
    assert DEFAULT_DIR_TO_KB["科研管理制度"] == "kb_regulation"
    assert DEFAULT_DIR_TO_KB["模版"] == "kb_template"
    assert DEFAULT_DIR_TO_KB["集团历史项目-v2"] == "kb_project"


def test_scan_assigns_kb_by_default_mapping():
    inv = scan_kb_dirs(FIXTURE)
    by_doc = {f.doc_name: f.kb for f in inv.files}
    assert by_doc["十四五规划"] == "kb_policy_national"
    assert by_doc["集团科技创新纲要.pdf"] == "kb_policy_group"
    assert by_doc["项目导出.xlsx"] == "kb_project"


def test_scan_dedups_by_content_hash():
    inv = scan_kb_dirs(FIXTURE)
    # 十四五规划.md 在两个目录内容相同 → 只保留一条，另一条进 duplicates
    docs = [f for f in inv.files if f.doc_name == "十四五规划"]
    assert len(docs) == 1
    # duplicates 的主 path 命中保留的那条
    assert any("政策二" in dup for dups in inv.duplicates.values() for dup in dups)


def test_scan_only_collects_md_files():
    inv = scan_kb_dirs(FIXTURE)
    for f in inv.files:
        assert f.path.endswith(".md")
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_scanner.py -v`
Expected: FAIL — `ImportError: cannot import name 'scan_kb_dirs' ...`

- [ ] **Step 4: 实现 `app/kbmap/scanner.py`**

```python
"""目录扫描器：扫 merged/ 下所有 *_<中文名>/ 目录，按 content-hash 去重，
按默认规则映射到 KB，产出 KbInventory。"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# 目录后缀名（去掉 UUID 前缀）→ KB 名；对应 spec §3.3
DEFAULT_DIR_TO_KB: dict[str, str] = {
    "政策文件": "kb_policy_national",
    "政策二": "kb_policy_national",
    "规划政策文件": "kb_policy_national",
    "政策_会议_讲话汇编": "kb_policy_national",
    "集团规划": "kb_policy_group",
    "操作指引_常见问题": "kb_ops",
    "科小星-操作文档": "kb_ops",
    "科研管理制度": "kb_regulation",
    "模版": "kb_template",
    "集团历史项目-v2": "kb_project",
}


@dataclass
class FileRecord:
    path: str          # 相对 merged 根的相对路径
    doc_name: str      # 文件名去 .md
    content_hash: str  # 整文件 SHA-256
    kb: str


@dataclass
class KbInventory:
    files: list[FileRecord] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    # duplicates: 保留的主 path → 同 content-hash 被丢弃的其他 path 列表


def strip_uuid_prefix(dir_name: str) -> str:
    """'aaa1_政策文件' → '政策文件'；无前缀的原样返回。

    UUID 前缀判定：第一个下划线之前的部分含字母数字且像 hash（长度>=8）。
    简化规则：若含 '_' 且第一个 _ 前的部分长度 >= 8，视为 UUID 前缀。
    """
    if "_" in dir_name:
        prefix, _, rest = dir_name.partition("_")
        if len(prefix) >= 8 and rest:
            return rest
    return dir_name


def _doc_name_from(filename: str) -> str:
    # 去掉 .md 后缀；.pdf.md 这种保留 .pdf
    if filename.endswith(".md"):
        filename = filename[:-3]
    return filename


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def scan_kb_dirs(root: Path) -> KbInventory:
    """扫描 root 下所有一级子目录（形如 *_<中文名>/），收集 .md 文件，
    按 content-hash 去重，按 DEFAULT_DIR_TO_KB 映射到 KB。"""
    root = Path(root)
    inv = KbInventory()

    # hash → 主 FileRecord（先到先得，保留更短路径的）
    hash_to_main: dict[str, FileRecord] = {}
    # 收集顺序：按相对路径排序，保证确定性
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for subdir in subdirs:
        kb_name = DEFAULT_DIR_TO_KB.get(strip_uuid_prefix(subdir.name))
        if kb_name is None:
            # 未知目录：跳过（不报错，便于增量）
            continue
        md_files = sorted([p for p in subdir.iterdir() if p.is_file() and p.name.endswith(".md")],
                          key=lambda p: p.name)
        for md in md_files:
            rel = str(md.relative_to(root)).replace("\\", "/")
            ch = _content_hash(md)
            if ch in hash_to_main:
                main = hash_to_main[ch]
                inv.duplicates.setdefault(main.path, []).append(rel)
            else:
                rec = FileRecord(
                    path=rel,
                    doc_name=_doc_name_from(md.name),
                    content_hash=ch,
                    kb=kb_name,
                )
                hash_to_main[ch] = rec
                inv.files.append(rec)
    return inv
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_scanner.py -v`
Expected: PASS（4 passed）

- [ ] **Step 6: 提交**

```bash
git add app/kbmap/scanner.py tests/unit/kbmap/test_scanner.py tests/fixtures/kbmap/
git commit -m "feat(kbmap): 任务2 目录扫描器（去重+默认映射）"
```

---

## Task 3: Manifest 模型与 IO

**Files:**
- Create: `app/kbmap/manifest.py`
- Test: `tests/unit/kbmap/test_manifest.py`

**Interfaces:**
- Consumes: `app.kbmap.scanner.KbInventory`
- Produces:
  - `KbDefine`（pydantic）：`description: str`、`chunker: str | None = None`、`serializer: str | None = None`
  - `FileEntry`（pydantic）：`path: str`、`kb: str`、`doc_name: str`、`duplicates: list[str] = []`
  - `Manifest`（pydantic）：`version: int`、`embedding_model: str`、`kb_defines: dict[str, KbDefine]`、`files: list[FileEntry]`
  - `DEFAULT_KB_DEFINES: dict[str, KbDefine]`
  - `build_manifest_from_inventory(inv, embedding_model="bge-m3") -> Manifest`
  - `load_manifest(path: Path) -> Manifest`
  - `save_manifest(manifest, path: Path) -> None`
  - `validate_manifest(manifest) -> None`（kb 字段必须在 kb_defines；collection 名合规）

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_manifest.py`:

```python
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.kbmap.scanner import scan_kb_dirs, KbInventory
from app.kbmap.manifest import (
    Manifest, build_manifest_from_inventory, load_manifest, save_manifest,
    validate_manifest, DEFAULT_KB_DEFINES,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "kbmap" / "merged"


def test_default_kb_defines_cover_six_kbs():
    assert set(DEFAULT_KB_DEFINES.keys()) == {
        "kb_policy_national", "kb_policy_group", "kb_ops",
        "kb_regulation", "kb_template", "kb_project",
    }
    # kb_project 走 jsonl 切块器
    assert DEFAULT_KB_DEFINES["kb_project"].chunker == "jsonl"
    assert DEFAULT_KB_DEFINES["kb_project"].serializer == "project_natural_language"


def test_build_manifest_from_inventory():
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    assert m.version == 1
    assert m.embedding_model == "bge-m3"
    assert "kb_policy_national" in m.kb_defines
    # 每条 file 的 kb 必须在 kb_defines
    for f in m.files:
        assert f.kb in m.kb_defines


def test_save_load_roundtrip(tmp_path):
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    p = tmp_path / "kb_manifest.yaml"
    save_manifest(m, p)
    m2 = load_manifest(p)
    assert m2 == m


def test_validate_rejects_unknown_kb(tmp_path):
    inv = scan_kb_dirs(FIXTURE)
    m = build_manifest_from_inventory(inv)
    # 篡改：把第一个 file 的 kb 改成不存在的
    m.files[0] = m.files[0].model_copy(update={"kb": "kb_nonexistent"})
    with pytest.raises(ValueError, match="kb_nonexistent"):
        validate_manifest(m)


def test_validate_rejects_bad_collection_name():
    # kb_defines 的 key 必须形如 kb_<小写字母数字下划线>
    m = Manifest(
        version=1, embedding_model="bge-m3",
        kb_defines={"Bad-Name": DEFAULT_KB_DEFINES["kb_ops"]},
        files=[],
    )
    with pytest.raises(ValueError, match="collection 名"):
        validate_manifest(m)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_manifest.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 `app/kbmap/manifest.py`**

```python
"""Manifest pydantic 模型 + YAML IO + 校验。"""
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.kbmap.scanner import KbInventory

# collection 命名规则：kb_ 前缀 + 小写字母/数字/下划线（对齐主 spec §5）
_COLLECTION_RE = re.compile(r"^kb_[a-z0-9_]+$")


class KbDefine(BaseModel):
    description: str
    chunker: str | None = None        # None = FixedChunker；"jsonl" = kb_project 例外
    serializer: str | None = None


class FileEntry(BaseModel):
    path: str
    kb: str
    doc_name: str
    duplicates: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    version: int
    embedding_model: str
    kb_defines: dict[str, KbDefine]
    files: list[FileEntry]


# 默认 KB 元信息（spec §3.2 + §4.1）
DEFAULT_KB_DEFINES: dict[str, KbDefine] = {
    "kb_policy_national": KbDefine(description="国家/部委/省的政策、规划、讲话"),
    "kb_policy_group": KbDefine(description="集团内部规划、办法"),
    "kb_ops": KbDefine(description="信息平台操作指引、常见问题、功能说明"),
    "kb_regulation": KbDefine(description="科研管理制度、办法、通知"),
    "kb_template": KbDefine(description="申报表单模板"),
    "kb_project": KbDefine(
        description="集团历史科研项目（jsonl，一项目一 chunk）",
        chunker="jsonl",
        serializer="project_natural_language",
    ),
}


def build_manifest_from_inventory(
    inv: KbInventory, embedding_model: str = "bge-m3"
) -> Manifest:
    """从扫描结果构造 manifest，附带 duplicates 字段。"""
    # 只保留出现在 inv 中的 KB（避免写出空 KB）
    used_kbs = {f.kb for f in inv.files}
    kb_defines = {k: v for k, v in DEFAULT_KB_DEFINES.items() if k in used_kbs}

    files = []
    # 按 path 排序保证确定性
    dup_map = {k: list(v) for k, v in inv.duplicates.items()}
    for rec in sorted(inv.files, key=lambda r: r.path):
        files.append(FileEntry(
            path=rec.path, kb=rec.kb, doc_name=rec.doc_name,
            duplicates=dup_map.get(rec.path, []),
        ))
    return Manifest(version=1, embedding_model=embedding_model,
                    kb_defines=kb_defines, files=files)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """写 YAML（人类可读，含注释式字段顺序）。"""
    path = Path(path)
    data = {
        "version": manifest.version,
        "embedding_model": manifest.embedding_model,
        "kb_defines": {
            k: v.model_dump(exclude_none=True)
            for k, v in manifest.kb_defines.items()
        },
        "files": [f.model_dump() for f in manifest.files],
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    """读 YAML → Manifest（带 pydantic 校验）。"""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)


def validate_manifest(manifest: Manifest) -> None:
    """语义校验：collection 名合规 + 每条 file.kb 必须在 kb_defines。失败抛 ValueError。"""
    for name in manifest.kb_defines:
        if not _COLLECTION_RE.match(name):
            raise ValueError(
                f"collection 名不合规（需 kb_<小写字母数字下划线>）: {name}"
            )
    for f in manifest.files:
        if f.kb not in manifest.kb_defines:
            raise ValueError(
                f"文件 {f.path} 的 kb='{f.kb}' 不在 kb_defines 中"
            )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_manifest.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/manifest.py tests/unit/kbmap/test_manifest.py
git commit -m "feat(kbmap): 任务3 Manifest 模型与 YAML IO"
```

---

## Task 4: kb_project JSONL 序列化器

**Files:**
- Create: `app/kbmap/project_serializer.py`
- Test: `tests/unit/kbmap/test_project_serializer.py`

**Interfaces:**
- Consumes: 无（独立工具）
- Produces:
  - `serialize_project(obj: dict) -> str`：把单个项目 JSON 序列化成自然语言描述
  - `iter_jsonl_chunks(path: Path) -> list[str]`：读 kb_project 文件，返回每个项目序列化后的文本列表（一项目一 chunk）

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_project_serializer.py`:

```python
from app.kbmap.project_serializer import serialize_project, iter_jsonl_chunks
from tests.fixpaths import FIXTURE_MERGED  # 见 Step 2 注释


def test_serialize_project_basic():
    obj = {
        "立项年份": "2019",
        "项目编号": "JT2019YB15",
        "项目名称": "悬索桥缆索研究",
        "承担单位": "广东省公路建设有限公司",
        "项目负责人": "熊锋",
        "技术领域": "桥梁工程",
        "主要研究内容": "缆索腐蚀防护研究",
        "预期成果指标": "1项发明专利",
    }
    s = serialize_project(obj)
    # 关键事实必须出现在序列化文本里（embedding 输入靠这些召回）
    assert "2019" in s
    assert "JT2019YB15" in s
    assert "悬索桥缆索研究" in s
    assert "广东省公路建设有限公司" in s
    assert "熊锋" in s
    assert "桥梁工程" in s
    assert "缆索腐蚀防护研究" in s


def test_serialize_project_skips_empty_fields():
    obj = {"立项年份": "2019", "项目名称": "X", "承担单位": ""}
    s = serialize_project(obj)
    assert "承担单位" not in s  # 空字段不出现


def test_iter_jsonl_chunks_one_project_per_chunk():
    # fixture 里 项目导出.xlsx.md 有 2 个 JSON 行
    path = FIXTURE_MERGED / "ddd4_集团历史项目-v2" / "项目导出.xlsx.md"
    chunks = iter_jsonl_chunks(path)
    assert len(chunks) == 2
    assert "测试项目一" in chunks[0]
    assert "测试项目二" in chunks[1]
```

- [ ] **Step 2: 建 tests 共享路径常量**

`tests/fixtures/__init__.py`（空）。`tests/fixpaths.py`:

```python
"""测试用 fixture 路径常量。"""
from pathlib import Path

FIXTURE_MERGED = Path(__file__).parent / "fixtures" / "kbmap" / "merged"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_project_serializer.py -v`
Expected: FAIL — ImportError

- [ ] **Step 4: 实现 `app/kbmap/project_serializer.py`**

```python
"""kb_project 的 JSONL → 自然语言序列化。

原文件每行一个项目 JSON（带 <!-- chunk --> 标记行），序列化成自然语言描述
作为 embedding 输入（spec §3.3-3）。
"""
import json
from pathlib import Path

# 序列化字段顺序（影响语义权重，重要的在前）
_FIELDS_ORDER = [
    "立项年份", "技术领域", "项目名称", "项目编号",
    "承担单位", "项目负责人",
    "主要研究内容", "预期成果指标", "实际产出成果",
]


def serialize_project(obj: dict) -> str:
    """把单个项目 dict 序列化成自然语言描述，跳过空字段。

    例：「2019年立项的【桥梁工程】类项目《大跨径...》（编号 JT2019YB15），
    由广东省公路建设有限公司承担，项目负责人熊锋。主要研究内容：…」
    """
    year = obj.get("立项年份", "")
    field_obj = obj.get("技术领域", "")
    name = obj.get("项目名称", "")
    number = obj.get("项目编号", "")
    org = obj.get("承担单位", "")
    leader = obj.get("项目负责人", "")

    parts: list[str] = []
    # 头部：年份 + 领域 + 项目名 + 编号
    head = ""
    if year:
        head += f"{year}年立项的"
    if field_obj:
        head += f"【{field_obj}】类项目"
    if name:
        head += f"《{name}》"
    if number:
        head += f"（编号 {number}）"
    if head:
        parts.append(head)

    if org:
        parts.append(f"由{org}承担")
    if leader:
        parts.append(f"项目负责人{leader}")

    # 正文：其余字段按顺序拼「字段名：值」
    body_fields = [f for f in _FIELDS_ORDER if f not in
                   ("立项年份", "技术领域", "项目名称", "项目编号", "承担单位", "项目负责人")]
    body_parts = []
    for f in body_fields:
        v = obj.get(f)
        if v and str(v).strip():
            body_parts.append(f"{f}：{v}")
    if parts:
        head_text = "，".join(parts) + "。"
    else:
        head_text = ""
    body_text = "；".join(body_parts)
    return (head_text + body_text).strip()


def iter_jsonl_chunks(path: Path) -> list[str]:
    """读 kb_project 文件：跳过 <!-- chunk --> 标记行和空行，每行 JSON
    序列化成一个 chunk 文本。返回顺序与文件一致。"""
    path = Path(path)
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("<!--"):
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunks.append(serialize_project(obj))
    return chunks
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_project_serializer.py -v`
Expected: PASS（3 passed）

- [ ] **Step 6: 提交**

```bash
git add app/kbmap/project_serializer.py tests/unit/kbmap/test_project_serializer.py tests/fixpaths.py tests/fixtures/__init__.py
git commit -m "feat(kbmap): 任务4 kb_project JSONL 自然语言序列化器"
```

---

## Task 5: 文件嵌入助手

**Files:**
- Create: `app/kbmap/embed.py`
- Test: `tests/unit/kbmap/test_embed.py`

**Interfaces:**
- Consumes:
  - `app.kbmap.manifest.Manifest`、`app.kbmap.manifest.FileEntry`
  - `app.kbmap.project_serializer.iter_jsonl_chunks`
  - `app.services.cloud_embedding_service.CloudEmbeddingService`（含子类 `MockCloudEmbeddingService`）
  - `settings.kbmap_chunk_preview_chars`、`settings.kbmap_embed_batch_size`
- Produces:
  - `extract_file_text(entry: FileEntry, root: Path) -> str`：标题 + 首 chunk 前 N 字（kb_project 例外：用 serialize_project 第一个项目）
  - `embed_files(manifest: Manifest, root: Path, embedder) -> dict[str, list[float]]`：path → 向量（按 batch 调 embedder）

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_embed.py`:

```python
from pathlib import Path
import numpy as np
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.embed import extract_file_text, embed_files
from tests.fixpaths import FIXTURE_MERGED


def test_extract_md_file_text_uses_title_and_first_chunk():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    # 找一个 kb_policy_national 的文件（十四五规划 或 交通强国纲要）
    entry = next(f for f in m.files if f.kb == "kb_policy_national")
    txt = extract_file_text(entry, FIXTURE_MERGED)
    # 标题（doc_name）必须出现在文本里
    assert entry.doc_name in txt


def test_extract_kb_project_uses_serializer():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    entry = next(f for f in m.files if f.kb == "kb_project")
    txt = extract_file_text(entry, FIXTURE_MERGED)
    # 用的是第一个项目序列化结果，应含「测试项目一」
    assert "测试项目一" in txt


def test_embed_files_returns_one_vector_per_file():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    vecs = embed_files(m, FIXTURE_MERGED, embedder)
    # 每个 file 一条向量
    assert len(vecs) == len(m.files)
    for v in vecs.values():
        assert len(v) == 1024
    # mock 基于 hash(text) 确定性：相同文本同向量；不同文本大概率不同
    unique_texts = {extract_file_text(f, FIXTURE_MERGED) for f in m.files}
    if len(unique_texts) > 1:
        # 至少有一对向量不同
        vs = list(vecs.values())
        assert any(vs[0] != vs[i] for i in range(1, len(vs)))


def test_embed_files_is_deterministic_same_input_same_output():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    e1 = MockCloudEmbeddingService(dimension=1024)
    e2 = MockCloudEmbeddingService(dimension=1024)
    v1 = embed_files(m, FIXTURE_MERGED, e1)
    v2 = embed_files(m, FIXTURE_MERGED, e2)
    # 同进程内 mock 确定性：两次结果逐位相同
    for path in v1:
        assert v1[path] == v2[path]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_embed.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 `app/kbmap/embed.py`**

```python
"""文件→向量助手。

- 普通 KB（md）：取标题 + 首个 <!-- chunk_idx=0 --> 之后的前 N 字
- kb_project（jsonl）：用 project_serializer 序列化第一个项目作为代表文本
"""
import re
from pathlib import Path

from app.config import settings
from app.kbmap.manifest import Manifest, FileEntry
from app.kbmap.project_serializer import iter_jsonl_chunks

_CHUNK_HEADER_RE = re.compile(r"<!--\s*chunk_idx=0")


def extract_file_text(entry: FileEntry, root: Path) -> str:
    """提取用于 embedding 的文本。"""
    full = Path(root) / entry.path
    text = full.read_text(encoding="utf-8")

    # kb_project：jsonl 文件，用第一个项目序列化结果
    if entry.kb == "kb_project":
        chunks = iter_jsonl_chunks(full)
        if chunks:
            return chunks[0]
        return entry.doc_name  # 兜底

    # 普通 md：标题 + 首个 chunk 之后的前 N 字
    title = entry.doc_name
    # 找首个 chunk_idx=0 标记
    m = _CHUNK_HEADER_RE.search(text)
    body = text[m.end():] if m else text
    # 去掉后续的 chunk 标记行
    body = re.split(r"<!--\s*chunk_idx=\d+", body)[0]
    preview = body.strip()[:settings.kbmap_chunk_preview_chars]
    return f"{title}\n{preview}"


def embed_files(manifest: Manifest, root: Path, embedder) -> dict[str, list[float]]:
    """对 manifest 中每个文件提取文本并 embed，返回 path → 向量(list)。

    按 settings.kbmap_embed_batch_size 分批调 embedder.encode。
    """
    root = Path(root)
    # 按 path 排序保证确定性
    entries = sorted(manifest.files, key=lambda f: f.path)
    texts = [extract_file_text(e, root) for e in entries]

    bs = settings.kbmap_embed_batch_size
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        if not batch:
            continue
        arr = embedder.encode(batch)  # np.ndarray shape [n, dim]
        all_vecs.extend([row.tolist() for row in arr])

    return {e.path: v for e, v in zip(entries, all_vecs)}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_embed.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/embed.py tests/unit/kbmap/test_embed.py
git commit -m "feat(kbmap): 任务5 文件嵌入助手（含 kb_project 例外）"
```

---

## Task 6: 验证指标与报告

**Files:**
- Create: `app/kbmap/metrics.py`
- Test: `tests/unit/kbmap/test_metrics.py`

**Interfaces:**
- Consumes:
  - `app.kbmap.manifest.Manifest`
  - `app.kbmap.embed.embed_files`
  - `app.services.cloud_embedding_service.CloudEmbeddingService`
  - `settings.kbmap_outlier_margin`、`settings.kbmap_cohesion_min`、`settings.kbmap_separation_max`
- Produces:
  - `cosine(a: list[float], b: list[float]) -> float`
  - `compute_centroids(manifest, path_to_vec) -> dict[str, list[float]]`（KB 名 → 质心）
  - `VerifyReport`（dataclass）：`cohesion: dict[str, float]`、`separation: dict[tuple[str,str], float]`、`outliers: list[Outlier]`
  - `Outlier`（dataclass）：`path: str`、`assigned_kb: str`、`nearest_other: str`、`margin: float`
  - `verify(manifest, root, embedder) -> VerifyReport`
  - `write_report_md(report: VerifyReport, manifest, out_path: Path) -> None`

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_metrics.py`:

```python
import math
from app.services.cloud_embedding_service import MockCloudEmbeddingService
from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.metrics import cosine, compute_centroids, verify
from tests.fixpaths import FIXTURE_MERGED


def test_cosine_identical_vectors_is_one():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    assert abs(cosine([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6


def test_compute_centroids_returns_one_per_kb():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    from app.kbmap.embed import embed_files
    embedder = MockCloudEmbeddingService(dimension=1024)
    p2v = embed_files(m, FIXTURE_MERGED, embedder)
    cents = compute_centroids(m, p2v)
    # 每个 KB 一个质心
    assert set(cents.keys()) == {f.kb for f in m.files}
    for c in cents.values():
        assert len(c) == 1024


def test_verify_returns_report_structure():
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    rep = verify(m, FIXTURE_MERGED, embedder)
    # cohesion 对每个 KB 有值
    assert set(rep.cohesion.keys()) == {f.kb for f in m.files}
    # outliers 是列表（可能为空，fixture 小）
    assert isinstance(rep.outliers, list)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_metrics.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 `app/kbmap/metrics.py`**

```python
"""分类验证：类内凝聚度 / 类间分离度 / 异常文件 + 报告生成。"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import settings
from app.kbmap.manifest import Manifest
from app.kbmap.embed import embed_files


@dataclass
class Outlier:
    path: str
    assigned_kb: str
    nearest_other: str
    margin: float          # 到本库质心余弦 - 到最近他库质心余弦


@dataclass
class VerifyReport:
    cohesion: dict[str, float] = field(default_factory=dict)              # kb -> 均值
    separation: dict[tuple[str, str], float] = field(default_factory=dict)  # (kb1,kb2) -> 余弦
    outliers: list[Outlier] = field(default_factory=list)


def cosine(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度。"""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va) + 1e-8
    nb = np.linalg.norm(vb) + 1e-8
    return float(np.dot(va, vb) / (na * nb))


def compute_centroids(
    manifest: Manifest, path_to_vec: dict[str, list[float]]
) -> dict[str, list[float]]:
    """按 KB 聚合：每 KB 内文件向量取均值 → 质心。"""
    by_kb: dict[str, list[list[float]]] = {}
    for f in manifest.files:
        by_kb.setdefault(f.kb, []).append(path_to_vec[f.path])
    cents: dict[str, list[float]] = {}
    for kb, vecs in by_kb.items():
        arr = np.asarray(vecs, dtype=np.float32)
        cents[kb] = arr.mean(axis=0).tolist()
    return cents


def verify(manifest: Manifest, root: Path, embedder) -> VerifyReport:
    """跑完整验证。读 manifest（只读），算三指标。"""
    root = Path(root)
    p2v = embed_files(manifest, root, embedder)
    cents = compute_centroids(manifest, p2v)
    rep = VerifyReport()

    # 类内凝聚度：每文件到本库质心余弦，取 KB 内均值
    coh: dict[str, list[float]] = {}
    for f in manifest.files:
        sim = cosine(p2v[f.path], cents[f.kb])
        coh.setdefault(f.kb, []).append(sim)
    rep.cohesion = {kb: sum(vs) / len(vs) for kb, vs in coh.items()}

    # 类间分离度：KB 质心两两余弦
    kbs = sorted(cents.keys())
    for i, k1 in enumerate(kbs):
        for k2 in kbs[i + 1:]:
            rep.separation[(k1, k2)] = cosine(cents[k1], cents[k2])

    # 异常文件：到本库质心余弦 - 到最近他库质心余弦 < margin
    margin = settings.kbmap_outlier_margin
    for f in manifest.files:
        own = cosine(p2v[f.path], cents[f.kb])
        best_other = None
        best_other_sim = -1.0
        for kb, c in cents.items():
            if kb == f.kb:
                continue
            s = cosine(p2v[f.path], c)
            if s > best_other_sim:
                best_other_sim = s
                best_other = kb
        if best_other is None:
            continue
        diff = own - best_other_sim
        if diff < margin:
            rep.outliers.append(Outlier(
                path=f.path, assigned_kb=f.kb,
                nearest_other=best_other, margin=diff,
            ))
    # 按 margin 升序
    rep.outliers.sort(key=lambda o: o.margin)
    return rep


def write_report_md(
    report: VerifyReport, manifest: Manifest, out_path: Path
) -> None:
    """写人类可读 markdown 报告。"""
    out_path = Path(out_path)
    lines: list[str] = ["# KB 分类验证报告", ""]

    lines.append("## 总览")
    lines.append("| KB | 文件数 | 类内凝聚度 |")
    lines.append("|----|--------|------------|")
    count_by_kb: dict[str, int] = {}
    for f in manifest.files:
        count_by_kb[f.kb] = count_by_kb.get(f.kb, 0) + 1
    for kb in sorted(report.cohesion):
        lines.append(f"| {kb} | {count_by_kb.get(kb, 0)} | {report.cohesion[kb]:.4f} |")
    lines.append("")

    lines.append("## 类间分离度矩阵（KB 质心两两余弦，越小越好）")
    kbs = sorted(report.cohesion)
    lines.append("| | " + " | ".join(kbs) + " |")
    lines.append("|---|" + "|".join(["---"] * len(kbs)) + "|")
    for k1 in kbs:
        row = [k1]
        for k2 in kbs:
            if k1 == k2:
                row.append("1.00")
            elif (k1, k2) in report.separation:
                row.append(f"{report.separation[(k1, k2)]:.2f}")
            else:
                row.append(f"{report.separation[(k2, k1)]:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"## 异常文件（margin < {settings.kbmap_outlier_margin}）")
    if not report.outliers:
        lines.append("无异常文件。")
    else:
        lines.append("| 文件 | 当前归属 | 最近他库 | margin |")
        lines.append("|------|----------|----------|--------|")
        for o in report.outliers:
            lines.append(
                f"| {o.path} | {o.assigned_kb} | {o.nearest_other} | {o.margin:+.4f} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_metrics.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/metrics.py tests/unit/kbmap/test_metrics.py
git commit -m "feat(kbmap): 任务6 分类验证指标与报告生成"
```

---

## Task 7: KB 质心构建与落盘

**Files:**
- Create: `app/kbmap/centroids.py`
- Test: `tests/unit/kbmap/test_centroids.py`

**Interfaces:**
- Consumes:
  - `app.kbmap.manifest.Manifest`、`app.kbmap.manifest.load_manifest`
  - `app.kbmap.embed.embed_files`
  - `app.kbmap.metrics.compute_centroids`
  - `app.services.cloud_embedding_service.CloudEmbeddingService`
- Produces:
  - `build_centroids(manifest: Manifest, root: Path, embedder, out_dir: Path) -> None`：写 `kb_centroids.npy`（shape [KB数, dim]）+ `kb_names.json`（行号→kb名）
  - `load_centroids(out_dir: Path) -> tuple[np.ndarray, list[str]]`：返回（质心矩阵, kb 名列表，行号对应矩阵行）

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_centroids.py`:

```python
import json
from pathlib import Path
import numpy as np
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.centroids import build_centroids, load_centroids
from tests.fixpaths import FIXTURE_MERGED


def test_build_and_load_centroids_roundtrip(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(dimension=1024)
    build_centroids(m, FIXTURE_MERGED, embedder, tmp_path)

    assert (tmp_path / "kb_centroids.npy").exists()
    assert (tmp_path / "kb_names.json").exists()

    mat, names = load_centroids(tmp_path)
    # 每个 KB 一行
    assert mat.shape[0] == len(names)
    assert mat.shape[1] == 1024
    assert set(names) == {f.kb for f in m.files}


def test_centroids_are_deterministic(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    build_centroids(m, FIXTURE_MERGED, MockCloudEmbeddingService(1024), tmp_path / "a")
    build_centroids(m, FIXTURE_MERGED, MockCloudEmbeddingService(1024), tmp_path / "b")
    mat_a, names_a = load_centroids(tmp_path / "a")
    mat_b, names_b = load_centroids(tmp_path / "b")
    assert names_a == names_b
    assert np.allclose(mat_a, mat_b)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_centroids.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 `app/kbmap/centroids.py`**

```python
"""KB 质心构建与落盘。

质心来源：manifest 冻结后，每文件取 embed 文本，按 KB 取均值。
落盘 kb_centroids.npy（shape [KB数, dim]）+ kb_names.json（行号→kb名）。
"""
import json
from pathlib import Path

import numpy as np

from app.kbmap.manifest import Manifest
from app.kbmap.embed import embed_files
from app.kbmap.metrics import compute_centroids


def build_centroids(
    manifest: Manifest, root: Path, embedder, out_dir: Path
) -> None:
    """构建 KB 质心并落盘到 out_dir。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p2v = embed_files(manifest, Path(root), embedder)
    cents = compute_centroids(manifest, p2v)

    # 按字典序固定行号顺序（确定性）
    names = sorted(cents.keys())
    mat = np.asarray([cents[n] for n in names], dtype=np.float32)

    np.save(out_dir / "kb_centroids.npy", mat)
    (out_dir / "kb_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_centroids(in_dir: Path) -> tuple[np.ndarray, list[str]]:
    """读质心矩阵与 kb 名列表（行号对应矩阵行）。"""
    in_dir = Path(in_dir)
    mat = np.load(in_dir / "kb_centroids.npy")
    names = json.loads((in_dir / "kb_names.json").read_text(encoding="utf-8"))
    return mat, names
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_centroids.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/centroids.py tests/unit/kbmap/test_centroids.py
git commit -m "feat(kbmap): 任务7 KB 质心构建与落盘"
```

---

## Task 8: KBClassifier 分类器

**Files:**
- Create: `app/kbmap/classifier.py`
- Test: `tests/unit/kbmap/test_classifier.py`

**Interfaces:**
- Consumes:
  - `app.kbmap.centroids.load_centroids`
  - `app.kbmap.metrics.cosine`
  - `app.services.cloud_embedding_service.CloudEmbeddingService`
- Produces:
  - `class KBClassifier`：
    - `__init__(self, centroids_dir: Path, embedder)`
    - `classify(self, text: str, top_k: int = 3) -> list[tuple[str, float]]`
    - `assign_file(self, path: str, root: Path, top_k: int = 3) -> list[tuple[str, float]]`（用 `extract_file_text` 取文本）

- [ ] **Step 1: 写失败测试**

`tests/unit/kbmap/test_classifier.py`:

```python
from pathlib import Path
from app.services.cloud_embedding_service import MockCloudEmbeddingService

from app.kbmap.scanner import scan_kb_dirs
from app.kbmap.manifest import build_manifest_from_inventory
from app.kbmap.centroids import build_centroids
from app.kbmap.classifier import KBClassifier
from tests.fixpaths import FIXTURE_MERGED


def _build(tmp_path):
    inv = scan_kb_dirs(FIXTURE_MERGED)
    m = build_manifest_from_inventory(inv)
    embedder = MockCloudEmbeddingService(1024)
    build_centroids(m, FIXTURE_MERGED, embedder, tmp_path)
    return m, embedder


def test_classify_returns_top_k_with_scores(tmp_path):
    m, embedder = _build(tmp_path)
    clf = KBClassifier(tmp_path, embedder)
    res = clf.classify("桥梁工程研究", top_k=3)
    assert len(res) <= 3
    assert all(isinstance(k, str) for k, _ in res)
    assert all(isinstance(s, float) for _, s in res)
    # 分数按降序
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_classify_is_deterministic(tmp_path):
    m, embedder = _build(tmp_path)
    clf1 = KBClassifier(tmp_path, MockCloudEmbeddingService(1024))
    clf2 = KBClassifier(tmp_path, MockCloudEmbeddingService(1024))
    r1 = clf1.classify("同一问题")
    r2 = clf2.classify("同一问题")
    assert r1 == r2  # 同进程 mock 确定性


def test_assign_file_returns_ranking(tmp_path):
    m, embedder = _build(tmp_path)
    clf = KBClassifier(tmp_path, embedder)
    # 任挑一个 fixture 文件
    entry = m.files[0]
    res = clf.assign_file(entry.path, FIXTURE_MERGED, top_k=2)
    assert len(res) <= 2
    # 分数降序
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_classifier.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: 实现 `app/kbmap/classifier.py`**

```python
"""KB 分类器：基于 KB 质心 + bge-m3，对问题或文件判归属。

确定性：质心落盘后固定，classify 仅做一次 embed + 余弦比较，无随机性。
"""
from pathlib import Path

import numpy as np

from app.kbmap.centroids import load_centroids
from app.kbmap.metrics import cosine
from app.kbmap.embed import extract_file_text
from app.kbmap.manifest import FileEntry


class KBClassifier:
    """基于 KB 质心的分类器。"""

    def __init__(self, centroids_dir: Path, embedder):
        self._mat, self._names = load_centroids(Path(centroids_dir))
        self._embedder = embedder

    def classify(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """对文本返回 top-k 候选 KB 及与其质心的余弦得分（降序）。"""
        vec = self._embedder.encode([text])[0].tolist()
        scores = [(self._names[i], float(cosine(vec, self._mat[i].tolist())))
                  for i in range(len(self._names))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def assign_file(
        self, path: str, root: Path, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """对单个文件判归属：用 extract_file_text 取文本后分类。"""
        # 用 FileEntry 占位取文本（kb 字段不影响 extract_file_text 的 md 分支；
        # kb_project 分支靠 kb=='kb_project' 判定，故这里需还原 kb）
        # 简化：直接根据 path 是否在 kb_project 目录判定 kb
        kb = "kb_project" if "集团历史项目" in path else "_md"
        entry = FileEntry(path=path, kb=kb, doc_name=Path(path).name)
        text = extract_file_text(entry, Path(root))
        return self.classify(text, top_k=top_k)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_classifier.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add app/kbmap/classifier.py tests/unit/kbmap/test_classifier.py
git commit -m "feat(kbmap): 任务8 KBClassifier 分类器"
```

---

## Task 9: CLI 入口（scan / verify / build-centroids / classify / assign）

**Files:**
- Create: `app/kbmap/__main__.py`
- Test: `tests/unit/kbmap/test_cli.py`

**Interfaces:**
- Consumes: Task 2–8 全部产物；`app.config.settings`
- Produces: `python -m app.kbmap <subcommand>` 的 5 个子命令

子命令：
- `scan --root <merged> --out <kb_manifest.yaml> [--draft <kb_taxonomy_draft.md>]`：扫描 + 写 manifest（+可选草案 md）
- `verify --manifest <yaml> --root <merged> --out <kb_verify_report.md>`：跑验证
- `build-centroids --manifest <yaml> --root <merged> --out-dir <dir>`：建质心
- `classify "<问题>" [--top-k 3] [--centroids-dir <dir>]`：分类文本
- `assign <文件相对路径> --root <merged> [--top-k 3]`：分类文件

- [ ] **Step 1: 写失败测试（用 subprocess 调 CLI）**

`tests/unit/kbmap/test_cli.py`:

```python
import os
import subprocess
import sys
from pathlib import Path
import yaml

from tests.fixpaths import FIXTURE_MERGED

_MOCK_ENV = {**os.environ, "KBMAP_EMBEDDER": "mock"}


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "app.kbmap", *args],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )


def test_cli_scan_writes_manifest(tmp_path):
    out = tmp_path / "kb_manifest.yaml"
    r = _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["embedding_model"] == "bge-m3"
    assert len(data["files"]) > 0
    # 每条 file 的 kb 在 kb_defines
    for f in data["files"]:
        assert f["kb"] in data["kb_defines"]


def test_cli_verify_writes_report(tmp_path):
    manifest = tmp_path / "kb_manifest.yaml"
    report = tmp_path / "kb_verify_report.md"
    _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(manifest))
    # verify 用 mock embedder（通过 KBMAP_EMBEDDER=mock 环境变量切换）
    r = _run("verify", "--manifest", str(manifest),
             "--root", str(FIXTURE_MERGED), "--out", str(report), env=_MOCK_ENV)
    assert r.returncode == 0, r.stderr
    assert report.exists()
    txt = report.read_text(encoding="utf-8")
    assert "类内凝聚度" in txt


def test_cli_build_centroids_and_classify(tmp_path):
    manifest = tmp_path / "kb_manifest.yaml"
    cents_dir = tmp_path / "cents"
    _run("scan", "--root", str(FIXTURE_MERGED), "--out", str(manifest))
    r = _run("build-centroids", "--manifest", str(manifest),
             "--root", str(FIXTURE_MERGED), "--out-dir", str(cents_dir), env=_MOCK_ENV)
    assert r.returncode == 0, r.stderr
    assert (cents_dir / "kb_centroids.npy").exists()

    r2 = _run("classify", "桥梁工程研究", "--centroids-dir", str(cents_dir), env=_MOCK_ENV)
    assert r2.returncode == 0, r2.stderr
    # 输出含至少一个 KB 名
    assert "kb_" in r2.stdout
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/unit/kbmap/test_cli.py -v`
Expected: FAIL — `No module named app.kbmap.__main__`

- [ ] **Step 3: 实现 `app/kbmap/__main__.py`**

```python
"""kbmap CLI 入口。

子命令：scan / verify / build-centroids / classify / assign
通过环境变量 KBMAP_EMBEDDER=mock 切换到 MockCloudEmbeddingService（测试用）。
"""
import argparse
import os
import sys
from pathlib import Path

from app.config import settings


def _make_embedder():
    """根据 KBMAP_EMBEDDER 环境变量返回 embedder。默认真实 CloudEmbeddingService。"""
    if os.environ.get("KBMAP_EMBEDDER") == "mock":
        from app.services.cloud_embedding_service import MockCloudEmbeddingService
        return MockCloudEmbeddingService(dimension=1024)
    from app.services.cloud_embedding_service import CloudEmbeddingService
    try:
        return CloudEmbeddingService()
    except Exception as e:
        sys.stderr.write(f"[kbmap] 无法初始化 CloudEmbeddingService：{e}\n")
        sys.stderr.write("[kbmap] 测试可设 KBMAP_EMBEDDER=mock 用 Mock。\n")
        raise


def cmd_scan(args):
    from app.kbmap.scanner import scan_kb_dirs
    from app.kbmap.manifest import build_manifest_from_inventory, save_manifest, validate_manifest
    root = Path(args.root)
    inv = scan_kb_dirs(root)
    m = build_manifest_from_inventory(inv)
    validate_manifest(m)
    save_manifest(m, Path(args.out))
    print(f"[kbmap] scan 完成：{len(m.files)} 个文件 → {args.out}")
    if args.draft:
        _write_draft_md(m, inv, Path(args.draft))


def _write_draft_md(manifest, inventory, path):
    """写人类可读草案 markdown（供用户编辑）。"""
    from collections import Counter
    cnt = Counter(f.kb for f in manifest.files)
    lines = ["# KB 分类体系草案（待用户编辑）", "",
             "## 1. KB 清单", "| KB | 文件数 | 描述 |",
             "|----|--------|------|"]
    for kb in sorted(manifest.kb_defines):
        lines.append(f"| {kb} | {cnt.get(kb, 0)} | {manifest.kb_defines[kb].description} |")
    lines += ["", "## 2. 去重记录（同 content-hash 被合并）"]
    if not inventory.duplicates:
        lines.append("无去重。")
    else:
        lines += ["| 保留 | 被丢弃（同内容） |", "|------|------------------|"]
        for main, dups in sorted(inventory.duplicates.items()):
            lines.append(f"| {main} | {', '.join(dups)} |")
    lines += ["", "## 3. 含混文件标记（需用户裁决）",
              "- 请用 `python -m app.kbmap verify` 跑验证后参考异常文件清单调整。"]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[kbmap] 草案写入 {path}")


def cmd_verify(args):
    from app.kbmap.manifest import load_manifest, validate_manifest
    from app.kbmap.metrics import verify, write_report_md
    m = load_manifest(Path(args.manifest))
    validate_manifest(m)
    embedder = _make_embedder()
    rep = verify(m, Path(args.root), embedder)
    write_report_md(rep, m, Path(args.out))
    print(f"[kbmap] verify 完成 → {args.out}")
    print(f"[kbmap] 凝聚度: { {k: round(v,3) for k,v in rep.cohesion.items()} }")
    print(f"[kbmap] 异常文件数: {len(rep.outliers)}")


def cmd_build_centroids(args):
    from app.kbmap.manifest import load_manifest, validate_manifest
    from app.kbmap.centroids import build_centroids
    m = load_manifest(Path(args.manifest))
    validate_manifest(m)
    embedder = _make_embedder()
    build_centroids(m, Path(args.root), embedder, Path(args.out_dir))
    print(f"[kbmap] 质心写入 {args.out_dir}")


def cmd_classify(args):
    from app.kbmap.classifier import KBClassifier
    embedder = _make_embedder()
    clf = KBClassifier(Path(args.centroids_dir), embedder)
    res = clf.classify(args.text, top_k=args.top_k)
    for kb, score in res:
        print(f"{kb}\t{score:.4f}")


def cmd_assign(args):
    from app.kbmap.classifier import KBClassifier
    embedder = _make_embedder()
    clf = KBClassifier(Path(args.centroids_dir), embedder)
    res = clf.assign_file(args.path, Path(args.root), top_k=args.top_k)
    for kb, score in res:
        print(f"{kb}\t{score:.4f}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m app.kbmap", description="KB 分类工具链")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="扫描目录生成 manifest")
    s.add_argument("--root", required=True, help="merged/ 根目录")
    s.add_argument("--out", required=True, help="输出 manifest yaml 路径")
    s.add_argument("--draft", help="（可选）输出人类可读草案 md 路径")
    s.set_defaults(func=cmd_scan)

    v = sub.add_parser("verify", help="跑分类验证")
    v.add_argument("--manifest", required=True)
    v.add_argument("--root", required=True)
    v.add_argument("--out", required=True, help="输出报告 md 路径")
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("build-centroids", help="构建 KB 质心")
    b.add_argument("--manifest", required=True)
    b.add_argument("--root", required=True)
    b.add_argument("--out-dir", required=True)
    b.set_defaults(func=cmd_build_centroids)

    c = sub.add_parser("classify", help="分类一段文本")
    c.add_argument("text", help="要分类的文本（如用户问题）")
    c.add_argument("--centroids-dir", required=True)
    c.add_argument("--top-k", type=int, default=3)
    c.set_defaults(func=cmd_classify)

    a = sub.add_parser("assign", help="分类单个文件")
    a.add_argument("path", help="相对 root 的文件路径")
    a.add_argument("--root", required=True)
    a.add_argument("--centroids-dir", required=True)
    a.add_argument("--top-k", type=int, default=3)
    a.set_defaults(func=cmd_assign)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/unit/kbmap/test_cli.py -v`
Expected: PASS（3 passed）。

- [ ] **Step 5: 全量跑 kbmap 测试**

Run: `pytest tests/unit/kbmap/ -v`
Expected: 所有 kbmap 测试 PASS（约 30 passed）

- [ ] **Step 6: 提交**

```bash
git add app/kbmap/__main__.py tests/unit/kbmap/test_cli.py
git commit -m "feat(kbmap): 任务9 CLI 入口（scan/verify/build-centroids/classify/assign）"
```

---

## 范围外（推迟）

以下属 spec 范围但**不在本计划**，需主 spec 的 `app/ingest/` 实现后再做：

1. **入库流水线 `--manifest` 模式**（spec §6.2）：主 spec 的 `app/ingest/` 尚未实现，本计划的 `kb_project` JSONL 序列化器（`app/kbmap/project_serializer.py`）已就位，待 ingest 实现时直接复用 `iter_jsonl_chunks` 作为 `chunker=jsonl` 的实现。
2. **真实 `merged/` 数据执行 P1→P2→P3**：工具链就位后，由用户/Claude 协作跑 `scan` → 编辑草案/manifest → `verify` → 迭代 → `build-centroids`，产出 `docs/kb_manifest.yaml`、`docs/kb_verify_report.md`、`docs/kb_taxonomy_draft.md`。这是**数据工作流**，非代码任务，不在本 TDD 计划内。

## 完工验收（对应 spec §9）

- 所有 `tests/unit/kbmap/` 通过。
- `python -m app.kbmap scan --root <真实 merged>` 能产出合法 manifest。
- `python -m app.kbmap verify ...` 能产出报告（真实 embedder 需配置 `CLOUD_EMBEDDING_URL` 等）。
- `KBClassifier.classify` 对同一问题多次调用结果完全一致（同进程 mock 测试已覆盖；真实 bge-m3 天然确定性）。
