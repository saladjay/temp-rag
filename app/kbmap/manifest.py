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
