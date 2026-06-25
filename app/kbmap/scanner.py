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
    """'<uuid>_<中文名>' → '<中文名>'；无 UUID 前缀的原样返回。

    UUID 前缀判定：第一个下划线之前的部分全为十六进制字符（真实 UUID 是 32-hex）。
    这样既能剥真实 UUID（0352fdaa...）、fixture 短 hex（aaa1），
    又不会误伤本身含下划线的 KB 名（如「操作指引_常见问题」「政策_会议_讲话汇编」）。
    """
    if "_" in dir_name:
        prefix, _, rest = dir_name.partition("_")
        if rest and prefix.isalnum() and all(c in "0123456789abcdefABCDEF" for c in prefix):
            return rest
    return dir_name


def _doc_name_from(filename: str) -> str:
    # 去掉 .md 后缀；.pdf.md 这种保留 .pdf
    if filename.endswith(".md"):
        filename = filename[:-3]
    return filename


def _content_hash(path: Path) -> str:
    """正文哈希：先剥掉 `<!-- chunk ... -->` 标记行再 SHA-256。

    整文件哈希会把「同一文档的不同导出」当成不同文件（chunk_id/时间戳不同），
    导致 13 对政策文档漏过去重。剥掉标记行后，正文相同即判为重复。
    """
    text = path.read_text(encoding="utf-8")
    body_lines = [ln for ln in text.splitlines()
                  if not ln.strip().startswith("<!-- chunk")]
    body = "\n".join(body_lines)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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
