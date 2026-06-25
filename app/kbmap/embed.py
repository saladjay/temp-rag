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
