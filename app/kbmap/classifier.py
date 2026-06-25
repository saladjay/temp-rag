"""KB 分类器：基于 KB 质心 + bge-m3，对问题或文件判归属。

确定性：质心落盘后固定，classify 仅做一次 embed + 余弦比较，无随机性。
"""
from pathlib import Path

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
        # dev-only heuristic: _md sentinel routes to md-text branch of extract_file_text; only kb_project needs the JSONL branch
        kb = "kb_project" if "集团历史项目" in path else "_md"
        entry = FileEntry(path=path, kb=kb, doc_name=Path(path).name)
        text = extract_file_text(entry, Path(root))
        return self.classify(text, top_k=top_k)
