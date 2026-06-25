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
