"""分类验证：类内凝聚度 / 类间分离度 / 异常文件 + 报告生成。"""
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
